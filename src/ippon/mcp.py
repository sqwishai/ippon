"""`ippon mcp`: ippon's tools for coding agents, over the Model Context Protocol on stdio.

    claude mcp add ippon -- ippon mcp kettei-core

Tools: `decide` (a System One request answered, with how it was answered), `models` (the models of the server
`decide` uses, and every named build, with what is downloaded here), `recommend` and `lint`. An MCP host starts one
stdio server per client (each agent session, each editor window), so this one loads no model and never imports MLX:
`decide` and `models` ask over HTTP the local `ippon serve` that runs this command's models as it names them
(`daemon.find`), and `decide` starts one in the background when none does, so every client of the same models
shares one server and its loaded models. Nothing is downloaded from here: a model not on this Mac yet is an error
naming the `ippon pull` to run, so a download of gigabytes stays the user's call.

The protocol is JSON-RPC 2.0, one message per line on stdin and stdout (MCP's stdio transport), and the server
answers `initialize`, `ping`, `tools/list` and `tools/call`. This tools-only stdio implementation uses the standard
library, keeping client startup independent of the optional SDK's web-server and validation dependencies. A tool's
failure is its result (`isError`, with the error's `type` and exit code) so that an agent can correct its call; an
unknown tool or method is a JSON-RPC error. Each tool call runs on a thread of its own, so an agent's parallel
`decide` calls reach the server together and are answered in shared passes.
"""

from __future__ import annotations

import base64
import contextlib
import json
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import IO, Any

from . import __version__, daemon, lint, recommend
from .device import Device, macos_version, max_body, physical_memory
from .errors import DecisionError, InvalidRequest, as_decision_error
from .locate import downloaded
from .media.decode import MediaKind, as_media
from .names import listing
from .options import precision_argv
from .schema import REQUEST, parse_request, read_json

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")  # newest first
WORKERS = 16  # tool calls answered at once; more wait for one to finish
INSTRUCTIONS = (
    "A decision model running on this Mac through ippon answers typed questions about a state (a ticket, a diff, "
    "a log, an agent's trace): a probability for every option and no generated text. "
    "Write a request, check it with `lint`, answer it with `decide`; `models` and `recommend` say "
    "which models this Mac has and fits. Clients of the same model share one local server and its loaded models."
)

EXAMPLE: dict[str, Any] = {
    "state": "Order #48213 was charged twice. Please reverse the second charge before Friday.",
    "questions": {
        "refund": {"type": "noul", "instructions": "Is the customer asking for money back?"},
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this message?",
            "criteria": {
                "billing": "Charges, refunds and invoices.",
                "shipping": "Delivery, tracking and returns.",
                "other": "Anything else.",
            },
        },
    },
}
MODEL: dict[str, Any] = {
    "type": "string",
    "description": "The model to ask: a short name (kettei-core, kettei-core:q8), a Hub id or a folder; `models` "
    "lists them. Leave it out for the model this server was started for, else the shared server's first.",
}
ARGUMENTS = {**REQUEST, "model": MODEL}  # a System One request (schema.REQUEST), its model as this server takes it
for _field, _what in (
    ("images", "Images as local file paths, base64 strings, or data: URIs."),
    ("videos", "Videos as local file paths, base64 strings, or data: URIs."),
    ("audio", "One audio clip or a list of clips, as local file paths, base64 strings, or data: URIs."),
):
    if _field in ARGUMENTS:
        ARGUMENTS[_field] = {**ARGUMENTS[_field], "description": _what}


def _load_local_media(value: object, kind: MediaKind) -> object:
    """Turns a local file path passed by a local MCP host into base64 for the server; leaves base64 strings, data:
    URIs, URLs and non-strings for parse_request to validate."""
    if not isinstance(value, str) or not value or value.startswith("data:") or "://" in value:
        return value
    expanded = Path(value).expanduser()
    try:
        found = expanded.is_file()
    except OSError:  # a name too long for a file, as base64 often is
        return value
    # base64 can start like a path (every JPEG's starts with "/9j/"): a path only when it is a file, or when it reads
    # as one and not as base64 (a path to a missing file, said as such)
    if not found and not (value.startswith(("/", "~/", "./", "../")) and not _base64(value)):
        return value
    media = as_media(expanded, kind)
    return base64.b64encode(media.data).decode("ascii")


def _base64(value: str) -> bool:
    try:
        base64.b64decode(value, validate=True)
    except ValueError:  # binascii.Error
        return False
    return True


def _resolve_media_paths(body: dict[str, Any]) -> dict[str, Any]:
    """Reads any local file paths given in `images`, `videos` or `audio` into base64 so MCP clients can pass photos,
    videos or recordings by path without encoding them into the model's context."""
    out = dict(body)
    fields: tuple[tuple[str, MediaKind], ...] = (("images", "image"), ("videos", "video"))
    for field_name, kind in fields:
        if isinstance(out.get(field_name), list):
            out[field_name] = [_load_local_media(item, kind) for item in out[field_name]]
    audio = out.get("audio")
    if isinstance(audio, str):
        out["audio"] = _load_local_media(audio, "audio")
    elif isinstance(audio, list):
        out["audio"] = [_load_local_media(item, "audio") for item in audio]
    return out


STATES: dict[str, Any] = {
    "type": "array",
    "items": {"type": "string"},
    "minItems": 1,
    "description": "Many states to ask the same questions, in place of `state`: one call asks them all, with no "
    "round trip through you for each. Returns `results`, one per state in order. For "
    "states already in a file, never copy them here: run `ippon decide MODEL request.json --states FILE > "
    "answers.jsonl` in a shell, which reads the file itself and asks the same server.",
}


def _tools() -> list[dict[str, Any]]:
    quiet = {"readOnlyHint": True, "openWorldHint": False}
    return [
        {
            "name": "decide",
            "title": "Decide",
            "description": "Answer typed questions about a state with a local decision model: a probability for "
            "every option, no generated text. For routing, triage, classification, moderation and yes/no checks an "
            "agent makes again and again. Returns System One's response: `answers` by question id (noul: `noul`, "
            "P(yes); choice: `choice`, its `confidence` and every option's `probabilities`; score: the expected "
            "level `score`, `confidence`, `legend`, `probabilities`), `usage`, `truncated` when input was cut, and "
            "`diagnostics` (the build and commit that answered, precision, the fallback's reason). Run `lint` on a "
            "new request first. The first call may start a local server and load the model (seconds). To ask the same "
            "questions about many states (every commit, every ticket), give `states` in one call, not a call each: "
            "it returns {results: [{answers} or {error}, in order], usage, diagnostics}. States that sit in a file "
            "go through the shell instead (`ippon decide MODEL request.json --states FILE`), not typed out here.",
            "inputSchema": {
                "type": "object",
                "properties": {**ARGUMENTS, "states": STATES},
                "required": ["questions"],
                "additionalProperties": False,
                "examples": [{"model": "kettei-core", **EXAMPLE}],
            },
            "annotations": {**quiet, "readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
        },
        {
            "name": "models",
            "title": "Models",
            "description": "What this Mac can answer with: the local server `decide` uses (its address, how it keeps "
            "models ready, and each model it serves: loaded or not, its build, precision, memory and idle time) and "
            "every named build ippon knows (its size, what has been checked of it, the media it reads, and whether "
            "it is downloaded here). Call it to choose `decide`'s model.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            "annotations": quiet,
        },
        {
            "name": "recommend",
            "title": "Recommend a model",
            "description": "Which model to start with for a kind of task on this Mac, before downloading: each "
            "family's build for it, its weights, what has been checked of it and whether it fits this Mac, best "
            "first. It doesn't say which answers the task best: `ippon eval` measures that on labelled "
            "examples.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "enum": list(recommend.TASKS), "description": "What the requests carry."}
                },
                "additionalProperties": False,
            },
            "annotations": quiet,
        },
        {
            "name": "lint",
            "title": "Lint a request",
            "description": "What to change in a request before a model answers it: an empty state; a question with "
            "no instructions, too few words, several questions in one, or a yes/no question asked as an open one; "
            "two questions asking the same; options written twice or overlapping; a choice of one option, or of "
            "three or more with no catch-all. Takes `decide`'s arguments; answers {problems: [{question, message}], "
            "ok}. A model's token counts: `ippon lint FILE --model M` on the command line.",
            "inputSchema": {
                "type": "object",
                "properties": ARGUMENTS,
                "required": ["questions"],
                "additionalProperties": False,
                "examples": [EXAMPLE],
            },
            "annotations": quiet,
        },
    ]


@cache
def _device() -> Any:
    """This Mac, as `device.detect` sees it, asked in a short process of its own: this one never imports MLX. When
    that process can't say (MLX doesn't load, it timed out), its memory alone, read from the OS, with no GPU named,
    and why on stderr, which the MCP host keeps in its log (stdout carries the protocol only)."""
    code = "import dataclasses, json; from ippon.device import detect; print(json.dumps(dataclasses.asdict(detect())))"
    try:
        done = subprocess.run([sys.executable, "-P", "-c", code], capture_output=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        why = str(error)
    else:
        try:
            found = json.loads(done.stdout)
            return Device(**{**found, "macos": tuple(found["macos"])})
        except (ValueError, TypeError, KeyError):
            said = done.stderr.decode(errors="replace").strip().splitlines()
            why = said[-1] if said else f"it exited with {done.returncode}"
    print(f"ippon: this Mac's GPU couldn't be read ({why}), so models are recommended by its memory", file=sys.stderr)
    return Device(None, None, physical_memory(), None, macos_version(), False)


class Session:
    """One client's session: its messages in, the answers out on `out`, tool calls on threads of their own."""

    def __init__(
        self,
        out: IO[bytes],
        wanted: daemon.Wanted | None = None,
        url: str | None = None,
        key: str | None = None,
        start_timeout: float | None = daemon.START_TIMEOUT_S,
    ):
        # the server a decision is asked of; given none, whichever runs (`daemon.find`)
        self.out, self.wanted = out, wanted or daemon.Wanted((), offline=True)
        self.url, self.key, self.start_timeout = url, key, start_timeout
        self.tools = {t["name"]: t for t in _tools()}
        self._write = threading.Lock()
        self._slots = threading.BoundedSemaphore(WORKERS)
        self._negotiated: str | None = None
        self._initialized = False

    def send(self, message: Any) -> None:
        data = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
        with self._write:
            self.out.write(data)
            self.out.flush()

    def receive(self, line: bytes) -> None:
        """Answer one newline-delimited message. The supported MCP versions do not allow JSON-RPC batches."""
        try:
            message = read_json(line)
        except InvalidRequest as error:
            self.send(_failed(None, -32700, f"Parse error: a message is one JSON object per line ({error})"))
            return
        if isinstance(message, list):
            self.send(_failed(None, -32600, "JSON-RPC batches are not supported"))
            return
        if isinstance(message, dict) and message.get("method") == "tools/call" and "id" in message:
            self._slots.acquire()  # at most WORKERS at once: the rest wait here, in order
            threading.Thread(target=self._in_thread, args=(message,), daemon=True).start()
            return
        answer = self.answer(message)
        if answer is not None:
            self.send(answer)

    def _in_thread(self, message: dict[str, Any]) -> None:
        try:
            answer = self.answer(message)
            if answer is not None:
                self.send(answer)
        finally:
            self._slots.release()

    def answer(self, message: Any) -> dict[str, Any] | None:
        """The response to a request; None for a notification (or a client's response), which gets none."""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _failed(None, -32600, "Invalid Request: a JSON-RPC 2.0 message")
        method, ident = message.get("method"), message.get("id")
        if not isinstance(method, str):
            return (
                None if "result" in message or "error" in message else _failed(ident, -32600, "method must be a string")
            )
        if "id" not in message:
            if method == "notifications/initialized" and self._negotiated is not None:
                self._initialized = True
            return None
        if "id" in message and (isinstance(ident, bool) or not isinstance(ident, (str, int))):
            return _failed(None, -32600, "id must be a string or integer")
        params = message.get("params", {})
        if not isinstance(params, dict):
            return _failed(ident, -32602, "params must be an object")
        if method == "initialize":
            if self._negotiated is not None:
                return _failed(ident, -32600, "session already initialized")
            asked = params.get("protocolVersion")
            client = params.get("clientInfo")
            if (
                not isinstance(asked, str)
                or not isinstance(params.get("capabilities"), dict)
                or not isinstance(client, dict)
                or not isinstance(client.get("name"), str)
                or not isinstance(client.get("version"), str)
            ):
                return _failed(ident, -32602, "initialize needs protocolVersion, capabilities and clientInfo")
            self._negotiated = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
            return _done(
                ident,
                {
                    "protocolVersion": self._negotiated,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "ippon", "title": "ippon", "version": __version__},
                    "instructions": INSTRUCTIONS,
                },
            )
        if method == "ping":
            return _done(ident, {})
        if not self._initialized:
            return _failed(ident, -32600, "send initialize, then notifications/initialized before calling tools")
        if method == "tools/list":
            if params.get("cursor") is not None:
                return _failed(ident, -32602, "invalid cursor: all tools fit on one page")
            return _done(ident, {"tools": list(self.tools.values())})
        if method == "tools/call":
            name, arguments = params.get("name"), params.get("arguments", {})
            if not isinstance(name, str) or name not in self.tools:
                return _failed(ident, -32602, f"Unknown tool: {name!r}; the tools are {', '.join(self.tools)}")
            if not isinstance(arguments, dict):
                return _failed(ident, -32602, "arguments must be an object")
            return _done(ident, self.call(str(name), arguments))
        return _failed(ident, -32601, f"Method not found: {method}")

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """A tool's result: its value as structured content and as JSON text, or its error with `isError`."""
        tools: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "decide": self.decide,
            "models": self.list_models,
            "recommend": self.recommend,
            "lint": self.lint,
        }
        structured = self._negotiated is None or self._negotiated >= "2025-06-18"
        try:
            allowed = set(self.tools[name]["inputSchema"]["properties"])
            if unknown := set(arguments) - allowed:
                raise InvalidRequest(f"unknown argument: {sorted(unknown)[0]!r}")
            value = tools[name](arguments)
        except Exception as error:  # a bug too is said, not raised (`internal`), so the session goes on
            failed = as_decision_error(error)
            result: dict[str, Any] = {
                "content": [{"type": "text", "text": f"{failed.kind}: {failed}"}],
                "isError": True,
            }
            if structured:
                result["structuredContent"] = failed.to_json(exit_code=True)
            return result
        text = json.dumps(value, ensure_ascii=False, indent=1)
        result = {"content": [{"type": "text", "text": text}], "isError": False}
        if structured:
            result["structuredContent"] = value
        return result

    def _server(self, model: str | None) -> daemon.Wanted:
        """The server `model` is asked of: the one this command was started for (`ippon mcp MODEL --precision q8`),
        or for a model it wasn't started with, one that serves that model alone, at the precision its name gives,
        never at this command's."""
        if model is None or model in self.wanted.models:
            return self.wanted
        return replace(self.wanted, models=(model,), precision="auto")

    def decide(self, arguments: dict[str, Any]) -> dict[str, Any]:
        body = _resolve_media_paths(arguments)
        model = body.pop("model", None)
        if model is not None and not isinstance(model, str):
            raise InvalidRequest("'model' must be a string")
        states = body.pop("states", None)
        if states is not None:
            if not isinstance(states, list) or not states or not all(isinstance(s, str) for s in states):
                raise InvalidRequest("'states' must be a list of strings, one or more")
            if "state" in body:
                raise InvalidRequest("give 'state' or 'states', not both")
        parse_request({**body, "state": states[0]} if states else body)  # one that isn't one starts no server
        # a request naming no model asks for the one this server was configured with (`ippon mcp kettei-core:q8`),
        # never whatever the shared server happens to run: model_not_found when it runs another
        asked = model or (self.wanted.models[0] if self.wanted.models else None)
        client, served = daemon.answering(asked, self._server(asked), self.key, self.start_timeout, self.url)
        if served is not None:
            body["model"] = served.id
        if states is None:
            return client.decide(body)
        # what every state shares is said once, so hundreds of states don't repeat it to the agent
        results: list[dict[str, Any]] = []
        usage: dict[str, int] = {}
        diagnostics = None
        for got in client.decide_all({**body, "state": state} for state in states):
            if isinstance(got, DecisionError):
                results.append(got.to_json())
                continue
            for key, value in got.get("usage", {}).items():
                usage[key] = usage.get(key, 0) + value
            diagnostics = diagnostics or got.get("diagnostics")
            results.append({k: v for k, v in got.items() if k not in ("usage", "diagnostics")})
        return {"results": results, "usage": usage, "diagnostics": diagnostics}

    def list_models(self, arguments: dict[str, Any]) -> dict[str, Any]:
        server: dict[str, Any] | None = None
        if self.url:
            server = {"url": self.url, "models": [m.to_json() for m in daemon.Client(self.url, self.key).models()]}
        elif (found := daemon.find(self.wanted, self.key)) is not None:  # the server `decide` would use
            server = daemon.describe(found, self.key)
        builds = listing(lambda n: {"media": sorted(n.media), "downloaded": downloaded(n.build.load)})
        wanted = self.wanted
        options = [*precision_argv(wanted.precision), *wanted.options]
        starts = None if server else {"models": list(wanted.models), "options": options}
        return {"server": server, "starts": starts, "families": builds}

    def recommend(self, arguments: dict[str, Any]) -> dict[str, Any]:
        task = arguments.get("task", "text")
        if not isinstance(task, str) or task not in recommend.TASKS:
            raise InvalidRequest(f"task {task!r}: one of {', '.join(recommend.TASKS)}")
        device = _device()
        return recommend.as_json(task, recommend.recommend(task, device), device)

    def lint(self, arguments: dict[str, Any]) -> dict[str, Any]:
        found = [f.to_json() for f in lint.lint(_resolve_media_paths(arguments))]
        return {"problems": found, "ok": not found}


def _done(ident: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": ident, "result": result}


def _failed(ident: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": ident, "error": {"code": code, "message": message}}


def run(
    wanted: daemon.Wanted | None = None,
    url: str | None = None,
    key: str | None = None,
    start_timeout: float | None = daemon.START_TIMEOUT_S,
    stdin: IO[bytes] | None = None,
    stdout: IO[bytes] | None = None,
) -> int:
    """Serve one client on stdin and stdout until its host closes stdin. Anything else printed while it runs goes to
    stderr: stdout carries the protocol's messages only."""
    out = stdout or sys.stdout.buffer
    source = stdin or sys.stdin.buffer
    with contextlib.redirect_stdout(sys.stderr):
        session = Session(out, wanted, url, key, start_timeout)
        limit = max_body()  # a message's bytes at most: as the HTTP server's body, by this Mac's memory
        while line := source.readline(limit + 1):
            if len(line) > limit:
                session.send(_failed(None, -32600, f"message exceeds {limit >> 20} MiB"))
                while line and not line.endswith(b"\n"):
                    line = source.readline(limit + 1)
            elif line.strip():
                session.receive(line)
        # A host can close stdin after sending its calls: finish their responses before leaving stdout.
        for _ in range(WORKERS):
            session._slots.acquire()
    return 0
