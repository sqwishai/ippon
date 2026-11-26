"""The command line:

    ippon setup [NAME] [--yes]        choose a decision model for this Mac (each listed with what it reads and whether
                                            it fits), download it, answer a first question, and say what to run next
    ippon claude|codex|opencode|pi    the agent for one session with the model beside it (ippon's MCP server, and in
                                            Claude Code ippon's skill); arguments after -- go to the agent, and
                                            nothing is written to its config
    ippon start [MODEL]               the shared server in the background, for your own code (port 8000)
    ippon status | stop | logs        what it serves, stop it, follow its log
    ippon remove AGENT|login|all      undo what setup added to an agent's settings, or the server from login, or
                                      everything ippon added (all, --models with the models it downloaded)
    ippon doctor                      this Mac (chip, memory, profile, code paths, installed adapters), what stands
                                            between it and full speed, and what to change
    ippon decide MODEL [REQUEST|-]    answer a System One request (a JSON file, or stdin); MODEL is a short name
                                            (kettei-core, kettei-core:q8; `pull --list`), a Hub id
                                            (org/name[@revision]) or a directory; `run` is the same command;
                                            --states FILE.jsonl asks every line's state the request's
                                            questions, in batches, JSON lines out
    ippon recommend [TASK]            which builds this Mac can run for a task (text, images, audio), and the one,
                                            from what has been checked of each build, before downloading
    ippon pull NAME                   download a model by its short name (or Hub id), saying first which build
                                            and why, what it takes and whether it fits this Mac, then answer a sample
                                            request with it; again: its newest revision, which loads then read;
                                            --list: every name; --dry-run: all but the download
    ippon lint REQUEST [--model M]    what to change in a request before a model answers it (exit 1: found);
                                            --prompt with a model: the text it reads
    ippon eval DATA.jsonl --model A   labelled examples answered by each model in turn: accuracy, macro-F1,
                                            confusion, confidence thresholds, where models disagree
    ippon verify MODEL --reference R  check a model against its creator's reference answers (--invariance: that
                                            its answers don't depend on chunks, kept states, order or batches)
    ippon bench MODEL                 time it on this Mac (20 requests per shape after 2 warm-ups); --ab "A" "B"
                                            compares two configurations in fresh processes,
                                            --compare mlx-vlm-model=PYTHON compares it with another runtime of the model
    ippon serve MODEL [--port 8000]   serve it over HTTP (127.0.0.1 by default): POST /v1/decide
                                            (the Sqwish Labs SDK's), /v1/systemone, and SGLang's decisions format at
                                            /v1/decisions
    ippon new-adapter NAME --arch A   a package that plugs a decision model in, ready to fill in
    ippon profile MODEL               where a request's time goes, by operation (--size, --questions)
    ippon suite --models DIR --out D  verify and time every suite model on this Mac (docs/suite.md)

Every command that loads a model takes --profile m1..m5 (force a chip profile), --switch op=path,... (one operation's
code path, over the profile's), --precision auto|bf16|fp16|fp32|q8|q6|q4 (auto, the default: fp16 where the model's
activations fit, else bf16; q8 to q4 quantize the model's weights, so they change it, and q6 and q4 warn how far it
drifts from its creator's answers), --chunk N (tokens per pass of a long state), --prefix-budget B (bytes of kept
prefixes; 0 keeps none) and --truncation creator|error (input past the window cut as the creator's code cuts it and
listed in the response's `truncated`, the default, or refused).
`ippon <command> --help` lists each command's options.
"""

from __future__ import annotations

import argparse
import base64
import functools
import json
import math
import os
import platform
import re
import shlex
import shutil
import sys
import time
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, replace
from importlib.resources import files
from pathlib import Path
from typing import Any, NoReturn

# none of these loads MLX, numpy, packaging or huggingface_hub: --help, --version and fast subcommands stay lean, and
# commands import their own heavier modules (ab, bench, decider, doctor, pull, registry, serve, verify ...) inside
# their functions
from . import (
    __version__,
    agents,
    brand,
    daemon,
    names,
    onboarding,
    policy,
    recommend,
    terminal,
)
from .answers import DEFAULT_SIZE as ANSWERS_KEPT
from .errors import DecisionError, InsufficientMemory, InvalidRequest, RuntimeUnavailable, Unavailable, from_json
from .licences import accepting
from .options import AWAKE_WINDOW_S, PERIOD_S, TRUNCATIONS, Reads, precision_argv, validate_load_options
from .plan import DEFAULT_CHUNK
from .served import DEFAULT_PORT, ServedModel, url_of
from .verified import DEFAULT_MODEL

STATUS_EVERY_S = 1.0  # how often `ippon status` redraws at a terminal

MODEL = (  # what names a model on the command line
    "a short name (kettei-core, kettei-core:q8 at 8 bits; `ippon pull --list`), a Hub id (org/name[@revision]), a "
    "GGUF build (repo:Q8_0) or a downloaded folder"
)


START = f"""start here:
  ippon setup     choose a decision model for this Mac and see it answer
  ippon claude    Claude Code with the model beside it (or codex, opencode)
  ippon start     serve it to your own code at {url_of(DEFAULT_PORT)}"""
# each command's --help ends with it in use: the forms people type, the commonest first, each with what it does
EXAMPLES: dict[str, tuple[tuple[str, str], ...]] = {
    "setup": (
        ("ippon setup", "choose with the arrow keys, Enter takes each default"),
        ("ippon setup kettei-core --yes --use claude", "no questions, for a script or an agent"),
    ),
    "start": (
        ("ippon start", "the model setup chose"),
        ("ippon start kettei-core:q8 --port 8001", "another model, on a port of its own"),
    ),
    "decide": (
        ("ippon decide kettei-core request.json", "a request file"),
        ("ippon decide kettei-core - < request.json", "a request on stdin"),
        ("ippon decide kettei-core request.json --server", "the shared server answers it"),
        ("ippon decide d1-2b request.json --image photo.jpg", "with a photo, read from its file"),
        ("ippon decide kettei-core questions.json --states rows.jsonl", "every row, a JSON line each"),
    ),
    "status": (
        ("ippon status", "live at a terminal, until Ctrl-C"),
        ("ippon status --once", "printed once"),
        ("ippon status --json", "every detail, for a script"),
    ),
    "stop": (
        ("ippon stop", "the only server running"),
        ("ippon stop kettei-core", "the one serving this model"),
        ("ippon stop --all", "every server"),
    ),
    "logs": (
        ("ippon logs", "follow it, until Ctrl-C"),
        ("ippon logs -n 200 --no-follow", "its last 200 lines"),
    ),
    "pull": (
        ("ippon pull kettei-core", "download it, then answer a sample request"),
        ("ippon pull kettei-core --dry-run", "what it needs and whether it fits, nothing downloaded"),
        ("ippon pull --list", "every name ippon knows"),
    ),
    "recommend": (
        ("ippon recommend", "for text"),
        ("ippon recommend images", "for pictures and scans"),
    ),
    "serve": (
        ("ippon serve", "the model setup chose, in this terminal, a line per answer"),
        ("ippon serve kettei-core kettei-dot --port 8001", "two models, the second loaded on its first request"),
    ),
    "lint": (
        ("ippon lint request.json", "what to change in it"),
        ("ippon lint request.json --model kettei-core --prompt", "as this model reads it, with the text it reads"),
    ),
    "eval": (("ippon eval labelled.jsonl --model kettei-core --model kettei-dot", "two models on the same rows"),),
    "verify": (
        ("ippon verify kettei-core", "against the reference ippon ships for it"),
        ("ippon verify kettei-core:q8", "the same check at 8 bits"),
    ),
    "bench": (
        ("ippon bench kettei-core", "single requests of four sizes"),
        ("ippon bench kettei-core --scenario concurrent --clients 1,8", "1 client, then 8 at once"),
    ),
    "doctor": (
        ("ippon doctor", "this Mac, then each check"),
        ("ippon doctor --json", "the checks, for a script"),
    ),
    "remove": (
        ("ippon remove claude", "take ippon out of Claude Code's settings"),
        ("ippon remove all --models", "everything ippon added, and the models it downloaded"),
    ),
    "mcp": (
        ("ippon mcp kettei-core", "the stdio server an MCP client starts"),
        ("claude mcp add --transport stdio ippon -- ippon mcp kettei-core", "add it to Claude Code"),
    ),
    "skills": (
        ("ippon skills", "list them"),
        ("ippon skills ippon", "print one for an agent to read"),
        ("ippon skills --path", "the folder that holds them, to copy or link"),
    ),
} | {
    agent: (
        (f"ippon {agent}", "with the model setup chose"),
        (f"ippon {agent} -- --help", f"what follows -- goes to {agent} unchanged"),
    )
    for agent in agents.AGENTS
}
# what a request is, under `decide`'s examples: the least a newcomer needs to write one
REQUEST = """a request:
  {"state": "I was charged twice. Please refund the second charge today.",
   "questions": {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}}}"""


def examples(command: str) -> str | None:
    """`command`'s examples as its --help shows them, each with what it does beside it."""
    lines = EXAMPLES.get(command)
    if not lines:
        return None
    width = max(len(typed) for typed, _ in lines)
    return "examples:\n" + "\n".join(f"  {typed.ljust(width)}   {does}" for typed, does in lines)


# the commands `ippon --help` lists, in the order `main` adds them: a first run's, then checking and timing, then tools
HELP_ORDER = (
    "setup", "claude", "codex", "opencode", "pi", "start", "decide", "status", "stop", "logs", "pull", "recommend",
    "serve", "lint", "eval", "verify", "bench", "doctor", "remove", "mcp", "skills",
)  # fmt: skip


def _bytes(text: str) -> int:
    """A byte count: 4096, 512M, 4G."""
    units = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    return int(float(text[:-1]) * units[text[-1].upper()]) if text[-1:].upper() in units else int(text)


def load_options(args: argparse.Namespace) -> dict[str, Any]:
    """`Decider.load`'s options from the common options (`common_options`): every command that loads a model loads it
    with all of them, so none is silently dropped."""
    options: dict[str, Any] = {
        "profile": args.profile,
        "precision": args.precision,
        "switches": args.switch,
        "chunk": args.chunk,
        "accept_license": args.accept_license,
    }
    for name in ("literal_state", "truncation", "window", "prefix_budget", "keep_awake", "answer_cache"):
        if getattr(args, name) is not None:
            options[name] = getattr(args, name)
    if args.settings is not None:
        options["settings"] = _settings(args)
    return options


def forward_options(
    args: argparse.Namespace, parser: argparse.ArgumentParser | None = None, skip: Sequence[str] = ()
) -> list[str]:
    """The options of `parser` (default: the common ones) given on this command line, as arguments again, but those
    in `skip` (by dest): for another process that loads the model as this one would (ippon's own arm of a
    comparison, an A/B round, a server)."""
    out: list[str] = []
    for action in (parser or common_options())._actions:
        value = getattr(args, action.dest, None)
        if not action.option_strings or action.dest in skip or value is None or value == action.default:
            continue
        flag = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            out.append(flag)
        elif isinstance(action, argparse._AppendAction):
            out += [word for item in value for word in (flag, str(item))]
        else:
            out += [flag, str(value)]
    return out


def load(args: argparse.Namespace, model: str | None = None, **overrides: Any) -> Any:
    """The decider of `model` (default: `args.model`) with the command's common options, and `overrides` over them;
    a context manager that closes it."""
    from .decider import Decider

    return Decider.load(model or args.model, **(load_options(args) | overrides))


def error_json(error: DecisionError) -> dict[str, Any]:
    """An error as JSON output gives it: its kind, message and exit code (errors.py)."""
    error_body: dict[str, Any] = error.to_json(exit_code=True)["error"]
    return error_body


def _api_key(args: argparse.Namespace | None = None) -> str | None:
    """The server's API key: --api-key, else the environment's (`daemon.api_key`); None when neither is set."""
    return (args.api_key if args else None) or daemon.api_key()


def asked_reads(args: argparse.Namespace) -> dict[str, Any]:
    """The options that change what a model answers (`Reads`) as this command line gives them, each one it leaves out
    absent: `--truncation creator` given asks for the creator's cut, while no --truncation takes a server's either
    way."""
    given = {name: getattr(args, name) for name in ("literal_state", "truncation", "window")}
    return {name: value for name, value in (given | {"settings": _settings(args)}).items() if value is not None}


def _settings(args: argparse.Namespace) -> Any:
    """--settings: a bring-your-own model's ippon.json, from a file anywhere."""
    if args.settings is None:
        return None
    value = _request(args.settings)
    if not isinstance(value, dict):
        raise InvalidRequest("--settings must contain a JSON object")
    return value


def _gib(n: int | None) -> str:
    return f"{n / 2**30:.0f} GiB" if n else "unknown"


def _info(args: argparse.Namespace) -> int:
    from importlib.metadata import version

    from . import choices, machine, profiles
    from .decider import default_prefix_budget
    from .registry import adapters

    p = profiles.resolve(args.profile, switched=args.switch)
    device = p.device
    assert device is not None
    prefix_budget = args.prefix_budget if args.prefix_budget is not None else default_prefix_budget(device.memory_bytes)
    facts = {
        "ippon": __version__,
        "mlx": version("mlx"),
        "python": platform.python_version(),
        "chip": {
            "name": device.name,
            "architecture": device.architecture,
            "tier": device.tier,
            "gpu_cores": device.gpu_cores,
        },
        "memory_bytes": device.memory_bytes,
        "working_set_bytes": device.working_set_bytes,
        "macos": ".".join(map(str, device.macos)),
        "neural_accelerators": bool(device.neural_accelerators),
        "profile": {"name": p.name, "detected": p.detected, "forced": p.forced, "emulated": p.emulated},
        "switches": p.switches.as_dict(),
        "precision": args.precision,
        "chunk": args.chunk,
        "prefix_budget_bytes": prefix_budget,
        "keep_awake_s": args.keep_awake,
        "serve_heartbeat": {
            "thread": "model",
            "every_s": PERIOD_S,
            "window_s": policy.POLICIES[policy.DEFAULT][0],
            "policy": policy.DEFAULT,
        },
        "kernels": choices.summary(p),
        "environment": machine.environment(),
        "adapters": sorted(adapters()),
    }
    if getattr(args, "json", False):
        print(json.dumps(facts, indent=2))
        return 0
    print(f"ippon {__version__}, mlx {facts['mlx']}, Python {facts['python']}")
    cores = f"; {device.gpu_cores} GPU cores" if device.gpu_cores else ""
    print(f"chip: {device.name or 'no Metal device'} ({device.architecture or '-'}; {device.tier or '-'}{cores})")
    print(f"memory: {_gib(device.memory_bytes)}, GPU working set {_gib(device.working_set_bytes)}")
    print(f"macOS: {facts['macos']}")
    unused = " (unused: MLX runs an older family)" if p.emulated else ""
    print(f"neural accelerators: {'yes' if device.neural_accelerators else 'no'}{unused}")
    print(f"profile: {p.name}{f' (forced; this chip is {p.detected})' if p.forced else ''}")
    if p.forced:
        mlx = f"{device.emulated_architecture}'s" if p.emulated else "this chip's (set IPPON_PROFILE to emulate)"
        print(f"  MLX kernels: {mlx}")
    for (
        key,
        value,
    ) in p.switches.as_dict().items():  # where each path comes from, when not the profile (`Profile.sources`)
        source = p.sources.get(key)
        print(f"  {key}={value}{f'  ({source})' if source else ''}")
    print(f"precision: {args.precision}{' (default)' if args.precision == 'auto' else ''}")
    print(f"chunk: {args.chunk} tokens{' (default)' if args.chunk == DEFAULT_CHUNK else ''}")
    grows = "" if args.prefix_budget is not None else " (default; one growing state may hold more while requests come)"
    print(f"kept prefixes: up to {prefix_budget / 2**30:.1f} GiB{grows}")
    window = f"{AWAKE_WINDOW_S:g} s"
    awake = f"every {args.keep_awake or PERIOD_S:g} s for {window} after an answer"
    awake += "" if args.keep_awake else " (default)"
    served = f"every {PERIOD_S:g} s for {policy.POLICIES[policy.DEFAULT][0]:g} s"
    print(f"keep-awake: {awake}; serve: one model-thread heartbeat {served} after a request (--policy)")
    print(f"kernels: {facts['kernels']}")
    if facts["environment"]:
        print("environment: " + ", ".join(f"{k}={v}" for k, v in facts["environment"].items()))
    print(f"adapters: {', '.join(facts['adapters']) or 'none installed'}")
    return 0


def _home(path: Path) -> str:
    """`path` with the home folder written as ~."""
    try:
        return str(Path("~") / path.relative_to(Path.home()))
    except ValueError:
        return str(path)


def _request(source: str | None) -> Any:
    """A request from `source`: JSON text (it starts with "{"), a JSON file, or "-" for stdin; InvalidRequest, naming
    what to change, when there is none or it isn't JSON. Stdin is read only when "-" asks for it: an agent's shell
    leaves stdin open, so reading it unasked would wait forever."""
    if source is None:
        raise InvalidRequest("give the request: a JSON file, JSON text, or - to read it from stdin")
    if source == "-":
        if sys.stdin.isatty():
            print("ippon: reading a request from stdin (end it with Ctrl-D)", file=sys.stderr)
        text, where = sys.stdin.read(), "stdin"
    elif source.lstrip().startswith("{"):
        text, where = source, "the request given"
    else:
        try:
            text, where = Path(source).read_text(encoding="utf-8"), source
        except OSError as error:
            raise InvalidRequest(f"can't read {source} ({error.strerror}): give a JSON file, JSON text or -") from None
        except UnicodeError:
            raise InvalidRequest(f"can't read {source} as UTF-8: save the request as UTF-8 JSON") from None
    if not text.strip():
        raise InvalidRequest(f"no request in {where}: a System One request is a JSON object, {{\"state\": ..., "
                             '"questions": {...}}')  # fmt: skip
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise InvalidRequest(f"{where} is not JSON: {error.msg} at line {error.lineno}, column {error.colno}") from None


MEDIA_FLAGS = {"image": "images", "video": "videos", "audio": "audio"}  # decide's --image PATH ..., by request field


def with_files(body: Any, args: argparse.Namespace) -> Any:
    """The request with the files that --image, --video and --audio name added to its images, videos and audio, in
    base64, after any it holds: a photo is asked about without writing its base64 into the JSON by hand."""
    given = {field: getattr(args, flag, None) or [] for flag, field in MEDIA_FLAGS.items()}
    if not any(given.values()):
        return body
    if not isinstance(body, dict):
        raise InvalidRequest("a request is a JSON object")
    body = dict(body)
    for field, paths in given.items():
        if not paths:
            continue
        held = body.get(field) or []
        sent = [held] if isinstance(held, str) else list(held)
        for path in paths:
            try:
                sent.append(base64.b64encode(Path(path).expanduser().read_bytes()).decode())
            except OSError as error:
                raise InvalidRequest(f"can't read {path} ({error.strerror})") from None
        body[field] = sent
    return body


def _run(args: argparse.Namespace) -> int:
    from . import provenance, quantization
    from .schema import parse_request

    if args.server and "://" not in args.server and args.request is None:
        # `decide MODEL --server request.json`: --server takes an optional URL, so it took the request; give it back
        args.request, args.server = args.server, ""
    body = with_files(_request(args.request), args)  # before the model loads: a request that isn't one costs no load
    licence: dict[str, str] | None = None  # the model's, shown under the answers
    if args.states:
        return _states(args, body)
    if args.server is not None:
        parse_request(body)
        client, name = _server_client(args)
        answer = client.decide({**body, "model": name})
    else:
        quantization.get(args.precision)  # the command's own options first, then its request
        parse_request(body)  # a request that can't be answered costs no download and no load
        with load(args) as decider:
            answer = decider.systemone(body)
            licence = provenance.licence(decider.details)
    if args.json or not sys.stdout.isatty():  # scripts and agents read a pipe: the response as it is
        print(json.dumps(answer, indent=2, ensure_ascii=False))
    else:
        print(readable(answer, terminal.colour_on(sys.stdout)))
        if args.server is None and licence:
            show(f"model licence: {licence['name']}" + (f", {licence['url']}" if licence.get("url") else ""))
    return 0


def readable(answer: Mapping[str, Any], colour: bool = False) -> str:
    """A System One response for a person at a terminal: a line per question, its answer and how sure the model is
    (a choice with every option's probability, a score with its nearest level named), then what was read. With
    `colour` the answers stand out as they do in `serve`'s log: each in the accent of what is settled, the rest dim,
    and a question under its `unsure_below` in the accent of what to do next."""
    rows = []
    for qid, a in answer.get("answers", {}).items():
        said = terminal.styled(terminal.said(a), colour, "1", "mint")
        if a.get("type") == "choice":  # the runners-up too, here where a line can hold them
            ranked = sorted(a["probabilities"].items(), key=lambda kv: -kv[1])
            others = ", ".join(f"{label} {terminal.percent(p)}" for label, p in ranked[1:4])
            more = f", {len(ranked) - 4} more" if len(ranked) > 4 else ""
            said += terminal.styled(f"  ({others}{more})", colour, "2") if others else ""
        rows.append((qid, said))
    width = max((len(q) for q, _ in rows), default=0)
    unsure = set(answer.get("unsure", ()))  # asked for with unsure_below: the ones to send on
    flag = terminal.styled("  (unsure)", colour, "amber")
    lines = [f"{q.ljust(width)}  {said}" + (flag if q in unsure else "") for q, said in rows]
    usage = answer.get("usage", {}).get("input_tokens")
    if usage is not None:
        cut = ", part of it cut to fit, see --json)" if answer.get("truncated") else ")"
        lines.append(terminal.styled(f"({usage:,} tokens read{cut}", colour, "2"))
    for note in answer.get("warnings") or ():
        lines.append(terminal.styled(f"warning: {note}", colour, "amber"))
    return "\n".join(lines)


def _states(args: argparse.Namespace, body: Any) -> int:
    """`decide --states FILE.jsonl`: every line's state asked the request's questions, answered in batches
    (`batches.answer_all`), one JSON line out per line in, in order: {"line", "id"?, and the response's fields} or
    {"line", "id"?, "error": {"type", "message"}}, a line's error its own. 1 when any line failed."""
    import heapq

    from .schema import STATE_FIELDS

    questions = body.get("questions") if isinstance(body, dict) else None
    if not isinstance(questions, dict) or not questions:
        raise InvalidRequest("--states asks every line the request's questions: give a request with 'questions'")
    try:
        lines = Path(args.states).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise InvalidRequest(f"can't read {args.states} ({error.strerror})") from None
    except UnicodeError:
        raise InvalidRequest(f"can't read {args.states} as UTF-8: save the states as UTF-8 JSON lines") from None
    # every line is the request with its state swapped for the line's, so unsure_below and the request's other
    # options apply to each line as they do to one decision; the request's media too (a reference photo every line
    # is compared with), unless a line gives that field itself, as MCP's decide takes them
    template = {k: v for k, v in body.items() if k != "state"}
    ids: dict[int, Any] = {}
    bodies: dict[int, dict[str, Any]] = {}
    failed: dict[int, DecisionError] = {}
    for n, text in enumerate(lines, 1):
        if not text.strip():
            continue
        try:
            item = json.loads(text)
        except json.JSONDecodeError as error:
            failed[n] = InvalidRequest(f"not JSON: {error.msg}")
            continue
        if isinstance(item, dict):
            ids[n] = item.get("id")
            unknown = sorted(set(item) - {"id", *STATE_FIELDS})
            if unknown:
                failed[n] = InvalidRequest(f"unknown field {unknown[0]!r}: a line gives {', '.join(STATE_FIELDS)}, id")
                continue
            given = {k: item[k] for k in STATE_FIELDS if k in item}
            if not given:  # an id alone would be asked about nothing
                failed[n] = InvalidRequest(f"nothing to ask about: a line gives {', '.join(STATE_FIELDS)}, id")
                continue
        else:
            given = {"state": item}
        bodies[n] = {**template, **given}

    total = len(bodies) + len(failed)

    def write(lines: Iterable[tuple[int, Any]]) -> int:  # one JSON line out per line in, as each is answered
        errors = 0
        for n, got in _counted(heapq.merge(lines, sorted(failed.items())), total, len(questions)):
            out: dict[str, Any] = {"line": n, **({"id": ids[n]} if ids.get(n) is not None else {})}
            if isinstance(got, DecisionError):
                out["error"] = error_json(got)
                errors += 1
            else:
                out |= got if isinstance(got, dict) else got.response
            print(json.dumps(out, ensure_ascii=False), flush=True)
        return errors

    def asked(client: daemon.Client, name: str) -> Iterator[tuple[int, Any]]:
        sent = ({**given, "model": name} for given in bodies.values())
        return zip(bodies, client.decide_all(sent, args.batch or policy.IN_FLIGHT), strict=True)

    # the server is found or started, or the model loaded, before the count starts: its rate is the answering's
    if bodies and args.server is not None:
        errors = write(asked(*_server_client(args)))
    elif bodies:
        from .batches import answer_all

        with load(args) as decider:
            order = list(bodies)
            errors = write((order[i], got) for i, got in answer_all(decider, list(bodies.values()), args.batch))
    else:
        errors = write(iter(()))
    if errors:
        print(f"ippon: {errors} of {total} lines failed (their 'error')", file=sys.stderr)
    return 1 if errors else 0


def _counted(lines: Iterable[tuple[int, Any]], total: int, decisions: int) -> Iterator[tuple[int, Any]]:
    """`lines` passed through, with how many are answered and the time per decision so far (`decisions` a line) kept
    on one line of a terminal's standard error, and the total time once all are: a long --states run shows it is
    moving. Only when the answers go to a
    file or a pipe and stderr is a terminal, so the counter never lands among the answers or in a log."""
    if not sys.stderr.isatty() or sys.stdout.isatty():
        yield from lines
        return
    begun = shown = time.monotonic()
    done = 0
    for line in lines:
        yield line
        done += 1
        if (now := time.monotonic()) - shown >= 0.1 or done == total:
            shown = now
            each = 1000 * (now - begun) / (done * max(decisions, 1))
            print(f"\r\x1b[2Kippon: {done:,} of {total:,} answered · {each:.0f} ms/decision", end="", file=sys.stderr)
    took = time.monotonic() - begun
    each = 1000 * took / (max(done, 1) * max(decisions, 1))
    print(f"\r\x1b[2Kippon: {done:,} answered in {took:.1f} s · {each:.0f} ms/decision", file=sys.stderr)


def _check(args: argparse.Namespace) -> int:
    from . import check

    found = check.check(args.model, args.adapter)
    print(json.dumps(found, indent=2) if args.json else check.report(found))
    return 0


# the starter requests and how to label examples, which the package doesn't carry
EVALUATING = "https://github.com/sqwishai/ippon/blob/main/docs/evaluating.md#starter-requests"


def _pull(args: argparse.Namespace) -> int:
    """Download what loading a model reads, after saying which build a name chose and why, what it takes, whether it
    fits this Mac (`check`) and the family's other builds; a build that doesn't fit is refused, naming the family's
    smallest that does."""
    from . import check, pull

    if args.list or not args.model:
        # at a terminal no line is longer than it; a pipe gets every line whole
        width = shutil.get_terminal_size().columns - 1 if sys.stdout.isatty() else None
        print(
            json.dumps(names.listing(), indent=2)
            if args.json
            else terminal.commands(names.table(terminal.colour_on(sys.stdout), width), terminal.colour_on(sys.stdout))
        )
        return 0

    plan = pull.prepare(args.named)
    chosen, why = names.find(args.named) if names.is_name(args.named) else (names.of_load(args.model), "")
    others = [n for n in names.families()[chosen.family] if n.name != chosen.name] if chosen else []

    found = plan.report
    if not args.json:
        head = f"{args.named} -> {chosen.name}" if chosen and chosen.name != args.named else args.named
        print(f"{head} ({args.model}){', ' + why if why else ''}")
        print(found["support"])
        print(f"creator: {found['creator']}")
        print(check.report(found))
        disk = found["disk"]
        left = f"{disk['needed_bytes'] / 1e9:.1f} GB" if disk["needed_bytes"] else "nothing, it is on this Mac"
        print(f"to download: {left} · disk free: {disk['free_bytes'] / 1e9:.1f} GB")
        for licence in plan.licenses:
            print(f"licence: {licence.name} — {licence.url}")
        if others:
            print(f"other builds of {chosen.family if chosen else ''}:")  # the family's, for choosing another
            width = max(len(n.name) for n in others)
            print("\n".join(f"  {n.name.ljust(width)}  {n.build.weights}, {n.status}" for n in others), flush=True)
    if args.dry_run:
        if args.json:
            print(json.dumps(found, indent=2))
        return 0

    def approve(licence: Any) -> bool:
        if args.json or not (sys.stdin.isatty() and sys.stdout.isatty()):
            return False
        try:
            return input(f"Accept {licence.name}? [y/N] ").strip().lower() in {"y", "yes"}
        except EOFError:
            return False

    sys.stdout.flush()
    progress = None if args.json else pull.TerminalProgress(sys.stderr, tty=sys.stdout.isatty() and sys.stderr.isatty())
    try:
        path, revision = plan.run(accept_license=args.accept_license, approve=approve, progress=progress)
    except KeyboardInterrupt:
        message = "download stopped; partial files kept. Run the same pull command to resume."
        if args.json:
            print(json.dumps({"error": {"type": "download_interrupted", "message": message, "exit_code": 130}}))
        else:
            print(f"\nippon: {message}", file=sys.stderr)
        return 130
    was = plan.previous_revision
    try:
        at = precision_argv(names.precision_of(args.named))  # kettei-core:q8 answers at 8 bits, as it will run
        sample = None if args.no_sample else _sample(args, quiet=args.json, options=at)
    except InsufficientMemory as error:  # downloaded: it loads once this Mac has the memory free
        print(f"downloaded; not loaded for a sample: {error}", file=sys.stderr)
        sample = None
    if args.json:
        picked = {"name": chosen.name, "why": why, "others": [n.name for n in others]} if chosen else {}
        got = {"asked": args.named, "path": path, "revision": revision, "previous_revision": was, "sample": sample}
        print(json.dumps(found | got | picked, indent=2))
        return 0
    if path:
        print(f"downloaded: {_home(Path(path))}")
    if revision and was and was != revision:
        print(f"revision {revision[:12]}: the newest, which loads read from now on (it was {was[:12]})")
    elif revision:
        print(
            f"revision {revision[:12]}: the newest; loads read it from the cache, without the Hub, until the next pull"
        )
    print(f"next: a starter request ({EVALUATING}), then ippon lint request.json --model {args.named}")
    print(f"      ippon eval labelled.jsonl --model {args.named}   (your own labelled examples)")
    return 0


PACE_CALLS = 5  # a sample's pace: the median of these calls after the first (the second can be slower)


def _sample(args: argparse.Namespace, quiet: bool = False, options: Sequence[str] = ()) -> list[dict[str, Any]]:
    """`pull`'s sample: the model loaded as `decide` loads it, a short request answered (and an image one for a model
    that reads images, an audio one for one that hears); each printed with its options' probabilities and the
    time it took, unless `quiet`, loaded with `options` (`--precision q8`). The requests and answers as JSON."""
    from .templates import SAMPLE_TEXT, sample_audio, sample_image  # Pillow, which --help does without

    start = time.perf_counter()
    import statistics

    from . import bench

    decider = load(common_options().parse_args(list(options)), args.model)  # as `decide` loads it
    loaded = time.perf_counter() - start
    with decider, decider.uncached():  # each call read, so its time is the model's
        own = decider.own_questions  # a model asked its own questions (a classifier's labels)
        bodies = [{"state": SAMPLE_TEXT["state"], "questions": own(2, 0)} if own else SAMPLE_TEXT]
        media: frozenset[str] = decider.format.media
        bodies += [sample() for kind, sample in (("images", sample_image), ("audio", sample_audio)) if kind in media]
        out = []
        for body in bodies:
            took = []  # the first answer at a request's shapes compiles them; the median of the next 5 is its pace
            for _ in range(1 + PACE_CALLS):
                result, ms = bench.timed(decider, body)
                took.append(ms)
            pace = statistics.median(took[1:])
            out.append({"request": body, "response": result.response, "ms": round(pace, 1)})
            out[-1]["first_ms"] = round(took[0], 1)
            if quiet:
                continue
            shown = (
                "an image" if "images" in body else "an audio clip" if "audio" in body else json.dumps(body["state"])
            )
            print(
                f"sample decision about {shown}: {pace:.0f} ms, the median of {PACE_CALLS} ({took[0]:.0f} ms the first "
                f"time, compiling its shapes; loaded in {loaded:.1f} s)"
            )
            for qid, answer in result.answers.items():
                ranked = sorted(answer.by_option.items(), key=lambda kv: -kv[1])
                words = {"true": "yes", "false": "no"} if answer.question.type == "noul" else {}
                levels = dict(answer.question.criteria) if answer.question.type == "score" else {}
                said = ", ".join(f"{words.get(o) or levels.get(o) or o} {p:.2f}" for o, p in ranked)
                print(f"  {qid} ({answer.question.instructions}): {said}")
        if not quiet:
            print("`ippon verify` checks the chosen build against its creator's reference bounds.")
        return out


def _recommend(args: argparse.Namespace) -> int:
    from .device import detect

    device = detect()
    picks = recommend.recommend(args.task, device)
    if args.json:
        print(json.dumps(recommend.as_json(args.task, picks, device), indent=2))
        return 0
    width = shutil.get_terminal_size().columns - 1 if sys.stdout.isatty() else None  # a pipe gets every row whole
    print(recommend.report(args.task, picks, device, terminal.colour_on(sys.stdout), width))
    return 0


def _skills(args: argparse.Namespace) -> int:
    root = files("ippon").joinpath("skills")
    if args.path:
        print(root)
        return 0
    skills = sorted(entry.name for entry in root.iterdir() if entry.joinpath("SKILL.md").is_file())
    if not args.name:
        about = {}
        for name in skills:
            head = root.joinpath(name, "SKILL.md").read_text(encoding="utf-8").split("---")[1]
            about[name] = " ".join(head.split("description:", 1)[1].replace(">", " ").split()).split(". ")[0] + "."
        if args.json:
            print(json.dumps({"skills": [{"name": n, "about": a} for n, a in about.items()], "path": str(root)}))
            return 0
        for name, line in about.items():
            print(f"{name.ljust(max(map(len, skills)))}  {line}")
        print("ippon skills NAME prints one for an agent to read; ippon skills --path is the folder to copy or link")
        return 0
    if args.name not in skills:
        raise InvalidRequest(f"{args.name!r}: not a skill; they are {', '.join(skills)}")
    folder = root.joinpath(args.name)
    if args.reference:
        references = sorted(e.name.removesuffix(".md") for e in folder.joinpath("references").iterdir())
        if args.reference not in references:
            raise InvalidRequest(
                f"{args.reference!r}: not a reference of {args.name}; they are {', '.join(references)}"
            )
        print(folder.joinpath("references", f"{args.reference}.md").read_text(encoding="utf-8"), end="")
        return 0
    print(folder.joinpath("SKILL.md").read_text(encoding="utf-8"), end="")
    return 0


def _lint(args: argparse.Namespace) -> int:
    """What to change in a request (lint.py), and with --model what that model can't be asked or would cut, with its
    prompts' token counts; 1 when anything is found."""
    from . import lint

    body = with_files(_request(args.request), args)
    found = lint.lint(body)
    seen: dict[str, Any] = {}
    if args.model and not any(isinstance(f, lint.NotARequest) for f in found):
        with load(args, warm_up=False) as decider:
            more, seen = lint.against(decider, body)
            found += more
            if args.prompt:
                seen["read"] = [{k: v for k, v in p.items() if k != "ids"} for p in decider.inspect(body)["prompts"]]
    if args.json:
        print(json.dumps({"problems": [f.to_json() for f in found], **seen}, indent=2, ensure_ascii=False))
        return 1 if found else 0
    window = f" of the model's {seen['max_tokens']}" if seen.get("max_tokens") else ""
    for p in seen.get("prompts", []):
        print(f"prompt for {', '.join(p['questions'])}: {p['tokens']} tokens{window}")
    for p in seen.get("read", []):
        print(json.dumps(p, indent=2, ensure_ascii=False))
    for f in found:
        print(f)
    print(f"{len(found)} to change" if found else "nothing to change")
    return 1 if found else 0


def _eval(args: argparse.Namespace) -> int:
    """Each model's answers to the labelled rows (evaluate.py), loaded one at a time; every row's answers to a results
    file and the report to stdout."""
    from . import evaluate

    if len(set(args.model)) < len(args.model):
        raise InvalidRequest("a model is named twice with --model: each is evaluated once")
    if not (0 <= args.min_precision <= 1 and 0 <= args.min_recall <= 1) or args.folds < 2:
        raise InvalidRequest("--min-precision and --min-recall are between 0 and 1, and --folds is 2 or more")
    shared = None
    if args.questions:
        given = _request(args.questions)
        shared = given.get("questions") if isinstance(given, dict) else None
        if not isinstance(shared, dict) or not shared:
            raise InvalidRequest(f"{args.questions}: --questions is a request whose 'questions' every row is asked")
    rows = evaluate.load(args.data, shared)  # before any model loads: data that can't be scored costs no load
    models = []
    for named in args.model:
        precision = names.named_precision(named, args.precision)
        models.append((named, functools.partial(load, args, str(names.resolve(named)), precision=precision)))
    results, about = evaluate.run(models, rows, args.batch, log=lambda m: print(m, file=sys.stderr))
    if not results:  # no model loaded: the first one's error, with its exit code
        first = from_json(about[args.model[0]])
        raise type(first)(f"no model could be loaded: {first}")
    out = Path(args.out) if args.out else Path(args.data).with_suffix(".results.jsonl")
    try:
        with out.open("w", encoding="utf-8") as f:
            for name, answers in results.items():
                for a in answers:
                    f.write(json.dumps(a.to_json(name), ensure_ascii=False) + "\n")
    except OSError as error:
        raise InvalidRequest(f"can't write evaluation results to {out}: {error}") from None
    found = evaluate.report(results, about, args.min_precision, args.min_recall, args.folds)
    print(json.dumps(found, indent=2, ensure_ascii=False) if args.json else evaluate.text(found))
    print(f"every row's answers, each option's probability as the model gave it: {out}", file=sys.stderr)
    return 1 if len(results) < len(args.model) else 0  # a model left out: the comparison isn't the one asked for


def _prompt(args: argparse.Namespace) -> int:
    """What the model reads for a request: each prompt's text and tokens, and what was cut to fit its window."""
    body = _request(args.request)
    with load(args, warm_up=False) as decider:
        found = decider.inspect(body)
    if not args.ids:
        for p in found["prompts"]:
            del p["ids"]
    print(json.dumps(found, indent=2, ensure_ascii=False))
    return 0


def _verify(args: argparse.Namespace) -> int:
    from .verify import BOUND, load_reference, verify_against, verify_model

    if args.against:
        if args.reference:
            cases = load_reference(args.reference)["requests"]
        elif args.requests:
            cases = _request(args.requests)
        else:
            raise InvalidRequest("--against answers a reference's requests (--reference) or a request set (--requests)")
        report: Any = verify_against(args.model, args.against, cases, bound=args.bound, **load_options(args))
    elif args.invariance:
        from . import invariance

        if args.reference:
            reference = load_reference(args.reference)
            cases, tolerance = reference["requests"], float(reference.get("bound", BOUND))
        elif args.requests:
            cases, tolerance = _request(args.requests), BOUND
        else:
            raise InvalidRequest(
                "--invariance answers a reference's requests (--reference) or a request set (--requests)"
            )
        with load(args, pin_memory=False) as decider:
            report = invariance.run(decider, cases, tolerance, args.model)
    else:
        shipped = args.reference or names.reference(args.model)
        if shipped is None:
            raise InvalidRequest(
                f"{args.model} has no reference ippon ships: give one with --reference (or use --invariance or "
                "--against, with --requests)"
            )
        report = verify_model(args.model, shipped, **load_options(args))
    print(json.dumps(report.to_json(), indent=2) if args.json else report.text())
    return 0 if report.passed else 1


def _bench(args: argparse.Namespace) -> int:
    from . import ab, bench

    if args.compare:
        from . import compare

        text, data = compare.run(
            args.model,
            args.compare,
            forward_options(args),
            bench.Spec.from_args(args),
            args.rounds,
            lambda m: print(m, file=sys.stderr),
        )
        print(text)
        if args.out:
            bench.save(args.out, data)
        return 0
    if args.ab:
        shared = forward_options(args, args.options, skip=("ab", "compare", "rounds", "n", "seed", "out"))
        log = lambda m: print(m, file=sys.stderr)  # noqa: E731
        text, data = ab.run(args.model, args.ab, shared, args.n, args.rounds, args.seed, log)
        print(text)
        if args.out:
            bench.save(args.out, data)
        return 0
    spec = bench.Spec.from_args(args)
    with load(args) as decider:
        cells = bench.run(decider, spec, args.batch_window_ms, args.max_batch)
        print(bench.table(cells))
        if args.out:
            bench.save(args.out, bench.report(args.model, decider, cells, spec))
    return 0


def _new_adapter(args: argparse.Namespace) -> int:
    from .scaffold import write

    written = write(args.name, args.arch, Path(args.out or args.name.split("/")[-1] + "-ippon"))
    for path in written:
        print(path)
    return 0


def _profile(args: argparse.Namespace) -> int:
    from . import bench
    from .profile import profile

    given = json.loads(Path(args.request).read_text(encoding="utf-8")) if args.request else None
    if args.phases:
        from .profile import phases

        if given is not None and args.scenario != "fresh":
            raise InvalidRequest("--scenario changed times bench's requests with new questions; leave out --request")
        start = time.perf_counter()
        with load(args) as decider:
            loaded = (time.perf_counter() - start) * 1000
            scenario = {"fresh": "fresh", "changed": "repeat"}[args.scenario]
            bodies = (
                [given] * (args.repeat + 1)
                if given is not None
                else list(bench.bodies(scenario, args.size, args.questions, 0, args.repeat + 1, decider.asks))
            )
            print(f"loaded in {loaded:.0f} ms (weights, warm-up and memory wiring)")
            print(phases(decider, bodies, args.scenario).table())
        return 0
    with load(args) as decider:
        body = given or next(bench.bodies("fresh", args.size, args.questions, 0, 1, decider.asks))  # what it's asked
        print(profile(decider, body, args.repeat).table())
    return 0


def _suite(args: argparse.Namespace) -> int:
    from . import suite

    report = suite.run(
        args.models,
        args.out,
        args.quick,
        args.arm or (),
        [x for x in (args.only or "").split(",") if x],
        lambda m: print(m, file=sys.stderr),
    )
    failed = [s for s in report["steps"] if not s["ok"] or s.get("passed") is False]
    print(f"{len(report['steps'])} steps, {len(failed)} failed; results in {args.out} (summary.md)")
    return 1 if failed else 0


def _doctor(args: argparse.Namespace) -> int:
    from . import doctor

    found = doctor.run()
    if not args.json and not any(c.status == "fail" for c in found):  # this Mac and how ippon runs on it, first
        _info(common_options().parse_args([]))  # needs MLX, which a failed check may be about
        print()
    colour = terminal.colour_on(sys.stdout)
    print(json.dumps([asdict(c) for c in found], indent=2) if args.json else doctor.report(found, colour))
    return 1 if any(c.status == "fail" for c in found) else 0


def _keep_alive(text: str) -> float:
    """--keep-alive: seconds (90), or with a unit (30s, 10m, 1h); 0 unloads right after answering, a negative
    number or "forever" never (infinity)."""
    if text.strip().lower() == "forever":
        return float("inf")
    units = {"s": 1, "m": 60, "h": 3600}
    try:
        seconds = float(text[:-1]) * units[text[-1].lower()] if text[-1:].lower() in units else float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: seconds, or 30s, 10m, 1h; 0, or -1 (forever)") from None
    if math.isnan(seconds):
        raise argparse.ArgumentTypeError("duration cannot be NaN")
    return float("inf") if seconds < 0 else seconds


def _policy(args: argparse.Namespace) -> tuple[float | None, float]:
    """serve's heartbeat window (None: no heartbeat) and keep-alive: --policy's, with --keep-awake,
    --keep-awake-window and --keep-alive over it (--keep-awake turns battery's heartbeat on, for its window)."""
    window, keep_alive = policy.POLICIES[args.policy]
    if args.keep_awake_window is not None or (window is None and args.keep_awake is not None):
        window = args.keep_awake_window if args.keep_awake_window is not None else AWAKE_WINDOW_S
    if args.no_keep_awake:
        window = None
    return window, keep_alive if args.keep_alive is None else args.keep_alive


def _serve(args: argparse.Namespace) -> int:
    if not args.model:  # none named: the model setup chose, at its precision, as `ippon start` serves it
        model, chosen = onboarding.chosen_model()
        args.model = [model]
        if chosen and args.precision == "auto":
            args.precision = chosen
    key = _api_key(args)
    daemon.require_key_off_loopback(args.host, key)
    if args.detach:
        wanted = _wanted(args, args.model, keep_alive=None)
        # a server answering these models as asked is the one to use, as `ippon start` finds: a second copy would only
        # take the memory again; a --port given asks for a server there
        found = daemon.find(wanted, key) if args.port is None else None
        if found is None:
            _approve_before_detach(args.model, args)
        ready = found or daemon.start(wanted, key, args.start_timeout)
        if found and wanted.options:  # a running server keeps the options it started with
            print(
                f"ippon: the running server keeps its own options, not {shlex.join(wanted.options)}: "
                f"`ippon stop --port {found.port}`, then serve again to start it with them",
                file=sys.stderr,
            )
        if args.json:
            print(json.dumps({**ready.to_json(), "started": found is None}, indent=2))
        elif found:
            show(
                f"ippon is already serving {', '.join(args.model)} at {ready.url} (pid {ready.pid})"
                " · `ippon status` says what, `ippon stop` stops it"
            )
        else:
            print(
                f"ippon: server pid {ready.pid} at {ready.url}; log: {ready.log}"
                + daemon.moved(ready.port, args.port)
                + daemon.network_note(args.host, ready.port)
            )
        return 0

    # said before the seconds MLX and the first model take, so a person sees the command is under way
    begun = time.monotonic()
    person = sys.stderr.isatty()  # at a terminal: the cat and a short header once the model is ready, then the answers
    colour = terminal.colour_on(sys.stderr)
    titles = [names.title(spec.split("=", 1)[-1]) for spec in args.model]  # NAME=MODEL: the model it serves
    print(terminal.styled(f"ippon: loading {', '.join(titles)}...", colour, "2"), file=sys.stderr, flush=True)
    from .residency import Heartbeat
    from .serve import run, specs

    # the server's model thread sends the heartbeat, between passes: no model gets a keep-awake thread of its own
    window, keep_alive = _policy(args)
    heartbeat = None if window is None else Heartbeat(args.keep_awake or PERIOD_S, window)
    served = specs(args.model, args.precision)
    port = daemon.free_port(args.host, args.port)
    record = daemon.Server(
        os.getpid(),
        port,
        url_of(port, args.host),
        tuple((s.name, s.loads) for s in served),
        started=daemon.started_at(os.getpid()) or time.time(),
        policy=policy.policy_json(args.policy, args.keep_awake or PERIOD_S, window, keep_alive),
        api_key=bool(key),
        log=args.record_log,
    )
    daemon.record(record)  # from before its first model loads, so a client that started it finds it starting

    def listening(host: str, port: int, first: Any) -> None:
        nonlocal record
        record = replace(record, port=port, url=url_of(port, host), status="serving")
        daemon.record(record)
        others = f" and {len(served) - 1} more on their first requests" if len(served) > 1 else ""
        url, took = f"http://{host}:{port}", f"ready in {time.monotonic() - begun:.1f} s"
        notes = [n.strip() for n in (daemon.moved(port, args.port), daemon.network_note(host, port)) if n.strip()]
        if not person:  # a log file: one line
            print(f"ippon: serving {titles[0]}{others} on {url}, {took}", *notes, sep="\n", file=sys.stderr)
            return

        def dim(text: str) -> str:
            return terminal.styled(text, colour, "2")

        header = [
            terminal.styled("ippon", colour, "1") + dim(f" {__version__}"),
            "",
            terminal.styled(titles[0], colour, "1") + dim(others),
            "serving at " + terminal.styled(url, colour, "pink") + dim(f" · {took}"),
            *(dim(n) for n in notes),
            "",
            dim("Ctrl-C stops it · ") + terminal.commands("`ippon status` watches it", colour),
        ]
        art = brand.small_cat(colour) if colour else []
        width = shutil.get_terminal_size().columns - 1
        # the loading line gives way to the header, and the answers follow under it
        print("\x1b[1A\x1b[2K" + "\n".join(["", *brand.beside(art, header, width), ""]), file=sys.stderr, flush=True)

    try:
        run(
            served,
            load_options(args),
            args.memory_gb,
            args.host,
            port,
            listening,
            max_batch=args.max_batch,
            log=args.log or sys.stderr.isatty(),  # at a terminal, each answer as it is sent
            window_ms=args.batch_window_ms,
            max_queue=args.max_queue,
            timeout_s=args.timeout_s,
            api_key=key,
            allow_origins=args.allow_origin,
            heartbeat=heartbeat,
            keep_alive=keep_alive,
            max_body=None if args.max_body_mb is None else args.max_body_mb << 20,
        )
    finally:
        daemon.forget(record)
    return 0


# what a server started from the command line is given apart from its options: its address on its own, its API key
# in its environment, never on its command line, which `ps` shows, and its precision as such (`daemon.Wanted`)
SERVER_OWN = ("host", "port", "detach", "json", "start_timeout", "record_log", "api_key", "precision")


def _wanted(args: argparse.Namespace, models: Sequence[str], **how: Any) -> daemon.Wanted:
    """The server a command wants for `models`: at the command's precision, started with its other options."""
    options = forward_options(args, args.options, skip=SERVER_OWN)
    reads = asked_reads(args)  # what they change in answers, as given
    precision = args.precision or "auto"
    return daemon.Wanted(tuple(models), precision, tuple(options), args.host, args.port, reads=reads, **how)


def _server_client(args: argparse.Namespace) -> tuple[daemon.Client, str]:
    """A client of the server `--server` names (or the shared one, started when none runs, beside any that answers
    the model otherwise) and the name it answers the model by. ModelNotFound from the server it names when that one
    doesn't answer as the options given ask (another --precision, a server started without --literal-state): the
    server's own options decide, so a client's would be silently dropped."""
    client, served = daemon.answering(
        args.named, _wanted(args, [args.named]), _api_key(), args.start_timeout, args.server or None
    )
    assert served is not None
    return client, served.id


def _loaded_already(args: argparse.Namespace) -> str | None:
    """The address of a running server that answers for the model as this command asks, for a `decide` given no
    --server: the model is in memory there, so a second copy is never loaded beside it (an agent's one-off call
    inside `ippon claude` loaded Kettei Core's 7.6 GB again and its server saw nothing). None when none runs."""
    try:
        found = daemon.find(_wanted(args, [args.named]), _api_key())
    except DecisionError:  # one still starting that ps can't place: this command loads its own, as before
        return None
    return found.url if found else None


def show(text: str) -> None:
    """A line for a person to read on standard output, the `commands` in it in bold amber at a terminal."""
    print(terminal.commands(text, terminal.colour_on(sys.stdout)))


def _status(args: argparse.Namespace) -> int:
    """The model chosen and each server's state. At a terminal it is redrawn every second until Ctrl-C, so a pane can
    watch a server work, and the last state stays on screen after; once with --once or --json, or where the output
    isn't a terminal (an agent's shell, a pipe)."""
    if args.json:
        found = daemon.running()
        servers = [daemon.describe(s, _api_key()) for s in found]
        print(json.dumps({"servers": servers, "setup": asdict(onboarding.Settings.load())}, indent=2))
        return 0
    if not sys.stdout.isatty() or args.once:
        print("\n".join(_status_lines(live=False)))
        return 0
    sys.stdout.write("\x1b[?1049h")  # the terminal's other screen, as top uses: wrapped lines can't break a redraw
    try:
        while True:
            sys.stdout.write("\x1b[H\x1b[J" + "\n".join(_status_lines(live=True)) + "\n")
            sys.stdout.flush()
            time.sleep(STATUS_EVERY_S)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[?1049l")  # back to the user's own screen, which keeps the last state
        sys.stdout.flush()
    print("\n".join(_status_lines(live=False)))
    return 0


BAR_CELLS = 24  # a client's bar at its longest


def _client_bars(clients: Mapping[str, int], colour: bool) -> list[str]:
    """Each client's decisions as a bar against the busiest client's, one client a line, busiest first."""
    name, peak = max(map(len, clients)), max(clients.values())
    count = len(f"{peak:,}")
    return [
        f"{n:<{name}}  {terminal.styled(terminal.bar(v / peak, BAR_CELLS), colour, 'pink')}  {v:>{count},} decisions"
        for n, v in sorted(clients.items(), key=lambda kv: -kv[1])
    ]


def _status_lines(live: bool) -> list[str]:
    """What `ippon status` shows: each server's models by name, its address, then its traffic; the model setup chose
    when none runs. Watched live, the cat beside it, and the clients, how many are asking and each one's decisions as a
    bar, under it."""
    colour = terminal.colour_on(sys.stdout)

    def dim(text: str) -> str:
        return terminal.styled(text, colour, "2")

    def bold(text: str) -> str:
        return terminal.styled(text, colour, "1")

    found = daemon.running()
    servers = [daemon.describe(s, _api_key()) for s in found]
    settings = onboarding.Settings.load()
    top: list[str] = []  # beside the cat, live
    under: list[str] = []  # the client bars, the width of the terminal
    if not servers:
        started = "from login" if onboarding.from_login() else "when something needs it (`ippon start`, an agent)"
        chosen = settings.model and " ".join([names.title(settings.model), *settings.options()])
        top.append(
            bold(chosen) + dim(", chosen in setup")
            if chosen
            else terminal.commands("no model chosen yet · `ippon setup` chooses one", colour)
        )
        top.append(terminal.commands(f"not serving · it starts {started}", colour))
    if settings.everywhere:
        top.append(dim("in every session of " + ", ".join(agents.KINDS[a].name for a in settings.everywhere)))
    for record, server in zip(found, servers, strict=True):
        if top:
            top.append("")
        state = server.get("memory") or {}  # the server's own /v1/status
        held = {m["name"]: m["weights"] for m in state.get("models", []) if m.get("loaded")}
        for model in map(ServedModel.from_json, server["models"]):
            if "error" in server and server["status"] != "starting":  # a stuck server can't say what it has loaded
                top.append(bold(names.title(model.id)))
                continue
            shown = ("loaded" if model.loaded else "not loaded") + (
                f" · {held[model.id] / 1e9:.1f} GB" if held.get(model.id) else ""
            )
            licence = f" · licence: {model.licence}" if model.licence and not live else ""
            top.append(bold(names.title(model.id)) + dim(f" · {shown}{licence}"))
        top.append(terminal.styled(server["url"], colour, "pink") + dim(f" · pid {server['pid']} · {server['status']}"))
        if record.log and Path(record.log).name == onboarding.LOGIN_LOG and record.port != DEFAULT_PORT:
            # code that calls port 8000 reaches the program that held it at login, not ippon
            top.append(f"started at login on port {record.port}: another program held {DEFAULT_PORT} then")
        if "error" in server:
            top.append(server["error"]["message"])
        if traffic := state.get("traffic"):  # in progress too: a server reading a long batch answers none for a while
            top.append("")
            # the model's time per decision: how fast the models people run answer
            if (each := traffic.get("ms_per_decision")) is not None:
                quiet = traffic.get("last_answer_s")  # the server has gone quiet: its last speed, and how long ago
                top.append(
                    f"{each:.0f} ms per decision" + (dim(f" · last answer {terminal.ago(quiet)}") if quiet else "")
                )
            # a rate is a measure of steady traffic: an agent that asks now and then has its count alone, not "0.2"
            pace = traffic["questions_per_second"]
            steady = f"{pace:g} questions a second · " if pace >= 1 else ""
            top.append(f"{steady}{traffic['questions']:,} answered")
            if tokens := traffic.get("tokens"):
                reused = sum(m.get("tokens_reused", 0) for m in state.get("models", []))
                # 1% or more spared reading again: said beside what was read
                kept = f" · {reused / tokens:.0%} from earlier requests" if reused * 100 >= tokens else ""
                top.append(f"{terminal.compact(tokens)} tokens read{kept}")
            asking = traffic.get("clients_asking", traffic["in_progress"])
            # a client that opens a connection for each request (curl, most scripts) holds none between them, yet
            # one asking now is connected
            heading = f"{max(traffic['connections'], asking)} clients connected · {asking} asking"
            clients = traffic.get("by_client") or {}
            if live:
                under += ["", heading, *_client_bars(clients, colour)] if clients else ["", heading]
            else:
                top.append(heading)
                if clients:
                    top.append(" · ".join(f"{name} {count:,} decisions" for name, count in clients.items()))
    footer = terminal.commands("`ippon status --json` shows every detail", colour)
    if not live:
        return [*top, footer]
    header = [bold("ippon") + dim(f" {__version__} · status"), ""]
    art = brand.small_cat(colour) if colour else []
    width = shutil.get_terminal_size().columns - 1
    # the footer's longest form that fits the terminal: it is redrawn each second, so it never wraps
    said = next(
        (lead for lead in ("every second · Ctrl-C stops · ", "Ctrl-C stops · ") if len(lead) + 39 <= width),
        "Ctrl-C stops",
    )
    tail = footer if said.endswith("· ") else ""
    # a blank line over the cat, as `serve` prints one: the other screen starts it on the terminal's top line
    return ["", *brand.beside(art, header + top, width), *under, "", dim(said) + tail]


def _start(args: argparse.Namespace) -> int:
    # the name as typed (kettei-dot:q8), which `ippon status` shows, as for a server an agent starts
    wanted, key = _wanted(args, [args.named], keep_alive=None), _api_key(args)
    if daemon.find(wanted, key) is None:
        if others := daemon.running():
            # a server of another model (Kettei Dot for an app) is no reason not to start
            print(f"ippon: the server at {others[0].url} serves other models, so {args.named} starts beside it")
        _approve_before_detach([args.named], args)
    ready, started = daemon.ensure(wanted, key, args.start_timeout)
    if not started:
        daemon.match(daemon.Client(ready.url, key).models(), args.named, wanted.precision, wanted.reads)
        show(
            f"ippon is already serving {args.named} at {ready.url} (pid {ready.pid})"
            " · `ippon status` says what, `ippon stop` stops it"
        )
        return 0
    print(f"ippon is serving {args.named} at {ready.url} (pid {ready.pid}): POST /v1/decide or /v1/systemone")
    if note := daemon.moved(ready.port, args.port):
        print(note.lstrip("\n"))
    show("  `ippon logs`    follow what it does\n  `ippon stop`    stop it")
    return 0


def _logs(args: argparse.Namespace) -> int:
    servers = [s for s in daemon.running() if args.port is None or s.port == args.port]
    if not servers:
        raise Unavailable(
            "no ippon server is running" + (f" on port {args.port}" if args.port else " · `ippon start` starts one")
        )
    log = servers[0].log
    if not log:
        raise Unavailable(f"the server at {servers[0].url} runs in a terminal of its own, which shows its log")
    try:
        import subprocess

        return subprocess.run(
            ["tail", "-n", str(args.lines), *([] if args.no_follow else ["-f"]), log], check=False
        ).returncode
    except KeyboardInterrupt:
        return 0


INSTALLER_LINK_DIRS = ("~/.local/bin", "~/bin", "/opt/homebrew/bin", "/usr/local/bin")  # install.sh's, in its order


def installer_links() -> list[Path]:
    """The links install.sh made to this ippon command in a directory already on the PATH: a link to uv's own link
    of the command (never uv's link itself, which `uv tool uninstall` removes), which would dangle after it."""
    command = (Path(sys.executable).parent / "ippon").resolve()
    found = []
    for folder in INSTALLER_LINK_DIRS:
        link = Path(folder).expanduser() / "ippon"
        if link.is_symlink() and (link.parent / link.readlink()).is_symlink() and link.resolve() == command:
            found.append(link)
    return found


def _remove(args: argparse.Namespace) -> int:
    if args.what == "all":
        for line in onboarding.remove_all(args.models) or ["ippon had added nothing to take out"]:
            print(line)
        if not args.models:
            print("the models ippon downloaded stay in the Hugging Face cache: ippon remove all --models deletes them")
        for link in installer_links():
            link.unlink()
            print(f"removed {link}, the installer's link to the ippon command")
        show("now `uv tool uninstall ippon` removes the ippon command")
        return 0
    settings = onboarding.Settings.load()
    if args.what == "login":
        settings.serve("on demand")
        print("ippon no longer starts at login; ippon start starts it when you want it")
        return 0
    gone = settings.leave(args.what)
    name = agents.KINDS[args.what].name
    print(f"{name}: {gone}" if gone else f"ippon wasn't in {name}'s settings, so nothing changed")
    print(f"ippon {args.what} still brings them for a session")
    return 0


def _setup(args: argparse.Namespace) -> int:
    from . import pull
    from .device import detect

    asker: list[terminal.Terminal | None] = [None]  # who accepts a licence: a person at the terminal, --yes or not

    def get(name: str, approve: Callable[[Any], bool] | None, progress: Callable[[Any], None] | None) -> int:
        """Download `name`; the bytes it fetched (0: it was here already)."""
        plan = pull.prepare(name)
        needed = int(plan.report["disk"]["needed_bytes"])
        if approve is None and asker[0] is not None:  # --yes takes defaults, and a licence has none
            approve = onboarding.licence_question(asker[0])
        plan.run(accept_license=args.accept_license, approve=approve, progress=progress if needed else None)
        return needed

    def sample(name: str, options: Sequence[str]) -> str:
        """The first question of pull's sample, answered: what it asked, the answer and how long it took."""
        carried = names.precision_of(name)  # kettei-core:q8: its build, asked at 8 bits as it will run
        options = precision_argv(carried) if carried else options
        first = _sample(argparse.Namespace(model=str(names.resolve(name))), quiet=True, options=options)[0]
        qid, answer = next(iter(first["response"]["answers"].items()))
        asked = first["request"]["questions"][qid]["instructions"]
        said = f"yes {answer['noul']:.0%}" if answer["type"] == "noul" else f"{answer.get(answer['type'])}"
        return f"It answered “{asked}” {said}, in {first['ms']:.0f} ms"

    use = tuple(word.strip() for word in args.use.split(",")) if args.use is not None else None
    if use is not None and (unknown := set(use) - {*agents.AGENTS, "code"}):
        raise InvalidRequest(f"--use takes {', '.join([*agents.AGENTS, 'code'])}; not {', '.join(sorted(unknown))}")
    for agent in use or ():  # an agent that isn't installed is said before anything downloads
        if agent != "code":
            agents.installed(agent)
    answers = onboarding.Answers(args.name, use, args.every_session, args.from_login)
    with terminal.Terminal.open() as term:
        asker[0] = term
        onboarding.run(None if args.yes else term, detect(), onboarding.Steps(get, sample), answers)
    return 0


def _stop(args: argparse.Namespace) -> int:
    servers = daemon.running()
    if args.port is not None:
        selected = [s for s in servers if s.port == args.port]
        if not selected:
            raise Unavailable(f"no recorded ippon server uses port {args.port}")
    elif args.model is not None:  # by the name as typed: a revision it pins, and only that, must match
        selected = [s for s in servers if s.answers(args.named)]
        if not selected:
            raise Unavailable(f"no ippon server runs {args.named}")
    else:
        selected = servers
    if len(selected) > 1 and not args.all:
        # one server may be the command check's or an app's, so with several the user picks
        listed = "\n".join(f"  {', '.join(n for n, _ in s.models)} (port {s.port})" for s in selected)
        raise InvalidRequest(
            f"{len(selected)} ippon servers are running, so none was stopped:\n{listed}\n"
            f"name one (`ippon stop {selected[0].models[0][0]}`), or stop them all with `ippon stop --all`"
        )
    # a shared server's clients lose their model with it, so what is stopped says how many it had
    stopped = [{**daemon.stop(s), "connections": n} for s, n in [(s, _connections(s)) for s in selected]]
    if args.json:
        print(json.dumps({"stopped": stopped}, indent=2))
        return 0
    for server, was in zip(selected, stopped, strict=True):
        had = was["connections"]
        clients = f" · {had} client{'s were' if had > 1 else ' was'} connected to it" if had else ""
        print(f"stopped {', '.join(n for n, _ in server.models)} (port {server.port}){clients}")
    if not selected:
        print("no ippon server is running")
    return 0


def _connections(server: daemon.Server) -> int:
    """The clients connected to `server` now, besides this one; 0 for a server that doesn't say."""
    try:
        status = daemon.Client(server.url, _api_key()).call("/v1/status", timeout=daemon.DESCRIBE_TIMEOUT_S)
    except DecisionError:
        return 0
    return int(status["traffic"]["connections"])


def _mcp(args: argparse.Namespace) -> int:
    from . import mcp

    wanted = _wanted(args, args.model, offline=True)
    return mcp.run(wanted, args.url, _api_key(), args.start_timeout)


# what an agent command keeps for itself: the model is the server's first, and the key goes in its environment
AGENT_OWN = ("model", "start_timeout", "api_key")


def _agent(args: argparse.Namespace) -> int:
    agents.installed(args.command)
    key = _api_key(args)
    server = forward_options(args, server_options(), skip=AGENT_OWN)  # the shared server's own, should it start
    wanted = _wanted(args, [args.named], offline=True)
    wanted = replace(wanted, options=(*wanted.options, *server))
    client, model = daemon.answering(args.named, wanted, key, args.start_timeout)
    assert model is not None
    # what `ippon mcp` takes too, should it start the server again
    ippon = agents.mcp_command(args.named, forward_options(args, skip=AGENT_OWN))
    status = agents.run(
        lambda folder: agents.plan(args.command, ippon, client.url, model.id, key, args.agent_args, folder)
    )
    print(
        terminal.commands(
            f"ippon: the server at {client.url} keeps running for other clients; `ippon stop` stops it",
            terminal.colour_on(sys.stderr),
        ),
        file=sys.stderr,
    )
    return status


def common_options() -> Parser:
    """The options of every command that loads a model (a parent parser)."""
    common = Parser(add_help=False)
    common.set_defaults(loads=True)
    loading = common.add_argument_group("loading the model (defaults suit most uses)")
    loading.add_argument(
        "--accept-license", action="store_true", help="accept model licences before downloading new weights"
    )
    # the tuning options run unlisted (help=SUPPRESS): --profile forces a chip profile, m1..m5 (a lower one than this
    # Mac's); --switch sets operations' code paths over the profile's (profiles.Switches: "conv=mlx,gemm=mlx");
    # --chunk is the tokens per pass of a long state; --settings a bring-your-own model's ippon.json, from anywhere
    # (docs/bring-your-own.md); --keep-awake sends the GPU a trivial command every SECONDS for AWAKE_WINDOW_S after
    # each answer (default PERIOD_S)
    loading.add_argument("--profile", help=argparse.SUPPRESS)
    loading.add_argument("--switch", help=argparse.SUPPRESS)
    loading.add_argument(
        "--precision",
        default="auto",
        help="auto (default: fp16 where the model's activations fit, else bf16), bf16, fp16, fp32; or, changing the"
        " model (its weights quantized here, its answers checked with verify), q8, or q6 and q4, which warn of their"
        " drift from the creator's answers. A short name can carry it instead, as kettei-core:q8",
    )
    loading.add_argument("--chunk", type=int, default=DEFAULT_CHUNK, help=argparse.SUPPRESS)
    loading.add_argument("--settings", help=argparse.SUPPRESS)
    loading.add_argument("--keep-awake", type=float, metavar="SECONDS", help=argparse.SUPPRESS)
    loading.add_argument(
        "--prefix-budget",
        type=_bytes,
        help="bytes of kept prefixes (4G, 512M; 0 keeps none; default: min(4G, 1/16 of memory))",
    )
    loading.add_argument(
        "--answer-cache",
        type=int,
        nargs="?",
        const=ANSWERS_KEPT,
        metavar="N",
        help=f"keep the answers to the last N requests (default: {ANSWERS_KEPT:,}) and answer an identical request "
        "again without reading it; 0 keeps none, so every request is read",
    )
    loading.add_argument(
        "--literal-state",
        action="store_true",
        default=None,  # None: not given, so a server started otherwise answers too (`asked_reads`)
        help="read special tokens' strings inside a state as text, never as tokens (prompt injection; default: off)",
    )
    loading.add_argument(
        "--truncation",
        choices=TRUNCATIONS,
        help="input past the model's window: cut as its creator's code cuts it, listed in the response's 'truncated' "
        "(creator, the default), or refuse the request (error)",
    )
    loading.add_argument(
        "--window",
        type=int,
        metavar="TOKENS",
        help="a prompt's tokens at most, in place of the window its creator's code sets (`ippon lint --model M` shows "
        "it): longer prompts are read whole, unverified past the creator's",
    )
    return common


SERVER_GROUP = "the server (defaults suit most uses)"


def server_options() -> Parser:
    """The options of a server, beyond its models and address (a parent parser): `serve`'s, and the agent commands'
    for the shared server they start."""
    server = Parser(add_help=False)
    tuning = server.add_argument_group(SERVER_GROUP)
    tuning.add_argument(
        "--memory-gb",
        type=float,
        help="memory the loaded models may hold together, past which the least recently used unload "
        "(default: what one model may take of this Mac's GPU working set)",
    )
    tuning.add_argument("--max-batch", type=int, default=policy.MAX_BATCH, help="requests answered together at most")
    tuning.add_argument(
        "--batch-window-ms",
        type=float,
        default=0.0,
        help="wait this long after a request for more to answer with it (default 0: never wait)",
    )
    tuning.add_argument(
        "--max-queue",
        type=int,
        default=policy.MAX_QUEUE,
        help="requests waiting at most; past that a request gets a 503",
    )
    tuning.add_argument(
        "--timeout-s", type=float, help="seconds a request may take before its caller gets a 504 (default: no deadline)"
    )
    tuning.add_argument(
        "--log", action="store_true", help="a line per answer on stderr even when it isn't a terminal (a log file)"
    )
    # unlisted (help=SUPPRESS): --no-keep-awake sends the GPU no trivial command while idle, and --keep-awake-window
    # sends it for this long after the last request, then none until the next (default: --policy's)
    tuning.add_argument("--no-keep-awake", action="store_true", help=argparse.SUPPRESS)
    tuning.add_argument(
        "--policy",
        choices=tuple(policy.POLICIES),
        default=policy.DEFAULT,
        help="what the server keeps ready between requests: balanced (the default) keeps the models' memory wired for "
        f"{AWAKE_WINDOW_S:g} s after a request, latency keeps it wired as long as the server runs, and battery keeps "
        f"nothing wired and unloads a model that has answered nothing for {AWAKE_WINDOW_S:g} s. After an idle spell "
        "the first request is slower under balanced and battery. --keep-alive overrides when a model unloads",
    )
    tuning.add_argument("--keep-awake-window", type=float, metavar="SECONDS", help=argparse.SUPPRESS)
    tuning.add_argument(
        "--keep-alive",
        type=_keep_alive,
        metavar="DURATION",
        help="unload a model that has answered nothing for this long (30s, 10m, 1h; 0: right after answering; -1 or "
        "forever: never), loading it again on its next request (default: --policy's, never but for battery's 5m)",
    )
    tuning.add_argument(
        "--api-key",
        metavar="KEY",
        help="every request needs 'Authorization: Bearer KEY' (default: $IPPON_API_KEY, which `ps` doesn't "
        "show; unset: no key)",
    )
    tuning.add_argument(
        "--max-body-mb",
        type=int,
        metavar="MB",
        help="the most MiB a request body may hold, its images, clips and videos in base64 included (default: a "
        "32nd of this Mac's memory, from 64 to 1,536: 256 on an 8 GB Mac, 1,024 on 32 GB)",
    )
    tuning.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        metavar="ORIGIN",
        help="a web origin (https://app.example) whose pages may call the server, again for each; '*' for any "
        "(default: none, so a request a browser marks with an Origin is refused)",
    )
    return server


def starting_options(address: bool = False) -> Parser:
    """The options of a command that may start a server (a parent parser): how long to wait for it, and with
    `address` where it listens."""
    starting = Parser(add_help=False)
    if address:
        starting.add_argument("--host", default="127.0.0.1", help="the address to listen on (default: this Mac only)")
        starting.add_argument("--port", type=int, help="port (default: 8000, or a free port if occupied)")
    starting.add_argument_group(SERVER_GROUP).add_argument(
        "--start-timeout",
        type=float,
        default=daemon.START_TIMEOUT_S,
        help=f"seconds to wait for the server to start and load its first model (default {daemon.START_TIMEOUT_S:g})",
    )
    return starting


class Help(argparse.HelpFormatter):
    """A command's --help: its usage is its positionals and [options], since each option is listed below with what
    it does, and text written in lines (the examples) keeps them, while a description wraps to the terminal."""

    def _format_usage(
        self,
        usage: str | None,
        actions: Iterable[argparse.Action],
        groups: Iterable[argparse._MutuallyExclusiveGroup],
        prefix: str | None,
    ) -> str:
        listed = list(actions)
        positionals = [action for action in listed if not action.option_strings]
        text = super()._format_usage(usage, positionals, [], prefix)
        return text.replace(self._prog, f"{self._prog} [options]", 1) if len(listed) > len(positionals) else text

    def _fill_text(self, text: str, width: int, indent: str) -> str:
        if "\n" in text:
            return "".join(indent + line for line in text.splitlines(keepends=True))
        return super()._fill_text(text, width, indent)


class Parser(argparse.ArgumentParser):
    """ippon's command line: its help as `Help` writes it, and a mistake in it said in one line with the command that
    shows how to use it, instead of the whole usage block above the error."""

    def __init__(self, *args: Any, **how: Any) -> None:
        super().__init__(*args, **{"formatter_class": Help, **how})

    def error(self, message: str) -> NoReturn:
        colour = terminal.colour_on(sys.stderr)
        typed = re.search(r"invalid choice: '([^']*)'", message)
        if typed and self.prog == "ippon":  # a mistyped command: the nearest of the ones --help lists
            import difflib

            near = difflib.get_close_matches(typed[1], HELP_ORDER, n=1)
            guess = f" Did you mean `ippon {near[0]}`?" if near else ""
            self.exit(
                2, terminal.commands(f"ippon: {typed[1]} isn't a command.{guess} `ippon --help` lists them.\n", colour)
            )
        command = self.prog
        if message.startswith("unrecognized arguments"):
            if command == "ippon":  # found after the command's own parse
                named = next((word for word in sys.argv[1:] if not word.startswith("-")), None)
                command = f"ippon {named}" if named in HELP_ORDER else command
            # the options' names only: what else was typed is never echoed, since a prompt in it can hold a secret
            options = dict.fromkeys(re.findall(r"(?<!\S)--[a-z][a-z-]*", message))
            are = "aren't options" if len(options) > 1 else "isn't an option"
            message = f"{', '.join(options)} {are} it takes" if options else "unrecognized arguments"
        hint = terminal.commands(f"`{command} --help` shows how to use it", colour)
        self.exit(2, f"{command}: {message}\n  {hint}\n")


def stopped(command: str) -> str:
    """What Ctrl-C leaves behind, said when a command stops: setup says how to start again, since a model it was
    downloading keeps the files it has (the next download carries on from them)."""
    if command == "setup":
        return (
            "\n\n  Setup stopped before it finished. Run `ippon setup` to start again whenever you like, and a model it"
            "\n  was downloading carries on from where it stopped.\n"
        )
    return f"\n\n  Stopped `ippon {command}`. Run it again whenever you like.\n"


def _ippon_warnings_once(show: Callable[..., None]) -> Callable[..., None]:
    """`warnings.showwarning` that prints each warning ippon's own code raises once, as ippon prints its other
    messages (one line, no file and line of the code), and hands another library's to `show`."""
    shown: set[str] = set()
    package = str(Path(__file__).resolve().parent)

    def showing(message: Warning | str, category: type[Warning], filename: str, *rest: Any, **named: Any) -> None:
        if not str(Path(filename).resolve()).startswith(package):
            show(message, category, filename, *rest, **named)
        elif str(message) not in shown:
            shown.add(str(message))
            print(f"ippon: ⚠ {message}", file=sys.stderr)

    return showing


def _approve_licence(args: argparse.Namespace) -> Callable[[Any], bool] | None:
    """Interactive terminal licence prompt for any CLI command that downloads a custom-licence model. Prompts on
    stderr so stdout stays clean; disabled for --json, stdio MCP, or non-interactive/piped shells."""
    if args.json or args.command == "mcp" or not (sys.stdin.isatty() and sys.stderr.isatty()):
        return None

    def _ask(licence: Any) -> bool:
        print(f"\n{licence.name}", file=sys.stderr)
        print(f"  {licence.url}", file=sys.stderr, flush=True)
        try:
            sys.stderr.write("Accept licence and download? [y/N] ")
            sys.stderr.flush()
            return sys.stdin.readline().strip().lower() in {"y", "yes"}
        except EOFError:
            return False

    return _ask


def _approve_before_detach(specs: Sequence[str], args: argparse.Namespace) -> None:
    """For `ippon start` / `ippon serve --detach` in an interactive terminal, prompt for any uncached custom-licence
    model before spawning the detached child (which has stdin=/dev/null)."""
    if args.accept_license or _approve_licence(args) is None:
        return
    from . import locate, pull

    for spec in specs:
        model = spec.split("=", 1)[-1]
        resolved = names.resolve(model)
        if isinstance(resolved, Path) or locate.downloaded(resolved) is not None:
            continue
        plan = pull.prepare(model)
        for licence in plan.licenses:
            licence.require(model)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    typed = ["ippon", *argv]  # a licence's hint: this command again, accepting (before an agent's own words after --)
    typed.insert(typed.index("--") if "--" in typed else len(typed), "--accept-license")
    common, server = common_options(), server_options()
    waiting, listening = starting_options(), starting_options(address=True)
    parser = Parser(
        prog="ippon",
        description="Run decision models on this Mac: answer, serve, pull, check and time them.",
        epilog=START,
    )
    parser.add_argument("--version", action="version", version=f"ippon {__version__}")
    # what main reads of every command, each command's parser declaring its own: whether it loads a model with the
    # common options, takes setup's model when none is named, and the options only some commands have
    parser.set_defaults(
        loads=False, from_setup=False, model=None, named=None, precision=None, server=None, start_timeout=None,
        port=None, host="127.0.0.1", accept_license=None, api_key=None, options=None, json=False,
    )  # fmt: skip
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND", title="commands")

    def subcommand(name: str, **how: Any) -> argparse.ArgumentParser:
        """A command, whose own --help opens with what `ippon --help` says of it and ends with it in use."""
        said = how.get("help")
        if said and "description" not in how:
            how["description"] = f"{said[0].upper()}{said[1:]}."
        return commands.add_parser(name, **{"epilog": examples(name), **how})

    setting_up = subcommand(
        "setup",
        help="choose a decision model for this Mac, download it, and connect your agents",
        description="Choose a decision model for this Mac (each listed with what it reads and whether it fits, "
        "Sqwish Labs' first), download it, answer a first question, then choose what to use it with and how the "
        "server runs. Enter alone takes each step's default. Run it again to change any of it.",
    )
    setting_up.set_defaults(handler=_setup)
    setting_up.add_argument(
        "name", nargs="?", help="a model's short name, to skip choosing (kettei-core:q8: at 8 bits)"
    )
    setting_up.add_argument("--yes", "-y", action="store_true", help="ask nothing: each step's default")
    setting_up.add_argument(
        "--use",
        metavar="LIST",
        help="what to use it with, without asking: claude, codex, opencode, pi, code (your own)",
    )
    setting_up.add_argument(
        "--every-session",
        action=argparse.BooleanOptionalAction,
        help="add it to those agents' own settings, for every session (default: only with ippon claude and the like)",
    )
    setting_up.add_argument(
        "--from-login",
        action=argparse.BooleanOptionalAction,
        help="serve it from login (default: start it when something needs it)",
    )
    setting_up.add_argument("--accept-license", action="store_true", help="accept the model's licence")
    for agent in agents.AGENTS:
        launching = subcommand(
            agent,
            allow_abbrev=False,  # an agent's own --print is never taken for one of ippon's options
            parents=[common_options(), server, waiting],
            help=f"{agents.KINDS[agent].name} with the decision model beside it, for one session",
            description=f"Start {agents.KINDS[agent].name} for this session with the decision model beside it: "
            "ippon's MCP server, which starts the shared ippon server or reuses the one running"
            + (", and ippon's skill" if agent == "claude" else "")
            + ". Nothing is written to its configuration (ippon setup can add it to every session), and the server "
            "keeps running after the agent exits (ippon stop stops it).",
        )
        launching.set_defaults(handler=_agent, precision=None, agent_args=[], from_setup=True)
        launching.add_argument(
            "--model",
            help=f"the model it asks (default: the one ippon setup chose, else {DEFAULT_MODEL}); {MODEL}",
        )
    starting = subcommand(  # common options of its own: its --precision defaults to setup's choice
        "start",
        parents=[common_options(), server, listening],
        help="serve the model to your own code, in the background",
        description="Serve the model to your own code, in the background. It answers POST /v1/decide (the Sqwish Labs "
        "SDK's) and /v1/systemone at the address printed, and ippon status, ippon logs and ippon stop look after it.",
    )
    starting.set_defaults(handler=_start, options=starting, precision=None, from_setup=True)
    starting.add_argument("model", nargs="?", help=f"default: the one ippon setup chose; {MODEL}")
    deciding = subcommand(
        "decide",
        aliases=["run"],
        parents=[common, waiting],
        help="answer a System One request",
        description="Answer a System One request: a state to judge and the questions to ask about it, each a yes or "
        "no (noul), one of named options (choice) or a level on a scale (score). At a terminal it prints a line per "
        "answer with its probability, and to a pipe the response as JSON.",
        epilog=f"{examples('decide')}\n\n{REQUEST}",
    )
    deciding.set_defaults(handler=_run)
    deciding.add_argument("model", help=MODEL)
    deciding.add_argument("request", nargs="?", help="a JSON request file, JSON text, or - to read it from stdin")
    deciding.add_argument(
        "--states",
        metavar="FILE.jsonl",
        help="ask every line's state (a JSON string, or an object with any of "
        "'state', 'files', 'images', 'audio', 'videos', and optionally an 'id') the request's questions, in batches; "
        "one JSON line out per line in: its response, or its error",
    )
    for flag, field in MEDIA_FLAGS.items():
        deciding.add_argument(
            f"--{flag}",
            action="append",
            metavar="PATH",
            help=f"a file added to the request's {field}, for a model that reads them (repeat it for more)",
        )
    deciding.add_argument(
        "--batch",
        type=int,
        help=f"with --states: lines asked at once (default {policy.MAX_BATCH}, {policy.IN_FLIGHT} with --server)",
    )
    deciding.add_argument(
        "--json",
        action="store_true",
        help="the full System One response as JSON, errors too (the default when the output isn't a terminal)",
    )
    deciding.add_argument(
        "--server",
        nargs="?",
        const="",
        metavar="URL",
        help="use the shared server; optionally its local HTTP URL; start one if absent",
    )
    listing = subcommand("status", aliases=["ps"], help="the model chosen, the server and what it has loaded, live")
    listing.set_defaults(handler=_status)
    listing.add_argument("--json", action="store_true", help="print every server and its models as JSON")
    listing.add_argument("--once", action="store_true", help="print it once (at a terminal it refreshes every second)")
    stopping = subcommand("stop", help="stop a background server and give its memory back")
    stopping.set_defaults(handler=_stop)
    stopping.add_argument(
        "model", nargs="?", help="the server running this model, as `ippon status` names it (default: the only one)"
    )
    stopping.add_argument("--all", action="store_true", help="stop every server running")
    stopping.add_argument("--port", type=int, help="the server on this port")
    stopping.add_argument("--json", action="store_true", help="print what was stopped as JSON")
    following = subcommand("logs", help="follow the background server's log")
    following.set_defaults(handler=_logs)
    following.add_argument("--port", type=int, help="the server on this port (default: the oldest running)")
    following.add_argument("-n", "--lines", type=int, default=40, help="lines of the log to show first (default 40)")
    following.add_argument("--no-follow", action="store_true", help="show them and stop")
    pulling = subcommand("pull", help="download a model by its short name, if it fits this Mac")
    pulling.set_defaults(handler=_pull)
    pulling.add_argument("model", nargs="?", help="a short name (kettei-core, kettei-core:q8) or a Hub id")
    pulling.add_argument(
        "--list", action="store_true", help="every short name by family: weights, status, licence, the recommended one"
    )
    pulling.add_argument("--accept-license", action="store_true", help="accept the shown model licence for downloads")
    pulling.add_argument("--json", action="store_true", help="print it as JSON")
    pulling.add_argument("--no-sample", action="store_true", help="don't answer a sample request once it is downloaded")
    pulling.add_argument(
        "--dry-run",
        action="store_true",
        help="say what it needs, whether it fits and its licence, then download nothing",
    )
    recommending = subcommand(
        "recommend", help="the builds this Mac can run for text, images, audio or videos, and where to start"
    )
    recommending.set_defaults(handler=_recommend)
    recommending.add_argument(
        "task", nargs="?", default="text", choices=list(recommend.TASKS), help="what the requests carry (default: text)"
    )
    recommending.add_argument("--json", action="store_true", help="print it as JSON")
    serving = subcommand("serve", parents=[common, server, listening], help="serve models' System One API over HTTP")
    serving.set_defaults(handler=_serve, options=serving)
    serving.add_argument(
        "model",
        nargs="*",
        help="the model (default: the one ippon setup chose), or several (NAME=MODEL names one as requests name it), "
        "each a short name (kettei-core, kettei-dot), a Hub id or a folder: the first loads at start, each other on "
        "its first request",
    )
    serving.add_argument("--detach", action="store_true", help="start in the background and print its address and PID")
    serving.add_argument(
        "--json", action="store_true", help="with --detach: print the server record as JSON; errors as JSON"
    )
    serving.add_argument("--record-log", help=argparse.SUPPRESS)
    linting = subcommand("lint", parents=[common], help="what to fix in a request before a model answers it")
    linting.set_defaults(handler=_lint)
    linting.add_argument("request", nargs="?", help="a JSON request file, JSON text, or - to read it from stdin")
    linting.add_argument(
        "--model", help=f"also render it as this model reads it (its tokens, what it can't be asked): {MODEL}"
    )
    for flag, field in MEDIA_FLAGS.items():
        linting.add_argument(
            f"--{flag}",
            action="append",
            metavar="PATH",
            help=f"a file added to the request's {field}, as on decide (repeat it for more)",
        )
    linting.add_argument("--json", action="store_true", help="print the problems (and prompts) as JSON")
    linting.add_argument("--prompt", action="store_true", help="with --model: also print the text the model reads")
    evaluating = subcommand(
        "eval", parents=[common], help="labelled examples answered by each model: accuracy, F1, thresholds"
    )
    evaluating.set_defaults(handler=_eval)
    evaluating.add_argument(
        "data", help='a JSONL file of labelled rows: {"state", "questions", "labels": {id: expected}}'
    )
    evaluating.add_argument(
        "--model",
        action="append",
        required=True,
        help=f"a model to evaluate, again for each (one loaded at a time): {MODEL}",
    )
    evaluating.add_argument("--questions", help="a request file whose questions every row without its own is asked")
    evaluating.add_argument(
        "--out", help="the results file, a JSON line per row and model (default: DATA.results.jsonl)"
    )
    evaluating.add_argument("--batch", type=int, help=f"rows answered together at most (default {policy.MAX_BATCH})")
    evaluating.add_argument(
        "--min-precision",
        type=float,
        default=0.0,
        help="a yes/no question's flagging threshold must keep precision at least this (0 to 1; default 0)",
    )
    evaluating.add_argument(
        "--min-recall",
        type=float,
        default=0.0,
        help="a yes/no question's flagging threshold must keep recall at least this (0 to 1; default 0)",
    )
    evaluating.add_argument(
        "--folds", type=int, default=5, help="held-out folds that check the chosen flagging threshold (default 5)"
    )
    evaluating.add_argument("--json", action="store_true", help="print the report as JSON")
    verify = subcommand("verify", parents=[common], help="check a model against its creator's reference")
    verify.set_defaults(handler=_verify)
    verify.add_argument("model", help=MODEL)
    verify.add_argument("--reference", help="a reference file (.json or .json.gz)")
    verify.add_argument(
        "--invariance",
        action="store_true",
        help="check instead that the answers don't depend on how they are run (chunks, kept states, order, batches)",
    )
    verify.add_argument(
        "--requests", help="with --invariance or --against and no reference: a request set ([{name, request}])"
    )
    verify.add_argument(
        "--against",
        metavar="ARM=PYTHON",
        help="check a build against the runtime it was published for, on the same weights: an arm of bench --compare "
        "(community-clef, ...) and the Python that has it",
    )
    verify.add_argument(
        "--bound",
        type=float,
        help="with --against: the largest probability difference (default: the arm's own precision's distance from "
        "its fp32, at least 0.01)",
    )
    verify.add_argument("--json", action="store_true", help="print the report as JSON")
    timing = subcommand("bench", parents=[common], help="time a model on this Mac")
    timing.set_defaults(handler=_bench, options=timing)
    timing.add_argument("model", help=MODEL)
    timing.add_argument("--sizes", default="300,1000,2000,4000", help="state sizes in words")
    timing.add_argument("--questions", default="1,3,5", help="questions per request to time, comma separated")
    timing.add_argument(
        "--scenario",
        default="fresh",
        choices=("fresh", "repeat", "grow", "idle", "concurrent"),
        help="how each request relates to the one before",
    )
    timing.add_argument("--clients", default="1,4,8", help="--scenario concurrent: clients sending at once")
    timing.add_argument(
        "--batch-window-ms", type=float, default=0.0, help="--scenario concurrent: serve's batching window"
    )
    timing.add_argument(
        "--max-batch", type=int, help="--scenario concurrent: serve's --max-batch, requests answered together at most"
    )
    timing.add_argument("--gaps", default="1,5,15,30", help="seconds idle before each request (--scenario idle)")
    timing.add_argument("--n", type=int, default=20, help="timed requests per cell")
    timing.add_argument("--warmup", type=int, default=2, help="untimed requests before each timed shape")
    timing.add_argument("--seed", type=int, default=1, help="the requests' seed: the same seed times the same requests")
    timing.add_argument("--ab", nargs=2, metavar=("A", "B"), help="compare two arms: extra options and NAME=value env")
    timing.add_argument(
        "--compare",
        nargs="+",
        metavar="ARM=PYTHON",
        help="time ippon against other runtimes of the model: mlx-vlm-model=/path/bin/python, community-clef=...",
    )
    timing.add_argument("--rounds", type=int, default=4, help="fresh processes per arm with --ab or --compare")
    timing.add_argument("--out", help="write the results as JSON")
    doctoring = subcommand("doctor", help="this Mac, how ippon runs on it, and what stands in its way")
    doctoring.set_defaults(handler=_doctor)
    doctoring.add_argument("--json", action="store_true", help="print the checks as JSON: [{status, what, fix}]")
    removing = subcommand(
        "remove", help="take ippon out of an agent's settings, stop starting at login, or (all) take everything out"
    )
    removing.set_defaults(handler=_remove)
    removing.add_argument(
        "what",
        choices=(*agents.AGENTS, "login", "all"),
        help="an agent setup added ippon to, login (stop serving from login), or all (everything ippon added)",
    )
    removing.add_argument("--models", action="store_true", help="with all: delete the models ippon downloaded too")
    mcping = subcommand(
        "mcp", parents=[common, waiting], help="ippon's tools for MCP clients, all sharing one model server"
    )
    mcping.set_defaults(handler=_mcp)
    mcping.add_argument("model", nargs="*", help="downloaded models to serve if no server is running")
    mcping.add_argument("--url", help=f"an existing local HTTP server, e.g. {url_of(DEFAULT_PORT)}")
    skilling = subcommand("skills", help="guides that teach your agent to use ippon, listed when none is named")
    skilling.set_defaults(handler=_skills)
    skilling.add_argument("name", nargs="?", help="the skill to print")
    skilling.add_argument("reference", nargs="?", help="one of its references to print")
    skilling.add_argument("--path", action="store_true", help="print the folder holding the skills, to copy or link")
    skilling.add_argument("--json", action="store_true", help="print the list and errors as JSON")
    # run but not listed (no help): what `pull --dry-run` (check), `lint --model --prompt` (prompt) and `doctor`
    # (info) show, and the maintainers' tools (profile, suite, new-adapter)
    informing = subcommand("info", parents=[common], description="this Mac, its profile and the installed adapters")
    informing.set_defaults(handler=_info)
    informing.add_argument("--json", action="store_true", help="print it as JSON")
    scaffold = subcommand("new-adapter", description="write an adapter package for a decision model")
    scaffold.set_defaults(handler=_new_adapter)
    scaffold.add_argument("name", help="the adapter's name, creator/model")
    scaffold.add_argument(
        "--arch", default="qwen35", help="the layers' module (default qwen35; another is named if unknown)"
    )
    scaffold.add_argument("--out", help="the directory to write (default: <model>-ippon)")
    checking = subcommand("check", description="what a model needs and what is verified, before downloading it")
    checking.set_defaults(handler=_check)
    checking.add_argument("model", help=MODEL)
    checking.add_argument("--adapter", help="the adapter to run it with (default: the one that matches it)")
    checking.add_argument("--json", action="store_true", help="print the report as JSON")
    prompting = subcommand(
        "prompt", parents=[common], description="show what a model reads for a request, and what was cut to fit"
    )
    prompting.set_defaults(handler=_prompt)
    prompting.add_argument("model", help=MODEL)
    prompting.add_argument("request", nargs="?", help="a JSON request file, JSON text, or - to read it from stdin")
    prompting.add_argument("--ids", action="store_true", help="the token ids too")
    profiling = subcommand("profile", parents=[common], description="where a request's time goes, by operation")
    profiling.set_defaults(handler=_profile)
    profiling.add_argument("model", help=MODEL)
    profiling.add_argument("--request", help="a System One request file (default: bench's request of --size words)")
    profiling.add_argument("--size", type=int, default=1000, help="state size in words of bench's request")
    profiling.add_argument("--questions", type=int, default=3, help="questions in bench's request")
    profiling.add_argument("--repeat", type=int, default=5, help="times the request is profiled")
    profiling.add_argument("--phases", action="store_true", help="time by phase: render, plan, graph, GPU, readout")
    profiling.add_argument(
        "--scenario", choices=("fresh", "changed"), default="fresh", help="with --phases: new states, or new questions"
    )
    suite = subcommand("suite", description="verify, check and time every suite model on this Mac")
    suite.set_defaults(handler=_suite)
    suite.add_argument("--models", required=True, help="the folder holding the models (see docs/suite.md)")
    suite.add_argument("--out", required=True, help="the folder to write results into")
    suite.add_argument("--quick", action="store_true", help="fewer shapes and requests, no invariance checks")
    suite.add_argument("--arm", nargs="+", metavar="ARM=PYTHON", help="runtimes to compare with: mlx-vlm-model=PYTHON")
    suite.add_argument("--only", help="comma-separated suite model names")
    if not argv:  # ippon alone: who it is, and where to start
        colour = terminal.colour_on(sys.stdout)
        if sys.stdout.isatty():
            print("\n".join(["", *brand.banner(colour, [f"ippon {__version__}"]), ""]))
        print(f"  {terminal.styled('Start here', colour, '1')}")
        for line in START.splitlines()[1:]:
            command, _, what = line.strip().partition("  ")
            print(f"    {terminal.styled(command.ljust(15), colour, '1', 'amber')}{what.strip()}")
        print("\n  ippon --help lists every command, and ippon COMMAND --help each one's options\n")
        return 0
    passed: list[str] = []
    if argv[:1] and argv[0] in agents.AGENTS and "--" in argv:  # what follows -- is the agent's, unparsed
        argv, passed = argv[: argv.index("--")], argv[argv.index("--") + 1 :]
    if argv[:1] and argv[0] in agents.AGENTS:  # what ippon doesn't take is the agent's too: ippon claude -p "..."
        args, theirs = parser.parse_known_args(argv)
        passed = theirs + passed
    else:
        args = parser.parse_args(argv)
    if passed:
        args.agent_args = passed
    try:
        if args.loads:
            validate_load_options(
                chunk=args.chunk,
                window=args.window,
                prefix_budget=args.prefix_budget,
                answer_cache=args.answer_cache if args.answer_cache is not None else ANSWERS_KEPT,
                keep_awake=args.keep_awake,
                truncation=args.truncation or Reads.truncation,
            )
        if args.from_setup:  # the model setup chose, unless one is named
            if args.model is None:
                args.model, chosen = onboarding.chosen_model()
                args.precision = args.precision or chosen
            args.precision = args.precision or "auto"
        if isinstance(args.model, str):  # one model, by a short name: its build, from here on (names.py)
            args.named = args.model
            if args.precision is not None:  # kettei-core:q8 asks for 8 bits as --precision q8 does
                args.precision = names.named_precision(args.model, args.precision)
            if args.command == "decide" and args.server is None:
                args.server = _loaded_already(args)
            if args.server is None:
                args.model = str(names.resolve(args.model))
        if args.start_timeout is not None and (not math.isfinite(args.start_timeout) or args.start_timeout <= 0):
            raise InvalidRequest("--start-timeout must be finite and positive")
        if args.port is not None and not 0 <= args.port <= 65535:
            raise InvalidRequest("--port must be between 0 and 65535")
        handler: Callable[[argparse.Namespace], int] = args.handler
        # a licence's hint repeats the name as typed, and a model below 8 bits says so once
        rerun = shlex.join(typed) if args.accept_license is not None else None
        with accepting(asked=args.named, rerun=rerun, approve=_approve_licence(args)), warnings.catch_warnings():
            warnings.showwarning = _ippon_warnings_once(warnings.showwarning)
            return handler(args)
    except (DecisionError, ImportError) as error:  # the same typed body for startup and command failures
        failure = error if isinstance(error, DecisionError) else RuntimeUnavailable.of(error)
        if failure is None:  # a fault inside ippon, not the user's installation
            raise
        if args.json:
            print(json.dumps({"error": error_json(failure)}))
        print(terminal.commands(f"ippon: {failure}", terminal.colour_on(sys.stderr)), file=sys.stderr)
        return failure.exit_code
    except KeyboardInterrupt:  # Ctrl-C while waiting (a download, another process's lock): stopped, not a crash
        print(terminal.commands(stopped(args.command), terminal.colour_on(sys.stderr)), file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
