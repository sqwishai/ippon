"""One ippon server per Mac, shared by every client: `ippon serve --detach`, `ps`, `stop`,
`decide --server` and `ippon mcp` find it, start it and talk to it here.

Every `serve` on this Mac records itself in the ippon cache (servers/PORT.json: its pid, its address, the
models it answers for and how it keeps them, and its log) from before its first model loads until it stops; a reader
drops a record whose process has gone (`running`). A client finds the server there and asks it over HTTP
(`Client`), sending the API key in IPPON_API_KEY when there is one, so that every client (each coding agent's
MCP server, each `decide --server`) asks one pool of loaded models within one memory budget and none loads a copy of
its own. When none runs, `ensure` starts one in a session of its own, so it outlives the client that started it,
logging to servers/PORT.log; a lock file makes clients that start together share the one they start.

A server listens on port 8000 unless that is taken, then on a free one, and records the port it got.
"""

from __future__ import annotations

import fcntl
import http.client
import json
import os
import shlex
import signal
import socket
import subprocess
import threading
import time
import urllib.parse
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing, contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any
from urllib.parse import urlsplit

from . import __version__, errors, names, processes
from .errors import DecisionError, InvalidRequest, ModelNotFound, Unavailable, UnsupportedModel
from .options import precision_argv
from .policy import IN_FLIGHT
from .served import CLIENT_ENV, CLIENT_HEADER, DEFAULT_PORT, LOOPBACK, ServedModel, is_loopback, reach, url_of
from .storage import cache_dir, write_json_atomic

API_KEY_ENV = "IPPON_API_KEY"  # the API key clients send and a server started without --api-key needs
START_TIMEOUT_S = 120.0  # seconds a client waits for a server it started to record itself
POLL_S = 0.1  # seconds between looks at a server that is starting or stopping
HTTP_TIMEOUT_S = 300.0  # a short call's wait (status, models); a decision waits for its answer (serve has no deadline)
DESCRIBE_TIMEOUT_S = 3.0  # `describe`'s wait: a live server answers it from its HTTP threads in ms, however busy
STOP_TIMEOUT_S = 30.0  # seconds `stop` waits for a server to answer what waits and unload, before it is killed
# keep-alive of a server a client starts (`ensure`: mcp, decide --server): a server nobody started by hand gives back
# its models' memory after STARTED_IDLE_MIN minutes without a request, and loads them again on the next
STARTED_IDLE_MIN = 30
STARTED_KEEP_ALIVE = f"{STARTED_IDLE_MIN}m"
SERVERS = "servers"  # the folder in ippon's cache servers record themselves and log in: all this module writes


def folder() -> Path:
    """Where servers record themselves and log: servers/ in the ippon cache, made readable by this user only."""
    path = cache_dir() / SERVERS
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


@dataclass(frozen=True)
class Server:
    """A running `ippon serve`, as it recorded itself: `models` are (the name requests use, what it loads),
    `policy` how it keeps them (`policy.policy_json`), `api_key` whether every request needs one (never the key itself),
    `status` "starting" until it listens, then "serving"."""

    pid: int
    port: int
    url: str
    models: tuple[tuple[str, str], ...]
    status: str = "starting"
    started: float = field(default_factory=time.time)
    policy: Mapping[str, Any] = field(default_factory=dict)
    api_key: bool = False
    log: str | None = None
    version: str = __version__

    def to_json(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "port": self.port,
            "url": self.url,
            "status": self.status,
            "started": self.started,
            "models": [{"name": name, "model": model} for name, model in self.models],
            "policy": dict(self.policy),
            "api_key": self.api_key,
            "log": self.log,
            "version": self.version,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Server:
        return cls(
            pid=int(data["pid"]),
            port=int(data["port"]),
            url=str(data["url"]),
            models=tuple((str(m["name"]), str(m["model"])) for m in data["models"]),
            status=str(data["status"]),
            started=float(data["started"]),
            policy=dict(data.get("policy") or {}),
            api_key=bool(data.get("api_key")),
            log=data.get("log"),
            version=str(data.get("version", "")),
        )

    def answers(self, asked: str) -> bool:
        """Whether the server answers for `asked` at any weights (`ServedModel.answers`), by the record it keeps: the
        name a model was started with (kettei-dot:q8, as `ippon status` shows it) or what it loads, so kettei-core
        finds a server started as kettei-core:q8."""
        return any(ServedModel(name, model).answers(asked) for name, model in self.models)


def record(server: Server) -> None:
    """Write `server`'s record, whole or not at all."""
    write_json_atomic(folder() / f"{server.port}.json", server.to_json(), indent=1, mode=0o600)


def forget(server: Server) -> None:
    """Remove `server`'s record, if it is still its own (not a later server's on the same port)."""
    path = folder() / f"{server.port}.json"
    try:
        if json.loads(path.read_text())["pid"] == server.pid:
            path.unlink()
    except (OSError, ValueError, KeyError):
        pass


def started_at(pid: int) -> float | None:
    """When process `pid` started (seconds since the epoch, to the second), or None when `ps` can't say."""
    try:
        text = subprocess.run(
            ["/bin/ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            env={**os.environ, "LC_ALL": "C"},
            timeout=5,
            check=False,
        ).stdout.strip()
        return time.mktime(time.strptime(text, "%a %b %d %H:%M:%S %Y"))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _identity(server: Server) -> str:
    """Alive, gone, or unknown: only a confirmed identity may be signalled; unknown must not spawn a duplicate."""
    if server.pid <= 0:
        return "gone"
    try:
        os.kill(server.pid, 0)
    except ProcessLookupError:
        return "gone"
    except PermissionError:
        # another user's process: a server records itself in this user's own cache, so it ran as this user, and its
        # pid has since gone to someone else's process
        return "gone"
    began = started_at(server.pid)
    if began is None:  # ps can't say: a serving server's own port does (nothing there, nothing to signal)
        if server.status != "serving":
            return "unknown"
        return "alive" if _answers(server.url) else "gone"
    return "alive" if abs(began - server.started) <= 2 else "gone"


def _answers(url: str) -> bool:
    """Whether something takes a connection at `url`'s host and port, within a quarter of a second."""
    parts = urllib.parse.urlsplit(url)
    if parts.hostname is None or parts.port is None:
        return False
    try:
        socket.create_connection((parts.hostname, parts.port), timeout=0.25).close()
    except OSError:
        return False
    return True


def alive(server: Server) -> bool:
    """Whether the pid and start time match the recorded process."""
    return _identity(server) == "alive"


def running() -> list[Server]:
    """The servers running on this Mac, oldest first; a record whose process has gone is removed."""
    return _records()[0]


def _records() -> tuple[list[Server], list[Server]]:
    """The servers running, oldest first, and those still starting whose process ps can't place (`_identity`), kept:
    those never stand in the way of the others, and no second server starts beside one (`ensure`)."""
    found, unconfirmed = [], []
    for path in sorted(folder().glob("*.json")):
        try:
            server = Server.from_json(json.loads(path.read_text()))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        identity = _identity(server)
        if identity == "alive":
            found.append(server)
        elif identity == "gone":
            forget(server)
        else:
            unconfirmed.append(server)
    return sorted(found, key=lambda s: s.started), unconfirmed


def api_key() -> str | None:
    """The API key in IPPON_API_KEY, or None when it is unset or empty."""
    return os.environ.get(API_KEY_ENV) or None


def _taken(host: str, port: int) -> bool:
    """Whether something already listens at `host`:`port` or the port can't be bound there."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with closing(socket.socket(family, socket.SOCK_STREAM)) as probe:
        probe.settimeout(0.2)
        if probe.connect_ex((reach(host), port)) == 0:
            return True
    with closing(socket.socket(family, socket.SOCK_STREAM)) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # as the server binds
        try:
            probe.bind((host, port))
        except OSError:
            return True
    return False


def require_key_off_loopback(host: str, key: str | None) -> None:
    """InvalidRequest for a server that listens beyond this Mac (0.0.0.0, a network address) with no API key: any
    machine on the network could then ask it, and a web page could reach it through DNS rebinding, which the Host
    check stops only on loopback. With a key, every request must carry it, which neither can."""
    if not is_loopback(host) and not key:
        raise InvalidRequest(
            f"serving on {host} reaches the network: give --api-key KEY (or IPPON_API_KEY), which every request "
            "must then carry, or serve on 127.0.0.1"
        )


def network_note(host: str, port: int) -> str:
    """What to say of a server listening beyond this Mac: where other machines reach it, and that they need the key."""
    if is_loopback(host):
        return ""
    where = "this Mac's network address" if host in ("0.0.0.0", "::") else host
    return f"\nippon: other machines reach it at {where}, port {port}, with the API key in every request"


def moved(port: int, asked: int | None) -> str:
    """What to say of a server on `port` when no port was `asked` for: nothing on DEFAULT_PORT, else that it was
    taken (by another program, or another ippon server) and the clients that aren't ippon's need the URL (ippon's own
    find the server by its record)."""
    if asked is not None or port == DEFAULT_PORT:
        return ""
    return (
        f"\nippon: port {DEFAULT_PORT} is taken, so the server is on port {port}. ippon's own "
        "commands, MCP server and hooks find it, and other clients need this URL"
    )


def free_port(host: str, wanted: int | None = None) -> int:
    """The port a server on `host` listens on: `wanted`, refused when it is taken; else DEFAULT_PORT unless it is
    taken, then a free one."""
    if wanted is not None:
        if not 0 <= wanted <= 65535:
            raise InvalidRequest("port must be between 0 and 65535")
        if wanted and _taken(host, wanted):
            raise DecisionError(
                f"port {wanted} on {host} is in use: give another --port, or leave it out for a free one"
            )
        if wanted:
            return wanted
    if wanted is None and not _taken(host, DEFAULT_PORT):
        return DEFAULT_PORT
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with closing(socket.socket(family, socket.SOCK_STREAM)) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def _error(status: int, data: bytes, url: str) -> DecisionError:
    """The typed error for a server's error response (`errors.from_json`), or a DecisionError naming the status for
    a response that isn't one."""
    try:
        return errors.from_json(json.loads(data))
    except (ValueError, KeyError, TypeError):
        return DecisionError(f"HTTP {status} from {url}")


class Client:
    """A server's HTTP API, from this Mac: answers as JSON, errors as their typed DecisionError, and Unavailable when
    no server answers at `url`."""

    def __init__(self, url: str, key: str | None = None):
        try:
            parsed = urlsplit(url)
        except ValueError as error:
            raise InvalidRequest(f"invalid server URL: {error}") from None
        if (
            parsed.scheme != "http"
            or parsed.hostname not in LOOPBACK
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise InvalidRequest(f"the shared server URL must be local HTTP, e.g. {url_of(DEFAULT_PORT)}")
        try:
            port = parsed.port
            if port is not None and port == 0:
                raise ValueError("port must be between 1 and 65535")
        except ValueError as error:
            raise InvalidRequest(f"invalid server URL: {error}") from None
        self.url, self.key = url.rstrip("/"), key or None
        self._connections = threading.local()

    def close(self) -> None:
        """Close this thread's connection. Other threads' connections end with those threads or their own close()."""
        connection: http.client.HTTPConnection | None = getattr(self._connections, "connection", None)
        if connection is not None:
            connection.close()
            self._connections.connection = None

    def call(self, path: str, body: Any = None, timeout: float | None = HTTP_TIMEOUT_S) -> Any:
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        if name := os.environ.get(CLIENT_ENV):
            headers[CLIENT_HEADER] = name
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
        try:
            # Reuse one connection per calling thread so sustained requests cannot exhaust local TCP ports.
            # Direct loopback HTTP never uses proxies or follows redirects with the API key.
            connection: http.client.HTTPConnection | None = getattr(self._connections, "connection", None)
            if connection is None:
                connection = http.client.HTTPConnection(urlsplit(self.url).netloc, timeout=timeout)
                self._connections.connection = connection
            connection.timeout = timeout
            if connection.sock is not None:
                connection.sock.settimeout(timeout)
            # A rejected upload can close early, with a useful error response still buffered.
            with suppress(BrokenPipeError, ConnectionResetError):
                connection.request("GET" if data is None else "POST", path, body=data, headers=headers)
            with connection.getresponse() as response:
                payload = response.read()
                if not 200 <= response.status < 300:
                    raise _error(response.status, payload, self.url)
                return json.loads(payload)
        except DecisionError:
            raise
        except (ValueError, TypeError) as error:
            self.close()
            raise Unavailable(f"invalid JSON response from {self.url}: {error}") from None
        except (OSError, http.client.HTTPException) as error:
            self.close()
            raise Unavailable(
                f"no complete response from the ippon server at {self.url} ({error}): retry the request"
            ) from None

    def healthy(self) -> bool:
        try:
            return bool(self.call("/health", timeout=2.0).get("status") == "ok")
        except DecisionError:
            return False

    def models(self) -> list[ServedModel]:
        """`GET /v1/models`'s list: each model the server answers for."""
        return [ServedModel.from_json(m) for m in self.call("/v1/models")["data"]]

    def decide(self, body: Mapping[str, Any], diagnostics: bool = True) -> dict[str, Any]:
        """The server's System One response to `body`, with how it was answered (`diagnostics`) unless told not."""
        return dict(self.call("/v1/systemone" + ("?diagnostics=1" if diagnostics else ""), body, timeout=None))

    def decide_all(
        self, bodies: Iterable[Mapping[str, Any]], in_flight: int = IN_FLIGHT, diagnostics: bool = True
    ) -> Iterator[dict[str, Any] | DecisionError]:
        """Each body's response, or its own error, in order, with up to `in_flight` asked at once. One at a time by
        default: the server reads requests that arrive together in one batch (serve.Batcher), which is up to a fifth
        faster and makes every one of them, and any other caller's, wait for the whole batch (policy.IN_FLIGHT)."""

        def one(body: Mapping[str, Any]) -> dict[str, Any] | DecisionError:
            try:
                return self.decide(body, diagnostics)
            except DecisionError as error:
                return error
            finally:
                self.close()  # this worker's connection, which no later call would use

        with ThreadPoolExecutor(in_flight) as pool:
            pending: deque[Future[dict[str, Any] | DecisionError]] = deque()
            for body in bodies:
                pending.append(pool.submit(one, body))
                if len(pending) == in_flight:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()


def match(
    listed: Sequence[ServedModel], asked: str, precision: str = "auto", reads: Mapping[str, Any] | None = None
) -> ServedModel:
    """The model a server answers `asked` (a short name, Hub id or folder, at `precision`, loaded with `reads`, None
    for none given) with, among `listed` (`Client.models`): the one served as `asked` or loading what it loads, at
    the weights asked for (`ServedModel.answers`, names.weights_at: kettei-core:q8 or precision q8 8-bit, otherwise
    as published), and loaded as asked (`ServedModel.unlike`: a compute precision or a load option given outright is
    the server's own); ModelNotFound naming the served builds otherwise, before the request is sent."""
    weights, chosen = names.weights_at(asked, precision), names.named_precision(asked, precision)
    reads = reads or {}
    named = [m for m in listed if m.answers(asked, weights)]
    found = next((m for m in named if not m.unlike(chosen, reads)), None)
    if found is not None:
        return found
    if named:
        flags = " ".join(named[0].unlike(chosen, reads))
        raise ModelNotFound(
            f"{asked!r}: the running server answers {named[0].id} loaded with other {flags} than asked, and a "
            f"server's own options decide how it answers: `ippon stop`, then start it with them (ippon serve "
            f"{named[0].id} {flags} ... --detach), or leave them out"
        )
    served = ", ".join(f"{m.id} ({m.weights})" for m in listed) or "nothing"
    spec = f"{asked}:{weights}" if weights != "full" and names.precision_of(asked) is None else asked
    raise ModelNotFound(
        f"{asked!r} ({weights}): the running server answers for {served}; `ippon stop`, then start it with every "
        f"model its clients ask for (ippon serve {' '.join([*(m.id for m in listed), spec])} --detach)"
    )


@dataclass
class Started:
    """A server this process started: its process, port and log, and where in the log its run begins."""

    process: subprocess.Popen[bytes]
    port: int
    log: Path
    offset: int


@dataclass(frozen=True)
class Wanted:
    """The server a client wants: one that answers for every one of `models` (each as named: a short name, Hub id or
    folder) at `precision`, loaded with `reads` (`match`), and how to start one that does when none runs: on `host`
    and `port` (None: DEFAULT_PORT, else a free one), giving back a model's memory after `keep_alive` without a
    request (None: its policy's), with serve's other `options` as arguments (forward_options, which give it `reads`;
    never --precision, which `precision` gives), and with no download when `offline` (an agent's server must never
    fetch gigabytes unasked)."""

    models: tuple[str, ...]
    precision: str = "auto"
    options: tuple[str, ...] = ()
    host: str = "127.0.0.1"
    port: int | None = None
    keep_alive: str | None = STARTED_KEEP_ALIVE
    offline: bool = False
    reads: Mapping[str, Any] = field(default_factory=dict)  # the load options given (`cli.asked_reads`)

    def argv(self) -> list[str]:
        """`ippon serve`'s arguments for these models, but where it listens: the options after the keep-alive, so
        their own --keep-alive wins."""
        kept = ["--keep-alive", self.keep_alive] if self.keep_alive is not None else []
        return [*self.models, *kept, *precision_argv(self.precision), *self.options]


def spawn(wanted: Wanted, key: str | None = None) -> Started:
    """Start `ippon serve` for `wanted` in the background on its host (`free_port`), in a session of its own so it
    outlives this process and its terminal, its output appended to servers/PORT.log. The API key goes to it in its
    environment, never on its command line, which `ps` shows."""
    port = free_port(wanted.host, wanted.port)
    log = folder() / f"{port}.log"
    child = [
        *processes.command(),
        "serve",
        *wanted.argv(),
        "--host",
        wanted.host,
        "--port",
        str(port),
        "--record-log",
        str(log),
    ]
    env = {
        **os.environ,
        **({API_KEY_ENV: key} if key else {}),
        **({"HF_HUB_OFFLINE": "1"} if wanted.offline else {}),
    }
    with log.open("ab") as out:
        log.chmod(0o600)
        out.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} {shlex.join(child)}\n".encode())
        out.flush()
        offset = out.tell()
        process = subprocess.Popen(
            child, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True, env=env
        )
    threading.Thread(target=process.wait, daemon=True, name="ippon-reap").start()  # no zombie when it ends
    return Started(process, port, log, offset)


def _failure(started: Started) -> DecisionError:
    """Why a server this process started ended before it served: the last error its log gives, as the error its
    exit code names."""
    try:
        with started.log.open("rb") as f:
            f.seek(started.offset)
            lines = f.read().decode(errors="replace").splitlines()
    except OSError:
        lines = []
    said = next((line.removeprefix("ippon: ") for line in reversed(lines) if line.startswith("ippon: ")), "")
    code = started.process.returncode or 1
    tail = said or (lines[-1] if lines else f"it exited with {code}")
    return errors.BY_EXIT_CODE.get(code, DecisionError)(f"the server didn't start: {tail} (log: {started.log})")


def _recorded(port: int, pid: int) -> Server | None:
    try:
        server = Server.from_json(json.loads((folder() / f"{port}.json").read_text()))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return server if server.pid == pid else None


def ready(server: Server, key: str | None, timeout: float | None, started: Started | None = None) -> Server:
    """`server` once it answers (`Client.healthy`), waiting while it loads its first model: at most `timeout`
    seconds (None: as long as it is starting). A server that ends first raises its error (`started`: one this
    process started, whose log and exit code say why); one still starting past `timeout` raises Unavailable."""
    deadline = None if timeout is None else time.monotonic() + timeout
    client = Client(server.url, key)
    while True:
        if started is not None and started.process.poll() is not None:
            raise _failure(started)
        if started is None:
            identity = _identity(server)
            if identity == "unknown":
                raise Unavailable(f"cannot confirm server pid {server.pid}'s identity; its record is preserved")
            if identity == "gone":
                raise Unavailable(f"the server on {server.url} (pid {server.pid}) stopped while it started")
        now = _recorded(server.port, server.pid) or server
        if now.status == "serving":
            health = client.call("/health", timeout=2.0)
            if isinstance(health, dict) and health.get("status") == "ok":
                return now
        if deadline is not None and time.monotonic() > deadline:
            first = server.models[0][0] if server.models else "its first model"
            raise Unavailable(
                f"the server on {server.url} (pid {server.pid}) is still loading {first} after {timeout:g} s; it "
                f"goes on in the background (log: {server.log}): send the request again shortly"
            )
        time.sleep(POLL_S)


@contextmanager
def _lock() -> Iterator[IO[str]]:
    """This user's lock on starting a server: a client that finds none running starts one while holding it, so
    another that looks at the same moment finds that one."""
    path = folder() / "start.lock"
    with path.open("w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield f
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def find(wanted: Wanted, key: str | None = None) -> Server | None:
    """The oldest running server that answers for every one of `wanted.models` as it asks (`match`; any server when
    none is named), or None. A server whose list can't be read now (still loading its first model) counts as one, so
    a second is never started beside a server only because it was slow to answer: its client learns once it is
    ready (`ready`) whether it answers for them (`match`). Unavailable while a server ps can't place is starting
    (`_identity`): no second server starts beside that one either."""
    found = next((s for s in running() if _serves(s, wanted, key)), None)
    unconfirmed = [] if found else _records()[1]
    if unconfirmed:
        raise Unavailable(
            f"server pid {unconfirmed[0].pid} is starting and ps can't confirm it yet, so its record is preserved "
            f"and no second server starts: wait for it, or `ippon stop --port {unconfirmed[0].port}`"
        )
    return found


def _serves(server: Server, wanted: Wanted, key: str | None) -> bool:
    """Whether `server` answers every one of `wanted.models` as it asks (`match`): a running server of other models,
    of the same model at other weights (setup's Kettei Core in bf16 for a client of kettei-core:q8), or loaded with
    other options (another compute precision or settings file), is no server for them. When its list can't be read
    (still loading, or it needs another key), its record decides by name alone (`find`)."""
    try:
        listed = Client(server.url, key).models()
    except DecisionError:
        recorded = [ServedModel(name, loads) for name, loads in server.models]
        return all(any(m.answers(asked) for m in recorded) for asked in wanted.models)
    try:
        for model in wanted.models:
            match(listed, model, wanted.precision, wanted.reads)
    except ModelNotFound:
        return False
    return True


def ensure(wanted: Wanted, key: str | None, timeout: float | None) -> tuple[Server, bool]:
    """The server every client shares, ready, and whether it was started here: the one `find` finds for `wanted`,
    else one started for it (`spawn`), beside any other, waiting at most `timeout` seconds for it to load its first
    model (`ready`). Unavailable when none runs and no model is named to start one with; UnsupportedModel, before
    anything starts, for a model an `offline` server would have to download."""
    with _lock():
        server, started = find(wanted, key), None
        if server is None:
            if not wanted.models:
                raise Unavailable(
                    "no ippon server is running and no model was named to start one with: name one, or start "
                    "one (ippon serve MODEL --detach)"
                )
            if wanted.offline:
                _downloaded(wanted.models)
            started = spawn(wanted, key)
            server = registered(started, timeout)
    return ready(server, key, timeout, started), started is not None


def _downloaded(models: Sequence[str]) -> None:
    """UnsupportedModel for the first of `models` not downloaded on this Mac, naming the `ippon pull` that fetches
    it: a server an agent starts never downloads, so a download of gigabytes stays the user's call."""
    from .locate import downloaded

    for model in models:
        if not downloaded(str(names.resolve(model))):
            size = f" ({names.find(model)[0].build.weights})" if names.is_name(model) else ""
            raise UnsupportedModel(
                f"{model} isn't downloaded on this Mac yet{size}: ask the user to run `ippon pull {model}`, "
                "which says what it takes and whether it fits before downloading"
            )


def answering(
    model: str | None, wanted: Wanted, key: str | None, timeout: float | None, url: str | None = None
) -> tuple[Client, ServedModel | None]:
    """A client of the server that answers for `model` as `wanted` asks (its precision and load options), and the
    model it answers with: the server at `url` when one is given, else the shared one (`ensure`); None for the model
    when none is named. ModelNotFound when the server doesn't answer for it so (`match`)."""
    if url:
        client = Client(url, key)
    else:
        server, _ = ensure(wanted, key, timeout)
        client = Client(server.url, key)
    return client, None if model is None else match(client.models(), model, wanted.precision, wanted.reads)


def start(wanted: Wanted, key: str | None, timeout: float | None) -> Server:
    """`ippon serve MODEL --detach`: a server started for `wanted` (`spawn`), ready (`ready`), whatever else runs. It
    starts under the start lock, so a client that looks meanwhile finds it rather than starting another."""
    with _lock():
        started = spawn(wanted, key)
        server = registered(started, timeout)
    return ready(server, key, timeout, started)


def stop(server: Server, timeout: float = STOP_TIMEOUT_S) -> dict[str, Any]:
    """Stop `server` as Ctrl-C does (SIGTERM: what waits is answered, its models unloaded, its record removed), and
    kill it (SIGKILL) if it hasn't ended within `timeout` seconds; what was stopped, how long it took, and whether
    it had to be killed."""
    start = time.monotonic()
    killed = False
    identity = _identity(server)
    if identity != "alive":
        if identity == "gone":
            forget(server)
        raise Unavailable(f"server pid {server.pid} no longer has a confirmed recorded identity; no signal sent")
    try:
        os.kill(server.pid, signal.SIGTERM)
        while alive(server) and time.monotonic() - start < timeout:
            time.sleep(POLL_S)
        if alive(server):
            os.kill(server.pid, signal.SIGKILL)
            killed = True
            while alive(server) and time.monotonic() - start < timeout + 5:
                time.sleep(POLL_S)
    except ProcessLookupError:
        pass
    if _identity(server) == "unknown":
        raise Unavailable(f"cannot confirm server pid {server.pid} stopped; its record is preserved")
    forget(server)
    took = round(time.monotonic() - start, 2)
    return {"pid": server.pid, "url": server.url, "port": server.port, "seconds": took, "killed": killed}


def registered(started: Started, timeout: float | None) -> Server:
    """Wait for the child to publish its identity, bounded even before model loading begins."""
    deadline = time.monotonic() + (START_TIMEOUT_S if timeout is None else timeout)
    while (server := _recorded(started.port, started.process.pid)) is None:
        if started.process.poll() is not None:
            raise _failure(started)
        if time.monotonic() >= deadline:
            started.process.terminate()
            try:
                started.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                started.process.kill()
            raise Unavailable(f"server did not register before the startup timeout (log: {started.log})")
        time.sleep(POLL_S)
    return server


def describe(server: Server, key: str | None = None) -> dict[str, Any]:
    """The saved identity and live models/memory status, or the reason it can't be reached: within
    DESCRIBE_TIMEOUT_S, so a server that is stopped or stuck shows as that, never holds `ippon status` still."""
    result = server.to_json()
    try:
        client = Client(server.url, key)
        listed = client.call("/v1/models", timeout=DESCRIBE_TIMEOUT_S)["data"]
        result["models"] = [ServedModel.from_json(m).to_json() for m in listed]
        result["memory"] = client.call("/v1/status", timeout=DESCRIBE_TIMEOUT_S)
    except Unavailable as error:
        result["models"] = [ServedModel(name, model).to_json() for name, model in server.models]
        stuck = Unavailable(  # it said it serves, yet no reply in time
            f"not answering within {DESCRIBE_TIMEOUT_S:g} s: stopped or stuck · `ippon logs` shows what it does, "
            "`ippon stop` stops it"
        )
        result |= (error if server.status == "starting" else stuck).to_json(exit_code=True)
    except DecisionError as error:
        result["models"] = [ServedModel(name, model).to_json() for name, model in server.models]
        result |= error.to_json(exit_code=True)
    return result
