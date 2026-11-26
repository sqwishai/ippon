"""Agent clients share HTTP models, survive malformed messages, and never load weights themselves."""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar

import mlx.core as mx
import pytest
from PIL import Image

from ippon import Decider, cli, daemon, hub, locate, mcp, registry
from ippon import check as checks
from ippon.errors import DecisionError, InvalidRequest, ModelNotFound, Unavailable
from ippon.options import Reads
from ippon.serve import Batcher
from ippon.served import ServedModel, url_of
from test_check import Found, mac
from test_decider import REQUEST
from test_serve import running as running


def rpc_session(calls: list[tuple[str, dict[str, Any]]]) -> bytes:
    """The same initialized JSON-lines session for in-process and separate stdio clients."""
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
    ]
    messages.extend(
        {"jsonrpc": "2.0", "id": i, "method": method, "params": params} for i, (method, params) in enumerate(calls, 2)
    )
    return b"\n".join(json.dumps(m).encode() for m in messages) + b"\n"


def test_mcp_piped_calls_finish_and_bad_arguments_do_not_end_the_session() -> None:
    source = io.BytesIO(
        rpc_session(
            [
                ("tools/list", {}),
                ("tools/call", {"name": "recommend", "arguments": {}}),
                ("tools/call", {"name": [], "arguments": {}}),
                ("tools/call", {"name": "recommend", "arguments": []}),
                ("tools/call", {"name": "recommend", "arguments": {"task": []}}),
                ("ping", {}),
                ("tools/call", {"name": "lint", "arguments": {**REQUEST, "files": ["R0lGODlhAQABAAAAACw="]}}),
            ]
        )
    )
    out, stdout = io.BytesIO(), sys.stdout
    assert mcp.run(stdin=source, stdout=out) == 0
    assert sys.stdout is stdout
    messages = {m["id"]: m for m in map(json.loads, out.getvalue().splitlines())}
    assert messages.keys() == {1, 2, 3, 4, 5, 6, 7, 8}
    assert messages[1]["result"]["protocolVersion"] == "2025-11-25"
    assert messages[3]["result"]["structuredContent"]["picks"]
    assert messages[4]["error"]["code"] == messages[5]["error"]["code"] == -32602
    assert messages[6]["result"]["isError"]
    tools = {t["name"]: t for t in messages[2]["result"]["tools"]}
    assert tools["decide"]["annotations"]["readOnlyHint"] is False
    assert messages[7]["result"] == {}
    assert messages[8]["result"]["isError"] is False  # a request's every field, its files too, as the CLI takes it


def test_two_stdio_clients_use_one_http_model_without_importing_mlx(running: tuple[str, Batcher, Decider]) -> None:
    url, batcher, model = running
    code = """
import sys
class NoMLX:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise RuntimeError("MCP client imported MLX")
sys.meta_path.insert(0, NoMLX())
from ippon.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    request = rpc_session([("tools/call", {"name": "decide", "arguments": REQUEST})])
    clients = [
        subprocess.Popen(
            [sys.executable, "-c", code, "mcp", "--url", url],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(2)
    ]
    try:
        for client in clients:
            out, err = client.communicate(request, timeout=30)
            assert client.returncode == 0, err.decode()
            response = {m["id"]: m for m in map(json.loads, out.splitlines())}[2]["result"]
            assert response["isError"] is False, response
            assert response["structuredContent"]["answers"] == model.systemone(REQUEST)["answers"]
            assert "diagnostics" in response["structuredContent"]
        assert batcher.models.models[0].decider is model
    finally:
        for client in clients:
            if client.poll() is None:
                client.kill()
                client.wait()


def test_an_mcp_message_too_deep_to_read_is_answered_and_the_session_goes_on(
    running: tuple[str, Batcher, Decider],
) -> None:
    """A message nested past Python's recursion is a parse error, and the next message is answered."""
    deep = b"[" * 100_000 + b"]" * 100_000 + b"\n"
    session = deep + rpc_session([("tools/call", {"name": "decide", "arguments": REQUEST})])
    command = [sys.executable, "-P", "-m", "ippon", "mcp", "--url", running[0]]
    done = subprocess.run(command, input=session, capture_output=True, timeout=60, check=True)
    said = [json.loads(line) for line in done.stdout.splitlines()]
    assert said[0]["error"]["code"] == -32700
    assert {m.get("id"): m for m in said}[2]["result"]["isError"] is False


def test_mcp_answers_only_with_the_model_it_was_given(running: tuple[str, Batcher, Decider]) -> None:
    """`ippon mcp MODEL` asks for MODEL even when a request names none: a shared server running another model
    answers model_not_found, never with its own model in MODEL's place."""
    url = running[0]

    def decide(*configured: str) -> dict[str, Any]:
        session = rpc_session([("tools/call", {"name": "decide", "arguments": REQUEST})])
        command = [sys.executable, "-P", "-m", "ippon", "mcp", *configured, "--url", url]
        done = subprocess.run(command, input=session, capture_output=True, timeout=60, check=True)
        return dict({m["id"]: m for m in map(json.loads, done.stdout.splitlines())}[2]["result"])

    assert decide()["isError"] is False  # given no model, whatever the server runs
    refused = decide("kettei-dot")
    assert refused["isError"] is True
    assert refused["structuredContent"]["error"]["type"] == "model_not_found"


def test_cached_checks_read_recorded_files_without_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IPPON_CACHE", str(tmp_path / "cache"))
    snapshot = tmp_path / ("a" * 40)
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"model_type":"toy"}')
    mx.save_safetensors(str(snapshot / "model.safetensors"), {"w": mx.zeros((8, 8))})
    hub.record_download("org/model", hub.probe(), snapshot)
    hub.record_download("org/model@" + snapshot.name, Found.files, snapshot)
    monkeypatch.setattr(registry, "select", lambda directory, adapter: Found())
    monkeypatch.setattr(locate, "_remote", lambda *a: pytest.fail("cached check used network"))
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **kw: pytest.fail("download attempted"))
    result = checks.check("org/model", device=mac(4 * 2**30))
    assert result["revision"] == snapshot.name
    assert result["download"] == {"files": 1, "bytes": (snapshot / "model.safetensors").stat().st_size}
    assert locate.downloaded("org/model")
    (snapshot / "model.safetensors").unlink()
    assert not locate.downloaded("org/model")


@pytest.mark.parametrize("identity", [200.0, None], ids=["reused-pid", "unknown-start"])
def test_unconfirmed_process_identity_never_signals_or_duplicates_a_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identity: float | None
) -> None:
    monkeypatch.setenv("IPPON_CACHE", str(tmp_path))
    recorded = daemon.Server(123, 8123, "http://127.0.0.1:8123", (("toy", "toy"),), started=100.0, api_key=True)
    daemon.record(recorded)
    monkeypatch.setattr(daemon, "started_at", lambda pid: identity)
    signals: list[int] = []
    monkeypatch.setattr(os, "kill", lambda pid, signum: signals.append(signum))
    with pytest.raises(Unavailable, match="no signal sent"):
        daemon.stop(recorded, timeout=0)
    assert signals == [0]
    assert not daemon.alive(recorded)
    if identity is None:
        monkeypatch.setattr(daemon, "spawn", lambda *a, **kw: pytest.fail("unknown server duplicated"))
        with pytest.raises(Unavailable, match="record is preserved"):
            daemon.ensure(daemon.Wanted(("toy",)), None, 1)
        assert (daemon.folder() / "8123.json").exists()
    assert signal.SIGTERM not in signals
    assert signal.SIGKILL not in signals


def test_a_record_whose_pid_is_now_another_users_is_cleared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A server records itself in its user's own cache, so a record whose pid now belongs to another user (pid 1 is
    launchd's) is a server gone: cleared, never a block on every shared-server command."""
    monkeypatch.setenv("IPPON_CACHE", str(tmp_path))
    daemon.record(daemon.Server(1, 8124, "http://127.0.0.1:8124", (("toy", "toy"),), started=100.0))
    assert daemon.running() == []
    assert not (daemon.folder() / "8124.json").exists()


def test_stop_with_several_servers_stops_only_the_one_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With Kettei Core serving the command check and Kettei Dot an app, a bare `ippon stop` stops neither and lists
    them. A model's name stops its server alone, kettei-core finds the server started as kettei-core:q8, and with one
    server left a bare stop takes it."""
    monkeypatch.setenv("IPPON_CACHE", str(tmp_path))
    pids = {}
    for port, name in ((8131, "kettei-core:q8"), (8132, "kettei-dot:q8"), (8133, "toy")):
        # a server is no child of the CLI that stops it: the shell leaves `sleep` to launchd, which reaps it
        pid = int(
            subprocess.run(
                ["/bin/sh", "-c", "/bin/sleep 120 >/dev/null 2>&1 & echo $!"], capture_output=True, check=True
            ).stdout
        )
        pids[name] = pid
        time.sleep(0.05)
        started = daemon.started_at(pid) or 0.0
        daemon.record(daemon.Server(pid, port, url_of(port), ((name, name),), started=started))
    servers = {s.models[0][0]: s for s in daemon.running()}

    def running() -> set[str]:
        return {name for name, server in servers.items() if daemon.alive(server)}

    try:
        assert cli.main(["stop"]) == InvalidRequest.exit_code
        listed = "3 ippon servers are running, so none was stopped:\n  kettei-core:q8 (port 8131)"
        assert listed in capsys.readouterr().err
        assert running() == {"kettei-core:q8", "kettei-dot:q8", "toy"}
        assert cli.main(["stop", "kettei-dot:q8"]) == 0
        assert capsys.readouterr().out == "stopped kettei-dot:q8 (port 8132)\n"
        assert running() == {"kettei-core:q8", "toy"}
        assert cli.main(["stop", "kettei-core"]) == 0
        assert running() == {"toy"}
        assert cli.main(["stop"]) == 0
        assert running() == set()
        assert daemon.running() == []
    finally:
        for pid in pids.values():
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


def test_concurrent_clients_start_one_shared_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IPPON_CACHE", str(tmp_path))
    starts: list[object] = []
    server = daemon.Server(os.getpid(), 8000, "http://127.0.0.1:8000", (("toy", "toy"),))
    running: list[daemon.Server] = []

    def spawned(*a: Any, **kw: Any) -> object:
        obj = object()
        starts.append(obj)
        return obj

    def registered(*a: Any) -> daemon.Server:
        running.append(server)
        return server

    monkeypatch.setattr(daemon, "spawn", spawned)
    monkeypatch.setattr(daemon, "registered", registered)
    monkeypatch.setattr(daemon, "running", lambda: running)
    monkeypatch.setattr(daemon, "ready", lambda server, *a: server)
    monkeypatch.setattr(daemon, "_serves", lambda *a: True)  # never ask whatever really listens on port 8000
    results: list[Any] = []
    threads = [
        threading.Thread(target=lambda: results.append(daemon.ensure(daemon.Wanted(("toy",)), None, 5)))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert len(starts) == 1
    assert len(results) == 4
    assert sum(started for _, started in results) == 1


def test_start_timeout_and_ports_are_validated_without_starting_a_model(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for argv in (["serve", "d1-2b", "--port", "65536"], ["serve", "d1-2b", "--detach", "--start-timeout", "nan"]):
        assert cli.main([*argv, "--json"]) == 3
        assert json.loads(capsys.readouterr().out)["error"]["type"] == "invalid_request"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        taken = listener.getsockname()[1]
        monkeypatch.setattr(daemon, "DEFAULT_PORT", taken)
        assert daemon.free_port("127.0.0.1") != taken
        with pytest.raises(DecisionError, match="in use"):
            daemon.free_port("127.0.0.1", taken)
    for url in ("https://example.com", "http://127.0.0.1:8000/secret", "http://user:pw@localhost:8000", "http://["):
        with pytest.raises(InvalidRequest):
            daemon.Client(url, "secret")


def test_lora_is_downloaded_only_with_its_base_and_cycles_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IPPON_CACHE", str(tmp_path / "cache"))
    adapter, base = tmp_path / "adapter", tmp_path / "base"
    adapter.mkdir()
    base.mkdir()
    (adapter / "adapter.safetensors").write_text("weights")
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(base)}))
    assert not locate.downloaded(str(adapter))
    (base / "model.safetensors").write_text("weights")
    assert locate.downloaded(str(adapter))
    (base / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(adapter)}))
    assert not locate.downloaded(str(adapter))


def test_startup_wait_is_bounded_before_the_child_records_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typing import cast

    class Child:
        pid = 123
        terminated = False

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float) -> int:
            return -15

    child = Child()
    started = daemon.Started(cast(Any, child), 8000, tmp_path / "server.log", 0)
    monkeypatch.setattr(daemon, "_recorded", lambda *a: None)
    with pytest.raises(Unavailable, match="startup timeout"):
        daemon.registered(started, timeout=0)
    assert child.terminated


def test_waiting_for_a_server_ends_at_its_timeout_or_when_its_process_ends(tmp_path: Path) -> None:
    """A recorded server still starting past the wait raises Unavailable; one whose process ends first raises the
    error its exit code and log name (a server this process started) or says it stopped (one found recorded)."""
    me = os.getpid()
    url = url_of(1)  # nothing listens: a starting server is never asked
    starting = daemon.Server(me, 1, url, (("toy", "toy"),), started=daemon.started_at(me) or 0.0, log="toy.log")
    daemon.record(starting)
    with pytest.raises(Unavailable, match=r"still loading toy after 0.2 s;.*log: toy.log"):
        daemon.ready(starting, None, 0.2)

    log = tmp_path / "server.log"
    with log.open("wb") as out:
        said = "print('loading'); print('ippon: no model at /nowhere'); raise SystemExit(10)"
        child = subprocess.Popen([sys.executable, "-c", said], stdout=out)
        child.wait()
    with pytest.raises(ModelNotFound, match=r"didn't start: no model at /nowhere \(log: "):
        daemon.ready(starting, None, None, daemon.Started(child, 1, log, 0))
    gone = daemon.Server(child.pid, 1, url, (("toy", "toy"),))
    with pytest.raises(Unavailable, match=f"pid {child.pid}\\) stopped while it started"):
        daemon.ready(gone, None, None)


def test_http_client_does_not_follow_redirects_with_a_key() -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Redirect(BaseHTTPRequestHandler):
        paths: ClassVar[list[str]] = []

        def do_GET(self) -> None:
            self.paths.append(self.path)
            self.send_response(302)
            self.send_header("Location", "/stolen-key")
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = daemon.Client(url_of(server.server_port), "secret")
        with pytest.raises(DecisionError, match="HTTP 302"):
            client.models()
        assert Redirect.paths == ["/v1/models"]
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("status", [200, 503])
def test_http_client_reports_a_dropped_response_and_can_retry(status: int) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    interrupted = True
    connections: set[tuple[str, int]] = set()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            connections.add(self.client_address)
            if self.path == "/slow":
                time.sleep(0.05)
            payload = b'{"status":"ok"}'
            self.send_response(status if interrupted else 200)
            self.send_header("Content-Length", str(len(payload) + 100 if interrupted else len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self.close_connection = interrupted

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    client = daemon.Client(f"http://127.0.0.1:{server.server_port}")
    try:
        with pytest.raises(Unavailable, match="retry"):
            client.call("/health")
        assert not client.healthy()
        interrupted = False
        assert client.healthy()
        connections.clear()
        barrier = threading.Barrier(2)

        def requests() -> None:
            barrier.wait(timeout=5)
            try:
                for _ in range(20):
                    assert client.healthy()
            finally:
                client.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [pool.submit(requests) for _ in range(2)]
            for future in pending:
                future.result(timeout=10)
        assert len(connections) == 2, "repeated requests must not exhaust local TCP ports"
        client.call("/health", timeout=0.01)
        assert client.call("/slow", timeout=1)["status"] == "ok"
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        serving.join()


def test_cli_remote_decision_and_status_use_the_existing_server(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], running: tuple[str, Batcher, Decider]
) -> None:
    url, _, model = running
    with pytest.raises(InvalidRequest):
        daemon.Client(url).decide({"questions": []})
    me = os.getpid()  # its true start time, so a client finding it by its record trusts it
    recorded = daemon.Server(
        me, int(url.rsplit(":", 1)[1]), url, (("toy", "toy"),), status="serving", started=daemon.started_at(me) or 0.0
    )
    monkeypatch.setattr(cli, "load", lambda *a, **kw: pytest.fail("client loaded a model"))
    monkeypatch.setattr(daemon, "running", lambda: [recorded])
    # an agent that runs serve --detach for a model already served gets that server, not a second copy of it
    monkeypatch.setattr(daemon, "start", lambda *a, **kw: pytest.fail("a second server started"))
    assert cli.main(["serve", "toy", "--detach"]) == 0
    assert f"already serving toy at {url}" in capsys.readouterr().out
    assert cli.main(["decide", "toy", json.dumps(REQUEST), "--server", url, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["answers"] == model.systemone(REQUEST)["answers"]
    # the request after --server, as agents often write it, is the request, not the server's address
    assert cli.main(["decide", "toy", "--server", json.dumps(REQUEST), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["answers"] == model.systemone(REQUEST)["answers"]
    # with no --server at all, the server that has the model loaded still answers: no second copy is loaded
    assert cli.main(["decide", "toy", json.dumps(REQUEST), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["answers"] == model.systemone(REQUEST)["answers"]
    assert cli.main(["ps", "--json"]) == 0
    status = json.loads(capsys.readouterr().out)["servers"][0]
    assert status["models"][0]["loaded"] is True
    assert "memory" in status
    assert cli.main(["status", "--once"]) == 0  # the two decisions above, both answered: none in progress
    assert " clients connected · 0 asking" in (said := capsys.readouterr().out)
    assert " · 9 answered" in said  # three requests of 3 questions each
    # a server still starting answers nothing yet: status says so, and lists what it was started with
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        idle = free.getsockname()[1]
    starting = daemon.Server(me, idle, url_of(idle), (("toy", "toy"),), started=recorded.started)
    monkeypatch.setattr(daemon, "running", lambda: [starting])
    assert cli.main(["status"]) == 0
    assert "toy · not loaded" in capsys.readouterr().out
    assert cli.main(["recommend", "--json"]) == 0
    recommended = json.loads(capsys.readouterr().out)  # as the MCP tool gives it, the build to start with included
    assert recommended == mcp.Session(io.BytesIO()).recommend({})
    assert recommended["start_with"] is not None
    assert cli.main(["decide", "toy", "{oops", "--server", url, "--json"]) == 3
    assert json.loads(capsys.readouterr().out)["error"]["exit_code"] == 3


FAKE_AGENT = """#!{python}
import json, os, shutil, sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname
config = json.loads(os.environ.get("OPENCODE_CONFIG_CONTENT", "{{}}"))
plugins = [url2pathname(urlparse(p).path) for p in config.get("plugin", [])]
paths = [a for a in sys.argv[1:] if os.path.isfile(a)] + plugins
folders = [a for a in sys.argv[1:] if os.path.isdir(a)]
dirs = {{a: sorted(str(f.relative_to(a)) for f in Path(a).rglob("*") if f.is_file()) for a in folders}}
files = {{p: Path(p).read_text() for p in paths}}
ippon = Path(shutil.which("ippon")).read_text()  # the ippon the agent's own commands run
record = {{"argv": sys.argv[1:], "env": dict(os.environ), "files": files, "dirs": dirs, "ippon": ippon}}
Path(os.environ["FAKE_RECORD"]).write_text(json.dumps(record))
sys.exit(7)
"""


def launched(
    agent: str, running: tuple[str, Batcher, Decider], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *more: str
) -> dict[str, Any]:
    """`ippon AGENT` run against the toy server, with a fake agent on PATH that records its arguments, environment
    and the files it was given, under an empty HOME; what it recorded."""
    url = running[0]
    me = os.getpid()
    port = int(url.rsplit(":", 1)[1])
    bin_dir, home, folder = tmp_path / "bin", tmp_path / "home", toy_folder(tmp_path)
    temporary = tmp_path / 'tmp #"🚀'  # the session's files in a folder whose name JSON, TOML and URLs must escape
    for made in (bin_dir, home, folder, temporary):
        made.mkdir(parents=True)
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    served = (("toy", str(folder)),)  # the server answers as "toy" for the model in this folder, at 8 bits
    toy_model = running[1].models.models[0]
    monkeypatch.setattr(toy_model, "loads", str(folder))
    monkeypatch.setattr(toy_model, "weights", "q8")  # as `ippon serve FOLDER --precision q8` serves it
    daemon.record(daemon.Server(me, port, url, served, started=daemon.started_at(me) or 0.0, status="serving"))
    fake = bin_dir / agent
    fake.write_text(FAKE_AGENT.format(python=sys.executable))
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("FAKE_RECORD", str(tmp_path / "record.json"))
    assert cli.main([agent, "--model", str(folder), "--precision", "q8", *more, "--", "--resume", "a b"]) == 7
    recorded = json.loads((tmp_path / "record.json").read_text())
    assert not any(home.iterdir()), "wrote into HOME"
    assert not any(Path(p).exists() for p in recorded["files"]), "left its temporary files"
    assert (recorded["env"]["IPPON_URL"], recorded["env"]["IPPON_MODEL"]) == (url, "toy")
    assert recorded["env"]["IPPON_CLIENT"] == agent  # its requests named in the server's log and status
    assert f"exec {shlex.quote(sys.executable)} -P -m ippon" in recorded["ippon"]  # this installation, not PATH's
    assert daemon.running(), "stopped the shared server"
    return dict(recorded)


def toy_folder(tmp_path: Path) -> Path:
    """Where the launcher tests' toy sits: a Hub cache folder, under a name each agent's config format must
    escape."""
    return tmp_path / 'hub #"🚀\x7f' / "models--Sqwish--Kettei-Core" / "snapshots" / ("a" * 40)


def mcp_args(tmp_path: Path) -> list[str]:
    return ["-P", "-m", "ippon", "mcp", str(toy_folder(tmp_path)), "--precision", "q8"]


def answers(mcp_command: list[str], env: dict[str, str], model: Decider) -> None:
    """The MCP server an agent was given answers a decision from the shared server."""
    session = rpc_session([("tools/call", {"name": "decide", "arguments": REQUEST})])
    done = subprocess.run(mcp_command, input=session, capture_output=True, env=env, timeout=60, check=True)
    result = {m["id"]: m for m in map(json.loads, done.stdout.splitlines())}[2]["result"]
    assert result["structuredContent"]["answers"] == model.systemone(REQUEST)["answers"]


def test_ippon_claude_gives_claude_code_the_tools_and_skill_for_one_session(
    running: tuple[str, Batcher, Decider], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded = launched("claude", running, tmp_path, monkeypatch)
    flag, servers, plugin_flag, plugin, *rest = recorded["argv"]
    assert (flag, plugin_flag, rest) == ("--mcp-config", "--plugin-dir", ["--resume", "a b"])
    assert {".claude-plugin/plugin.json", "skills/ippon/SKILL.md"} <= set(recorded["dirs"][plugin])
    ippon = json.loads(recorded["files"][servers])["mcpServers"]["ippon"]
    assert ippon["args"] == mcp_args(tmp_path)
    answers([ippon["command"], *ippon["args"]], recorded["env"], running[2])


def test_ippon_codex_gives_codex_the_tools_for_one_session(
    running: tuple[str, Batcher, Decider], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    recorded = launched("codex", running, tmp_path, monkeypatch, "--api-key", "a-key")
    *overrides, resume, session = recorded["argv"]
    assert (resume, session) == ("--resume", "a b")
    assert overrides[::2] == ["-c"] * (len(overrides) // 2)
    config: dict[str, Any] = {}
    for override in overrides[1::2]:
        key, _, value = override.partition("=")
        config[key] = tomllib.loads(f"value = {value}")["value"]
    assert config.pop("mcp_servers.ippon.args") == mcp_args(tmp_path)
    # Codex passes its MCP server only the variables it names: the session's own (the key --api-key gave) too
    passed = {"HF_HUB_OFFLINE", "IPPON_CACHE", "IPPON_URL", "IPPON_MODEL", "IPPON_API_KEY"}
    assert passed <= set(config.pop("mcp_servers.ippon.env_vars"))
    assert recorded["env"]["IPPON_API_KEY"] == "a-key"
    command = config.pop("mcp_servers.ippon.command")
    assert config == {}  # the MCP server and nothing else
    answers([command, *mcp_args(tmp_path)], recorded["env"], running[2])


def test_ippon_opencode_gives_opencode_the_tools_for_one_session(
    running: tuple[str, Batcher, Decider], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded = launched("opencode", running, tmp_path, monkeypatch)
    assert recorded["argv"] == ["--resume", "a b"]
    config = json.loads(recorded["env"]["OPENCODE_CONFIG_CONTENT"])
    assert set(config) == {"$schema", "mcp"}  # the MCP server and nothing else
    ippon = config["mcp"]["ippon"]
    assert (ippon["type"], ippon["command"][1:]) == ("local", mcp_args(tmp_path))
    answers(ippon["command"], recorded["env"], running[2])


def test_ippon_pi_gives_pi_the_tools_skill_and_extension_for_one_session(
    running: tuple[str, Batcher, Decider], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded = launched("pi", running, tmp_path, monkeypatch)
    ext_flag, ext_file, skill_flag, skill_dir, *rest = recorded["argv"]
    assert (ext_flag, skill_flag, rest) == ("-e", "--skill", ["--resume", "a b"])
    assert recorded["files"][ext_file].startswith("// ippon in Pi")
    assert "SKILL.md" in recorded["dirs"][skill_dir]
    mcp_cmd = json.loads(recorded["env"]["IPPON_MCP"])
    assert mcp_cmd[1:] == mcp_args(tmp_path)
    answers(mcp_cmd, recorded["env"], running[2])


def test_mcp_negotiates_older_protocol_versions_without_structured_content() -> None:
    """Older MCP hosts (2024-11-05, 2025-03-26) negotiate their own version and receive JSON text without the
    2025-06-18 `structuredContent` field."""
    lines = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "old", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "lint", "arguments": REQUEST}},
    ]
    payload = ("\n".join(json.dumps(line) for line in lines) + "\n").encode()
    stdout = io.BytesIO()
    assert mcp.run(stdin=io.BytesIO(payload), stdout=stdout) == 0
    messages = {m["id"]: m for m in map(json.loads, stdout.getvalue().splitlines())}
    assert messages[1]["result"]["protocolVersion"] == "2024-11-05"
    result = messages[2]["result"]
    assert result["isError"] is False
    assert "structuredContent" not in result
    parsed = json.loads(result["content"][0]["text"])
    assert "ok" in parsed
    assert "problems" in parsed


def test_a_server_of_other_models_is_no_server_for_this_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Kettei Dot on port 8000 (for the user's own app) is no reason for a client of Kettei Core, or of Core at other
    weights, to get model_not_found on every call: Core starts beside it. So is Dot loaded otherwise than a client
    asks, in another compute precision or with another settings file, which would answer it otherwise: a client that
    gives one reuses only a server loaded with it, and one that leaves it out takes the server's. A server whose list
    can't be read is kept."""
    monkeypatch.setenv("IPPON_CACHE", str(tmp_path))
    dot = daemon.Server(os.getpid(), 8000, "http://127.0.0.1:8000", (("kettei-dot", "Sqwish/Kettei-Dot"),))
    core = daemon.Server(os.getpid(), 8123, "http://127.0.0.1:8123", (("kettei-core", "Sqwish/Kettei-Core"),))
    monkeypatch.setattr(daemon, "running", lambda: [dot])
    settings = {"labels": ["no", "yes"]}
    listed = [ServedModel("kettei-dot", "Sqwish/Kettei-Dot", reads=Reads(settings=settings), precision="fp16")]
    monkeypatch.setattr(daemon.Client, "models", lambda self: listed)
    monkeypatch.setattr(daemon, "spawn", lambda *a, **kw: object())
    monkeypatch.setattr(daemon, "registered", lambda *a: core)
    monkeypatch.setattr(daemon, "ready", lambda server, *a: server)
    assert daemon.ensure(daemon.Wanted(("kettei-core",)), None, 5) == (core, True)
    assert daemon.ensure(daemon.Wanted(("kettei-dot",)), None, 5) == (dot, False)
    assert daemon.ensure(daemon.Wanted(("kettei-dot",), "q8"), None, 5) == (core, True)  # Dot at 8 bits isn't served
    assert daemon.ensure(daemon.Wanted(("kettei-dot",), "fp16", reads={"settings": settings}), None, 5) == (
        dot,
        False,
    )
    for other in (daemon.Wanted(("kettei-dot",), "fp32"), daemon.Wanted(("kettei-dot",), reads={"settings": {}})):
        assert daemon.ensure(other, None, 5) == (core, True)
    with pytest.raises(ModelNotFound, match=r"other --precision --settings than asked"):  # a server named outright
        daemon.match(listed, "kettei-dot", "fp32", {"settings": {"labels": ["yes", "no"]}})
    with pytest.raises(ModelNotFound, match=r"other --precision than asked"):  # the long name means the same
        daemon.match(listed, "kettei-dot", "float32", {"settings": settings})
    with pytest.raises(InvalidRequest, match="precision 'fp64'"):
        daemon.match(listed, "kettei-dot", "fp64", {"settings": settings})
    # an option given at its default is still given: --truncation creator never reuses a server that refuses long input
    refusing = [ServedModel("kettei-dot", "Sqwish/Kettei-Dot", reads=Reads(truncation="error"), precision="fp16")]
    given = cli.asked_reads(cli.common_options().parse_args(["--truncation", "creator"]))
    with pytest.raises(ModelNotFound, match=r"other --truncation than asked"):
        daemon.match(refusing, "kettei-dot", reads=given)
    assert daemon.match(refusing, "kettei-dot", reads=cli.asked_reads(cli.common_options().parse_args([])))
    # the MCP tool that lists models describes the server a decision would use: none yet for Core, Dot for Dot
    monkeypatch.setattr(mcp, "listing", lambda describe: [])
    monkeypatch.setattr(daemon, "describe", lambda server, key: {"url": server.url})
    assert mcp.Session(io.BytesIO(), daemon.Wanted(("kettei-core",))).list_models({})["server"] is None
    assert mcp.Session(io.BytesIO(), daemon.Wanted(("kettei-dot",))).list_models({})["server"] == {"url": dot.url}

    def unreachable(self: daemon.Client) -> list[ServedModel]:
        raise Unavailable("still loading")

    monkeypatch.setattr(daemon.Client, "models", unreachable)  # still loading, or needing another key: its record
    assert daemon.ensure(daemon.Wanted(("kettei-dot",)), None, 5) == (dot, False)
    assert daemon.ensure(daemon.Wanted(("kettei-core",)), None, 5) == (core, True)


def test_a_server_that_never_replies_is_described_as_not_answering_without_holding_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = socket.socket()  # accepts connections and never replies, as a stopped or stuck server does
    frozen.bind(("127.0.0.1", 0))
    frozen.listen()
    port = frozen.getsockname()[1]
    monkeypatch.setattr(daemon, "DESCRIBE_TIMEOUT_S", 0.3)
    try:
        start = time.monotonic()
        serving = daemon.Server(1, port, url_of(port), (("toy", "toy"),), status="serving", started=100.0)
        found = daemon.describe(serving)
        assert time.monotonic() - start < 3
        assert found["error"]["message"].startswith("not answering within 0.3 s")
        assert [m["id"] for m in found["models"]] == ["toy"]
    finally:
        frozen.close()


def test_mcp_accepts_local_media_file_paths(tmp_path: Path) -> None:
    """MCP lint and decide resolve local image/video/audio file paths to base64 before talking to the server."""
    photo = Path(__file__).resolve().parent / "data/media/images/plain.png"
    session = mcp.Session(io.BytesIO())
    checked = session.call(
        "lint",
        {
            "state": "Is this photo sharp?",
            "images": [str(photo)],
            "questions": {"sharp": {"type": "noul", "instructions": "Is the photo sharp?"}},
        },
    )
    assert checked["isError"] is False
    assert checked["structuredContent"]["ok"] is True
    missing = session.call(
        "lint",
        {
            "state": "Missing photo",
            "images": [str(tmp_path / "missing.png")],
            "questions": {"sharp": {"type": "noul", "instructions": "Is the photo sharp?"}},
        },
    )
    assert missing["isError"] is True
    assert "no such file" in missing["content"][0]["text"]
    jpeg = io.BytesIO()
    Image.open(photo).convert("RGB").save(jpeg, "JPEG")
    sent = base64.b64encode(jpeg.getvalue()).decode()
    assert sent.startswith("/9j/")  # every JPEG's base64 starts like a path, and is still an image
    raw = session.call(
        "lint",
        {"state": "Is this photo sharp?", "images": [sent],
         "questions": {"sharp": {"type": "noul", "instructions": "Is the photo sharp?"}}},
    )  # fmt: skip
    assert raw["isError"] is False
    assert raw["structuredContent"]["ok"] is True
