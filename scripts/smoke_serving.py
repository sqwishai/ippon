"""Real-model serving smoke checks. Run under gpulock.py; all started servers are stopped before exit.

    uv run python scripts/smoke_serving.py mcp MODEL --out RESULT.json
    uv run python scripts/smoke_serving.py pressure MODEL --idle-model CAUSAL_MODEL --out RESULT.json

Pressure is injected only into this script's child process, never through a production HTTP endpoint.
The idle model must keep causal prefixes; a bidirectional model has no prefix cache to exercise.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

from ippon._portable import dumps

REQUEST: dict[str, Any] = {
    "state": "The customer was charged twice and asks for the duplicate charge to be refunded.",
    "questions": {"refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"}},
}
COMMAND = [sys.executable, "-m", "ippon"]


def cli(args: list[str], env: dict[str, str], data: str | None = None) -> str:
    result = subprocess.run(
        [*COMMAND, *args], input=data, capture_output=True, text=True, env=env, timeout=180, check=False
    )
    if result.returncode:
        raise RuntimeError(f"{args}: exit {result.returncode}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def mcp_smoke(model: str, env: dict[str, str]) -> dict[str, Any]:
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "smoke", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "decide",
                "arguments": REQUEST,
            },
        },
    ]
    payload = "\n".join(json.dumps(m) for m in messages) + "\n"
    assert json.loads(cli(["ps", "--json"], env)) == {"servers": []}
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [pool.submit(cli, ["mcp", model], env, payload) for _ in range(2)]
            replies = [[json.loads(line) for line in f.result().splitlines()] for f in pending]
        answers = []
        for reply in replies:
            result = next(m["result"] for m in reply if m["id"] == 2)
            assert result["isError"] is False, result
            answers.append(result["structuredContent"]["answers"])
        assert answers[0] == answers[1]
        status = json.loads(cli(["ps", "--json"], env))
        assert len(status["servers"]) == 1, status
        server = status["servers"][0]
        assert len(server["models"]) == 1
        assert server["models"][0]["loaded"]
        stopped = json.loads(cli(["stop", "--json"], env))
        assert stopped["stopped"][0]["pid"] == server["pid"]
        assert not stopped["stopped"][0]["killed"]
        after = json.loads(cli(["ps", "--json"], env))
        assert after == {"servers": []}
        return {
            "model": model,
            "client_messages": messages,
            "clients": replies,
            "ps": status,
            "stop": stopped,
            "ps_after_stop": after,
        }
    finally:
        cli(["stop", "--json"], env)


def pressure_child(model: str, idle_model: str, scratch: Path, port: int) -> int:
    import mlx.core as mx

    from ippon import cli as command
    from ippon import pressure, serve
    from ippon.prefixes import Prefixes

    trace_lock = threading.Lock()

    def setting() -> dict[str, Any]:
        return dict(json.loads((scratch / "reading.json").read_text()))

    def trace(step: str, **values: Any) -> None:
        if setting()["level"] == "normal":
            return
        event = {"step": step, "thread": threading.current_thread().name, **values}
        with trace_lock, (scratch / "trace.jsonl").open("a") as out:
            out.write(json.dumps(event) + "\n")

    def read() -> pressure.Reading:
        actual = pressure.read()
        level = setting()["level"]
        fraction = {"normal": 0.5, "soft": 0.05, "hard": 0.01}[level]
        return pressure.Reading(actual.footprint, actual.memory, int(actual.memory * fraction), actual.compressed, 1)

    class Injected(pressure.Monitor):
        def __init__(self) -> None:
            super().__init__(read, busy=0.05, idle=0.05)

    clear, limit, unload = mx.clear_cache, Prefixes.limit, serve.Models.unload

    def cleared() -> None:
        clear()
        trace("cache")

    def limited(prefixes: Prefixes, budget: int) -> None:
        previous, before = prefixes.budget, prefixes.nbytes
        limit(prefixes, budget)
        if budget < previous:
            trace("prefixes", before=before, after=prefixes.nbytes, previous_budget=previous, budget=budget)

    def unloaded(models: serve.Models, model: serve.Model) -> None:
        unload(models, model)
        trace("unload", model=model.name)

    with (
        patch.object(serve, "Monitor", Injected),
        patch.object(mx, "clear_cache", cleared),
        patch.object(Prefixes, "limit", limited),
        patch.object(serve.Models, "unload", unloaded),
    ):
        return command.main(
            [
                "serve",
                f"primary={model}",
                f"secondary={idle_model}",
                "--port",
                str(port),
                "--prefix-budget",
                str(64 << 20),
                "--no-keep-awake",
            ]
        )


def pressure_smoke(model: str, idle_model: str, env: dict[str, str], scratch: Path) -> dict[str, Any]:
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(path: str, body: Any = None) -> tuple[int, dict[str, str], Any]:
        request = urllib.request.Request(
            url + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with opener.open(request, timeout=30) as response:
                return response.status, dict(response.headers), json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers), json.load(error)

    def inject(level: str) -> None:
        temp = scratch / "reading.next"
        temp.write_text(json.dumps({"level": level}))
        temp.replace(scratch / "reading.json")

    def status_until(predicate: Any) -> dict[str, Any]:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError((scratch / "server.log").read_text())
            try:
                code, _, status = call("/v1/status")
                if code == 200 and predicate(status):
                    return dict(status)
            except OSError:
                pass
            time.sleep(0.05)
        raise RuntimeError("server status did not reach the expected state")

    inject("normal")
    with (scratch / "server.log").open("w") as log:
        child = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "child",
                model,
                "--idle-model",
                idle_model,
                "--scratch",
                str(scratch),
                "--port",
                str(port),
            ],
            env=env,
            stdout=log,
            stderr=log,
        )
        try:
            status_until(lambda s: s["memory"]["pressure"] == "normal")
            assert call("/v1/systemone", {**REQUEST, "model": "secondary"})[0] == 200
            for i in range(32):
                body = {**REQUEST, "model": "secondary", "state": f"Ticket {i}: " + REQUEST["state"] * 70}
                assert call("/v1/systemone", body)[0] == 200
                before = call("/v1/status")[2]
                if before["models"][1]["prefixes"] > (32 << 20):
                    break
            assert call("/v1/systemone", {**REQUEST, "model": "primary"})[0] == 200
            before = call("/v1/status")[2]
            assert all(m["loaded"] for m in before["models"])
            assert before["models"][1]["prefixes"] > (16 << 20), "idle model must retain causal prefixes"
            inject("soft")
            soft = status_until(lambda s: s["memory"]["pressure"] == "soft" and not s["models"][1]["loaded"])
            steps = [e["step"] for e in soft["given_back"]]
            assert steps.index("prefixes") < steps.index("cache") < steps.index("unload")
            assert soft["models"][0]["loaded"]
            assert soft["models"][1]["prefixes"] == 0
            inject("hard")
            hard = status_until(lambda s: s["memory"]["pressure"] == "hard")
            code, headers, rejected = call("/v1/systemone", {**REQUEST, "model": "primary"})
            assert (code, rejected["error"]["type"], headers["Retry-After"]) == (503, "overloaded", "5")
            inject("normal")
            status_until(lambda s: s["memory"]["pressure"] == "normal")
            code, _, answer = call("/v1/systemone", {**REQUEST, "model": "primary"})
            assert code == 200
            events = [json.loads(line) for line in (scratch / "trace.jsonl").read_text().splitlines()]
            assert {e["thread"] for e in events} == {"ippon-model"}, events
            assert any(e["step"] == "prefixes" and e["after"] < e["before"] for e in events)
            child.send_signal(signal.SIGTERM)
            assert child.wait(timeout=30) == 0
            with socket.socket() as probe:
                assert probe.connect_ex(("127.0.0.1", port)) != 0
            return {
                "model": model,
                "idle_model": idle_model,
                "pid": child.pid,
                "url": url,
                "before": before,
                "soft": soft,
                "hard": hard,
                "rejected": {"status": 503, "retry_after": headers["Retry-After"], "body": rejected},
                "recovered_answer": answer,
                "trace": events,
                "sigterm_exit_code": 0,
                "socket_closed": True,
            }
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("mcp", "pressure", "child"))
    parser.add_argument("model")
    parser.add_argument("--idle-model", help="pressure check: a downloaded causal model whose prefixes can be evicted")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.kind == "child":
        raise SystemExit(pressure_child(args.model, args.idle_model or args.model, args.scratch, args.port))
    if args.out is None:
        parser.error("--out is required")
    with tempfile.TemporaryDirectory(prefix="ippon-serving-smoke-") as temporary:
        scratch = Path(temporary)
        env = {**os.environ, "IPPON_CACHE": str(scratch / "cache"), "HF_HUB_OFFLINE": "1"}
        result = (
            mcp_smoke(args.model, env)
            if args.kind == "mcp"
            else pressure_smoke(args.model, args.idle_model or args.model, env, scratch)
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dumps(result, indent=2) + "\n")
    print(dumps({"smoke": args.kind, "passed": True, "transcript": str(args.out)}))


if __name__ == "__main__":
    main()
