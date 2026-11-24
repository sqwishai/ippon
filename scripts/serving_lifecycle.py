"""Exercise an installed server through CLI, HTTP and MCP, then crash it and check automatic recovery.

Run with a text reference, for example:
    python scripts/serving_lifecycle.py MODEL --reference references/d1/d1-2B.json.gz --seconds 600 --out run.json

Uses an isolated cache, offline model loading and bounded HTTP waits. All owned servers stop on exit.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from types import FrameType
from typing import Any

from ippon import daemon
from ippon.errors import InvalidRequest, Overloaded
from ippon.schema import parse_request


def command(arguments: list[str], expected: int = 0, data: str | None = None) -> Any:
    process = subprocess.run(
        [str(Path(sys.executable).parent / "ippon"), *arguments],
        input=data,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if process.returncode != expected or "Traceback" in process.stderr:
        raise AssertionError(f"{arguments}: exit {process.returncode}\n{process.stdout}\n{process.stderr}")
    return json.loads(process.stdout) if "--json" in arguments else process.stdout


def until(predicate: Callable[[], bool], timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError("the server did not reach its expected state before the deadline")


def check_answer(case: dict[str, Any], answer: dict[str, Any], bound: float) -> float:
    maximum = 0.0
    for question in parse_request(case["request"]).questions:
        actual = answer["answers"][question.id]
        got = (
            [actual["noul"], 1 - actual["noul"]]
            if question.type == "noul"
            else [actual["probabilities"][option] for option in question.options]
        )
        want = case["probabilities"][question.id]
        assert len(got) == len(want), actual
        assert all(math.isfinite(value) for value in got), actual
        difference = max(abs(a - b) for a, b in zip(got, want, strict=True))
        assert difference <= bound, (case["name"], difference, bound, got, want)
        ordered = sorted(want, reverse=True)
        if ordered[0] - ordered[1] > bound:
            assert max(range(len(got)), key=got.__getitem__) == max(range(len(want)), key=want.__getitem__), actual
        maximum = max(maximum, difference)
    if "input_tokens" in case:
        assert answer["usage"]["input_tokens"] == case["input_tokens"], case["name"]
    return maximum


def soak(
    client: daemon.Client, cases: list[dict[str, Any]], bound: float, seconds: float, clients: int
) -> dict[str, Any]:
    """Bounded concurrent requests, typed backpressure, periodic health checks and memory samples."""
    counts: Counter[str] = Counter()
    lock, stopped = threading.Lock(), threading.Event()
    latencies: list[float] = []
    samples: list[dict[str, Any]] = []
    maximum = 0.0
    deadline = time.monotonic() + seconds
    next_report = time.monotonic() + 30

    def worker(index: int) -> None:
        nonlocal maximum
        iteration = index
        try:
            while time.monotonic() < deadline and not stopped.is_set():
                case = cases[iteration % len(cases)]
                start = time.monotonic()
                try:
                    answer = client.call("/v1/systemone", case["request"], timeout=30)
                except Overloaded as error:
                    with lock:
                        counts[error.kind] += 1
                    stopped.wait(float(error.retry_after or "1"))
                else:
                    difference = check_answer(case, answer, bound)
                    with lock:
                        counts["answered"] += 1
                        maximum = max(maximum, difference)
                        latencies.append(time.monotonic() - start)
                if iteration % 31 == 0:
                    try:
                        client.call("/v1/systemone", {"questions": []}, timeout=30)
                    except InvalidRequest:
                        with lock:
                            counts["invalid_request"] += 1
                    except Overloaded as error:
                        with lock:
                            counts[error.kind] += 1
                        stopped.wait(float(error.retry_after or "1"))
                    else:
                        raise AssertionError("malformed request was accepted")
                iteration += 1
        except BaseException:
            stopped.set()
            raise
        finally:
            client.close()

    with ThreadPoolExecutor(max_workers=clients) as pool:
        pending = [pool.submit(worker, index) for index in range(clients)]
        try:
            while not stopped.is_set() and not all(future.done() for future in pending):
                assert client.healthy(), "server became unhealthy under load"
                status = client.call("/v1/status", timeout=5)
                samples.append({"remaining_s": max(0.0, deadline - time.monotonic()), **status})
                if time.monotonic() >= next_report:
                    with lock:
                        print(
                            json.dumps(
                                {
                                    "remaining_s": round(deadline - time.monotonic()),
                                    "counts": dict(counts),
                                    "footprint": status["memory"]["footprint"],
                                }
                            ),
                            flush=True,
                        )
                    next_report = time.monotonic() + 30
                stopped.wait(min(5, max(0.1, deadline - time.monotonic())))
            for future in pending:
                future.result()
        finally:
            stopped.set()
    assert counts["answered"] >= clients, counts
    assert counts["invalid_request"], counts
    footprints = [sample["memory"]["footprint"] for sample in samples if sample["memory"]["footprint"] is not None]
    assert footprints, "server footprint measurements are unavailable"
    quarter = max(1, len(footprints) // 4)
    growth = statistics.median(footprints[-quarter:]) - statistics.median(footprints[:quarter])
    assert growth <= 256 << 20, f"server footprint grew by {growth / 2**20:.1f} MiB across the steady load"
    return {
        "seconds": seconds,
        "clients": clients,
        "counts": dict(counts),
        "max_probability_error": maximum,
        "latency_ms": {"median": statistics.median(latencies) * 1000, "max": max(latencies) * 1000},
        "footprint_growth_bytes": growth,
        "samples": samples,
    }


def mcp(client: daemon.Client, case: dict[str, Any], bound: float) -> None:
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "lifecycle", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "decide", "arguments": case["request"]}},
    ]
    payload = "\n".join(json.dumps(message) for message in messages) + "\n"
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(command, ["mcp", "--url", client.url], data=payload) for _ in range(2)]
        for future in pending:
            replies = {message["id"]: message for message in map(json.loads, future.result().splitlines())}
            result = replies[2]["result"]
            assert not result["isError"], result
            check_answer(case, result["structuredContent"], bound)


def exercise(args: argparse.Namespace) -> dict[str, Any]:
    raw = args.reference.read_bytes()
    reference = json.loads(gzip.decompress(raw) if args.reference.suffix == ".gz" else raw)
    assert reference.get("schema") == "ippon-reference/1"
    cases = [case for case in reference["requests"] if "probabilities" in case and not case.get("media")]
    assert cases, "a text probability reference is required"
    assert not reference.get("tokens_only"), "a text probability reference is required"
    bound = float(reference["bound"])
    case = cases[0]
    command(["--help"])
    assert not any(check["status"] == "fail" for check in command(["doctor", "--json"]))
    assert not daemon.running(), "the lifecycle cache must be isolated"
    options = ("--no-keep-awake", "--prefix-budget", str(64 << 20), "--answer-cache", "0", "--max-queue", "4",
               "--max-batch", "4", "--max-body-mb", "1")  # fmt: skip
    wanted = daemon.Wanted((args.model,), args.precision, options, port=0, keep_alive="1s", offline=True)
    server = daemon.start(wanted, None, 120)
    client = daemon.Client(server.url)
    try:
        for sample in cases:
            check_answer(sample, client.call("/v1/systemone", sample["request"], timeout=30), bound)
        until(lambda: not any(model.loaded for model in client.models()))
        check_answer(case, client.call("/v1/systemone", case["request"], timeout=30), bound)
        large = {**case["request"], "state": "x" * (2 << 20)}
        failure = command(["decide", args.model, "-", "--server", client.url, "--json"], 3, json.dumps(large))
        assert failure["error"]["type"] == "invalid_request"
        assert "--max-body-mb" in failure["error"]["message"]
        mcp(client, case, bound)
        assert [item.pid for item in daemon.running()] == [server.pid]
        result = soak(client, cases, bound, args.seconds, args.clients)
        assert daemon.alive(server), "refuse to signal a PID whose identity changed"
        os.kill(server.pid, signal.SIGKILL)
        until(lambda: not daemon.running())
        recovered = command(
            ["decide", args.model, "-", "--server", "--precision", args.precision, "--json"],
            data=json.dumps(case["request"]),
        )
        check_answer(case, recovered, bound)
        replacement = daemon.running()
        assert len(replacement) == 1
        assert replacement[0].pid != server.pid
        stopped = command(["stop", "--json"])
        assert len(stopped["stopped"]) == 1, stopped
        assert not stopped["stopped"][0]["killed"], stopped
        assert not daemon.running()
        return {
            "passed": True,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mlx": version("mlx"),
            "model": args.model,
            "reference": str(args.reference),
            "precision": args.precision,
            "bound": bound,
            "reference_cases": len(cases),
            "checks": [
                "doctor",
                "inference",
                "idle_reload",
                "large_body",
                "two_mcp_clients",
                "sustained_load",
                "forced_termination",
                "stale_record_cleanup",
                "automatic_restart",
                "clean_stop",
            ],
            "load": result,
        }
    finally:
        client.close()
        for owned in daemon.running():
            daemon.stop(owned)


def interrupted(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 1 or args.clients < 1:
        parser.error("--seconds and --clients must be positive, with at least one second of load")
    args.out = args.out.resolve()
    args.reference = args.reference.resolve()
    model_path = Path(args.model).expanduser()
    if model_path.exists():
        args.model = str(model_path.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, interrupted)
    with tempfile.TemporaryDirectory(prefix="ippon-lifecycle-") as directory:
        scratch = Path(directory)
        os.environ.update(IPPON_CACHE=str(scratch / "cache"), IPPON_CONFIG=str(scratch / "config"), HF_HUB_OFFLINE="1")
        os.environ.pop("IPPON_API_KEY", None)
        os.chdir(scratch)
        try:
            result = exercise(args)
        except BaseException as error:
            result = {"passed": False, "error": f"{type(error).__name__}: {error}"}
            raise
        finally:
            args.out.write_text(json.dumps(result, indent=2) + "\n")
            for log in (scratch / "cache/servers").glob("*.log"):
                shutil.copyfile(log, args.out.parent / f"{args.out.stem}-{log.name}")
    print(json.dumps({"passed": True, "report": str(args.out), "answered": result["load"]["counts"]["answered"]}))


if __name__ == "__main__":
    main()
