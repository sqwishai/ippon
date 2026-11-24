"""Run the kit's HTTP workload, stopping this harness's server with SIGTERM.

The kit's request generation, concurrency, timing and result serialization are used
unchanged. This wrapper owns only the server lifetime; it never edits the kit.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/kit"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/kit/adapters"))
import common
from llama_systemone import free_port, get, post


def main() -> None:
    parser = common.add_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--ippon", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--settle-seconds", type=float, default=0, help="untimed pause after the last warm-up")
    args = parser.parse_args()
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    log_path = Path(args.out).with_suffix(".server.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [args.ippon, "serve", args.model, "--host", "127.0.0.1", "--port", str(port)]
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Server exited; see {log_path}")
            try:
                if get(f"{base}/health").get("status") == "ok":
                    break
            except (OSError, ValueError):
                pass
            time.sleep(0.5)
        else:
            raise TimeoutError("Server did not start within 900 seconds")

        warmups = args.warmup * args.clients * len(args.sizes.split(",")) * len(args.questions.split(","))
        completed = 0
        guard = threading.Lock()

        def decide(request):
            nonlocal completed
            response = post(f"{base}/v1/systemone", request["systemone"])
            last_warmup = False
            if request["phase"] == "warmup":
                with guard:
                    completed += 1
                    last_warmup = completed == warmups
            if last_warmup and args.settle_seconds:
                time.sleep(args.settle_seconds)
            return common.systemone_answers(response), response.get("usage", {}).get("input_tokens"), None

        details = get(f"{base}/v1/models")
        meta = {"model": args.model, "server_cmd": command, "details": details, "settle_seconds": args.settle_seconds}
        common.run(args, decide, meta, server_pid=process.pid)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)


if __name__ == "__main__":
    main()
