"""A decision model behind `ippon serve`, POST /v1/systemone {state, questions}, timed exactly as
llama_systemone.py times llama-server: its own server process (127.0.0.1, a free port), client-side time around each
request (JSON encoding, HTTP and decoding included). The comparison with llama.cpp then pays the same HTTP costs on
both sides.

    envs/harness/bin/python -I adapters/ippon_serve.py --engine d1-2B-gguf-q8-ippon-serve \
        --ippon envs/ippon/bin/ippon --model ~/models/d1-2B-GGUF/d1-2B-Q8_0.gguf --max-input 32768
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common
sys.path.insert(0, str(Path(__file__).resolve().parent))  # adapters/: llama_systemone's HTTP helpers
import common
from llama_systemone import free_port, get, post


def main() -> None:
    p = common.add_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    p.add_argument("--ippon", required=True, help="the ippon command (envs/ippon/bin/ippon)")
    p.add_argument("--model", required=True, help="a model directory, a GGUF file, or a build's repo:QUANT")
    p.add_argument("--log", help="server log file (default: beside --out)")
    p.add_argument("--ippon-precision", help="ippon serve's --precision (default: its own, auto)")
    args = p.parse_args()

    port = free_port()
    log_path = common.server_log(args)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    model = str(Path(args.model).expanduser()) if Path(args.model).expanduser().exists() else args.model
    cmd = [str(Path(args.ippon).expanduser()), "serve", model, "--host", "127.0.0.1", "--port", str(port)]
    cmd += ["--precision", args.ippon_precision] if args.ippon_precision else []
    base = f"http://127.0.0.1:{port}"
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 900
        while True:
            if proc.poll() is not None:
                sys.exit(f"ippon serve exited ({proc.returncode}); see {log_path}")
            try:
                if get(f"{base}/health").get("status") == "ok":
                    break
            except (OSError, ValueError):
                pass
            if time.time() > deadline:
                sys.exit(f"ippon serve not healthy after 900 s; see {log_path}")
            time.sleep(0.5)
        models = get(f"{base}/v1/models")

        def decide(req):
            out = post(f"{base}/v1/systemone", req["systemone"])
            return common.systemone_answers(out), out.get("usage", {}).get("input_tokens"), None

        def memory():
            fp = common.footprint(proc.pid)
            return {"server_phys_footprint": fp["phys_footprint"]} if fp else None

        details = (models.get("data") or [{}])[0].get("details", {})
        runtime = json.loads(
            subprocess.check_output(
                [
                    str(Path(args.ippon).with_name("python")),
                    "-I",
                    "-c",
                    "import json; from ippon.machine import describe; print(json.dumps(describe()))",
                ],
                text=True,
            )
        )
        meta = {
            "runtime": f"ippon {runtime['ippon']} + mlx {runtime['mlx']}",
            "revision": runtime["revision"],
            "mlx": runtime["mlx"],
            "display_runtime": f"ippon {runtime['ippon']}",
            "server_cmd": [Path(cmd[0]).name, *cmd[1:]],
            "model": model,
            "precision": details.get("precision"),
            "adapter": details.get("adapter"),
        }
        common.run(args, decide, meta, memory=memory, server_pid=proc.pid)
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=30)


if __name__ == "__main__":
    main()
