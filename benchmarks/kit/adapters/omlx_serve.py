"""A decision model behind oMLX's server (jundot/omlx), POST /v1/systemone {model, state, questions}, timed exactly
as llama_systemone.py and ippon_serve.py time theirs: its own server process (127.0.0.1, a free port), the
models it serves in one folder (`omlx serve --model-dir`), client-side time around each request.

    envs/harness/bin/python -I adapters/omlx_serve.py --engine clef-flash-mlx-8bit-omlx --omlx envs/omlx/bin/omlx \
        --model-dir ~/models/omlx --model clef-flash-mlx-8bit --max-input 16384
"""

from __future__ import annotations

import argparse
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
    p.add_argument("--omlx", required=True, help="the omlx command (envs/omlx/bin/omlx)")
    p.add_argument("--model-dir", required=True, help="the folder of model folders oMLX serves")
    p.add_argument("--model", required=True, help="the model's folder name in it, as requests name it")
    p.add_argument("--log", help="server log file (default: beside --out)")
    args = p.parse_args()

    port = free_port()
    log_path = common.server_log(args)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(Path(args.omlx).expanduser()), "serve", "--model-dir", str(Path(args.model_dir).expanduser()),
           "--host", "127.0.0.1", "--port", str(port)]  # fmt: skip
    base = f"http://127.0.0.1:{port}"
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 900
        while True:
            if proc.poll() is not None:
                sys.exit(f"omlx serve exited ({proc.returncode}); see {log_path}")
            try:
                if any(m.get("id") == args.model for m in get(f"{base}/v1/models").get("data", [])):
                    break
            except (OSError, ValueError):
                pass
            if time.time() > deadline:
                sys.exit(f"omlx serve didn't list {args.model} after 900 s; see {log_path}")
            time.sleep(0.5)

        def decide(req):
            body = {"model": args.model, **req["systemone"]}
            if "files" in body:  # oMLX 0.7.2 ignores System One's `files` silently: it reads pictures from `images`
                files = body.pop("files")  # and, for d3, videos from `videos` (omlx/models/d3.py)
                for field, kind in (("images", "data:image/"), ("videos", "data:video/")):
                    if chosen := [url for url in files if url.startswith(kind)]:
                        body[field] = chosen
            out = post(f"{base}/v1/systemone", body)
            return common.systemone_answers(out), (out.get("usage") or {}).get("input_tokens"), None

        def memory():
            fp = common.footprint(proc.pid)
            return {"server_phys_footprint": fp["phys_footprint"]} if fp else None

        version = subprocess.check_output(
            [
                str(Path(args.omlx).with_name("python")),
                "-I",
                "-c",
                "from importlib.metadata import version; print(version('omlx'))",
            ],
            text=True,
        ).strip()
        meta = {"runtime": f"oMLX {version}", "server_cmd": [Path(cmd[0]).name, *cmd[1:]], "model": args.model}
        common.run(args, decide, meta, memory=memory, server_pid=proc.pid)
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=30)


if __name__ == "__main__":
    main()
