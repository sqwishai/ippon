"""A GGUF decision model behind llama.cpp's llama-server, POST /v1/systemone {state, questions} (stdlib only).

    envs/harness/bin/python -I adapters/llama_systemone.py --engine d1-2B-llama-Q8_0 \
        --server bin/llama-b11960/llama-server --gguf ~/models/d1-2B-GGUF/d1-2B-Q8_0.gguf \
        --max-input 32768 -- -c 32768

Starts its own llama-server (127.0.0.1, a free port, the arguments after `--`), waits for /health, times
each request on the client side (JSON encoding, HTTP and decoding included), and stops the server it
started. The prompt cache stays on (llama.cpp's default) so a request's questions can share its state as
llama.cpp intends; every state starts with a unique ID, so nothing is reused across requests.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common, gen_requests
import common


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get(url: str, timeout: float = 5) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def post(url: str, body: dict, timeout: float = 600) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read()[:1000].decode(errors='replace')}") from None


def main() -> None:
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1 :] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    p = common.add_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    p.add_argument("--server", required=True, help="the llama-server binary")
    p.add_argument("--gguf", required=True)
    p.add_argument("--log", help="server log file (default: beside --out)")
    args = p.parse_args(argv)

    server = str(Path(args.server).expanduser())
    gguf = str(Path(args.gguf).expanduser())
    port = free_port()
    log_path = common.server_log(args)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [server, "-m", gguf, "--host", "127.0.0.1", "--port", str(port), *extra]
    version = subprocess.run([server, "--version"], capture_output=True, text=True, check=False).stderr.strip()
    base = f"http://127.0.0.1:{port}"
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(common.RESULTS))
    try:
        deadline = time.time() + 600
        while True:
            if proc.poll() is not None:
                sys.exit(f"llama-server exited ({proc.returncode}); see {log_path}")
            try:
                if get(f"{base}/health").get("status") == "ok":
                    break
            except (OSError, ValueError):
                pass
            if time.time() > deadline:
                sys.exit(f"llama-server not healthy after 600 s; see {log_path}")
            time.sleep(0.5)
        props = get(f"{base}/props")
        models = get(f"{base}/v1/models")

        def decide(req):
            out = post(f"{base}/v1/systemone", req["systemone"])
            return common.systemone_answers(out), out.get("usage", {}).get("input_tokens"), None

        def memory():
            fp = common.footprint(proc.pid)
            return {"server_phys_footprint": fp["phys_footprint"]} if fp else None

        settings = props.get("default_generation_settings", {})
        meta = {
            "runtime": "llama.cpp llama-server (Metal)",
            "llama_server_version": version,
            "server_cmd": [os.path.basename(server), *cmd[1:]],
            "gguf": gguf,
            "gguf_bytes": os.path.getsize(gguf),
            "n_ctx": settings.get("n_ctx") or props.get("n_ctx"),
            "total_slots": props.get("total_slots"),
            "build_info": props.get("build_info"),
            "model_architecture": (models.get("data") or [{}])[0].get("architecture")
            if isinstance(models.get("data"), list)
            else None,
            "prompt_cache": "on (default); every state starts with a unique ID",
        }
        common.run(args, decide, meta, memory=memory, server_pid=proc.pid)
    finally:
        proc.send_signal(signal.SIGINT)  # llama-server shuts down cleanly on SIGINT
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=30)


if __name__ == "__main__":
    main()
