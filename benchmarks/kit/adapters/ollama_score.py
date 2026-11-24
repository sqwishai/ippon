"""A decision model behind Ollama's server (ollama/ollama), POST /v1/systemone {model, state, questions}, the route
Ollama documents for decision models (docs/capabilities/decision.mdx, docs/openapi.yaml), timed exactly as
llama_systemone.py and omlx_serve.py time theirs: its own server process (127.0.0.1, a free port), its own model
store (OLLAMA_MODELS, never ~/.ollama/models), client-side time around each request.

    envs/harness/bin/python -I adapters/ollama_score.py --engine d3-mini-mxfp8-ollama \
        --ollama bin/ollama-v0.40.11/ollama --models-dir ~/models/ollama --model d3:4b-mxfp8 --max-input 4096

Ollama loads a model on its first request into a runner process of its own (`ollama runner`), which holds the
weights: the adapter loads it with one untimed request before warming up, keeps it loaded for the whole run
(OLLAMA_KEEP_ALIVE=-1, one model at a time) and measures the runner's memory beside the server's. The server gets no
other OLLAMA_* settings but OLLAMA_NUM_PARALLEL, set to the most clients that send at once (--clients, or a --plan's
busiest workload), so everything else is Ollama's default; each package's context is its own (its Modelfile's
num_ctx), and Ollama never truncates a prompt, so --max-input is that context and sizes past it are skipped.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common, gen_requests
sys.path.insert(0, str(Path(__file__).resolve().parent))  # adapters/: llama_systemone's HTTP helpers
import common
import gen_requests as g
from llama_systemone import free_port, get, post


def runner_pid(server: int) -> int | None:
    """The `ollama runner` the server started for the model (its child process), if there is one."""
    children = subprocess.run(["pgrep", "-P", str(server)], capture_output=True, text=True, check=False).stdout.split()
    for pid in children:
        command = subprocess.run(
            ["ps", "-o", "command=", "-p", pid], capture_output=True, text=True, check=False
        ).stdout
        if " runner" in command:
            return int(pid)
    return None


def stop(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGINT)  # ollama serve stops its runners and exits on SIGINT
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(timeout=30)


def main() -> None:
    p = common.add_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    p.add_argument("--ollama", required=True, help="the ollama binary (bin/ollama-v0.40.11/ollama)")
    p.add_argument("--models-dir", required=True, help="the model store it serves (OLLAMA_MODELS)")
    p.add_argument("--model", required=True, help="the build's name:tag in it, as requests name it")
    p.add_argument("--log", help="server log file (default: beside --out)")
    args = p.parse_args()

    port = free_port()
    log_path = common.server_log(args)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    store = str(Path(args.models_dir).expanduser())
    settings = {
        "OLLAMA_HOST": f"127.0.0.1:{port}",
        "OLLAMA_MODELS": store,
        "OLLAMA_KEEP_ALIVE": "-1",  # loaded until the server stops
        "OLLAMA_MAX_LOADED_MODELS": "1",
        "OLLAMA_NOPRUNE": "1",  # leave the store's blobs as they are
        "OLLAMA_NO_CLOUD": "1",
    }
    if (clients := common.most_clients(args)) > 1:  # Ollama serves that many at once only when told to
        settings["OLLAMA_NUM_PARALLEL"] = str(clients)
    env = {k: v for k, v in os.environ.items() if not k.startswith("OLLAMA_")} | settings
    cmd = [str(Path(args.ollama).expanduser()), "serve"]
    base = f"http://127.0.0.1:{port}"
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    runner = None
    try:
        deadline = time.time() + 120
        while True:
            if proc.poll() is not None:
                sys.exit(f"ollama serve exited ({proc.returncode}); see {log_path}")
            try:
                tags = get(f"{base}/api/tags").get("models", [])
                break
            except (OSError, ValueError):
                pass
            if time.time() > deadline:
                sys.exit(f"ollama serve not answering after 120 s; see {log_path}")
            time.sleep(0.5)
        listed = next((m for m in tags if args.model in (m.get("name"), m.get("model"))), None)
        if listed is None:
            sys.exit(f"{args.model} is not in {store}; pull it first (setup.sh ollama)")
        version = get(f"{base}/api/version").get("version")
        show = post(f"{base}/api/show", {"model": args.model})

        # load: one untimed request of its own (a state no later request shares), so warm-up starts loaded
        load = g.to_systemone(g.state(100, args.seed, "load"), g.questions(1))
        t0 = time.perf_counter()
        post(f"{base}/v1/systemone", {"model": args.model, **load}, timeout=900)
        load_s = time.perf_counter() - t0
        ps = get(f"{base}/api/ps").get("models", [])
        loaded = next((m for m in ps if args.model in (m.get("name"), m.get("model"))), {})
        runner = runner_pid(proc.pid)

        def decide(req):
            body = {"model": args.model, **req["systemone"]}
            # the kit's media are data URLs under "files", as llama.cpp takes them; Ollama 0.40.13 takes them base64
            # under "images", pictures and clips alike, and tells a clip by its bytes (its request has no audio field)
            for url in body.pop("files", []):
                body.setdefault("images", []).append(url.split(",", 1)[1])
            out = post(f"{base}/v1/systemone", body)
            cached = out.get("prompt_eval_cached_count")
            extra = {"prompt_eval_cached_count": cached} if cached else None
            return common.systemone_answers(out), (out.get("usage") or {}).get("input_tokens"), extra

        def memory():
            server, model = common.footprint(proc.pid), common.footprint(runner) if runner else None
            if not server:
                return None
            out = {"server_phys_footprint": server["phys_footprint"]}
            if model:
                out["runner_phys_footprint"] = model["phys_footprint"]
                out["total_phys_footprint"] = server["phys_footprint"] + model["phys_footprint"]
            return out

        details = show.get("details") or {}
        meta = {
            "runtime": f"Ollama {version} serve (HTTP, its {details.get('runner', '?')} runner)",
            "ollama_version": version,
            "server_cmd": [Path(cmd[0]).name, *cmd[1:]],
            "server_env": settings | {"OLLAMA_HOST": "127.0.0.1:<free port>"},
            "model": args.model,
            "manifest_digest": listed.get("digest"),
            "size_bytes": listed.get("size"),
            "format": details.get("format"),
            "quantization": details.get("quantization_level"),
            "renderer": show.get("renderer"),  # Ollama's prompt for the model's family
            "capabilities": show.get("capabilities"),
            "parameters": show.get("parameters"),
            "context_length": loaded.get("context_length"),
            "loaded_bytes": loaded.get("size"),
            "load_s": round(load_s, 2),
            "runner_pid_found": runner is not None,
        }
        common.run(args, decide, meta, memory=memory, server_pid=runner or proc.pid)
    finally:
        stop(proc)
        if runner and common.footprint(runner):  # a runner its server left behind
            os.kill(runner, signal.SIGTERM)


if __name__ == "__main__":
    main()
