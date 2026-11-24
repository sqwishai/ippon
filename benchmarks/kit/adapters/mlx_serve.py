"""Time the community Clef build through its own HTTP server.

    envs/harness/bin/python -I adapters/mlx_serve.py --engine clef-flash-mlx-8bit \
        --python envs/mlxvlm/bin/python --model ~/models/clef-flash-mlx-8bit --max-input 16384

The community Clef checkpoint's clef_mlx.py serves /v1/systemone with the same load/systemone functions as
clef_mlx_community.py. Questions are sent unchanged; only the result field names are normalized for comparison.
Loading and native startup warmups finish before common.run begins timing client-side encoding, HTTP, and decoding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from llama_systemone import free_port, get, post


def main() -> None:
    parser = common.add_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    parser.add_argument("--python", required=True, help="the installed mlx-vlm environment's Python interpreter")
    parser.add_argument("--model", required=True, help="a downloaded local MLX build")
    parser.add_argument("--log", help="server log file (default: beside --out)")
    args = parser.parse_args()
    python = str(Path(args.python).expanduser())
    model = Path(args.model).expanduser().resolve()
    if not model.is_dir():
        parser.error(f"--model must name a local directory: {model}")
    script = model / "clef_mlx.py"
    if not script.is_file():
        parser.error(f"the selected build does not contain {script.name}")
    version_code = (
        "import json; from importlib.metadata import distributions; "
        "wanted = {'mlx-vlm', 'mlx-lm', 'mlx', 'fastapi', 'uvicorn', 'starlette', 'pydantic'}; "
        "print(json.dumps({d.metadata['Name']: d.version for d in distributions() "
        "if d.metadata['Name'].lower().replace('_', '-') in wanted}))"
    )
    versions = json.loads(subprocess.check_output([python, "-I", "-c", version_code], text=True, timeout=30))
    normalized = {name.lower().replace("_", "-"): version for name, version in versions.items()}
    port = free_port()
    command = [python, "-I", str(script), "serve", "--model", str(model),
               "--host", "127.0.0.1", "--port", str(port)]  # fmt: skip
    endpoint = "/v1/systemone"
    env = {key: value for key, value in os.environ.items() if not key.startswith("MLX_VLM_")}
    base = f"http://127.0.0.1:{port}"
    log_path = common.server_log(args)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 900
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"clef_mlx.py's server exited ({proc.returncode}); see {log_path}")
            health = None
            with suppress(OSError, ValueError):
                health = get(f"{base}/health")
            if health and health.get("status") == "ok":
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"clef_mlx.py's server not healthy after 900 s; see {log_path}")
            time.sleep(0.5)

        def decide(req):
            output = post(f"{base}{endpoint}", {**req["systemone"], "model": str(model)})
            answers = common.systemone_answers(output)
            return answers, (output.get("usage") or {}).get("input_tokens"), None

        def memory():
            footprint = common.footprint(proc.pid)
            return {"server_phys_footprint": footprint["phys_footprint"]} if footprint else None

        commit_file = Path(python).parent.parent / "MLX_VLM_COMMIT"
        meta = {
            "runtime": f"mlx-vlm {normalized['mlx-vlm']} + mlx {normalized['mlx']}",
            "versions": versions,
            "mlx_vlm_commit": commit_file.read_text().strip() if commit_file.is_file() else None,
            "server_cmd": [Path(python).name, *command[1:]],
            "endpoint": endpoint,
            "model_dir": str(model),
            "revision": common.repo_revision(str(model)),
            "health": health,
            "question_method": f"one native {endpoint} request containing all questions",
        }
        meta.update(
            build="community",
            server_source="clef_mlx.py",
            server_source_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        )
        common.run(args, decide, meta, memory=memory, server_pid=proc.pid)
    finally:
        if proc.poll() is None:
            with suppress(ProcessLookupError):
                proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)


if __name__ == "__main__":
    main()
