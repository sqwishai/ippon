"""Time a model through the HTTP server its creator ships with it, on MPS.

    envs/harness/bin/python -I adapters/maker_serve.py --engine d3-mini-tf --family d3 \
        --python envs/vllmsr/bin/python --model ~/models/d3-mini --max-input 32768
    envs/harness/bin/python -I adapters/maker_serve.py --engine torchcast-4b-maker-mps --family torchcast \
        --python envs/torchcast/bin/python --model ~/models/torchcast-decision-4b --max-input 16384

d3: vllm-sr's d3_server.py, in its default batches of eight questions, as in tf_systemone.py. torchcast: the
checkpoint's torchcast_decision.server (v1.3.1), every question of a request in one padded batch, bf16 under
autocast. glide: Fastino's glide.server (v1.2.0, shipped in each GLiDE checkpoint) on MPS in its default bf16,
with its micro-batching of concurrent requests at its defaults. kev: Kev's own kev.serve (KEV_SOURCE: its checkout,
installed in --python's venv from its uv.lock) on the Mac, where it serves through its MLX backend in bf16 by default
(LoadOptions.backend "auto"), the LoRA merged; it has no /health, so it is ready once /v1/models answers. Each
server loads its runtime (d3 and torchcast also warm it) before /health reports it ready; requests go unchanged to
/v1/systemone. Timing includes client JSON encoding, HTTP, and response decoding. The adapter stops only the server
it started.
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


def d3_server(python: str, model: Path, port: int) -> list[str]:
    return [python, "-I", str(model / "d3_server.py"), "--model", str(model),
            "--device", "mps", "--batch-size", "8", "--host", "127.0.0.1", "--port", str(port)]  # fmt: skip


def torchcast_server(python: str, model: Path, port: int) -> list[str]:
    # the package imported from the checkpoint, where its card runs `python -m torchcast_decision.server`
    run = (
        f"import runpy, sys; sys.path.insert(0, {str(model)!r}); "
        "runpy.run_module('torchcast_decision.server', run_name='__main__')"
    )
    return [
        python,
        "-I",
        "-c",
        run,
        "--model",
        str(model),
        "--device",
        "mps",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def qev_server(python: str, model: Path, port: int) -> list[str]:
    # OneJev's qev with the checkpoint's serve.patch applied (QEV_SOURCE: that checkout, at the commit setup.sh pins):
    # its text engine, the one serve.patch fixes and the checkpoint's engine.py runs, in its production dtypes (bf16
    # weights, fp32 head) on MPS, behind qev's own server app. `qev serve` picks the multimodal engine for a checkpoint
    # with a vision tower, which serve.patch leaves causal and whose start-up check fails on MPS (0.055, M5 Pro).
    run = (
        f"import sys; sys.path.insert(0, {os.environ['QEV_SOURCE']!r}); import uvicorn; "
        "from qev.engine import DecisionEngine; from qev.server import create_app; "
        f"engine = DecisionEngine({str(model)!r}, device='mps', dtype='bfloat16', head_dtype='float32'); "
        f"uvicorn.run(create_app(engine, {model.name!r}), host='127.0.0.1', port={port})"
    )
    return [python, "-I", "-c", run]


def jeff_server(python: str, model: Path, port: int) -> list[str]:
    # Jeff's own server (jeff.serve, v1.3.0 installed from JEFF_SOURCE), whose JeffModel defaults to CUDA and whose
    # command line has no device: the model loaded on "mps" instead, everything else as the server runs it
    run = (
        "import functools, sys; import jeff.model as m, jeff.serve as s; "
        "s.JeffModel = functools.partial(m.JeffModel, device='mps'); "
        f"sys.argv = ['jeff-serve', {str(model)!r}, '--host', '127.0.0.1', '--port', '{port}']; s.main()"
    )
    return [python, "-I", "-c", run]


def glide_server(python: str, model: Path, port: int) -> list[str]:
    # the package imported from the checkpoint, where its card runs `python -m glide.server`; its device comes from
    # GLIDE_DEVICE (default cuda), which the kit sets to mps, and its dtype is its default, bf16
    run = (
        f"import runpy, sys; sys.path.insert(0, {str(model)!r}); runpy.run_module('glide.server', run_name='__main__')"
    )
    return [python, "-I", "-c", run, "--model", str(model), "--host", "127.0.0.1", "--port", str(port)]


def kev_server(python: str, model: Path, port: int) -> list[str]:
    # Kev's own server on the Mac, as its README starts it, its serving defaults left as they are
    return [python, "-I", "-m", "kev.serve", "--run", str(model), "--device", "mps",
            "--host", "127.0.0.1", "--port", str(port)]  # fmt: skip


def strands_server(python: str, model: Path, port: int) -> list[str]:
    # AWS Strands' own server (strands-decider serve) on MPS in its default bf16, with prefix cache and warmup on
    run = (
        "import sys; from strands_decider.cli import main; "
        f"sys.argv = ['strands-decider', 'serve', {str(model)!r}, '--device', 'mps', "
        f"'--host', '127.0.0.1', '--port', '{port}']; main()"
    )
    return [python, "-I", "-c", run]


# a family's server command, the sources it runs (hashed into the result), /health's status once it is warm (None: it is
# ready once /v1/models answers), and how
# it runs a request's questions
FAMILIES = {
    "d3": (d3_server, ("d3_server.py", "d3_runtime.py"), "ready", "in batches of up to eight questions"),
    "torchcast": (
        torchcast_server,
        ("torchcast_decision/server.py", "torchcast_decision/model.py", "torchcast_decision/render.py"),
        "ok",
        "every question in one padded batch",
    ),
    "qev": (
        qev_server,
        ("qev/server.py", "qev/engine.py", "qev/readout.py", "qev/prompt.py"),
        "ok",
        "as forks of the block",
    ),
    "jeff": (
        jeff_server,
        ("src/jeff/serve.py", "src/jeff/model.py", "src/jeff/batching.py", "src/jeff/prompting.py"),
        "ok",
        "batched with other requests' questions (jeff.batching)",
    ),
    "glide": (
        glide_server,
        ("glide/server.py", "glide/model.py", "glide/prompt.py"),
        "ok",
        "batched with other requests' questions (its MicroBatcher, under a 16,384-token budget)",
    ),
    "kev": (
        kev_server,
        ("kev/serve.py", "kev/model.py", "kev/mlx_model.py", "kev/api.py", "kev/checkpoint.py"),
        None,
        "the state once and every question a row after it, batched with other requests (its MLX backend)",
    ),
    "strands": (
        strands_server,
        (
            "src/strands_decider/server.py",
            "src/strands_decider/decision.py",
            "src/strands_decider/backends/transformers.py",
            "src/strands_decider/core/prompt.py",
        ),
        "ok",
        "the state once and its questions in chunks of up to 32 (strands-decider on MPS)",
    ),
}
SOURCES = {
    "qev": lambda model: Path(os.environ["QEV_SOURCE"]),
    "jeff": lambda model: Path(os.environ["JEFF_SOURCE"]),
    "kev": lambda model: Path(os.environ["KEV_SOURCE"]),
    "strands": lambda model: Path(os.environ["STRANDS_SOURCE"]),
}  # where a family's sources are, if not the model
IMAGES = ("d3", "torchcast")  # servers that take a request's pictures as "images" (data URLs), not the kit's "files"
VIDEOS = ("d3",)  # and its videos as "videos" (d3_server.py's MEDIA_FIELDS)


def main() -> None:
    parser = common.add_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--python", required=True, help="the creator runtime's Python interpreter")
    parser.add_argument("--model", required=True, help="a downloaded checkpoint with its creator's server")
    parser.add_argument("--log", help="server log file (default: beside --out)")
    args = parser.parse_args()
    python = str(Path(args.python).expanduser())
    model = Path(args.model).expanduser().resolve()
    server, sources, ready, method = FAMILIES[args.family]
    root = SOURCES.get(args.family, lambda folder: folder)(model)
    for name in sources:
        if not (root / name).is_file():
            parser.error(f"the selected package does not contain {name}")
    version_code = (
        "import json; from importlib.metadata import distributions; "
        "wanted = {'torch', 'transformers', 'safetensors', 'tokenizers', 'accelerate', 'mlx', 'mlx-lm', 'peft'}; "
        "print(json.dumps({d.metadata['Name']: d.version for d in distributions() "
        "if d.metadata['Name'].lower().replace('_', '-') in wanted}))"
    )
    versions = json.loads(subprocess.check_output([python, "-I", "-c", version_code], text=True, timeout=30))
    normalized = {name.lower().replace("_", "-"): version for name, version in versions.items()}
    port = free_port()
    command = server(python, model, port)
    env = {key: value for key, value in os.environ.items() if key != "DECISION_API_KEY"}
    base = f"http://127.0.0.1:{port}"
    log_path = common.server_log(args)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 900
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"{args.family} server exited ({proc.returncode}); see {log_path}")
            health = None
            with suppress(OSError, ValueError):
                health = get(f"{base}/health") if ready is not None else get(f"{base}/v1/models")
            # a model list: OpenAI's shape ("data") or TypeSafe's ("models", Kev's server)
            listed = health and (health.get("data") or health.get("models"))
            if health and (health.get("status") == ready if ready is not None else listed):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{args.family} server did not load and warm after 900 s; see {log_path}")
            time.sleep(0.5)

        def decide(req):
            body = req["systemone"]
            if args.family in IMAGES and "files" in body:
                files = body["files"]
                videos = [url for url in files if url.startswith("data:video/")] if args.family in VIDEOS else []
                body = {k: v for k, v in body.items() if k != "files"}
                body |= {"images": [url for url in files if url not in videos]} if len(videos) < len(files) else {}
                body |= {"videos": videos} if videos else {}
            output = post(f"{base}/v1/systemone", body)
            return common.systemone_answers(output), (output.get("usage") or {}).get("input_tokens"), None

        def memory():
            footprint = common.footprint(proc.pid)
            return {"server_phys_footprint": footprint["phys_footprint"]} if footprint else None

        meta = {
            "runtime": f"kev with mlx-lm {normalized['mlx-lm']} (mps)"
            if args.family == "kev"
            else f"transformers {normalized['transformers']} + torch {normalized['torch']} (mps)",
            "versions": versions,
            "family": args.family,
            "device": "mps",
            "server_cmd": [Path(python).name, *command[1:]],
            "endpoint": "/v1/systemone",
            "server_source": sources[0],
            "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
            "model_dir": str(model),
            "revision": common.repo_revision(str(model)),
            "health": health,
            "question_method": f"one native /v1/systemone request containing all questions, run {method}",
        }
        common.run(args, decide, meta, memory=memory, server_pid=proc.pid)
    finally:
        if proc.poll() is None:
            with suppress(ProcessLookupError):
                # The native server closes its listener on KeyboardInterrupt; it has no SIGTERM handler.
                proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)


if __name__ == "__main__":
    main()
