"""Time ippon against the other ways to run the same decision models on this Mac, on identical requests, one
runtime per process (benchmarks/README.md has the setup and the method).

    python3 -I run.py list                                      # every engine, and whether its files are here
    python3 -I run.py run d1-2B-llama-Q8_0 d1-2B-gguf-q8-ippon-serve --n 10 --sizes 100,300,1000,2000,4000 \
        --questions 1,3 --tag m5pro/r1/
    python3 -I summarize.py ../results/m5pro/r1          # results -> markdown

Every engine runs in its own subprocess with its own environment (runtimes never share a process, and its memory is
freed when it exits), loads its model, warms up on --warmup requests of every (size, questions) shape, then times
--n requests of each shape on the client side. Requests come from gen_requests.py: the same contexts and questions
for every engine, each state with a unique ID, so no prompt cache hits across requests.

Like is compared with like: ippon behind `ippon serve` (HTTP) against the servers (llama.cpp's
llama-server, oMLX, Ollama), and ippon in-process against the runtimes that run in-process (the creators' and the
community's MLX code, the creators' PyTorch code on MPS). With --http, use each available native server
and a documented HTTP wrapper for libraries without one. Paths: --models (the model folders, default ~/models, with
Ollama's store in its ollama/), --envs (one Python environment per runtime, default ../envs, made by setup.sh),
--llama-server (default ../bin/llama-b11960/llama-server), --ollama (default ../bin/ollama-v0.40.13/ollama).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ADAPTERS = HERE / "adapters"

# llama-server's flags: every layer on the GPU; -c holds the longest prompt; for d1 and Clef-Flash the whole prompt
# in one batch (-b/-ub: llama.cpp reads a prompt both ways, as d1's set attention does, only within one batch);
# 16 slots on one KV cache (-np 16 -kvu), so a request's questions run in parallel and share their prompt; and its
# prompt cache, which b11960 leaves off for a decision model unless --cache-ram is set (its log says so at start), so a
# context sent again can be answered from what it kept, as on the other runtimes. Everything else is llama.cpp's
# default (flash attention). H2O-Lightning-4B's GGUF reads one way, so it keeps llama.cpp's own batch sizes: with the
# whole window as one batch, llama.cpp ran out of Metal memory on every request on 16 GB Macs (M2 Pro,
# M5). Its card's command has one slot (-np 1 --no-kv-unified), which answers a request's questions one after
# another: 27.8 s for 50 questions against 4.2 s with 16 slots (M5 Pro), so the kit gives it the 16 slots.
SLOTS = ("-np", "16", "-kvu", "--cache-ram", "8192")
LLAMA_FLAGS = {
    "d1-2B": ("-ngl", "all", "-c", "32768", "-b", "32768", "-ub", "32768", *SLOTS),
    "d1-omni-2B": ("-ngl", "all", "-c", "32768", "-b", "32768", "-ub", "32768", *SLOTS),
    "clef-flash": ("-ngl", "all", "-c", "32768", "-b", "32768", "-ub", "32768", *SLOTS),
    "h2o-4b": ("-ngl", "all", "-c", "40960", *SLOTS),
}
# On a Mac with 8 GB or less, llama.cpp's buffers for a 32,768-token batch don't fit (Metal out of memory on an M1
# Air): there every GGUF build's window is SMALL_WINDOW, which holds the longest generated request
# (about 6,100 tokens), for llama.cpp and for ippon on the same file alike, and llama.cpp's prompt cache is 1 GB.
SMALL_MAC, SMALL_WINDOW, SMALL_CACHE_MB = 8 << 30, 8192, 1024
# llama.cpp's KV cache there: its 16 slots of the 50-question workload's ~1,100-token prompts at once, 17,600 tokens
# (with -c 8192, 164 of 200 of its top answers agreed with ippon's, which agreed with Ollama's 250 of 250: a shared
# cache too small for its slots is the likely cause, which the M1's next round checks)
SMALL_CONTEXT = 24576


def small_mac() -> bool:
    memory = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False).stdout
    return memory.strip().isdigit() and int(memory) <= SMALL_MAC


def llama_flags(name: str, small: bool) -> list[str]:
    flags = list(LLAMA_FLAGS[name])
    if small:
        for option, value in (
            ("-c", SMALL_CONTEXT),
            ("-b", SMALL_WINDOW),
            ("-ub", SMALL_WINDOW),
            ("--cache-ram", SMALL_CACHE_MB),
        ):
            if option in flags:  # H2O-Lightning-4B keeps llama.cpp's own batch sizes: it has no -b or -ub
                flags[flags.index(option) + 1] = str(value)
    return flags


GGUF = {  # the creators' Q8_0 GGUF builds: name, file under --models, longest prompt
    "d1-2B": ("d1-2B-GGUF/d1-2B-Q8_0.gguf", 32768),
    "d1-omni-2B": ("d1-Omni-2B-GGUF/d1-Omni-2B-Q8_0.gguf", 32768),
    "clef-flash": ("Clef-Flash-GGUF/Clef-Flash-Q8_0.gguf", 16384),  # llama.cpp's -c is 32768: Clef's own window
    "h2o-4b": ("h2o-lightning-4b-gguf/h2o-lightning-4b-Q8_0.gguf", 40944),  # H2O's Mac path: one way, its context
}
# each GGUF build's picture and audio parts, loaded as llama.cpp users serve a multimodal model (--mmproj); ippon finds
# the same file beside the model's
MMPROJ = {
    "d1-2B": "d1-2B-GGUF/mmproj-d1-2B-Q8_0.gguf",
    "d1-omni-2B": "d1-Omni-2B-GGUF/mmproj-d1-Omni-2B-Q8_0.gguf",
    "h2o-4b": "h2o-lightning-4b-gguf/mmproj-lightning-f16.gguf",
}
IPPON = {  # ippon in-process: name, folder under --models, longest prompt, a file it needs
    "d1-2B": ("d1-2B", 32768, "model.safetensors"),
    "d1-omni-2B": ("d1-Omni-2B", 32768, "model.safetensors"),
    "clef-flash": ("clef-flash", 16384, "joint_head.safetensors"),
    "pplx-flash": ("pplx-decider-flash-v1.3", 32768, "readout.safetensors"),
    "clef-flash-mlx-8bit": ("clef-flash-mlx-8bit", 16384, "joint_head.safetensors"),
    "d3-mini": ("d3-mini", 32768, "decision_config.json"),
    "torchcast-4b": ("torchcast-decision-4b", 16384, "decision_config.json"),
    "d3-flash": ("d3-flash", 32768, "decision_config.json"),
    "elephant-1": ("elephant-1", 32768, "config.json"),  # qev's longest branch (--max-branch-tokens)
    "kettei-core": ("kettei-core", 65536, "config.json"),  # the SDK's request limit
    "kettei-dot": ("kettei-dot", 65536, "config.json"),
    "jeff-medium": ("jeff-1.1-medium", 8192, "head.safetensors"),  # the creator's input limit
    "jeff-qwen-4b": ("jeff-1.1-qwen-4b", 8192, "head.safetensors"),
    # Kev's serving context (model.SERVE_MAX_STATE); its LoRA loads with the base head.pt pins, from the Hub cache
    **{f"kev-{size}": (f"kev-{size}", 65536, "head.pt") for size in ("4b", "9b")},
    **{f"glide-{size}": (f"glide-{size}", 32768, "readout.safetensors") for size in ("2b", "9b")},
    "strands-e2b": ("strands-e2b", 12288, "readout.safetensors"),
    "strands-e4b": ("strands-e4b", 12288, "readout.safetensors"),
    "strands-12b": ("strands-12b", 12288, "readout.safetensors"),
}
# 8-bit at load (`--precision q8`) of a checkpoint in IPPON, for Macs its full precision does not fit: Kettei Core's
# 10.1 GB at bf16 against an 8 GB Mac's GPU working set of about 5 GiB
EIGHT_BIT = {"kettei-core-q8": "kettei-core"}
OLLAMA = {  # Ollama's own MLX builds (setup.sh ollama): name, name:tag in its store, the package's context (its
    # Modelfile's num_ctx: Ollama refuses a longer prompt rather than cut it), which ippon gets too
    "clef-flash-mxfp8": ("clef-flash:9b-mxfp8", 16384),
    "d3-mini-mxfp8": ("d3:4b-mxfp8", 4096),  # Ollama's d3:4b is vllm-sr/d3-mini
    "d3-flash-mxfp8": ("d3:9b-mxfp8", 4096),  # Ollama's d3:9b is vllm-sr/d3-flash
    "d1-2B-bf16": ("d1:2b-mlx-bf16", 32768),
    "d1-omni-2B-bf16": ("d1-omni:2b-mlx-bf16", 32768),
    "d1-2B-mxfp8": ("d1:2b-mxfp8", 32768),  # Ollama's d1:2b is LiquidAI/d1-2B
    "d1-omni-2B-mxfp8": ("d1-omni:2b-mxfp8", 32768),  # Ollama's d1-omni:2b is LiquidAI/d1-Omni-2B
}
MPS = {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "1.0", "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.8"}


def engines(models: Path, envs: Path, llama_server: Path, ollama: Path) -> dict[str, dict]:
    def py(env: str) -> list[str]:
        return [str(envs / env / "bin" / "python"), "-I"]

    out: dict[str, dict] = {}
    small = small_mac()
    for name, (file, window) in GGUF.items():
        window = min(window, SMALL_WINDOW) if small else window
        out[f"{name}-llama-Q8_0"] = {
            "cmd": [
                *py("harness"),
                str(ADAPTERS / "llama_systemone.py"),
                "--server",
                str(llama_server),
                "--gguf",
                str(models / file),
                "--max-input",
                str(window),
            ],
            "server": [
                *llama_flags(name, small),
                *(("--mmproj", str(models / MMPROJ[name])) if name in MMPROJ else ()),
            ],
            "needs": [models / file, llama_server, *((models / MMPROJ[name],) if name in MMPROJ else ())],
        }
        out[f"{name}-gguf-q8-ippon-serve"] = {
            "cmd": [
                *py("harness"),
                str(ADAPTERS / "ippon_serve.py"),
                "--ippon",
                str(envs / "ippon" / "bin" / "ippon"),
                "--model",
                str(models / file),
                "--max-input",
                str(window),
            ],
            "needs": [models / file],
        }
    for name, (folder, window, need) in IPPON.items():
        out[f"{name}-ippon"] = {
            "cmd": [
                *py("ippon"),
                str(ADAPTERS / "ippon_engine.py"),
                "--model",
                str(models / folder),
                "--max-input",
                str(window),
            ],
            "needs": [models / folder / need],
        }
    # the community Clef-Flash build through its loader
    out["clef-flash-mlx-8bit"] = {
        "cmd": [
            *py("mlxvlm"),
            str(ADAPTERS / "clef_mlx_community.py"),
            "--repo",
            "mlx-community/clef-flash-8bit",
            "--model",
            str(models / "clef-flash-mlx-8bit"),
            "--max-input",
            "16384",
        ],
        "needs": [models / "clef-flash-mlx-8bit" / "clef_mlx.py"],
    }
    # the creators' own PyTorch code on MPS, as their cards document it (d1-2B's: bf16 on MPS, no device map)
    for name, repo, folder, window, env, extra in (
        ("d1-2B", "LiquidAI/d1-2B", "d1-2B", 32768, "d1", ("--dtype", "bfloat16", "--device-map", "none")),
        (
            "d1-omni-2B",
            "LiquidAI/d1-Omni-2B",
            "d1-Omni-2B",
            32768,
            "d1",
            ("--dtype", "bfloat16", "--device-map", "none"),
        ),
    ):
        out[f"{name}-tf"] = {
            "cmd": [
                *py(env),
                str(ADAPTERS / "tf_systemone.py"),
                "--family",
                env,  # the environment is the family: Liquid's LFM3 code (d1)
                "--repo",
                repo,
                "--model",
                str(models / folder),
                "--max-input",
                str(window),
                *extra,
            ],
            "needs": [models / folder / "config.json", models / folder / "model.safetensors"],
            "env": MPS,
        }
    for name, weights in (("d3-mini", "model-00002-of-00002.safetensors"), ("d3-flash", "readout.safetensors")):
        out[f"{name}-tf"] = {  # vllm-sr's own runtime (the checkpoint's code, transformers on MPS)
            "cmd": [
                *py("vllmsr"),
                str(ADAPTERS / "tf_systemone.py"),
                "--family",
                "d3",
                "--repo",
                f"vllm-sr/{name}",
                "--model",
                str(models / name),
                "--max-input",
                "32768",
            ],
            "needs": [models / name / "config.json", models / name / weights],
            "env": MPS,
        }
    out["clef-flash-tf"] = {
        "cmd": [
            *py("clef"),
            str(ADAPTERS / "clef.py"),
            "--repo",
            "Cloudflare/clef-flash",
            "--model",
            str(models / "clef-flash"),
            "--max-input",
            "16384",
        ],
        "needs": [models / "clef-flash" / "joint_schema_model.py"],
        "env": MPS,
    }
    # other runtimes of the same models (setup.sh rivals), on their own builds of the checkpoint where they have one
    for name, window in (("clef-flash-mlx-8bit", 16384), ("d3-mini", 32768), ("d3-flash", 32768)):
        out[f"{name}-omlx"] = {
            "cmd": [*py("harness"), str(ADAPTERS / "omlx_serve.py"), "--omlx", str(envs / "omlx" / "bin" / "omlx"),
                    "--model-dir", str(models / "omlx"), "--model", name, "--max-input", str(window)],
            "needs": [models / "omlx" / name, envs / "omlx" / "bin" / "omlx"],
        }  # fmt: skip
        out[f"{name}-ippon-serve"] = {  # oMLX is a server: ippon behind `ippon serve` beside it
            "cmd": [*py("harness"), str(ADAPTERS / "ippon_serve.py"), "--ippon",
                    str(envs / "ippon" / "bin" / "ippon"), "--model", str(models / name),
                    "--max-input", str(window)],
            "needs": [models / name],
        }  # fmt: skip
    store = models / "ollama"  # Ollama's model store (OLLAMA_MODELS), which `ollama://` reads too

    def manifest(tag: str) -> Path:
        name, version = tag.split(":")
        return store / "manifests" / "registry.ollama.ai" / "library" / name / version

    for name, (tag, window) in OLLAMA.items():
        out[f"{name}-ollama"] = {
            "cmd": [*py("harness"), str(ADAPTERS / "ollama_score.py"), "--ollama", str(ollama), "--models-dir",
                    str(store), "--model", tag, "--max-input", str(window)],
            "needs": [manifest(tag), ollama],
        }  # fmt: skip
        out[f"{name}-ollama-ippon-serve"] = {  # Ollama is a server: `ippon serve` on the same build's blobs
            "cmd": [*py("harness"), str(ADAPTERS / "ippon_serve.py"), "--ippon",
                    str(envs / "ippon" / "bin" / "ippon"), "--model", f"ollama://{tag}",
                    "--max-input", str(window)],
            "needs": [manifest(tag)],
            "env": {"OLLAMA_MODELS": str(store)},
        }  # fmt: skip
    out["jeff-medium-maker-mps"] = {  # Jeff's own server (v1.3.0, jeff.serve) on MPS, bf16, batching on
        "cmd": [*py("harness"), str(ADAPTERS / "maker_serve.py"), "--family", "jeff", "--python",
                str(envs / "jeff/bin/python"), "--model", str(models / "jeff-1.1-medium"), "--max-input", "8192"],
        "needs": [models / "jeff-1.1-medium" / "head.safetensors",
                  envs.parent / "rivals/jeff-v1.3.0/src/jeff/serve.py"],
        "env": {**MPS, "JEFF_SOURCE": str(envs.parent / "rivals/jeff-v1.3.0")},
    }  # fmt: skip
    out["torchcast-4b-maker-mps"] = {  # Torchcast's own server (v1.3.1) on MPS: its only path that runs this model
        "cmd": [*py("harness"), str(ADAPTERS / "maker_serve.py"), "--family", "torchcast", "--python",
                str(envs / "torchcast/bin/python"), "--model", str(models / "torchcast-decision-4b"),
                "--max-input", "16384"],
        "needs": [models / "torchcast-decision-4b" / "torchcast_decision" / "server.py"],
        "env": MPS,
    }  # fmt: skip
    out["elephant-1-maker-mps"] = {  # Base Labs' runtime (OneJev's qev with the checkpoint's serve.patch) on MPS
        "cmd": [*py("harness"), str(ADAPTERS / "maker_serve.py"), "--family", "qev", "--python",
                str(envs / "elephant/bin/python"), "--model", str(models / "elephant-1"), "--max-input", "32768"],
        "needs": [models / "elephant-1" / "config.json", envs.parent / "makers/onejev/qev/readout.py"],
        "env": {**MPS, "QEV_SOURCE": str(envs.parent / "makers/onejev")},
    }  # fmt: skip
    out["jeff-qwen-4b-maker-mps"] = {  # Jeff's own server (v1.4.0, jeff.serve) on MPS, bf16, batching on
        "cmd": [*py("harness"), str(ADAPTERS / "maker_serve.py"), "--family", "jeff", "--python",
                str(envs / "jeff/bin/python"), "--model", str(models / "jeff-1.1-qwen-4b"), "--max-input", "8192"],
        "needs": [models / "jeff-1.1-qwen-4b" / "head.safetensors",
                  envs.parent / "rivals/jeff-v1.4.0/src/jeff/serve.py"],
        "env": {**MPS, "JEFF_SOURCE": str(envs.parent / "rivals/jeff-v1.4.0")},
    }  # fmt: skip
    for size in ("4b", "9b"):  # Kev's own server (kev.serve) on the Mac: its MLX backend, bf16, LoRA merged
        out[f"kev-{size}-maker-mps"] = {
            "cmd": [*py("harness"), str(ADAPTERS / "maker_serve.py"), "--family", "kev", "--python",
                    str(envs / "kev/bin/python"), "--model", str(models / f"kev-{size}"), "--max-input", "65536"],
            "needs": [models / f"kev-{size}" / "head.pt", envs.parent / "rivals/kev/kev/serve.py"],
            "env": {**MPS, "KEV_SOURCE": str(envs.parent / "rivals/kev")},
        }  # fmt: skip
    for size in ("2b", "9b"):  # Fastino's own server (glide 1.2.0) on MPS, its default bf16
        out[f"glide-{size}-maker-mps"] = {
            "cmd": [*py("harness"), str(ADAPTERS / "maker_serve.py"), "--family", "glide", "--python",
                    str(envs / "glide/bin/python"), "--model", str(models / f"glide-{size}"), "--max-input", "32768"],
            "needs": [models / f"glide-{size}" / "glide" / "server.py", envs / "glide/bin/python"],
            "env": {**MPS, "GLIDE_DEVICE": "mps"},
        }  # fmt: skip
    for size in ("e2b", "e4b", "12b"):  # AWS Strands' own server (strands-decider) on MPS, its default bf16
        out[f"strands-{size}-maker-mps"] = {
            "cmd": [*py("harness"), str(ADAPTERS / "maker_serve.py"), "--family", "strands", "--python",
                    str(envs / "strands/bin/python"), "--model", str(models / f"strands-{size}"),
                    "--max-input", "12288"],
            "needs": [models / f"strands-{size}" / "readout.safetensors",
                      envs.parent / "rivals/strands-decider/src/strands_decider/server.py"],
            "env": {**MPS, "STRANDS_SOURCE": str(envs.parent / "rivals/strands-decider")},
        }  # fmt: skip
    for suffix, dtype in (("", "fp32"), ("-bf16", "bf16")):
        out[f"pplx-flash-maker-mps{suffix}"] = {
            "cmd": [
                *py("pplx"),
                str(ADAPTERS / "pplx_decider.py"),
                "--model",
                str(models / "pplx-decider-flash-v1.3"),
                "--max-input",
                "32768",
                "--dtype",
                dtype,
            ],
            "needs": [models / "pplx-decider-flash-v1.3" / "readout.safetensors"],
        }
    for name, (folder, window, need) in IPPON.items():
        out.setdefault(
            f"{name}-ippon-serve",
            {
                "cmd": [
                    *py("harness"),
                    str(ADAPTERS / "ippon_serve.py"),
                    "--ippon",
                    str(envs / "ippon" / "bin" / "ippon"),
                    "--model",
                    str(models / folder),
                    "--max-input",
                    str(window),
                ],
                "needs": [models / folder / need],
            },
        )
    for name, base in EIGHT_BIT.items():  # a creator's checkpoint that ippon quantizes to 8-bit as it loads
        for arm in (f"{name}-ippon", f"{name}-ippon-serve"):
            out[arm] = {**out[arm.replace(name, base, 1)]}
            out[arm]["cmd"] = [*out[arm]["cmd"], "--ippon-precision", "q8"]
    for name in [*IPPON, *EIGHT_BIT]:
        out[f"{name}-ippon"]["http_cmd"] = out[f"{name}-ippon-serve"]["cmd"]
    out["clef-flash-mlx-8bit"]["http_cmd"] = [
        *py("harness"),
        str(ADAPTERS / "mlx_serve.py"),
        "--python",
        str(envs / "mlxvlm/bin/python"),
        "--model",
        str(models / "clef-flash-mlx-8bit"),
        "--max-input",
        "16384",
    ]
    for name in ("d3-mini", "d3-flash"):
        out[f"{name}-tf"]["http_cmd"] = [
            *py("harness"),
            str(ADAPTERS / "maker_serve.py"),
            "--family",
            "d3",
            "--python",
            str(envs / "vllmsr/bin/python"),
            "--model",
            str(models / name),
            "--max-input",
            "32768",
        ]
        out[f"{name}-tf"]["needs"].append(models / name / "d3_server.py")
    return out


def power() -> tuple[str | None, bool]:
    """Return the macOS execution-mode readings used by the benchmark gate."""
    pm = subprocess.run(["pmset", "-g"], capture_output=True, text=True, check=False).stdout
    mode = next((ln.split()[-1] for ln in pm.splitlines() if ln.strip().startswith("powermode")), None)
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, check=False).stdout
    return mode, "AC Power" in batt


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--models", type=Path, default=Path.home() / "models")
    p.add_argument("--envs", type=Path, default=HERE.parent / "envs")
    p.add_argument("--llama-server", type=Path, default=HERE.parent / "bin" / "llama-b11960" / "llama-server")
    p.add_argument("--ollama", type=Path, default=HERE.parent / "bin" / "ollama-v0.40.13" / "ollama")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("engines", nargs="+")
    r.add_argument("--n", type=int, default=10)
    r.add_argument("--warmup", type=int, default=2)
    r.add_argument("--sizes", default="100,300,1000,2000,4000")
    r.add_argument("--questions", default="1,3")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument(
        "--scenario",
        default="fresh",
        help="fresh, a re-sent state (repeat, grow), or a question per line (lines); gen_requests.py",
    )
    r.add_argument("--http", action="store_true", help="wrap runtimes without a server in the benchmark HTTP adapter")
    r.add_argument("--tag", default="", help="prefix of the result names, e.g. m5pro/r1/")
    r.add_argument("--cooldown", type=float, default=30.0, help="idle seconds between engines")
    r.add_argument("--gap", type=float, default=0.0, help="idle seconds before each timed request, as an agent leaves")
    args = p.parse_args()
    table = engines(
        args.models.expanduser(), args.envs.expanduser(), args.llama_server.expanduser(), args.ollama.expanduser()
    )

    def missing(name: str) -> list[str]:
        return [str(n) for n in table[name]["needs"] if not Path(n).exists()]

    if args.command == "list":
        for name in table:
            gone = missing(name)
            print(f"{name:40s} {'ready' if not gone else 'missing ' + ', '.join(gone)}")
        return
    unknown = [n for n in args.engines if n not in table]
    if unknown:
        sys.exit(f"unknown engines {unknown}; see `list`")
    results = HERE.parent / "results"
    results.mkdir(exist_ok=True)
    for i, name in enumerate(args.engines):
        if missing(name):
            print(f"== {name}: skipped, missing {missing(name)}", flush=True)
            continue
        if i and args.cooldown:
            time.sleep(args.cooldown)
        mode, ac = power()
        if mode != "2" or not ac:
            sys.exit("plug in and set Energy Mode to High Power: timings outside it are not comparable")
        spec = table[name]
        cmd = [
            *(spec.get("http_cmd", spec["cmd"]) if args.http else spec["cmd"]),
            "--engine",
            f"{args.tag}{name}",
            "--n",
            str(args.n),
            "--warmup",
            str(args.warmup),
            "--sizes",
            args.sizes,
            "--questions",
            args.questions,
            "--seed",
            str(args.seed),
            "--scenario",
            args.scenario,
            *(["--gap", str(args.gap)] if args.gap else []),
        ]
        if args.http:
            cmd += ["--http"]
        if "server" in spec:
            cmd += ["--", *spec["server"]]
        started = time.time()
        code = subprocess.run(cmd, cwd=str(results), env={**os.environ, **spec.get("env", {})}, check=False).returncode
        print(f"== {name}: exit {code} after {time.time() - started:.0f} s", flush=True)
        if code:
            sys.exit(code)


if __name__ == "__main__":
    main()
