"""Generate benchmark tables and an evidence ledger from recorded requests, without running any model.

    python3 -I benchmarks/kit/summarize.py benchmarks/results/CAMPAIGN/{r1,r2} --out SUMMARY.md

For each pair, use the newest supplied campaign containing both complete arms. Pool its rounds within each
measured shape; never pool different campaigns. Speed change is (other median / ippon median - 1) * 100 percent.
Short and long results are geometric means of the available cell ratios; missing shapes are never interpolated.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_requests as requests

ROOT = Path(__file__).resolve().parents[2]
# Group, model/build, other engine, ippon engine. Runtime versions come from the selected results.
PAIRS = [
    ("llama.cpp", "Clef-Flash 9B · Q8_0", "clef-flash-llama-Q8_0", "clef-flash-gguf-q8-ippon-serve"),
    ("ollama", "Clef-Flash 9B · Ollama MXFP8", "clef-flash-mxfp8-ollama", "clef-flash-mxfp8-ollama-ippon-serve"),
    ("ollama", "d3-mini · Ollama MXFP8", "d3-mini-mxfp8-ollama", "d3-mini-mxfp8-ollama-ippon-serve"),
    ("omlx", "Clef-Flash 9B · MLX 8-bit", "clef-flash-mlx-8bit-omlx", "clef-flash-mlx-8bit-ippon-serve"),
    ("omlx", "d3-mini · checkpoint", "d3-mini-omlx", "d3-mini-ippon-serve"),
    ("mlx", "Clef-Flash 9B · MLX 8-bit", "clef-flash-mlx-8bit", "clef-flash-mlx-8bit-ippon"),
    ("maker", "Clef-Flash 9B · full checkpoint", "clef-flash-tf", "clef-flash-ippon"),
    ("maker", "pplx-decider flash v1.3 · BF16", "pplx-flash-maker-mps-bf16", "pplx-flash-ippon"),
    ("maker", "d3-mini · full checkpoint", "d3-mini-tf", "d3-mini-ippon"),
    ("llama.cpp", "d1-2B · Q8_0", "d1-2B-llama-Q8_0", "d1-2B-gguf-q8-ippon-serve"),
    ("llama.cpp", "d1-Omni-2B · Q8_0", "d1-omni-2B-llama-Q8_0", "d1-omni-2B-gguf-q8-ippon-serve"),
    ("llama.cpp", "H2O-Lightning-4B · Q8_0", "h2o-4b-llama-Q8_0", "h2o-4b-gguf-q8-ippon-serve"),
    ("ollama", "d1-2B · Ollama BF16", "d1-2B-bf16-ollama", "d1-2B-bf16-ollama-ippon-serve"),
    ("ollama", "d1-Omni-2B · Ollama BF16", "d1-omni-2B-bf16-ollama", "d1-omni-2B-bf16-ollama-ippon-serve"),
    ("ollama", "d1-2B · Ollama MXFP8", "d1-2B-mxfp8-ollama", "d1-2B-mxfp8-ollama-ippon-serve"),
    ("ollama", "d1-Omni-2B · Ollama MXFP8", "d1-omni-2B-mxfp8-ollama", "d1-omni-2B-mxfp8-ollama-ippon-serve"),
    ("ollama", "d3-flash · Ollama MXFP8", "d3-flash-mxfp8-ollama", "d3-flash-mxfp8-ollama-ippon-serve"),
    ("omlx", "d3-flash · checkpoint", "d3-flash-omlx", "d3-flash-ippon-serve"),
    ("maker", "d1-2B · BF16", "d1-2B-tf", "d1-2B-ippon"),
    ("maker", "d1-Omni-2B · BF16", "d1-omni-2B-tf", "d1-omni-2B-ippon"),
    ("maker", "d3-flash · full checkpoint", "d3-flash-tf", "d3-flash-ippon"),
    ("maker", "Torchcast Decision 4B · BF16", "torchcast-4b-maker-mps", "torchcast-4b-ippon"),
    ("maker", "elephant-1 · BF16", "elephant-1-maker-mps", "elephant-1-ippon"),
    ("maker", "Jeff-1.1-Medium · BF16", "jeff-medium-maker-mps", "jeff-medium-ippon"),
    ("maker", "Jeff-1.1-Qwen-4B · BF16", "jeff-qwen-4b-maker-mps", "jeff-qwen-4b-ippon"),
    ("maker", "Kev-4B · BF16", "kev-4b-maker-mps", "kev-4b-ippon"),
    ("maker", "Kev-9B · BF16", "kev-9b-maker-mps", "kev-9b-ippon"),
    ("maker", "GLiDE 9B · BF16", "glide-9b-maker-mps", "glide-9b-ippon"),
    ("maker", "GLiDE 2B · BF16", "glide-2b-maker-mps", "glide-2b-ippon"),
    ("maker", "Strands-Decider-12B · BF16", "strands-12b-maker-mps", "strands-12b-ippon"),
    ("maker", "Strands-Decider-E4B · BF16", "strands-e4b-maker-mps", "strands-e4b-ippon"),
    ("maker", "Strands-Decider-E2B · BF16", "strands-e2b-maker-mps", "strands-e2b-ippon"),
]
# Models timed on ippon alone (no other runtime runs them as published): name, ippon engine.
ALONE = [
    ("Kettei Core", "kettei-core-ippon"),
    ("Kettei Dot", "kettei-dot-ippon"),
]
# and their 8-bit builds for small Macs (run.EIGHT_BIT), timed when a round names them, never pending otherwise
SMALL_MAC_ALONE = [("Kettei Core · 8-bit", "kettei-core-q8-ippon")]
SHORT, LONG = (100, 300), (1000, 2000, 4000)
BLOCKS = {
    "fresh": ("<!-- README round: benchmarks/kit/summarize.py -->", "<!-- /README round -->"),
    "pitch": ("<!-- engine pitch: benchmarks/kit/summarize.py --readme -->", "<!-- /engine pitch -->"),
    "runtime": ("<!-- runtime comparisons: benchmarks/kit/summarize.py -->", "<!-- /runtime comparisons -->"),
    "maker": ("<!-- creators comparisons: benchmarks/kit/summarize.py -->", "<!-- /creators comparisons -->"),
    "headline": ("<!-- speed headline: benchmarks/kit/summarize.py --readme -->", "<!-- /speed headline -->"),
    **{
        f"charts-{key}": (f"<!-- {key} charts: benchmarks/kit/summarize.py --readme -->", f"<!-- /{key} charts -->")
        for key in ("text", "text-images", "text-images-video", "text-images-video-audio")
    },
    "more": ("<!-- more workloads: benchmarks/kit/summarize.py --readme -->", "<!-- /more workloads -->"),
}
# The README's other workloads (readme_round.py --workloads): each one's result-file suffix and what it compares.
MORE = {
    "typical": (".typical", "milliseconds for a typical call: about 500 tokens, 3 questions"),
    "concurrent": (".c8", "requests per second, 8 clients at once"),
    "image": (".image", "milliseconds for 3 questions about an image"),
    "video": (".video", "milliseconds for 3 questions about a 4-second video"),
    "audio": (".audio", "milliseconds for one question with 10 s of audio"),
    "repeat": (".repeat", "milliseconds for a new question about a state already sent"),
    "grow": (".grow", "milliseconds for a state sent again with one more message"),
    "burst": (".c16s", "requests per second, 16 clients sending short requests at once"),
    "pictures": (".pictures", "milliseconds for one question as the images grow from 1 to 8, the median request"),
    "image-again": (".image-again", "milliseconds for a new question about an image already sent"),
    "video-again": (".video-again", "milliseconds for a new question about a video already sent"),
}
CLIENTS = ("concurrent", "burst")  # the workloads of several clients at once: requests per second, and the waits
# The README's campaigns, oldest first. --recorded reads each one's accepted rounds; a campaign not recorded yet, or
# still running (its round an attempt-rN folder), is left out, and each pair comes from its newest complete campaign.
README_CAMPAIGNS = (
    "m5pro-2026-10-28-readme",
    "m5pro-2026-10-29-readme-targeted",
    "m5pro-2026-10-29-readme-agents",
    "m5pro-2026-10-30-new",
    "m5pro-2026-10-30-practical",
    "m5pro-2026-11-01-ollama-parallel",  # Ollama with OLLAMA_NUM_PARALLEL from the plan: its 8 and 16 clients again
    "m5pro-2026-11-03-h2o-llama",  # H2O-Lightning-4B on llama.cpp with its own batch sizes (they fit 16 GB Macs)
    "m5pro-2026-11-03-media-llama",  # pictures and audio on llama.cpp's files and mmproj, as ippon reads them
    "m5pro-2026-11-03-media-ollama",  # pictures on Ollama's own MLX builds, never timed before
    "m5pro-2026-11-04-jeff",  # Jeff-1.1-Medium beside its creators' own server, 1 and 50 questions
    "m5pro-2026-11-04-jeff-agents",  # the same pair on the agent workloads
    "m5pro-2026-11-07-glide",  # Fastino's GLiDE 2B and 9B beside its creator's own server
    "m5pro-2026-11-09-media-omlx",  # pictures on oMLX's d3 checkpoints, sent as its decision API reads them
    "m5pro-2026-11-09-image-3q",  # every picture pair again, 3 questions about each picture
    "m5pro-2026-11-10-creators",  # Jeff-1.1-Qwen-4B, Kev-4B and Kev-9B beside their creators' own servers
    "m5pro-2026-11-10-kev-4b-grow",  # Kev-4B's grow pair again: ippon's arm overlapped a busy process the first time
    "m5pro-2026-11-17-pplx-v1.3",  # pplx-decider flash v1.3 beside Perplexity's package: typical, repeat, grow, image
    "m5pro-2026-11-17-pplx-v1.3-grow",  # its grow pair again, on an idle Mac (the first round's grow had not settled)
    "m5pro-2026-11-18-pplx-v1.3-text",  # its 1,000-token text pair, 1 and 50 questions
    "m5pro-2026-11-17-pictures",  # a growing set of pictures, H2O's GGUF beside llama.cpp on the same file
    "m5pro-2026-11-18-pictures",  # the same on d1 and d3, beside Ollama's MLX builds and llama.cpp's Q8_0 files
    "m5pro-2026-11-18-pictures-omlx",  # the same on oMLX's d3 checkpoints
    "m5pro-2026-11-18-video",  # a 4-second video, 3 questions, on d1, d1-Omni and d3 beside their creators' code
    "m5pro-2026-11-19-video-rivals",  # the same video on oMLX (d3) and llama.cpp (d1), the other runtimes that read one
    "m5pro-2026-11-19-audio-ollama",  # a 10-second clip on Ollama (d1-Omni BF16), which hears one sent under images
    "m5pro-2026-11-19-image-again",  # a new question about an image already sent, on every runtime that reads one
    "m5pro-2026-11-19-image-again-more",  # the same on Torchcast-4B's own code and Ollama's two MXFP8 builds
    "m5pro-2026-11-19-video-again",  # a new question about a video already sent, on oMLX, llama.cpp and creators' code
    "m5pro-2026-11-24-strands",  # Strands-Decider-E2B beside AWS Strands' server on MPS
    "m5pro-2026-11-24-strands-e4b",  # Strands-Decider-E4B beside AWS Strands' server on MPS
    "m5pro-2026-11-24-strands-12b",  # Strands-Decider-12B beside AWS Strands' server on MPS
)
HEADLINE_MODELS = 4  # the speed headline names the first four of README_ORDER that were timed
README_CHARTS = "docs/assets/readme"  # chart.py --readme-round draws <stem>-<model>-light.svg and -dark.svg there
MEMORY_CAMPAIGN = "m5pro-2026-11-19-memory"  # the README's memory chart: ippon at fix 1 beside Ollama and oMLX
# JevBench's headline Capability Score (https://benchmarkheaven.com/jev-models).
# Exact releases absent from that board follow the ranked models; older releases are not substitutes.
README_ORDER = (
    "pplx-decider v1.2 (27B)",  # 80.8
    "Torchcast Decision 4B",  # 77.5, v1.3
    "Jeff-1.1-Medium",  # on the board's ranking (intelligence 74.5), not on its capability list
    "Clef (27B)",  # 75.3
    "elephant-1",  # 70.2
    "d3-mini",  # 68.8
    "Clef-Flash 9B",  # 63.8
    "pplx-decider flash v1.3",
    "d3-flash",
    "d1-2B",
    "d1-Omni-2B",
    "H2O-Lightning-4B",  # the board's v1.1 is not the supported v1.4.1
    "GLiDE 9B",
    "GLiDE 2B",
    "Jeff-1.1-Qwen-4B",
    "Kev-9B",
    "Kev-4B",
    "Strands-Decider-12B",
    "Strands-Decider-E4B",
    "Strands-Decider-E2B",
    "Kettei Core",
    "Kettei Dot",
)
# A model chart's panels, in order: workload, title, what each request carries, and the measure.
PANELS = (
    ("typical", "Typical call", "~500 tokens, 3 questions", "ms"),
    ("one", "1 question", "~1,000 tokens", "ms"),
    ("fifty", "50 questions", "~1,000 tokens", "ms"),
    ("repeat", "Same context again", "~1,000 tokens, a new question", "ms"),
    ("grow", "Context grown", "~1,000 tokens and one ~60-token message", "ms"),
    ("concurrent", "8 clients at once", "~1,000 tokens, 1 question each", "req/s"),
    ("burst", "16 clients at once", "short requests, 1 question each", "req/s"),
    ("image", "An image", "1024\u00d7768, 3 questions", "ms"),
    ("video", "A video", "4 seconds, 1280\u00d7720, 3 questions", "ms"),
    ("audio", "An audio clip", "10 seconds, 1 question", "ms"),
    ("pictures", "Growing images", "1 to 8, one more each request, median", "ms"),
    ("image-again", "Same image again", "1024\u00d7768, a new question", "ms"),
    ("video-again", "Same video again", "4 seconds, 1280\u00d7720, a new question", "ms"),
)
# The README's charts show the workloads agents send (docs/benchmarks.md has every workload): each input's panels.
INPUTS = (
    ("text", ("typical", "repeat", "grow", "burst")),
    ("images", ("image", "pictures", "image-again")),
    ("video", ("video", "video-again")),
    ("audio", ("audio",)),
)
README_PANELS = tuple(key for _, keys in INPUTS for key in keys)
# The README's Speed sections, one for each set of inputs a model is timed on, so every model has one chart in one
# section: its key (the inputs joined) and heading.
SECTIONS = (
    ("text", "Text only"),
    ("text-images", "Both text and images"),
    ("text-images-video", "Text, images and video"),
    ("text-images-video-audio", "Text, images, video and audio"),
)
SAME_BUILD = (  # what an ippon engine's name says about the build both arms ran, in the order the charts show them
    ("gguf-q8", "same Q8_0 file"),
    ("mxfp8-ollama", "same Ollama MXFP8 build"),
    ("bf16-ollama", "same Ollama BF16 build"),
    ("mlx-8bit", "same MLX 8-bit build"),
)
MACHINE_KEYS = ("chip", "memory_bytes", "gpu_cores", "macos", "macos_build")  # what tells two Macs apart


def family(model: str) -> str:
    """The model a PAIRS entry measures, without its build: "Clef-Flash 9B · Q8_0" is Clef-Flash 9B."""
    return model.split(" · ")[0]


def recorded_model(model: str, runs: list[dict]) -> str:
    """A PAIRS entry's model with the version its runs loaded, where the model's folder names one: an earlier
    creators' round ran pplx-decider flash v1.2 under the engine names that v1.3 has now."""
    folders = (Path(run["meta"].get("model_dir") or run["meta"].get("model") or "").name for run in runs)
    versions = {found.group() for folder in folders if (found := re.search(r"v\d+\.\d+", folder))}
    return re.sub(r"v\d+\.\d+", versions.pop(), model) if len(versions) == 1 else model


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def portable(path: Path) -> str:
    """Return a project-relative source path without recording the user's checkout location."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def load(folders: list[Path]) -> dict[str, dict[Path, dict]]:
    out: dict[str, dict[Path, dict]] = {}
    for folder in folders:
        for path in sorted(folder.glob("*.json")):
            result = json.loads(path.read_text())
            if "requests" in result and result.get("complete") and not result.get("errors"):
                result = {**result, "source": portable(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                out.setdefault(result["engine"].rsplit("/", 1)[-1], {})[folder] = result
    return out


def timed(runs: list[dict], size: int, nq: int) -> list[dict]:
    return [
        row
        for run in runs
        for row in run["requests"]
        if row["phase"] == "timed" and row["size"] == size and row["nq"] == nq and not row["problems"]
    ]


def median(runs: list[dict], size: int, nq: int) -> float | None:
    values = [row["ms"] for row in timed(runs, size, nq)]
    return statistics.median(values) if values else None


def gmean(values: list[float]) -> float | None:
    return math.exp(statistics.mean(map(math.log, values))) if values else None


def percent(ratio: float | None, digits: int = 0) -> str:
    """Format signed throughput change, using the same definition for positive and negative values."""
    if ratio is None:
        return "—"
    if ratio > 2:  # past twice: 2.6x, 13x
        return "+" + (f"{ratio:.1f}" if ratio < 10 else f"{ratio:.0f}").removesuffix(".0") + "x"
    value = (ratio - 1) * 100
    if abs(value) < 0.5 * 10**-digits:
        return "0%"
    sign = "+" if value > 0 else "\u2212"
    return f"{sign}{abs(value):,.{digits}f}%"


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:,.1f}"


def _answered(run: dict) -> dict:
    """A run's answered requests by the request itself, set-aside ones too (their timing was busy, not their answers):
    a set-aside request moves its arm on to the next one, so two arms' repetition numbers stop meaning the same
    request."""
    shapes = {(r["size"], r["nq"]) for r in run["requests"]}  # a caller may have kept one shape's requests
    aside = [r for r in run.get("set_aside", []) if (r["size"], r["nq"]) in shapes]
    rows = [r for r in run["requests"] + aside if r["phase"] == "timed" and not r["problems"]]
    return {r.get("request_sha256") or (r["size"], r["nq"], r["rep"]): r for r in rows}


def agreement(a: list[dict], b: list[dict]) -> dict:
    worst, flips, requests, questions = 0.0, 0, 0, 0
    for ra, rb in zip(a, b, strict=True):
        index = _answered(rb)
        for key, row in _answered(ra).items():
            other = index.get(key)
            if other is None:
                continue
            requests += 1
            for question in row["answers"].keys() | other["answers"].keys():
                p, q = row["answers"].get(question, {}), other["answers"].get(question, {})
                worst = max(worst, *(abs(p.get(k, 0) - q.get(k, 0)) for k in p.keys() | q.keys()))
                flips += (max(p, key=p.get) if p else None) != (max(q, key=q.get) if q else None)
                questions += 1
    return {
        "max_probability_difference": worst,
        "top_answer_flips": flips,
        "questions": questions,
        "requests": requests,
    }


def setup_pin(group: str) -> str | None:
    if group != "omlx":
        return None
    match = re.search(r"OMLX_COMMIT=(\w+).*?v([\d.]+)", (Path(__file__).parent / "setup.sh").read_text())
    return f"oMLX {match[2]} (setup pin)" if match else None


def runtime_label(group: str, runs: list[dict]) -> str:
    labels = set()
    for run in runs:
        meta = run["meta"]
        runtime = meta.get("runtime", "runtime not recorded")
        if group == "llama.cpp":
            build = meta.get("build_info", "") or meta.get("llama_server_version", "")
            match = re.search(r"\bb(\d+)|build (\d+)", build)
            runtime = f"llama.cpp b{next(x for x in match.groups() if x)}" if match else "llama.cpp (build unrecorded)"
        elif group == "ollama":
            runtime = f"Ollama {meta.get('ollama_version', 'version unrecorded')}"
        elif group == "omlx" and not re.search(r"\d+\.\d+", runtime):
            runtime = setup_pin(group) or "oMLX (version unrecorded)"
        labels.add(runtime)
    return "; ".join(sorted(labels))


# whose code a creator's arm runs, by its engine's name (run.py), as a chart names it
MAKER_CODE = (
    ("d1-", "Liquid's code"),
    ("d3-", "vllm-sr's server"),
    ("clef-flash", "Cloudflare's code"),
    ("torchcast", "Torchcast's server"),
    ("elephant", "Base Labs' runtime"),
    ("glide", "Fastino's server"),
    ("pplx-flash", "Perplexity's package"),
    ("jeff", "Jonathan Geuter's server"),
    ("kev", "Jared Palmer's server"),
    ("strands", "AWS Strands' server"),
)


def maker_code(engine: str) -> str:
    """The creator's code an arm runs, named for its creator: "Liquid's code", "Torchcast's server"."""
    return next(label for prefix, label in MAKER_CODE if engine.lower().startswith(prefix))


def runtime_display(record: dict) -> str:
    """The other runtime as a chart row names it: its version, and its own conversion where it ran one."""
    runtime = record["runtime"]
    if record["group"] == "mlx":
        runtime = runtime.split(" + mlx ")[0].split(", card loader")[0].split(", the repo")[0]
    elif record["group"] == "maker":
        found = [
            f"{name} {match[1]}"
            for name, pattern in (
                ("Transformers", r"transformers ([\d.]+)"),
                ("PyTorch", r"torch ([\d.]+)"),
            )
            if (match := re.search(pattern, runtime))
        ]
        runtime = f"{maker_code(record['other_engine'])} · " + (" · ".join(found) or runtime)
        if record["other_engine"] == "pplx-flash-maker-mps-bf16":
            runtime += " · BF16"
    runtime = runtime.replace(" (HTTP)", "").replace(" (MPS)", "").replace(" (mps)", "")
    if " / " in record["model"]:  # it ran its own conversion of the checkpoint ippon read
        own = record["model"].split(" · ", 1)[1].split(" / ")[0]
        own = own.replace("Ollama ", "")
        runtime += f" · its {own} build"
    return runtime


def evidence_links(sources: list[dict]) -> list[str]:
    """Link the archived requests without exposing internal paths as prose."""
    return [
        f"- [Recorded run {i}](https://github.com/sqwishai/ippon/blob/main/{source['source']})"
        for i, source in enumerate(sources, 1)
    ]


def provenance(run: dict, ours: bool) -> dict:
    meta, device = run["meta"], run["device"]
    mlx = re.search(r"\bmlx ([\d.]+)", meta.get("runtime", ""))
    return {
        "source": run["source"],
        "sha256": run["sha256"],
        "date": run["start"]["time"],
        "machine": {key: device.get(key) for key in MACHINE_KEYS},
        "runtime": meta.get("runtime"),
        "mlx": mlx[1] if mlx else None,
        "library_commit": meta.get("revision") if ours else None,
        "precision": meta.get("precision", meta.get("dtype")),
        "build": Path(meta.get("model", meta.get("model_dir", meta.get("gguf", "unrecorded")))).name,
        "build_info": meta.get("build_info"),
        "manifest_digest": meta.get("manifest_digest"),
        "config": run["config"],
        "skipped_sizes": run.get("skipped_sizes", {}),
    }


def comparisons(folders: list[Path]) -> list[dict]:
    data, output = load(folders), []
    for group, model, other, ours in PAIRS:
        common = [f for f in folders if f in data.get(other, {}) and f in data.get(ours, {})]
        if not common:
            continue
        # A campaign is one parent containing its rounds. A later campaign replaces, never pools with, an earlier one.
        newest = max(common, key=lambda f: max(data[other][f]["start"]["time"], data[ours][f]["start"]["time"])).parent
        selected = [f for f in common if f.parent == newest]
        a, b = [data[other][f] for f in selected], [data[ours][f] for f in selected]
        shapes = sorted({(r["size"], r["nq"]) for d in a + b for r in d["requests"] if r["phase"] == "timed"})
        cells = []
        for size, nq in shapes:
            ma, mb = median(a, size, nq), median(b, size, nq)
            if ma is None or mb is None:
                continue
            tokens = [r["tokens"] for r in timed(a + b, size, nq) if r.get("tokens") is not None]
            cells.append(
                {
                    "size": size,
                    "questions": nq,
                    "other_ms": ma,
                    "ippon_ms": mb,
                    "ratio": ma / mb,
                    "speed_percent": (ma / mb - 1) * 100,
                    "saved_ms": ma - mb,
                    "input_tokens": [min(tokens), max(tokens)] if tokens else None,
                    "samples": [len(timed(a, size, nq)), len(timed(b, size, nq))],
                    "round_speed_percent": [
                        (x / y - 1) * 100
                        for ra, rb in zip(a, b, strict=True)
                        if (x := median([ra], size, nq)) is not None and (y := median([rb], size, nq)) is not None
                    ],
                }
            )
        typical = min((c for c in cells if c["questions"] == 1), key=lambda c: abs(c["size"] - 1000), default=None)
        output.append(
            {
                "group": group,
                "model": recorded_model(model, b),
                "runtime": runtime_label(group, a),
                "runtime_label_source": (
                    "benchmarks/kit/setup.sh pin; arm version unrecorded"
                    if group == "omlx" and all(not re.search(r"\d+\.\d+", d["meta"].get("runtime", "")) for d in a)
                    else "selected result metadata"
                ),
                "campaign": portable(newest),
                "other_engine": other,
                "ippon_engine": ours,
                "rounds": len(selected),
                "cells": cells,
                "typical": typical,
                "short_ratio": gmean([c["ratio"] for c in cells if c["size"] in SHORT]),
                "long_ratio": gmean([c["ratio"] for c in cells if c["size"] in LONG]),
                "overall_ratio": gmean([c["ratio"] for c in cells]),
                "agreement": agreement(a, b),
                "other_sources": [provenance(d, False) for d in a],
                "ippon_sources": [provenance(d, True) for d in b],
            }
        )
    return output


def context(records: list[dict]) -> list[str]:
    sources = [s for record in records for s in record["ippon_sources"]]
    machines = sorted(
        {
            f"{s['machine']['chip']}, {s['machine']['memory_bytes'] / 2**30:g} GiB, macOS {s['machine']['macos']}"
            for s in sources
        }
    )
    dates = sorted({s["date"][:10] for s in sources})
    revisions = sorted({s["library_commit"] for s in sources if s["library_commit"]})
    versions = sorted({s["mlx"] for s in sources if s["mlx"]})
    lines = [f"Measured {', '.join(dates)} on {' and '.join(machines)}.", ""]
    if revisions:
        lines += [
            f"In-process ippon ran commit {', '.join(f'`{r}`' for r in revisions)} "
            f"with MLX {', '.join(versions) or 'not recorded'}."
        ]
    if revisions and any(s["library_commit"] is None for s in sources):
        lines += [
            "The HTTP arms did not record the library commit or the MLX version, so that commit applies only to the "
            "in-process arms. These timings do not measure the current commit."
        ]
    if not revisions:
        lines += [
            "These arms did not record the library commit or the MLX version, so their timings do not measure the "
            "current commit."
        ]
    if any("setup pin" in r["runtime"] for r in records):
        lines += ["oMLX's version is the `kit/setup.sh` pin, and its result metadata does not record a version."]
    return [*lines, ""]


def tables(records: list[dict], title: str) -> list[str]:
    if not records:
        return []
    lines = [
        f"## {title}",
        "",
        *context(records),
        "The typical cell is the measured nominal ~1,000-state-token request with one question. Input tokens include "
        "the model's prompt. Times are pooled medians in milliseconds.",
        "",
        "| Model / build | Other runtime | Input tokens | Other ms | ippon ms | Speed change |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for record in records:
        cell = record["typical"]
        if cell is None:
            continue
        tokens = cell["input_tokens"]
        token_text = "—" if not tokens else str(tokens[0]) if tokens[0] == tokens[1] else f"{tokens[0]} to {tokens[1]}"
        model = record["model"] + (f" (state size {cell['size']})" if cell["size"] != 1000 else "")
        lines.append(
            f"| {model} | {runtime_display(record)} | {token_text} | {fmt(cell['other_ms'])} | "
            f"{fmt(cell['ippon_ms'])} | {percent(cell['ratio'])} |"
        )
    lines += [
        "",
        "Speed change is `(other median / ippon median \u2212 1) \u00d7 100%`, so a negative value means ippon is "
        "slower. It is not a reduction in latency. Short and Long are geometric means of the cell ratios at the "
        "short and long state sizes, with the 1-question and 3-question shapes weighted equally. The Short / long "
        "sizes column lists the nominal state sizes that were measured.",
        "",
        "| Model / build | Other runtime | Short | Long | Short / long sizes | "
        "Max probability gap | Top answers changed |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for record in records:
        cover = [
            ", ".join(str(s) for s in sorted({c["size"] for c in record["cells"] if c["size"] in bucket}))
            for bucket in (SHORT, LONG)
        ]
        a = record["agreement"]
        lines.append(
            f"| {record['model']} | {runtime_display(record)} | {percent(record['short_ratio'])} | "
            f"{percent(record['long_ratio'])} | {' / '.join(cover)} | {a['max_probability_difference']:.6f} | "
            f"{a['top_answer_flips']}/{a['questions']} |"
        )
    lines += [
        "",
        "Agreement uses every matched timed question. The probability gap is the largest absolute difference, "
        "not a percentage. A matching top answer does not imply matching probabilities.",
        "",
    ]
    clef = next((r for r in records if r["other_engine"] == "clef-flash-mxfp8-ollama"), None)
    if clef:
        a = clef["agreement"]
        lines += [
            f"Clef-Flash MXFP8 is about level overall, at {percent(clef['overall_ratio'], 1)} across its measured "
            f"cells, against {percent(clef['typical']['ratio'], 1)} for the typical cell. "
            "Its maximum probability "
            f"gap is {a['max_probability_difference']:.6f}, "
            f"with {a['top_answer_flips']}/{a['questions']} top answers changed.",
            "",
        ]
    return lines


def details(records: list[dict]) -> list[str]:
    lines = ["## Per-shape evidence", ""]
    for record in records:
        lines += [f"### {record['model']} against {runtime_display(record)}", "", *context([record])]
        precision = sorted({s["precision"] for s in record["ippon_sources"] if s["precision"]})
        lines += [
            f"ippon compute: {', '.join(precision) or 'not recorded'}, {record['rounds']} paired rounds.",
            "",
            "| State size | Questions | Other ms | ippon ms | Saved ms | Speed change | Per-round speed change |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
        for c in record["cells"]:
            spread = ", ".join(percent(1 + p / 100) for p in c["round_speed_percent"])
            lines.append(
                f"| {c['size']} | {c['questions']} | {fmt(c['other_ms'])} | {fmt(c['ippon_ms'])} | "
                f"{fmt(c['saved_ms'])} | {percent(c['ratio'])} | {spread} |"
            )
        lines += ["", "Recorded requests:", ""]
        lines += evidence_links(record["other_sources"] + record["ippon_sources"])
        lines.append("")
    return lines


def headline(records: list[dict]) -> str:
    selected = [
        r for r in records if r["group"] in ("llama.cpp", "ollama") and r["typical"] and r["typical"]["size"] == 1000
    ]
    if not selected:
        return ""
    clauses = []
    for group in ("llama.cpp", "ollama"):
        pairs = [r for r in selected if r["group"] == group]
        if not pairs:
            continue
        ratios = sorted(r["typical"]["ratio"] for r in pairs)
        runtimes = ", ".join(sorted({r["runtime"] for r in pairs}))
        build = "Q8_0 " if group == "llama.cpp" else ""
        clauses.append(
            f"{runtimes} took between {compared(ratios[0])} and {compared(ratios[-1])} across {len(pairs)} tested "
            f"{build}builds"
        )
    dates = sorted({s["date"][:10] for r in selected for s in r["ippon_sources"]})
    machine = selected[0]["ippon_sources"][0]["machine"]["chip"]
    return (
        f"In the recorded {', '.join(dates)} {machine} comparison against ippon, {' and '.join(clauses)}, "
        "at nominal ~1,000 state tokens and one question. "
        "[Timings, answer agreement and provenance](docs/benchmarks.md)."
    )


def fill(path: Path, block: str, text: str) -> None:
    start, end = BLOCKS[block]
    page = path.read_text()
    head, found, rest = page.partition(start)
    _, closed, tail = rest.partition(end)
    if not found or not closed:
        raise SystemExit(f"{path}: missing {start} ... {end} block")
    path.write_text(head + start + "\n" + text.rstrip() + "\n" + end + tail)


def readme_pairs() -> list[tuple[str, str, str, str]]:
    """Use native ippon servers for every fresh comparison, including library-only alternatives."""
    return [
        (group, model, other, ours + "-serve" if ours.endswith("-ippon") else ours)
        for group, model, other, ours in PAIRS
    ]


def exclusions(values: list[str]) -> dict[str, str]:
    """Require a reason for every deliberately omitted pair; absence is never treated as an exclusion."""
    known = {other for _, _, other, _ in readme_pairs()}
    out = {}
    for value in values:
        engine, sep, reason = value.partition("=")
        if not sep or engine not in known or not reason.strip() or engine in out:
            raise ValueError(f"invalid exclusion {value!r}; use OTHER_ENGINE=REASON once per known pair")
        out[engine] = reason.strip()
    return out


def _readme_run(run: dict, source: str) -> None:
    """Refuse incomplete or mislabeled evidence before it can enter a table or a chart; busy readings stay."""

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"{source}: {message}")

    require(run.get("complete") is True and not run.get("errors"), "run is incomplete or has errors")
    require(isinstance(run.get("busy"), list), "busy readings are missing")
    require(not run.get("activity_errors"), "activity monitor failed")
    require(not run.get("skipped_sizes") and not run.get("skipped_cells"), "requested cells were skipped")
    meta, config = run.get("meta", {}), run.get("config", {})
    require(meta.get("transport") in ("native HTTP", "benchmark HTTP wrapper"), "HTTP transport is unrecorded")
    require(
        config.get("sizes") == [1000] and config.get("questions") == [1, 50],
        "expected 1000 tokens \u00d7 1/50 questions",
    )
    n, warmup = config.get("n"), config.get("warmup")
    require(isinstance(n, int) and n > 0 and isinstance(warmup, int) and warmup > 0, "invalid sample counts")
    aside = run.get("set_aside", [])
    require(all(row.get("phase") == "timed" and row.get("busy") for row in aside), "invalid set-aside request")
    replaced = {nq: {row.get("rep") for row in aside if row.get("nq") == nq} for nq in (1, 50)}
    require(sum(map(len, replaced.values())) == len(aside), "duplicate or unknown set-aside requests")
    counts = {nq: len(timed([run], 1000, nq)) for nq in (1, 50)}
    minimum = config.get("min_n") or n
    require(all(minimum <= count <= n for count in counts.values()), "missing timed requests")
    # Each set-aside timed request was replaced by its shape's next generated one.
    expected = {("warmup", 1000, nq, rep) for nq in (1, 50) for rep in range(warmup)} | {
        ("timed", 1000, nq, rep)
        for nq in (1, 50)
        for rep in range(counts[nq] + len(replaced[nq]))
        if rep not in replaced[nq]
    }
    seen = set()
    for row in run.get("requests", []):
        key = tuple(row.get(k) for k in ("phase", "size", "nq", "rep"))
        require(key in expected and key not in seen, f"unexpected or duplicate request {key}")
        seen.add(key)
        require(not row.get("problems"), f"invalid request {key}")
        ms = row.get("ms")
        require(isinstance(ms, (int, float)) and math.isfinite(ms) and ms > 0, f"invalid latency in {key}")
        qs = requests.questions(row["nq"], row.get("first", 0))
        require(len(qs) == row["nq"], "request generator does not supply the advertised question count")
        answers = row.get("answers", {})
        require(set(answers) == {q["id"] for q in qs}, f"partial or extra answers in {key}")
        for q in qs:
            distribution = answers[q["id"]]
            require(set(distribution) == set(requests.outcome_names(q)), f"wrong outcomes for {q['id']} in {key}")
            values = list(distribution.values())
            require(
                all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in values)
                and abs(math.fsum(values) - 1) <= 1e-3,
                f"invalid distribution for {q['id']} in {key}",
            )
    require(seen == expected, f"missing {len(expected - seen)} requests")


def _readme_runtime(group: str, runs: list[dict]) -> str:
    labels = {run["meta"].get("display_runtime") for run in runs}
    if None in labels:
        label = runtime_label(group, runs)
    elif len(labels) == 1:
        label = str(next(iter(labels)))
    else:
        raise ValueError("runtime label changed between rounds")
    if ";" in label or "unrecorded" in label or "setup pin" in label:
        raise ValueError(f"runtime version must come from each result: {label}")
    return label


def _readme_evidence(campaign: dict, folder: Path, ours: list[dict], engines: set[str]) -> None:
    """Validate the model fingerprints and how ippon chose its kernels (a campaign before 28 Oct 2026: the tune it ran
    first, its store and records), independently of the installed runtime or local cache."""

    def saved(name: str) -> Path:
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("campaign evidence must use a sibling filename")
        path = folder.parent / name
        if not path.is_file():
            raise ValueError(f"{portable(path)}: missing campaign evidence")
        return path

    def hashed(entry: dict) -> Path:
        path = saved(entry.get("file", ""))
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry.get("sha256"):
            raise ValueError(f"{portable(path)}: evidence hash mismatch")
        return path

    generator = Path(__file__).with_name("gen_requests.py")
    if campaign.get("generator_sha256") != hashlib.sha256(generator.read_bytes()).hexdigest():
        # The 28 Oct 2026 M5 Pro campaign predates repeat/grow; its fresh requests are unchanged.
        # Compare the actual payloads, so an unrelated generator extension cannot invalidate that archive.
        for run in ours:
            config = run["config"]
            if config.get("scenario", "fresh") != "fresh" or config["chars_per_token"] != requests.CHARS_PER_TOKEN:
                raise ValueError("request generator changed since measurement")
            for row in run["requests"] + run.get("set_aside", []):
                state = requests.state(row["size"], config["seed"], row["phase"], row["nq"], row["rep"])
                payload = requests.to_systemone(state, requests.questions(row["nq"]))
                digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                if digest != row.get("request_sha256"):
                    raise ValueError(f"{run['source']}: request generator changed the recorded payloads")
    identities = json.loads(hashed(campaign.get("model_builds", {})).read_text())
    if not engines <= identities.get("engines", {}).keys():
        raise ValueError("model build identities are missing for selected engines")
    if "tuning" not in campaign:  # kernels chosen from the chip by the installed library (choices.py)
        if not isinstance(campaign.get("machine", {}).get("kernel_rule"), str):
            raise ValueError("the campaign records neither its kernel rule nor a tune")
        return
    tuning = campaign["tuning"]
    key, kernels = tuning.get("chip_key"), tuning.get("kernels")
    if not isinstance(key, str) or not isinstance(kernels, int) or not key.endswith(f"kernels {kernels}"):
        raise ValueError("tuning chip/kernel key is missing or inconsistent")
    store = json.loads(hashed(tuning).read_text())
    if not store.get(key):
        raise ValueError("saved store has no measurements for its recorded chip/kernel key")
    tuned = {}
    for entry in tuning.get("models", []):
        record = json.loads(hashed({"file": entry["record"], "sha256": entry.get("sha256")}).read_text())
        revision = record.get("machine", {}).get("revision", "")
        if (
            record.get("complete") is not True
            or "busy" not in record  # busy readings are kept: contention can only make ippon's tiles slower
            or record.get("activity_errors")
            or record.get("chip_key") != key
            or record.get("kernels") != kernels
            or record.get("model") != entry["model"]
            or len(revision) < 7
            or not campaign["library_commit"].startswith(revision)
        ):
            raise ValueError(f"{entry['record']}: invalid tuning record")
        tuned[entry["model"]] = record
    for run in ours:
        if run["meta"].get("model") not in tuned:
            raise ValueError(f"{run['source']}: no successful fresh tune for this model build")


def _readme_accepted_rounds(folders: list[Path], campaign: dict, engines: set[str]) -> None:
    """Require the complete successful campaign, including adapter exit status after its last request."""
    parents = {folder.resolve().parent for folder in folders}
    if len(parents) != 1:
        raise ValueError("accepted rounds must belong to one campaign directory")
    parent = next(iter(parents))
    for name in ("CAMPAIGN.json", "ATTEMPTS.json"):
        if not (parent / name).is_file():
            raise ValueError(f"{portable(parent / name)}: missing campaign record")
    if json.loads((parent / "CAMPAIGN.json").read_text()) != campaign:
        raise ValueError("saved campaign differs from the measured campaign")
    count = campaign.get("rounds")
    if not isinstance(count, int) or count < 1:
        raise ValueError("campaign must declare a positive round count")
    accepted = [
        entry
        for entry in json.loads((parent / "ATTEMPTS.json").read_text())
        if entry.get("kind") == "round" and entry.get("accepted") is True
    ]
    numbers = sorted(entry.get("round") for entry in accepted)
    if "close" in campaign:  # rounds past the first ran only the pairs round 1 left close, and may not have run
        expected = set(range(1, len(numbers) + 1))
        if not numbers or numbers != sorted(expected) or len(numbers) > count:
            raise ValueError("attempt ledger does not establish its accepted rounds")
    else:
        expected = set(range(1, count + 1))
        if len(accepted) != count or set(numbers) != expected:
            raise ValueError("attempt ledger does not establish every accepted round")
    if {folder.resolve() for folder in folders} != {parent / f"r{n}" for n in expected}:
        raise ValueError("supply every accepted r1..rN folder, never partial or rejected attempts")
    for entry in accepted:
        outcomes = entry.get("engines", [])
        names = [outcome.get("engine") for outcome in outcomes]
        required = engines if entry["round"] == 1 or "scheduled" not in entry else engines & set(entry["scheduled"])
        if (
            len(names) != len(set(names))
            or not required <= set(names)
            or any(
                not outcome.get("failed") and (outcome.get("exit") != 0 or outcome.get("problems") != [])
                for outcome in outcomes
            )
        ):
            raise ValueError(f"round {entry['round']}: an engine failed or has no accepted exit status")


def _sample_range(record: dict) -> str:
    """The timed requests each arm's cells pooled: one count, or the range when cells stopped once settled."""
    counts = sorted({n for cell in record["cells"] for n in cell["samples"]})
    return str(counts[0]) if len(counts) == 1 else f"{counts[0]} to {counts[-1]}"


def scheduled_steps(folders: list[Path]) -> dict[Path, set[str] | None]:
    """Per round folder, the steps (engine and workload suffix) its round ran: None for all of them (the first
    round, an unfinished one, or a campaign whose every round ran every pair)."""
    out: dict[Path, set[str] | None] = dict.fromkeys(folders)
    parent = folders[0].resolve().parent
    ledger = parent / "ATTEMPTS.json"
    if not ledger.is_file():
        return out
    for entry in json.loads(ledger.read_text()):
        if (
            entry.get("kind") == "round"
            and entry.get("accepted")
            and entry.get("round", 1) > 1
            and "scheduled" in entry
        ):
            for folder in folders:
                if folder.resolve() == parent / f"r{entry['round']}":
                    out[folder] = set(entry["scheduled"])
    return out


# A run's errors by what its result says they were (`_error_kind`), as a sentence says what the runtime did
DID = {
    "out of memory": "ran out of memory on",
    "dropped connections": "dropped the connection on",
    "errors": "returned errors on",
}


def _error_kind(run: dict) -> str:
    """What a run's errors were: its requests' errors and, from a server, the exception lines it logged meanwhile
    (`server_errors`), where its reply ("Internal Server Error") says less."""
    said = " ".join([*(e["error"] for e in run["errors"]), *run.get("server_errors", [])]).lower()
    if "out of memory" in said:
        return "out of memory"
    if any(reset in said for reset in ("connection reset", "broken pipe", "remote end closed")):
        return "dropped connections"
    return "errors"


def _runtime_failure(run: dict, path: Path) -> dict:
    """A runtime whose own requests failed, which the campaign keeps as its result: why, how many of each phase's
    requests (failed, sent), and where it is recorded."""
    meta, kind = run.get("meta", {}), _error_kind(run)
    counts = {}
    for phase in ("warmup", "timed"):
        failed = sum(e["phase"] == phase for e in run["errors"])
        counts[phase] = [failed, failed + sum(r["phase"] == phase for r in run["requests"])]
    return {
        "reason": "out of memory" if kind == "out of memory" else "failed",
        "kind": kind,
        "counts": counts,
        "error": run["errors"][0]["error"][:400],
        "source": portable(path),
        "runtime": meta.get("display_runtime") or meta.get("runtime"),
    }


def failed_requests(failure: dict, phase: str, of: str = "") -> str:
    """What a runtime did with one phase's requests, counted: "ran out of memory on 2 of its 20 requests"."""
    failed, sent = failure["counts"][phase]
    return f"{DID[failure['kind']]} {failed} of its {sent} {of}requests"


def _readme_runs(
    folders: list[Path], needed: set[str], partial: bool
) -> tuple[dict[str, dict[Path, dict]], dict[str, dict]]:
    """Each needed engine's text-workload run by round folder, checked complete, and the runtimes that failed; a round
    missing an engine it scheduled is refused (partial: kept out instead)."""
    scheduled = scheduled_steps(folders)
    data: dict[str, dict[Path, dict]] = {}
    failures: dict[str, dict] = {}
    for folder in folders:
        for path in sorted(folder.glob("*.json")):
            run = json.loads(path.read_text())
            if "engine" not in run:
                continue
            engine = run["engine"].rsplit("/", 1)[-1]
            if engine not in needed or path.stem != engine or (partial and not run.get("complete")):
                continue  # path.stem: the text workload's result, never another workload's (readme_more)
            if run.get("errors") and run.get("complete"):
                failures.setdefault(engine, _runtime_failure(run, path))
                continue
            if folder in data.get(engine, {}):
                raise ValueError(f"{folder}: duplicate results for {engine}")
            _readme_run(run, portable(path))
            data.setdefault(engine, {})[folder] = {
                **run,
                "source": portable(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ran = scheduled[folder]
        missing = sorted(
            engine
            for engine in needed
            if folder not in data.get(engine, {}) and engine not in failures and (ran is None or engine in ran)
        )
        if missing and not partial:
            raise ValueError(f"{folder}: missing required engines: {', '.join(missing)}")
    return data, failures


CONFIG_KEYS = ("n", "warmup", "sizes", "questions", "seed", "chars_per_token", "min_n", "precision", "cell_seconds")


def _readme_identity(runs: list[dict]) -> tuple[str, str, dict, dict, dict]:
    """What every run of a README round shares: the library commit, the dates it was measured on, the machine, the
    request configuration and the campaign; README evidence never mixes two."""
    commits, dates, machines, configs, campaigns = set(), set(), set(), set(), set()
    measured = set()
    for run in runs:
        campaign = run.get("campaign", {})
        commit = campaign.get("library_commit")
        date = campaign.get("date", run["start"]["time"][:10])
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError(f"{run['source']}: campaign.library_commit must be a full git SHA")
        # A campaign's rounds can run past midnight: every run starts on or after the day timing began, within a day.
        start = dt.date.fromisoformat(date)
        for when in (run["start"]["time"][:10], run["end"]["time"][:10]):
            if not start <= dt.date.fromisoformat(when) <= start + dt.timedelta(days=1):
                raise ValueError(f"{run['source']}: measured outside the campaign's dates")
            measured.add(when)
        campaigns.add(json.dumps(campaign, sort_keys=True))
        commits.add(commit)
        dates.add(date)
        machines.add(json.dumps({k: run["device"].get(k) for k in MACHINE_KEYS}, sort_keys=True))
        configs.add(json.dumps({k: run["config"].get(k) for k in CONFIG_KEYS if k in run["config"]}, sort_keys=True))
    if any(len(values) != 1 for values in (commits, dates, machines, configs, campaigns)):
        raise ValueError("README evidence must share one commit, date, machine and request configuration")
    machine, config, campaign = (json.loads(next(iter(values))) for values in (machines, configs, campaigns))
    return next(iter(commits)), " to ".join(sorted(measured)), machine, config, campaign


def _ippon_mlx(run: dict, commit: str) -> str:
    """The MLX version an ippon arm ran on, after checking it ran the campaign's commit on ippon's own server."""
    revision, mlx = run["meta"].get("revision"), run["meta"].get("mlx")
    if not isinstance(revision, str) or len(revision) < 7 or not commit.startswith(revision) or not mlx:
        raise ValueError(f"{run['source']}: installed ippon revision or MLX version is missing/mismatched")
    if run["meta"]["transport"] != "native HTTP":
        raise ValueError(f"{run['source']}: ippon must use its native server")
    return mlx


def _readme_cell(a: list[dict], b: list[dict], nq: int) -> dict:
    """One ~1,000-token cell of a pair: both arms' medians over their rounds, the speed change, and each round's."""
    ma, mb = median(a, 1000, nq), median(b, 1000, nq)
    assert ma is not None  # _readme_run established complete cells
    assert mb is not None
    return {
        "size": 1000,
        "questions": nq,
        "other_ms": ma,
        "ippon_ms": mb,
        "other_questions_per_second": nq * 1000 / ma,
        "ippon_questions_per_second": nq * 1000 / mb,
        "ratio": ma / mb,
        "speed_percent": (ma / mb - 1) * 100,
        "samples": [len(timed(a, 1000, nq)), len(timed(b, 1000, nq))],
        "round_speed_percent": [
            (median([ra], 1000, nq) / median([rb], 1000, nq) - 1) * 100 for ra, rb in zip(a, b, strict=True)
        ],
        "agreement": agreement(
            [{**run, "requests": timed([run], 1000, nq)} for run in a],
            [{**run, "requests": timed([run], 1000, nq)} for run in b],
        ),
    }


def _readme_campaign(folders: list[Path], excluded: dict[str, str], partial: bool) -> dict | None:
    """Validate one campaign's declared text workloads before selecting any of its measurements."""
    manifest = json.loads((folders[0].parent / "CAMPAIGN.json").read_text())
    workload = next((w for w in manifest["workloads"] if w["name"] == "text"), None)
    if workload is None:  # a campaign of practical workloads only has no text comparisons to report
        return None
    declared = {tuple(pair) for pair in workload["pairs"]}
    selected = [p for p in readme_pairs() if p[2] not in excluded and p[2:] in declared]
    alone = [
        (name, engine + "-serve")
        for name, engine in ALONE + SMALL_MAC_ALONE
        if engine + "-serve" in workload.get("alone", [])
    ]
    needed = {engine for *_, other, ours in selected for engine in (other, ours)} | {e for _, e in alone}
    required = needed.copy()
    data, failures = _readme_runs(folders, needed, partial)
    if failed_ippon := failures.keys() & ({p[3] for p in selected} | {e for _, e in alone}):
        raise ValueError(f"ippon requests failed: {', '.join(sorted(failed_ippon))}")
    failed = []
    for group, model, other, ours in selected:
        if other in failures and folders[0] in data.get(ours, {}):
            b = [data[ours][folder] for folder in folders if folder in data[ours]]
            one, fifty = median(b, 1000, 1), median(b, 1000, 50)
            assert one is not None  # _readme_run established complete cells
            assert fifty is not None
            failed.append(
                {
                    "group": group,
                    "model": model,
                    "other_engine": other,
                    "ippon_engine": ours,
                    **failures[other],
                    "ippon_ms": one,
                    "ippon_questions_per_second": 50 * 1000 / fifty,
                    "samples": [len(timed(b, 1000, nq)) for nq in (1, 50)],
                    "ippon_sources": [provenance(run, True) for run in b],
                }
            )
    selected = [pair for pair in selected if pair[2] not in failures]
    if partial:
        scheduled = scheduled_steps(folders)
        selected = [
            p
            for p in selected
            if all(
                folder in data.get(e, {})
                for e in p[2:]
                for folder in folders
                if scheduled[folder] is None or e in scheduled[folder]
            )
        ]
        alone = [
            (name, e)
            for name, e in alone
            if all(
                folder in data.get(e, {}) for folder in folders if scheduled[folder] is None or e in scheduled[folder]
            )
        ]
    ours = {pair[3] for pair in selected} | {r["ippon_engine"] for r in failed} | {e for _, e in alone}
    needed = {engine for p in selected for engine in p[2:]} | ours
    data = {engine: runs for engine, runs in data.items() if engine in needed}
    runs = [run for by_folder in data.values() for run in by_folder.values()]
    if not runs:
        if partial:
            return None
        raise ValueError("no comparisons selected")
    commit, date, machine, config, campaign = _readme_identity(runs)
    if not partial:
        _readme_accepted_rounds(folders, campaign, required)
    for run in runs:
        if run["config"]["n"] != campaign.get("samples") or run["config"]["warmup"] != campaign.get("warmup"):
            raise ValueError(f"{run['source']}: sample counts differ from the declared campaign")
    for folder in folders:
        present = [data[engine][folder] for engine in ours if folder in data.get(engine, {})]
        _readme_evidence(campaign, folder, present, needed)
    versions = {_ippon_mlx(run, commit) for engine in ours for run in data.get(engine, {}).values()}
    records = []
    for group, model, other, ours in selected:
        both = [f for f in folders if f in data[other] and f in data[ours]]  # a later round ran only close pairs
        if not both or both[0] != folders[0]:
            raise ValueError(f"{other}: the first round has no finished pair")
        a, b = [data[other][f] for f in both], [data[ours][f] for f in both]
        versions.update(_ippon_mlx(run, commit) for run in b)
        for left, right in zip(a, b, strict=True):
            _same_payloads(left, right, other)
        records.append(
            {
                "group": group,
                "model": model,
                "runtime": _readme_runtime(group, a),
                "other_engine": other,
                "ippon_engine": ours,
                "rounds": len(both),
                "cells": [_readme_cell(a, b, nq) for nq in (1, 50)],
                "agreement": agreement(a, b),
                "other_sources": [
                    {
                        **provenance(run, False),
                        "transport": run["meta"]["transport"],
                        "method": run["meta"].get("question_method", "one request"),
                    }
                    for run in a
                ],
                "ippon_sources": [
                    {
                        **provenance(run, True),
                        "mlx": run["meta"]["mlx"],
                        "transport": run["meta"]["transport"],
                        "kernels": run["campaign"].get("tuning") or run["campaign"]["machine"].get("kernel_rule"),
                    }
                    for run in b
                ],
            }
        )
    if len(versions) != 1:
        raise ValueError("ippon arms must all use one recorded MLX version")
    identity = {
        "library_commit": commit,
        "date": date,
        "machine": machine,
        "mlx": next(iter(versions)),
        "campaign": portable(folders[0].parent),
        "measured_at": max(run["start"]["time"] for run in runs),
    }
    solo = []
    for name, engine in alone:
        b = list(data[engine].values())
        solo.append(
            {
                "model": name,
                "ippon_engine": engine,
                "rounds": len(b),
                "cells": [
                    {"questions": nq, "ippon_ms": median(b, 1000, nq), "samples": [len(timed(b, 1000, nq))]}
                    for nq in (1, 50)
                ],
                "ippon_sources": [{key: run[key] for key in ("source", "sha256", "config")} for run in b],
            }
        )
    for record in records + failed + solo:
        record.update(identity)
    return {
        **identity,
        "config": config,
        "close": campaign.get("close"),
        "excluded_pairs": excluded,
        "failed": failed,
        "partial": {
            "pairs": len(selected),
            "of": sum(p[2] not in excluded and p[2:] in declared for p in readme_pairs()),
            "rounds": len(folders),
            "of_rounds": campaign["rounds"],
        }
        if partial
        else None,
        "busy_requests": [
            sum(bool(r.get("busy")) for run in runs for r in run["requests"] if r["phase"] == "timed"),
            sum(r["phase"] == "timed" for run in runs for r in run["requests"]),
        ],
        "pairs": records,
        "alone": solo,
        "speed_definition": "(other_median / ippon_median - 1) * 100",
        "input_folders": [portable(folder) for folder in folders],
        "generator_sources": {
            portable(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(__file__).with_name("chart.py"),
                Path(__file__).with_name("gen_requests.py"),
            )
        },
    }


def campaign_folders(folders: list[Path]) -> list[list[Path]]:
    """Keep accepted rounds together, even when callers supply several targeted campaigns."""
    if not folders or len({f.resolve() for f in folders}) != len(folders):
        raise ValueError("supply distinct accepted round folders")
    grouped: dict[Path, list[Path]] = {}
    for folder in folders:
        if not folder.is_dir():
            raise ValueError(f"{folder}: campaign round folder does not exist")
        if not (folder.parent / "CAMPAIGN.json").is_file():
            raise ValueError(f"{folder.parent}: missing CAMPAIGN.json")
        grouped.setdefault(folder.resolve().parent, []).append(folder)
    return [sorted(rounds, key=lambda f: int(f.name.rsplit("r", 1)[-1])) for rounds in grouped.values()]


def readme_comparisons(folders: list[Path], excluded: dict[str, str] | None = None, partial: bool = False) -> dict:
    """Select each pair's newest supplied campaign, retaining that campaign's rounds and provenance."""
    excluded = exclusions([f"{engine}={reason}" for engine, reason in (excluded or {}).items()])
    reports = [
        report
        for rounds in campaign_folders(folders)
        if (report := _readme_campaign(rounds, excluded, partial)) is not None
    ]
    if not reports:
        raise ValueError("no finished text workloads in the supplied campaigns")
    chosen: dict[str, tuple[str, dict]] = {}
    for report in reports:
        for kind in ("pairs", "failed", "alone"):
            for record in report[kind]:
                key = record.get("other_engine", record["ippon_engine"])
                if key not in chosen or record["measured_at"] > chosen[key][1]["measured_at"]:
                    chosen[key] = kind, record
    result = dict(max(reports, key=lambda r: r["measured_at"]))
    for kind in ("pairs", "failed", "alone"):
        result[kind] = [record for selected, record in chosen.values() if selected == kind]
    result["campaigns"] = [
        {k: report[k] for k in ("date", "machine", "library_commit", "mlx", "config", "close", "input_folders")}
        for report in reports
        if any(r["campaign"] == report["campaign"] for _, r in chosen.values())
    ]
    result["pending_pairs"] = {
        other: model for _, model, other, _ in readme_pairs() if other not in chosen and other not in excluded
    }
    result["pending_alone"] = [name for name, engine in ALONE if engine + "-serve" not in chosen]
    result["input_folders"] = [portable(folder) for folder in folders]
    return result


def readme_older(folders: list[Path], report: dict) -> list[dict]:
    """The README pairs this round has not measured, from an older campaign's one-question ~1,000-token cell. Each
    keeps that campaign's own ippon time: an older time is never paired with this round's."""
    measured = {record["other_engine"] for record in report["pairs"] + report.get("failed", [])}
    wanted = {other for *_, other, _ in readme_pairs()}
    older = []
    for record in comparisons(folders):
        if record["other_engine"] in measured or record["other_engine"] not in wanted:
            continue
        cell = next(c for c in record["cells"] if c["size"] == 1000 and c["questions"] == 1)
        source = record["ippon_sources"][0]
        older.append(
            {
                **{key: record[key] for key in ("group", "model", "runtime", "other_engine", "ippon_engine")},
                "date": source["date"][:10],
                "library_commit": source["library_commit"],
                "mlx": source["mlx"],
                "machine": source["machine"],
                "samples": cell["samples"],
                "other_ms": cell["other_ms"],
                "ippon_ms": cell["ippon_ms"],
                "speed_percent": cell["speed_percent"],
            }
        )
    return older


def _same_payloads(left: dict, right: dict, label: str) -> None:
    """Both arms of a pair answered the same generated requests, set-aside replacements aside."""
    left_hashes, right_hashes = (
        {(r["phase"], r["nq"], r["rep"]): r.get("request_sha256") for r in arm["requests"] + arm.get("set_aside", [])}
        for arm in (left, right)
    )
    shared = left_hashes.keys() & right_hashes.keys()
    if (
        any(left_hashes[k] != right_hashes[k] for k in shared)
        or not all([*left_hashes.values(), *right_hashes.values()])
        or not {k for k in left_hashes if k[0] == "warmup"} <= shared
    ):
        raise ValueError(f"{label}: the paired arms did not record identical request payloads")


def _more_run(run: dict, workload: dict, source: str, warmup_errors: bool = False) -> None:
    """Refuse an incomplete or mislabeled run of one of the other workloads, or one with errors (with
    `warmup_errors`, a rival's, only in its timed requests); busy readings stay and are counted."""

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"{source}: {message}")

    options = workload["options"]
    config = run.get("config", {})
    errors = [e for e in run.get("errors", []) if e["phase"] == "timed" or not warmup_errors]
    require(run.get("complete") is True and not errors, "run is incomplete or has errors")
    require(isinstance(run.get("busy"), list) and not run.get("activity_errors"), "activity readings are missing")
    require(
        config.get("sizes") == [int(options[options.index("--sizes") + 1])]
        and config.get("questions") == [int(options[options.index("--questions") + 1])],
        "request shape differs from its workload",
    )
    clients = int(options[options.index("--clients") + 1]) if "--clients" in options else 1
    media = options[options.index("--media") + 1] if "--media" in options else None
    require(config.get("clients", 1) == clients, "client count differs from its workload")
    require(((config.get("media") or {}).get("kind")) == media, "media differ from its workload")
    scenario = options[options.index("--scenario") + 1] if "--scenario" in options else "fresh"
    require(config.get("scenario", "fresh") == scenario, "scenario differs from its workload")
    if clients > 1:
        require(run.get("concurrent", {}).get("span_s", 0) > 0, "no concurrent window was recorded")
    for row in run.get("requests", []):
        require(not row.get("problems") and isinstance(row.get("ms"), (int, float)), f"invalid request {row['rep']}")


def _matched(a: list[dict], b: list[dict], size: int, nq: int) -> tuple[list[dict], list[dict]]:
    """Each round's timed requests that both arms answered, for a growing state (text or images): a busy request set
    aside moves its arm on to the next generated request, a longer one, so unmatched arms would time different
    lengths."""
    left, right = [], []
    for x, y in zip(a, b, strict=True):
        shared = {r["request_sha256"] for r in timed([x], size, nq)} & {
            r["request_sha256"] for r in timed([y], size, nq)
        }
        for run, out in ((x, left), (y, right)):
            kept = [r for r in run["requests"] if r["phase"] != "timed" or r["request_sha256"] in shared]
            out.append({**run, "requests": kept})
    return left, right


def _more_value(workload: str, runs: list[dict], size: int, nq: int) -> float:
    """A pair arm's measure in one of the other workloads over its rounds: requests per second answered with every
    client sending at once (concurrent), else the request's median milliseconds."""
    if workload in CLIENTS:
        return sum(len(timed([run], size, nq)) for run in runs) / sum(run["concurrent"]["span_s"] for run in runs)
    value = median(runs, size, nq)
    assert value is not None  # _more_run established complete runs
    return value


def _clients(runs: list[dict], size: int, nq: int) -> dict:
    """With several clients at once: requests answered per second, and each request's wait from sending to answer.
    The median alone misleads: a server that answers some clients at once and makes the others wait has a low
    median and a long tail, so the 90th percentile (nearest rank) goes with it."""
    waits = sorted(row["ms"] for row in timed(runs, size, nq))
    return {
        "requests_per_second": _more_value("concurrent", runs, size, nq),
        "median_ms": statistics.median(waits),
        "p90_ms": waits[math.ceil(0.9 * len(waits)) - 1],
    }


def _readme_more_campaign(folders: list[Path], newer: frozenset[tuple[str, str]] = frozenset()) -> dict | None:
    """The campaign of the README's other workloads: each pair's requests per second with 8 clients at once, through
    each runtime's own server, and its one-question milliseconds with an image or an audio clip. Folders named
    attempt-rN are an unfinished round: its pairs with both runs finished are reported, labelled partial. A
    (workload, other engine) in newer was measured again by a later campaign, which replaces it here unread; a
    campaign whose every cell was measured again is None."""
    parents = {folder.resolve().parent for folder in folders}
    if len(parents) != 1:
        raise ValueError("the other workloads' rounds must belong to one campaign directory")
    parent = next(iter(parents))
    campaign = json.loads((parent / "CAMPAIGN.json").read_text())
    partial = any(folder.name.startswith("attempt-") for folder in folders)
    workloads = [w for w in campaign["workloads"] if w["name"] in MORE]
    if not partial:
        steps = {
            f"{e}{MORE[w['name']][0]}"
            for w in workloads
            for e in [*(e for pair in w["pairs"] for e in pair), *w.get("alone", [])]
        }
        _readme_accepted_rounds(folders, campaign, steps)
    groups = {other: (group, model) for group, model, other, _ in readme_pairs()}
    scheduled = scheduled_steps(folders)
    records, runs_seen = [], []
    for workload in workloads:
        name, (suffix, measure) = workload["name"], MORE[workload["name"]]
        size = int(workload["options"][workload["options"].index("--sizes") + 1])
        nq = int(workload["options"][workload["options"].index("--questions") + 1])
        for other, ours in workload["pairs"]:
            if other not in groups or (name, other) in newer:  # dropped from the kit, or measured again since
                continue
            group, model = groups[other]
            arms, failures, warmup = [], {}, None
            ran = [f for f in folders if scheduled[f] is None or f"{other}{suffix}" in scheduled[f]]
            for engine in (other, ours):
                found = []
                for folder in ran:
                    path = folder / f"{engine}{suffix}.json"
                    run = json.loads(path.read_text()) if path.is_file() else {}
                    if not run.get("complete"):
                        break
                    # a rival's errors in its warm-up leave its timed requests a measurement; ippon's never
                    timed_errors = any(e["phase"] == "timed" for e in run.get("errors", []))
                    if timed_errors or (run.get("errors") and engine == ours):
                        # named as a finished run of its runtime is (`_readme_runtime`)
                        failures[engine] = {**_runtime_failure(run, path), "runtime": _readme_runtime(group, [run])}
                        break
                    if run.get("errors"):
                        warmup = warmup or _runtime_failure(run, path)
                    _more_run(run, workload, portable(path), warmup_errors=engine == other)
                    found.append(
                        {**run, "source": portable(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    )
                arms.append(found)
            a, b = arms
            if failures and len(b if other in failures else a) == len(ran):
                failed = next(iter(failures.values()))
                done = b if other in failures else a
                value = _more_value(name, done, size, nq)
                records.append(
                    {
                        "workload": name,
                        "measure": measure,
                        "group": group,
                        "model": model,
                        "runtime": _readme_runtime(group, a) if done is a else failed["runtime"],
                        "other_engine": other,
                        "ippon_engine": ours,
                        "other": value if done is a else None,
                        "ippon": value if done is b else None,
                        "speed_percent": None,
                        "failed": failed,
                        "rounds": len(ran),
                        "samples": [len(timed(done, size, nq))],
                        "server_env": None,
                        "sources": [failed["source"], *(run["source"] for run in done)],
                    }
                )
                if name in CLIENTS:
                    records[-1]["clients"] = {"other" if done is a else "ippon": _clients(done, size, nq)}
                runs_seen += done
                continue
            if len(a) != len(ran) or len(b) != len(ran):
                if partial:
                    continue
                raise ValueError(f"{other}{suffix}: a round has no finished pair")
            for left, right in zip(a, b, strict=True):
                _same_payloads(left, right, f"{other}{suffix}")
            for run in b:
                if run["meta"].get("transport") != "native HTTP" or not campaign["library_commit"].startswith(
                    run["meta"].get("revision", "-")
                ):
                    raise ValueError(f"{run['source']}: ippon must be the campaign's library, on its own server")
            runs_seen += a + b
            if name in ("grow", "pictures"):  # a growing state: compare the requests both arms timed
                a, b = _matched(a, b, size, nq)
                if not timed(a, size, nq):
                    raise ValueError(f"{other}{suffix}: the two arms timed no request in common")
            other_value, ippon_value = _more_value(name, a, size, nq), _more_value(name, b, size, nq)
            # requests per second: higher is faster; milliseconds: lower is
            ratio = ippon_value / other_value if name in CLIENTS else other_value / ippon_value
            speed = (ratio - 1) * 100
            records.append(
                {
                    "workload": name,
                    "measure": measure,
                    "group": group,
                    "model": model,
                    "runtime": _readme_runtime(group, a),
                    "other_engine": other,
                    "ippon_engine": ours,
                    "other": other_value,
                    "ippon": ippon_value,
                    "speed_percent": speed,
                    "rounds": len(ran),
                    "samples": [len(timed(a, size, nq)), len(timed(b, size, nq))],
                    "agreement": agreement(a, b),
                    "server_env": (a[0]["meta"].get("server_env") or {}).get("OLLAMA_NUM_PARALLEL"),
                    "sources": [run["source"] for run in a + b],
                    "warmup_errors": warmup,
                }
            )
            if name in CLIENTS:
                records[-1]["clients"] = {"other": _clients(a, size, nq), "ippon": _clients(b, size, nq)}
        solo_names = {engine + "-serve": model for model, engine in ALONE + SMALL_MAC_ALONE}
        for engine in workload.get("alone", []):
            if engine not in solo_names:
                continue
            ran = [f for f in folders if scheduled[f] is None or f"{engine}{suffix}" in scheduled[f]]
            found = []
            for folder in ran:
                path = folder / f"{engine}{suffix}.json"
                run = json.loads(path.read_text()) if path.is_file() else {}
                if not run.get("complete"):
                    break
                _more_run(run, workload, portable(path))
                found.append({**run, "source": portable(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            if not ran or len(found) != len(ran):
                if partial:
                    continue
                raise ValueError(f"{engine}{suffix}: a round has no finished solo run")
            runs_seen += found
            records.append(
                {
                    "workload": name,
                    "measure": measure,
                    "model": solo_names[engine],
                    "alone": True,
                    "ippon_engine": engine,
                    "ippon": _more_value(name, found, size, nq),
                    "other": None,
                    "speed_percent": None,
                    "rounds": len(ran),
                    "samples": [len(timed(found, size, nq))],
                    "sources": [run["source"] for run in found],
                }
            )
            if name in CLIENTS:
                records[-1]["clients"] = {"ippon": _clients(found, size, nq)}
    if not records:
        if newer:
            return None
        raise ValueError("no finished additional workloads")
    first = runs_seen[0]
    busy = [row for run in runs_seen for row in run["requests"] if row["phase"] == "timed"]
    versions = {
        _ippon_mlx(run, campaign["library_commit"])
        for run in runs_seen
        if run["engine"].rsplit("/", 1)[-1].endswith("-ippon-serve")
    }
    if len(versions) != 1:
        raise ValueError("ippon arms must all use one recorded MLX version")
    identity = {
        "library_commit": campaign["library_commit"],
        "date": " to ".join(sorted({run["start"]["time"][:10] for run in runs_seen})),
        "machine": {k: first["device"].get(k) for k in MACHINE_KEYS},
        "mlx": next(iter(versions)),
        "campaign": portable(parent),
        "measured_at": max(run["start"]["time"] for run in runs_seen),
    }
    for record in records:
        record.update(identity)
    return {
        **identity,
        "partial": {"rounds": len(folders), "of_rounds": campaign["rounds"]} if partial else None,
        "busy_runs": [sum(bool(run["busy"]) for run in runs_seen), len(runs_seen)],
        "timed_requests": len(busy),
        "records": records,
        "sources": {run["source"]: run["sha256"] for run in runs_seen},
    }


def readme_more(folders: list[Path]) -> dict:
    """Select the newest complete campaign separately for each additional workload and pair: campaigns are supplied
    oldest first (README_CAMPAIGNS), so each is read newest first and skips the cells a later one measured."""
    reports: list[dict] = []
    for rounds in reversed(campaign_folders(folders)):
        manifest = json.loads((rounds[0].parent / "CAMPAIGN.json").read_text())
        if any(w["name"] in MORE for w in manifest["workloads"]):
            newer = frozenset(
                (r["workload"], r.get("other_engine", "")) for report in reports for r in report["records"]
            )
            if (report := _readme_more_campaign(rounds, newer)) is not None:
                reports.append(report)
    if not reports:
        raise ValueError("no campaign declares the other workloads")
    chosen: dict[tuple[str, str], dict] = {}
    for report in reversed(reports):  # oldest first, as the charts order their clusters
        for record in report["records"]:
            key = record["workload"], record.get("other_engine", record["ippon_engine"])
            if key not in chosen or record["measured_at"] > chosen[key]["measured_at"]:
                chosen[key] = record
    return {
        **max(reports, key=lambda r: r["date"]),
        "records": list(chosen.values()),
        "sources": {p: h for report in reports for p, h in report["sources"].items()},
    }


def readme_more_tables(report: dict) -> str:
    """One table per measured workload, with standalone models and failed rivals explicit."""
    lines = [
        "# Practical workload results",
        "",
        "Speed change is other milliseconds / ippon milliseconds minus one. With several clients at once it is ippon "
        "requests/s / other requests/s minus one. Waits run from sending a request to its answer, and the 90th "
        "percentile is the nearest rank.",
        "",
    ]
    for name, (_, measure) in MORE.items():
        rows = [r for r in report["records"] if r["workload"] == name]
        if not rows:
            continue
        lines += [f"## {name}", "", measure + ".", ""]
        if name in CLIENTS:
            lines += [
                "| Model / build | Rival | ippon req/s | Rival req/s | ippon median ms | Rival median ms | "
                "ippon 90th percentile ms | Rival 90th percentile ms | Speed change |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        else:
            lines += ["| Model / build | Rival | ippon | Rival | Speed change |", "|---|---|---:|---:|---:|"]
        for r in rows:
            rival = "ippon alone" if r.get("alone") else runtime_display(r)
            if name in CLIENTS:
                ours, theirs = (r.get("clients", {}).get(arm, {}) for arm in ("ippon", "other"))
                rates = ours.get("requests_per_second"), theirs.get("requests_per_second")
                speed = percent(rates[0] / rates[1], 1) if None not in rates else "—"
                cells = [
                    fmt(arm.get(k)) for k in ("requests_per_second", "median_ms", "p90_ms") for arm in (ours, theirs)
                ]
                lines.append(f"| {r['model']} | {rival} | {' | '.join(cells)} | {speed} |")
                continue
            speed = percent(1 + r["speed_percent"] / 100, 1) if r["speed_percent"] is not None else "—"
            lines.append(f"| {r['model']} | {rival} | {fmt(r['ippon'])} | {fmt(r['other'])} | {speed} |")
        lines += [""]
        for r in rows:
            if r.get("failed"):
                lines.append(f"- {r['model']}: {r['failed']['reason']} ({r['failed']['source']}).")
        for campaign in dict.fromkeys(r["campaign"] for r in rows):
            r = next(r for r in rows if r["campaign"] == campaign)
            lines.append(
                f"- {r['date']}, {r['machine']['chip']}, ippon `{r['library_commit'][:7]}`, MLX {r['mlx']}, "
                f"source campaign: `{campaign}`."
            )
        lines += [""]
    return "\n".join(lines)


def readme_more_block(report: dict) -> str:
    """The README's charts of the other workloads, as chart.py --more draws them, with what each one sends."""
    lines = []
    sections = (
        (
            "typical",
            "A typical call",
            "One client sends about 500 tokens of text and 3 questions per call. Models without a rival "
            "show their own request times.",
        ),
        (
            "concurrent",
            "8 clients at once",
            "8 clients send ~1,000-token one-question requests at once to each runtime's own server, each "
            "sending its next request as soon as its last is answered. Libraries without a native server sit "
            "this one out.",
        ),
        (
            "media",
            "Images and audio",
            "One question about ~100 tokens of text with a 1024\u00d7768 image or a 10-second clip, a new one in "
            "every request.",
        ),
        (
            "resent",
            "A state sent again",
            "One ~1,000-token state sent again and again, as an agent re-sends its context at each step: with a new "
            "question each time, or with one more ~60-token message. Each runtime keeps what its own cache keeps.",
        ),
    )
    sections += (
        (
            "burst",
            "16 clients, short requests",
            "Sixteen clients send short one-question requests at once. "
            "The chart reports median request latency in milliseconds, including the wait.",
        ),
    )
    shown = {
        "typical": ("typical",),
        "concurrent": ("concurrent",),
        "media": ("image", "audio"),
        "resent": ("repeat", "grow"),
        "burst": ("burst",),
    }
    for name, title, text in sections:
        if not any(r["workload"] in shown[name] for r in report["records"]):
            continue
        stem = f"{README_CHARTS}-{name}"
        lines += [
            f"### {title}",
            "",
            "<picture>",
            f'  <source media="(prefers-color-scheme: dark)" srcset="{stem}-dark.svg">',
            f'  <img alt="{title}: ippon against each runtime on the same build" src="{stem}-light.svg" width="830">',
            "</picture>",
            "",
            text,
            "",
        ]
    if report["partial"]:
        lines.append(
            f"Partial: round {report['partial']['rounds']} of {report['partial']['of_rounds']} of these workloads "
            "is still running."
        )
    return "\n".join(lines)


def readme_pitch(report: dict) -> str:
    """Name the build behind the opening latency: the quickest measured creator GGUF build."""
    candidates = [r for r in report["pairs"] if r["group"] == "llama.cpp"]
    opening = "ippon is an inference engine for decision models on your Mac, built for Apple silicon"
    if not candidates:
        return opening + "."
    record = min(candidates, key=lambda r: r["cells"][0]["ippon_ms"])
    return (
        f"{opening}: {record['cells'][0]['ippon_ms']:,.0f} ms for a "
        f"{record['model'].replace(' · ', ' ')} decision on {record['machine']['chip']}."
    )


def compared(ratio: float, rate: bool = False) -> str:
    """How the other runtime compares with ippon, in the terms a reader feels, with no subject (the caller names it).
    `ratio` is ippon's lead: their milliseconds over ippon's, or with `rate` ippon's requests per second over theirs.
    A time is the other runtime's: "24% longer" up to twice ippon's, "3.2x as long" beyond, "12% less time" where it
    is quicker. A rate is ippon's: "44% more answers a second", "2.6x the answers a second", "12% fewer". Never
    "faster" or "slower", which read as a cut in waiting: 24% longer is 19% less waiting with ippon, not 24%."""
    if round((ratio - 1) * 100) == 0:
        return "about the same time" if not rate else "about as many answers a second"
    times = (f"{ratio:.1f}" if ratio < 10 else f"{ratio:.0f}").removesuffix(".0")
    if rate:
        if ratio >= 2:
            return f"{times}x the answers a second"
        return f"{abs(ratio - 1) * 100:.0f}% {'more' if ratio > 1 else 'fewer'} answers a second"
    if ratio >= 2:
        return f"{times}x as long"
    return f"{(ratio - 1) * 100:.0f}% longer" if ratio > 1 else f"{(1 - ratio) * 100:.0f}% less time"


def long_date(text: str) -> str:
    """A recorded date or date range ("2026-10-29", "2026-10-29 to 2026-10-30") as prose: 29 and 30 October 2026."""
    days = sorted({dt.date.fromisoformat(day) for part in text.split(", ") for day in part.split(" to ")})
    first, last = days[0], days[-1]
    if first == last:
        return f"{first.day} {first:%B} {first.year}"
    if (first.month, first.year) == (last.month, last.year):
        joined = " and " if len(days) == 2 and (last - first).days == 1 else " to "
        return f"{first.day}{joined}{last.day} {last:%B} {last.year}"
    return f"{first.day} {first:%B} to {last.day} {last:%B} {last.year}"


def rate(value: float) -> str:
    return f"{value:.2f}" if value < 10 else f"{value:.1f}"


def bare(runtime: str) -> str:
    """A runtime as the README and its charts name it, without its version: the versions are in the docs."""
    return re.sub(r"\s+(v?\d[\w.\-]*|b\d+)$", "", runtime)


def _listed(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def readme_lead(more: dict | None) -> str:
    """A table of the agent's loop and the typical call against Ollama, on the same builds, from the newest recorded
    results: one build per model, Ollama's default where it was timed (d1's BF16), else its MLX build."""
    if not more:
        return ""
    order = list(README_ORDER)
    workloads = ("grow", "typical")
    chosen: dict[str, dict[str, dict]] = {}
    for r in more["records"]:
        if r["workload"] in workloads and r.get("group") == "ollama" and r.get("speed_percent") is not None:
            held = chosen.setdefault(family(r["model"]), {}).get(r["workload"])
            if held is None or ("BF16" in r["model"] and "BF16" not in held["model"]):
                chosen[family(r["model"])][r["workload"]] = r
    if not chosen:
        return ""
    models = sorted(chosen, key=lambda f: order.index(f) if f in order else len(order))[:HEADLINE_MODELS]
    runtime = runtime_display(next(iter(chosen[models[0]].values())))

    def cell(r: dict | None) -> str:
        name = runtime.split()[0]  # "Ollama" of "Ollama 0.40.13", which the line above the table names in full
        return (
            ""
            if r is None
            else f"{r['ippon']:,.0f} ms against {r['other']:,.0f}, {name} takes {compared(r['other'] / r['ippon'])}"
        )

    lines = [
        f"ippon against {runtime} on the same builds, in an agent's loop (the same ~1,000-token context sent again "
        "with one more message) and in a typical call (about 500 tokens, 3 questions).",
        "",
        "| Model, build | Agent's loop | Typical call |",
        "|---|---|---|",
    ]
    for f in models:
        build = next(iter(chosen[f].values()))["model"].partition(" · Ollama ")[2] or "Ollama's build"
        lines.append(f"| {f}, {build} | {cell(chosen[f].get('grow'))} | {cell(chosen[f].get('typical'))} |")
    return "\n".join(lines)


def readme_headline(report: dict) -> str:
    """The agent's loop and the typical call against Ollama, a concrete model's milliseconds against llama.cpp on
    the same file, and the Mac and dates behind every chart."""
    record = next((r for r in report["pairs"] if r["group"] == "llama.cpp"), None)
    lead = readme_lead(report.get("more"))
    paragraphs = []
    if record is not None:
        cell = record["cells"][0]
        paragraphs.append(
            f"{family(record['model'])} answers one question about ~1,000 tokens in {cell['ippon_ms']:,.0f} ms on "
            f"ippon, where {runtime_display(record)} takes {compared(cell['ratio'])} ({cell['other_ms']:,.0f} ms) on "
            "the same Q8_0 file."
        )
    machine = report["machine"]
    dates = [c["date"] for c in report.get("campaigns", [report])]
    dates += [r["date"] for r in (report.get("more") or {}).get("records", [])]
    paragraphs.append(
        f"Every result here was measured on an {machine['chip']} with {machine['memory_bytes'] / 2**30:g} GB, "
        f"macOS {machine['macos']} and MLX {report['mlx']}, on {long_date(', '.join(dates))}."
    )
    text = " ".join(paragraphs)
    if lead:
        text = lead + "\n\n" + text
    if partial := report.get("partial"):
        text += "\n\n" + readme_partial(report, partial)
    return text


def readme_partial(report: dict, partial: dict) -> str:
    """Label the finished pairs of an unfinished campaign until the complete round replaces them."""
    busy, total = report["busy_requests"]
    return (
        f"Partial: round {partial['rounds']} of {partial['of_rounds']}, {partial['pairs']} of {partial['of']} "
        f"model/runtime pairs so far, and the rest are still running. {busy} of {total} timed requests overlapped "
        "another busy process and are included."
    )


def alone_title(report: dict) -> str:
    """Title the chart only with the measured models' names and runtime."""
    return _listed([r["model"] for r in report["alone"]]) + " on ippon"


def measurement_note(record: dict) -> str:
    """The date and environment of this record's own selected campaign."""
    machine = record["machine"]
    return (
        f"Measured {record['date']} on {machine['chip']}, {machine['memory_bytes'] / 2**30:g} GiB, "
        f"macOS {machine['macos']}, MLX {record['mlx']}, ippon commit `{record['library_commit']}`."
    )


def recorded_campaigns(root: Path = ROOT / "benchmarks/results") -> tuple[list[Path], list[Path]]:
    """The accepted rounds of every README campaign recorded so far: those with the text workload, and those with
    the other workloads. A campaign not recorded yet, or with no accepted round, is left out."""
    text, more = [], []
    for name in README_CAMPAIGNS:
        folder = root / name
        rounds = sorted(p for p in folder.glob("r[0-9]*") if p.is_dir() and p.name[1:].isdigit())
        if not rounds or not (folder / "CAMPAIGN.json").is_file():
            continue
        workloads = {w["name"] for w in json.loads((folder / "CAMPAIGN.json").read_text())["workloads"]}
        if "text" in workloads:
            text += rounds
        if workloads & MORE.keys():
            more += rounds
    return text, more


def short_runtime(rival: dict) -> str:
    """A rival as a chart's column names it; the creators' library versions go in the chart's notes."""
    if rival["group"] == "maker":
        return str(rival["runtime"]).split(" · ")[0]
    return str(rival["runtime"]).replace(" (setup pin)", "")


def same_build(ippon_engine: str) -> str:
    return next((label for part, label in SAME_BUILD if part in ippon_engine), "same checkpoint")


# mlx-vlm runs one decision model, a community Clef-Flash build through that build's own loader, since it has no
# decision API: no rival for the other models. Its result stays in docs/benchmarks.md, out of the README's charts.
CHARTS_LEAVE_OUT = ("mlx",)


def chart_models(report: dict) -> list[dict]:
    """Every model with a measurement, in README_ORDER: its panels in PANELS order, each a list of clusters, one per
    ippon measurement (a build in one campaign) with the rivals measured against it. A failed rival, a cluster left
    without a rival and a panel or model with nothing measured are left out; models only ippon runs keep ippon alone."""
    entries = []
    for record in report.get("pairs", []):
        for key, cell in zip(("one", "fifty"), record["cells"], strict=True):
            entries.append((key, record, cell["ippon_ms"], cell["other_ms"], None))
    for record in report.get("alone", []):
        for key, cell in zip(("one", "fifty"), record["cells"], strict=True):
            entries.append((key, record, cell["ippon_ms"], None, None))
    for record in (report.get("more") or {}).get("records", []):
        if record.get("failed"):
            continue
        clients = record.get("clients")
        if record["workload"] in CLIENTS:
            ours, theirs = (
                clients["ippon"]["requests_per_second"],
                (clients.get("other") or {}).get("requests_per_second"),
            )
        else:
            ours, theirs = record["ippon"], record["other"]
        entries.append((record["workload"], record, ours, theirs, clients))
    pairs_order = [other for *_, other, _ in readme_pairs()]
    builds_order = [part for part, _ in SAME_BUILD]
    models: dict[str, dict] = {}
    for key, record, ours, theirs, clients in entries:
        if record.get("group") in CHARTS_LEAVE_OUT:
            continue
        name = family(record["model"])
        model = models.setdefault(name, {"name": name, "slug": slug(name), "panels": {}, "records": []})
        model["records"].append(record)
        panel = model["panels"].setdefault(key, {})
        engine = record["ippon_engine"]
        cluster = panel.setdefault(  # one ippon bar per measured value: rivals matched on other requests get their own
            (engine, record["campaign"], ours),
            {
                "build": None if theirs is None else same_build(engine),
                "date": record["date"],
                "ippon": ours,
                "ippon_clients": (clients or {}).get("ippon"),
                "rivals": [],
                "order": (
                    next((i for i, part in enumerate(builds_order) if part in engine), len(builds_order)),
                    record["date"],
                ),
            },
        )
        if theirs is not None:
            per_second = key in CLIENTS
            cluster["build"] = same_build(engine)
            cluster["rivals"].append(
                {
                    "runtime": runtime_display(record),
                    "group": record["group"],
                    "value": theirs,
                    "ratio": ours / theirs if per_second else theirs / ours,
                    "clients": (clients or {}).get("other"),
                    "order": pairs_order.index(record["other_engine"]),
                }
            )
    out = []
    for model in models.values():
        alone = not any(c["rivals"] for panel in model["panels"].values() for c in panel.values())
        panels = []
        for key, title, detail, unit in PANELS:
            clusters = [c for c in model["panels"].get(key, {}).values() if alone or c["rivals"]]
            for cluster in clusters:
                cluster["rivals"].sort(key=lambda r: r["order"])
            if clusters:
                clusters.sort(key=lambda c: c["order"])
                panels.append({"key": key, "title": title, "detail": detail, "unit": unit, "clusters": clusters})
        if panels:
            out.append({**model, "alone": alone, "panels": panels})
    rank = {name: i for i, name in enumerate(README_ORDER)}
    return sorted(out, key=lambda m: rank.get(m["name"], len(rank)))


def wait_sentence(model: dict) -> str:
    """One plain sentence for a chart whose rival has the lower median wait with several clients at once: a median
    alone misleads when a server answers some clients at once and makes the others wait, so the sentence gives
    that rival's 90th-percentile wait and requests per second beside ippon's."""
    cases: dict[tuple[str, str], list[tuple[str, dict, dict]]] = {}
    for panel in model["panels"]:
        if panel["key"] not in CLIENTS:
            continue
        clients = "8" if panel["key"] == "concurrent" else "16"
        for cluster in panel["clusters"]:
            ours = cluster["ippon_clients"]
            for rival in cluster["rivals"]:
                theirs = rival["clients"]
                if ours and theirs and theirs["median_ms"] < ours["median_ms"]:
                    cases.setdefault((rival["runtime"], cluster["build"]), []).append((clients, ours, theirs))
    sentences = []
    for (runtime, _), found in cases.items():
        ms = {
            (arm, key): _listed([f"{case[i][key]:,.0f}" for case in found])
            for i, arm in ((1, "ippon"), (2, "other"))
            for key in ("median_ms", "p90_ms")
        }
        per_second = {
            arm: _listed([rate(case[i]["requests_per_second"]) for case in found])
            for i, arm in ((1, "ippon"), (2, "other"))
        }
        counts = " and with ".join(clients for clients, _, _ in found)
        tail = all(theirs["p90_ms"] > ours["p90_ms"] for _, ours, theirs in found)
        waits = "waits are" if len(found) > 1 else "wait is"
        sentences.append(
            f"With {counts} clients at once, {bare(runtime)}'s median wait is lower than ippon's, at "
            f"{ms['other', 'median_ms']} ms against {ms['ippon', 'median_ms']} ms."
            + (" It answers some clients at once and makes others wait, so its" if tail else " Its")
            + f" 90th-percentile {waits} {ms['other', 'p90_ms']} ms against ippon's {ms['ippon', 'p90_ms']} ms. "
            f"It answers {per_second['other']} requests per second, and ippon answers {per_second['ippon']}."
        )
    return " ".join(sentences)


def failed_rivals(report: dict) -> list[dict]:
    """The other workloads' records whose rival failed timed requests while ippon answered every request, which the
    charts leave out (`chart_models`)."""
    return [r for r in (report.get("more") or {}).get("records", []) if r.get("failed") and r["ippon"] is not None]


# A README panel's workload as the sentence under a chart names what a rival failed at (`failure_sentence`)
FAILED_AT = {
    "typical": "a typical call",
    "repeat": "the same context again",
    "grow": "a growing context",
    "burst": "16 clients at once",
    "image": "an image",
    "pictures": "growing images",
    "image-again": "the same image again",
    "video": "a video",
    "video-again": "the same video again",
    "audio": "an audio clip",
}


def failure_sentence(report: dict, model: dict) -> str:
    """One plain sentence for each README workload of a model's chart whose rival failed timed requests while ippon
    answered every request: the chart leaves that pair out, so the sentence says what the rival did, counted."""
    sentences = []
    for r in failed_rivals(report):
        if family(r["model"]) == model["name"] and r["workload"] in FAILED_AT:
            name = bare(short_runtime({"group": r["group"], "runtime": runtime_display(r)}))  # as the charts name it
            mac = f"{r['machine']['memory_bytes'] / 2**30:g} GB {r['machine']['chip']}"
            sentences.append(
                f"On the {mac}, with {FAILED_AT[r['workload']]}, {name} {failed_requests(r['failed'], 'timed')}, "
                "while ippon answered every request."
            )
    return " ".join(sentences)


def readme_chart_models(report: dict) -> list[dict]:
    """The README's model charts: each model measured against a rival, with its README panels only."""
    out = []
    for model in chart_models(report):
        if model["alone"]:
            continue
        panels = [p for p in model["panels"] if p["key"] in README_PANELS]
        shown = {p["key"] for p in panels}  # the notes name only the campaigns of the panels shown
        if panels:
            out.append(
                {**model, "panels": panels, "records": [r for r in model["records"] if r.get("workload") in shown]}
            )
    return out


def section(model: dict) -> str:
    """The Speed section a model's chart goes in: the inputs it was timed on, joined (SECTIONS)."""
    keys = {p["key"] for p in model["panels"]}
    inputs = "-".join(name for name, panels in INPUTS if keys & set(panels))
    if inputs not in dict(SECTIONS):
        raise ValueError(f"{model['name']} is timed on {inputs}, which has no Speed section: add it to SECTIONS")
    return inputs


def readme_section_models(report: dict) -> list[tuple[str, str, list[dict]]]:
    """Each Speed section's charts: (section key, heading, its models' charts), as SECTIONS orders them."""
    models = readme_chart_models(report)
    return [(key, title, [m for m in models if section(m) == key]) for key, title in SECTIONS]


def alone_table(report: dict, section_key: str) -> str:
    """The models only ippon runs that belong in this section, as one table of their workloads instead of a chart
    each."""
    alone = []
    for model in chart_models(report):
        panels = [p for p in model["panels"] if p["key"] in README_PANELS]
        if model["alone"] and panels and section({**model, "panels": panels}) == section_key:
            alone.append({**model, "panels": panels})
    keys = [key for key, *_ in PANELS if any(p["key"] == key for m in alone for p in m["panels"])]
    if not keys:
        return ""
    titles = {key: (title, unit) for key, title, _, unit in PANELS}
    lines = [
        "| Model | " + " | ".join(titles[k][0] for k in keys) + " |",
        "|---|" + "---:|" * len(keys),
    ]
    for model in alone:
        cells = []
        for key in keys:
            panel = next((p for p in model["panels"] if p["key"] == key), None)
            value = panel["clusters"][0]["ippon"] if panel else None
            unit = titles[key][1]
            cells.append("" if value is None else f"{rate(value)} req/s" if unit == "req/s" else f"{value:,.0f} ms")
        lines.append(f"| {model['name']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def readme_charts(report: dict, section_key: str | None = None) -> str:
    """One Speed section's charts (`section_key`, every section's in turn if None): each model measured against a
    rival, folded under a line naming its rivals with the sentences its numbers need, then one table of the models
    only ippon runs."""
    lines = []
    for key, _, models in readme_section_models(report):
        if section_key not in (None, key):
            continue
        for model in models:
            stem = f"{README_CHARTS}-{model['slug']}"
            runtimes = sorted(
                {bare(short_runtime(r)) for p in model["panels"] for c in p["clusters"] for r in c["rivals"]}
            )
            workloads = _listed([p["title"].lower() for p in model["panels"]])
            named = escape_attr(_listed([bare(r) for r in runtimes]))
            alt = f"{model['name']}: ippon beside {_listed(runtimes)} on the same builds, for {workloads}"
            lines += [
                "<details>",
                f"<summary><b>{model['name']}</b> - ippon beside {named}</summary>",
                "",
                "<picture>",
                f'  <source media="(prefers-color-scheme: dark)" srcset="{stem}-dark.svg">',
                f'  <img alt="{escape_attr(alt)}" src="{stem}-light.svg" width="830">',
                "</picture>",
                "",
            ]
            if sentence := wait_sentence(model):
                lines += [sentence, ""]
            if sentence := failure_sentence(report, model):
                lines += [sentence, ""]
            if model["name"] == "pplx-decider flash v1.3":
                lines += [
                    "These timings use the full BF16 checkpoint. Perplexity publishes no GGUF or MLX build of flash "
                    "v1.3, so its own package is the other runtime.",
                    "",
                ]
            lines += ["</details>", ""]
        if table := alone_table(report, key):
            lines += ["No other Mac runtime runs these models, so they are timed on ippon alone.", "", table, ""]
    return "\n".join(lines).rstrip("\n")


def escape_attr(text: str) -> str:
    return text.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def stopping_rules(campaigns: list[dict]) -> list[str]:
    """One paragraph per stopping rule, naming the ippon commits whose rounds used it."""
    rules: dict[str, list[str]] = {}
    for campaign in campaigns:
        config = campaign["config"]
        if not config.get("min_n"):
            continue
        rule = (
            f"a shape stops once its median's 95% interval is within ±{config['precision']:.0%}, after at least "
            f"{config['min_n']} and at most {config['n']} timed requests, or after {config['cell_seconds']:g} s."
        )
        if campaign.get("close"):
            rule += (
                f" Pairs within {campaign['close']:.0%} in the first round, or with overlapping intervals, can have "
                "another round in the opposite order."
            )
        commit = f"`{campaign['library_commit'][:7]}`"
        commits = rules.setdefault(rule, [])
        if commit not in commits:
            commits.append(commit)
    return [
        line for rule, commits in rules.items() for line in (f"In the rounds with ippon {_listed(commits)}, {rule}", "")
    ]


def readme_tables(report: dict) -> str:
    """Render both metrics, agreement and provenance without hiding slower or unequal-answer comparisons."""
    lines = [
        "# README benchmark round",
        "",
        *dict.fromkeys(measurement_note(c) for c in report.get("campaigns", [report])),
        "",
        *([readme_partial(report, report["partial"]), ""] if report.get("partial") else []),
        "Both workloads send the same nominal ~1,000-token text. A has one yes/no question, and B has 50 mixed "
        "yes/no, choice and score questions. Latency includes client encoding, HTTP, model execution and "
        "response decoding. Each runtime is already loaded, and warm-up requests are untimed.",
        "",
        *stopping_rules(report.get("campaigns", [report])),
        "Each pair uses its newest supplied campaign with complete results, and only that campaign's paired rounds "
        "are pooled. A timed request that overlapped another busy process is set aside in its "
        "result and the next generated request of that shape is timed instead, at most n times per shape. Past "
        "that, the busy requests stay in the run and are counted. No observation is dropped for its time. "
        "Speed change is "
        "`(other median / ippon median \u2212 1) \u00d7 100%`, and it is not a reduction in latency. "
        "Both chart panels show milliseconds, and a 50-question request takes `50,000 / questions per second` ms. "
        "Each rival bar has ippon's time on the same build directly above it. Percentages use unrounded times, and "
        "negative values are shown as slower. The charts of 8 and 16 clients at once use requests per second, "
        "where higher is better.",
        "",
        "## One-question latency",
        "",
        "| Model / build | Other runtime | Other ms | ippon ms | Speed change | "
        "Top answers agree | Max probability gap |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for record in report["pairs"]:
        cell = record["cells"][0]
        agree = cell["agreement"]
        lines.append(
            f"| {record['model']} | {runtime_display(record)} | {fmt(cell['other_ms'])} | {fmt(cell['ippon_ms'])} | "
            f"{percent(cell['ratio'])} | {agree['questions'] - agree['top_answer_flips']}/{agree['questions']} | "
            f"{agree['max_probability_difference']:.6f} |"
        )
    lines += [
        "",
        "## 50-question request time",
        "",
        "| Model / build | Other runtime | Other ms | ippon ms | Other questions/s | ippon questions/s | "
        "Speed change | Top answers agree | Max probability gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in report["pairs"]:
        cell = record["cells"][1]
        agree = cell["agreement"]
        lines.append(
            f"| {record['model']} | {runtime_display(record)} | {fmt(cell['other_ms'])} | {fmt(cell['ippon_ms'])} | "
            f"{fmt(cell['other_questions_per_second'])} | {fmt(cell['ippon_questions_per_second'])} | "
            f"{percent(cell['ratio'])} | {agree['questions'] - agree['top_answer_flips']}/{agree['questions']} | "
            f"{agree['max_probability_difference']:.6f} |"
        )
    lines += [
        "",
        "Matching top answers do not establish numerical parity, and the maximum probability gap is an "
        "absolute difference. These comparisons measure runtime agreement, not task accuracy.",
        "",
        "## Request paths and sources",
        "",
    ]
    for record in report["pairs"]:
        transports = sorted({source["transport"] for source in record["other_sources"]})
        methods = sorted({source["method"] for source in record["other_sources"]})
        lines += [
            f"### {record['model']} against {runtime_display(record)}",
            "",
            f"The other runtime over {', '.join(transports)}, {' and '.join(methods)}, and ippon over native HTTP. "
            f"{record['rounds']} paired round{'s' if record['rounds'] != 1 else ''}, "
            f"{_sample_range(record)} timed requests per arm per workload.",
            "",
            measurement_note(record),
            "",
        ]
        for cell in record["cells"]:
            spread = ", ".join(percent(1 + value / 100, 1) for value in cell["round_speed_percent"])
            noun = "question" if cell["questions"] == 1 else "questions"
            lines.append(f"- {cell['questions']} {noun}, per-round speed change {spread}.")
        lines += evidence_links(record["other_sources"] + record["ippon_sources"])
        lines.append("")
    if report.get("alone"):
        lines += [
            f"## {alone_title(report)}",
            "",
            "| Model | 1 question, ms | 50 questions, ms | Requests per median |",
            "|---|---:|---:|---:|",
        ]
        for record in report["alone"]:
            one, fifty = record["cells"]
            lines.append(
                f"| {record['model']} | {fmt(one['ippon_ms'])} | {fmt(fifty['ippon_ms'])} | {_sample_range(record)} |"
            )
        for record in report["alone"]:
            lines += ["", f"{record['model']} - {measurement_note(record)}"]
            lines += evidence_links(record["ippon_sources"])
        lines.append("")
    if report.get("pending_pairs") or report.get("pending_alone"):
        lines += ["## Not yet measured", "", "These have no complete results in the supplied campaigns.", ""]
        lines += [f"- {model}" for model in dict.fromkeys(report.get("pending_pairs", {}).values())]
        lines += [f"- {name} on ippon" for name in report.get("pending_alone", [])]
        lines.append("")
    warmed = [r for r in (report.get("more") or {}).get("records", []) if r.get("warmup_errors")]
    if report.get("failed") or failed_rivals(report) or warmed:
        titles = {key: f"{title.lower()} ({what})" for key, title, what, _ in PANELS}
        lines += ["## Runtimes that failed requests", ""]
        lines += [
            f"- {record['model']} with {runtime_display(record)} - {record['reason']}. "
            f"[Recorded failure](https://github.com/sqwishai/ippon/blob/main/{record['source']})."
            for record in report.get("failed", [])
        ]
        lines += [
            f"- {r['model']} with {runtime_display(r)} - {failed_requests(r['failed'], 'timed', 'timed ')}, "
            f"{titles[r['workload']]}. [Recorded run](https://github.com/sqwishai/ippon/blob/main/{r['failed']['source']})."
            for r in failed_rivals(report)
        ]
        lines += [
            f"- {r['model']} with {runtime_display(r)} - {failed_requests(r['warmup_errors'], 'warmup', 'warm-up ')} "
            f"and then answered every timed request, {titles[r['workload']]}. "
            f"[Recorded run](https://github.com/sqwishai/ippon/blob/main/{r['warmup_errors']['source']})."
            for r in warmed
        ]
        lines.append("")
    if report["excluded_pairs"]:
        lines += ["## Not measured", ""]
        lines += [f"- `{engine}`: {reason}" for engine, reason in report["excluded_pairs"].items()]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folders", nargs="*", type=Path)
    parser.add_argument(
        "--recorded",
        action="store_true",
        help="with --readme-round: every README campaign recorded so far (README_CAMPAIGNS), the text rounds and "
        "the other workloads' rounds, besides any folders given",
    )
    parser.add_argument("--more-only", action="store_true", help="report additional workloads without text rows")
    parser.add_argument("--out", type=Path, help="write the complete Markdown summary")
    parser.add_argument("--ledger", type=Path, help="write source hashes, metadata and unrounded calculations as JSON")
    parser.add_argument("--docs", type=Path, help="update runtime and creator blocks in the benchmark documentation")
    parser.add_argument(
        "--readme",
        type=Path,
        help="update the compact README headline and, for a fresh round, its measured pitch and model charts",
    )
    parser.add_argument(
        "--readme-round",
        action="store_true",
        help="select complete 1/50-question HTTP pairs from their newest supplied campaigns",
    )
    parser.add_argument(
        "--partial",
        action="store_true",
        help="with --readme-round: the finished pairs of unfinished attempt-rN folders",
    )
    parser.add_argument(
        "--more",
        nargs="+",
        type=Path,
        default=[],
        help="with --readme-round --readme: the rounds of the other workloads' campaign (attempt-rN: partial)",
    )
    parser.add_argument(
        "--older",
        nargs="+",
        type=Path,
        default=[],
        help="with --readme-round: older campaign folders whose one-question times chart the pairs not yet measured",
    )
    parser.add_argument(
        "--exclude-pair",
        action="append",
        default=[],
        metavar="OTHER_ENGINE=REASON",
        help="document an unsupported pair instead of silently dropping missing evidence",
    )
    args = parser.parse_args()
    if args.recorded:
        if not args.readme_round:
            parser.error("--recorded needs --readme-round")
        text, more_rounds = recorded_campaigns()
        args.folders = [*text, *args.folders]
        args.more = [*more_rounds, *args.more]
    if not args.folders:
        parser.error("give result folders, or --readme-round --recorded")
    if args.more_only:
        try:
            report = readme_more(args.folders)
        except ValueError as exc:
            parser.error(str(exc))
        if args.out:
            args.out.write_text(readme_more_tables(report))
        else:
            print(readme_more_tables(report))
        if args.ledger:
            args.ledger.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        return
    if args.readme_round:
        try:
            report = readme_comparisons(args.folders, exclusions(args.exclude_pair), args.partial)
            more = readme_more(args.more) if args.more else None
        except ValueError as exc:
            parser.error(str(exc))
        if args.older:
            report["older"] = readme_older(args.older, report)
        if more:
            report["more"] = more
        summary = readme_tables(report) + ("\n" + readme_more_tables(more) if more else "")
        if args.out:
            args.out.write_text(summary)
        if args.docs:
            section = re.sub(r"^(#+) ", r"#\1 ", readme_tables(report), flags=re.MULTILINE)
            section = section.replace("## README benchmark round", "## Current README round", 1)
            fill(args.docs, "fresh", section)
        if args.ledger:
            args.ledger.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        for page in (p for p in (args.readme, args.docs) if p):  # the table against Ollama: in the docs, and in
            if BLOCKS["headline"][0] in page.read_text():  # the README only if it still has the block
                fill(page, "headline", readme_headline(report))
        if args.readme:
            if BLOCKS["pitch"][0] in args.readme.read_text():
                fill(args.readme, "pitch", readme_pitch(report))
            for key, _ in SECTIONS:
                if BLOCKS[f"charts-{key}"][0] in args.readme.read_text():
                    fill(args.readme, f"charts-{key}", readme_charts(report, key))
            if more and BLOCKS["more"][0] in args.readme.read_text():
                fill(args.readme, "more", readme_more_block(more))
        if not args.out and not args.readme:
            print(summary)
        return
    if missing := [str(folder) for folder in args.folders if not folder.is_dir()]:
        parser.error("no such result folder: " + ", ".join(missing))
    records = comparisons(args.folders)
    if not records:
        parser.error("no paired runs in these folders: give a campaign's rounds (round/r1 round/r2) or --readme-round")
    runtime = tables([r for r in records if r["group"] != "maker"], "Recorded runtime comparisons")
    maker = tables([r for r in records if r["group"] == "maker"], "Older creators' code comparison")
    text = "\n".join(["# Recorded benchmark summary", "", *runtime, *maker, *details(records)]).rstrip() + "\n"
    if args.out:
        args.out.write_text(text)
    if args.ledger:
        args.ledger.write_text(
            json.dumps(
                {
                    "speed_definition": "(other_median / ippon_median - 1) * 100",
                    "input_folders": [portable(folder) for folder in args.folders],
                    "generator_sources": {
                        portable(path): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (
                            Path(__file__),
                            Path(__file__).with_name("chart.py"),
                            Path(__file__).with_name("setup.sh"),
                        )
                    },
                    "pairs": records,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
    if args.docs:
        for name, lines in (("runtime", runtime), ("maker", maker)):
            if lines:
                fill(args.docs, name, "\n".join(lines))
    if args.readme and (text_headline := headline(records)):
        fill(args.readme, "headline", text_headline)
    if not args.out and not args.docs and not args.readme:
        print(text)


if __name__ == "__main__":
    main()
