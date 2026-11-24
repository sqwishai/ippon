#!/usr/bin/env python3
"""Reference answers for Base Labs' elephant-1 from its creator's code: the file `ippon verify` checks..

The creator's runtime is OneJev's `qev` at 81ce62f1597c91e46767d4d02d6ac2e18534fe94 with the checkpoint's serve.patch
applied (what the checkpoint's setup.sh installs and its engine.py and model card call). This script never installs
it: --onejev names the patched checkout, which goes first on sys.path. Run it outside ippon, on the CPU only:

    python -I elephant_reference.py --onejev DIR --model ~/models/elephant-1 \\
        --requests ../requests/standard.json --out elephant-1.json.gz \\
        --repo baselabs/elephant-1 --revision b3f8a93462fb1051a9de031e94a7ff380e254072 [--tokens-only] [--bf16]

Text requests run the creator's serving engine, qev.engine.DecisionEngine (text weights only, Qwen3_5ForCausalLM), in
fp32 on the CPU (about 36 GB of weights): the block-bidirectional mask comes from config.json's modeling_overrides
(qev/readout.py), the readout is the LM-head logits of the slot letters at the last token, the temperature is
calibration.json's (none ships at b3f8a93: 1.0). Recorded per request: every question's prompt (shared prefix +
its suffix, token ids), the prefix length (the attention block boundary), the creator's token count (prefix once plus
every suffix), and each question's probabilities softmax(slot logits / temperature) unrounded, in the request's
option order (a noul as true, false: the creator's slots A = yes, B = no, so no reordering), next to the creator's own
rounded answers. --bf16 runs the same engine again in its production dtypes (bfloat16 weights, float32 head), on the
Mac's GPU (--bf16-device), for the bound: the larger of 0.01 and bf16's largest distance from fp32 plus 0.0005.

With --media DIR, image requests run instead (a separate process: the vision model in fp32 is about 38 GB):

    python -I make_reference.py --onejev DIR --model ~/models/elephant-1 \\
        --requests ../requests/media-elephant.json --media ../media/images --place-images state-start \\
        --out elephant-1-media.json.gz --repo baselabs/elephant-1 --revision b3f8a93... [--tokens-only] [--bf16]

qev's served multimodal engine (qev/mm_engine.py) is not touched by serve.patch and keeps a causal mask, so the images
go through the path the checkpoint was trained and evaluated on: train.common.batch_scores with the readout attached
as train/eval_mm.py attaches it (block ids over the prompt, the image rows inside the prefix block), one forward per
question, with the processor's own pixel range (65,536 to 16,777,216) and train/mm.py's decode (no EXIF turn, alpha
dropped). An image is referenced in the state by `<image:N>`, once each, in order (qev/media.py: the creator refuses
anything else, which is recorded); a state that names none gets, with --place-images state-start, the placeholders at
the start of its text, a line before the rest (ippon's convention, recorded in the entry's notes). Recorded per
request: each question's prompt (the processor's input ids, every <|image_pad|> expanded to its image's rows), the
block boundary (the text prefix's tokens plus the images' extra rows, as batch_scores places it), the token count,
the image grids and the sha256 of the pixel values, and each question's probabilities. --tokens-only stops before
the weights; --bf16 runs again in qev's production dtypes (bfloat16 weights, the LM head in float32) on --bf16-device,
for the bound. Needs Pillow and, for the creator's image processor backend, torchvision.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from runpy import run_path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # the reference is the CPU's

SCHEMA = "ippon-reference/1"
ADAPTER = "baselabs/elephant"
ONEJEV_COMMIT = "81ce62f1597c91e46767d4d02d6ac2e18534fe94"


def serializer(ippon: Path | None):
    """ippon's stdlib-only serializer when its checkout is at hand (portable paths), else json.dumps."""
    here = Path(__file__).resolve()
    candidates = [ippon / "src/ippon/_portable.py"] if ippon is not None else []
    if len(here.parents) > 2:  # placed at ippon/references/<adapter>/
        candidates.append(here.parents[2] / "src/ippon/_portable.py")
    for path in candidates:
        if path.is_file():
            return run_path(str(path))["dumps"]
    return json.dumps


def onejev(root: Path) -> str:
    """Put the patched checkout first on sys.path; refuse an unpatched or different one."""
    if not (root / "qev" / "readout.py").is_file():
        sys.exit(f"{root}: qev/readout.py is missing, so serve.patch is not applied")
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    if head != ONEJEV_COMMIT:
        sys.exit(f"{root}: at {head or 'an unknown commit'}, not {ONEJEV_COMMIT}")
    sys.path.insert(0, str(root))
    return head


def maker_body(request: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """The request as qev.schema takes it. A choice given as a list of labels (an ippon extension; qev wants a map)
    is asked as the map of those labels without descriptions, which is recorded."""
    body = json.loads(json.dumps(request))
    notes = []
    for qid, q in (body.get("questions") or {}).items():
        if isinstance(q, dict) and q.get("type") == "choice" and isinstance(q.get("criteria"), list):
            q["criteria"] = dict.fromkeys(q["criteria"])
            notes.append(f"{qid}: choice labels given as a list, asked as a map without descriptions")
    body.pop("images", None)
    return body, notes


def tokenizer_engine(model: Path):
    """A DecisionEngine with only what prompt building needs (no weights): its tokenizer, template switches, slot
    check and readout config, set as DecisionEngine.__init__ sets them (qev/engine.py:93, 104-112)."""
    import transformers
    from qev import readout as ro
    from qev.engine import DecisionEngine
    from qev.prompt import DEFAULT_STYLE

    engine = DecisionEngine.__new__(DecisionEngine)
    engine.model_path = str(model)
    common = {"local_files_only": True, "trust_remote_code": False}
    config = transformers.AutoConfig.from_pretrained(str(model), **common)
    engine.tokenizer = transformers.AutoTokenizer.from_pretrained(str(model), **common)
    engine.style = DEFAULT_STYLE
    engine.model_type = config.get_text_config().model_type
    engine.template_kwargs = {"enable_thinking": False} if "qwen" in engine.model_type else {}
    engine.slot_ids = engine._verify_slots()
    engine.readout = ro.Readout(ro.ReadoutConfig.load(str(model)))  # config only: nothing is installed
    engine.max_branch_tokens, engine.max_request_tokens = 32768, 65536  # DecisionEngine's defaults
    return engine


def plan(engine, request: dict[str, Any], entry: dict[str, Any]):
    """The creator's branches and token layout for one request, or None when the creator refuses it (recorded)."""
    from pydantic import ValidationError
    from qev.prompt import render_question
    from qev.schema import SystemOneRequest

    body, notes = maker_body(request)
    if notes:
        entry["notes"] = notes
    try:
        req = SystemOneRequest.model_validate(body)
    except ValidationError as error:
        entry["error"] = "qev.schema: " + "; ".join(e["msg"] for e in error.errors())
        return None
    rendered = [render_question(qid, q, style=engine.style) for qid, q in req.questions.items()]
    if any(r.n_slots > engine.max_slots for r in rendered):  # never with 256 slots and at most 255 options
        entry["error"] = "a choice over the slot count: the creator's chunked path, not reproduced here"
        return None
    try:
        prefix, suffixes, _ = engine._split(req.state, rendered)  # the patched split, with its block checks
    except ValueError as error:
        entry["error"] = f"qev layout: {error}"
        return None
    size = len(prefix)
    longest, total = size + max(len(s) for s in suffixes), size + sum(len(s) for s in suffixes)
    if longest > engine.max_branch_tokens:
        entry["error"] = f"state plus longest question is {longest} tokens, limit is {engine.max_branch_tokens}"
        return None
    if total > engine.max_request_tokens:
        entry["error"] = f"request is {total} tokens, limit is {engine.max_request_tokens}"
        return None
    entry["prompts"] = [prefix + s for s in suffixes]
    entry["prefix_tokens"] = size
    entry["input_tokens"] = total
    return req, rendered


def text_pass(engine, cases, key: str, found: dict[str, dict[str, Any]], weights: bool) -> None:
    from qev.calibrate import softmax

    for case in cases:
        name, request = case["name"], case["request"]
        if request.get("images"):
            continue
        entry = found.setdefault(name, {"name": name, "request": request})
        planned = plan(engine, request, entry)
        if planned is None:
            print(f"{name}: refused ({entry['error']})", flush=True)
            continue
        if not weights:
            print(f"{name}: {entry['input_tokens']} tokens", flush=True)
            continue
        req, rendered = planned
        response, meta = engine.decide(req, debug=True)
        if response.usage.input_tokens != entry["input_tokens"]:
            raise RuntimeError(f"{name}: the engine counted {response.usage.input_tokens} tokens")
        probabilities = {}
        for r in rendered:
            diagnostics = meta["diagnostics"][r.qid]
            (logits,) = diagnostics["slot_logits"]  # one branch per question (debias 1)
            probabilities[r.qid] = softmax(logits, diagnostics["temperature"])  # as decide() computes them
        entry[key] = probabilities
        entry.setdefault("answers", {})[key] = {q: a.model_dump() for q, a in response.answers.items()}
        print(f"{key} {name}: {entry['input_tokens']} tokens", flush=True)


PLACEHOLDER = r"<(image|video):(\d+)>"  # qev/media.py's and train/mm.py's


def placed(state: Any, count: int, how: str) -> tuple[Any, list[str]]:
    """The state the creator is asked with: as given when it names any image (the creator's convention, which its own
    content_parts then checks), else, with --place-images state-start, the placeholders at the start of the state's
    text (qev.prompt.render_state's: a JSON state indented by 2), a line before the rest (ippon's convention)."""
    import re

    from qev.prompt import render_state

    text = render_state(state)
    if re.search(PLACEHOLDER, text):
        return state, []
    if how != "state-start":
        raise ValueError("the state must reference each image once, in order, as <image:N> (qev/media.py)")
    marks = "".join(f"<image:{i}>" for i in range(1, count + 1))
    return marks + ("\n" + text if text else ""), [f"placeholders {marks} put at the start of the state"]


def media_rows(a, bundle, cases, found: dict[str, dict[str, Any]]) -> dict[str, list[tuple[str, Any, Any]]]:
    """Each image request's prompts as the creator's training and evaluation path renders them (train.common
    render_example with the multimodal bundle's template, train.mm.mm_inputs's processor), recorded; returns each
    request's (question id, example, row) to score. A request the creator refuses is recorded with its error."""
    from PIL import Image
    from qev import readout as ro
    from qev.prompt import render_question
    from train import common, mm

    planned = {}
    for case in cases:
        name, request = case["name"], case["request"]
        if not request.get("images"):
            continue
        entry = found.setdefault(name, {"name": name, "request": request})
        entry["media"] = case.get("media", [])
        try:
            state, notes = placed(request.get("state"), len(request["images"]), a.place_images)
        except ValueError as error:
            entry["error"] = str(error)
            print(f"{name}: refused ({error})", flush=True)
            continue
        media = []
        for image in request["images"]:
            with Image.open(Path(a.media) / image) as im:  # its stored size: the creator turns no image by its EXIF
                width, height = im.size
            media.append({"type": "image", "path": image, "h": height, "w": width})
        body, more = maker_body({**request, "state": state})
        if notes + more:
            entry["notes"] = notes + more
        prompts, rows, grids, pixels, prefix = [], [], None, None, None
        try:
            for qid, q in body["questions"].items():
                question = common.build_question(q)
                labels = render_question("q", question).labels
                row = {"state": state, "question": q, "media": media, "target": {labels[0]: 1.0}}
                ex = common.render_example(row, bundle.template_for(row), 0, seed=None, augment=False)
                inputs, _ = mm.mm_inputs(bundle.processor, [ex.prompt], [media], "cpu")
                ids = inputs["input_ids"][0].tolist()
                _, text_prefix, _ = ro.token_layout(bundle.tokenizer, ex.prompt, ex.suffix_start, None)
                size = text_prefix + bundle.counter.media_extra(media)  # the images are in the state's block
                if prefix is not None and (size != prefix or ids[:size] != prompts[0][:size]):
                    raise RuntimeError(f"{name}: the questions' prefixes differ")
                prefix = size
                prompts.append(ids)
                rows.append((qid, ex, row))
                grids = inputs["image_grid_thw"].tolist()
                pixels = hashlib.sha256(inputs["pixel_values"].float().numpy().tobytes()).hexdigest()
        except ValueError as error:  # train.mm.content_parts: placeholders that are not each image once, in order
            entry["error"] = f"train.mm: {error}"
            print(f"{name}: refused ({error})", flush=True)
            continue
        entry.update(
            prompts=prompts,
            prefix_tokens=prefix,
            image_grids=grids,
            pixel_values_sha256=pixels,
            input_tokens=prefix + sum(len(p) - prefix for p in prompts),
        )
        planned[name] = rows
        print(f"{name}: {entry['input_tokens']} tokens, prefix {prefix}", flush=True)
    return planned


def media_scores(bundle, planned, found, key: str, device: str, calibration) -> None:
    """Each planned question scored by train.common.batch_scores (one forward per question, the block ids over its
    prompt), softmax(slot logits / the calibration's temperature) in the request's option order."""
    import time

    import torch
    from qev.calibrate import softmax
    from train import common

    for name, rows in planned.items():
        started = time.perf_counter()
        probabilities = {}
        for qid, ex, row in rows:
            with torch.inference_mode():
                scores, _ = common.batch_scores(bundle, bundle.model, [ex], [row], device)
            logits = scores[0, : len(ex.labels)].float().cpu().tolist()
            probabilities[qid] = softmax(logits, calibration.temperature(ex.kind, len(ex.labels)))
        found[name][key] = probabilities
        print(f"{key} {name}: {time.perf_counter() - started:.0f} s", flush=True)


def media_pass(a, cases, found: dict[str, dict[str, Any]], write) -> float | None:
    """The image requests through the creator's training and evaluation path; with --bf16, again in the creator's
    production dtypes on --bf16-device. Returns bf16's largest distance from fp32 (None without --bf16)."""
    import torch
    from qev import readout as ro
    from qev.calibrate import Calibration
    from train import common, mm

    mm.set_root(a.media)
    bundle = common.load_bundle(
        str(a.model), device="cpu", dtype="float32", multimodal=True, load_model=not a.tokens_only
    )
    planned = media_rows(a, bundle, cases, found)
    if a.tokens_only:
        return None
    bundle.readout = ro.Readout.attach(bundle.model, ro.ReadoutConfig.load(str(a.model)), slot_ids=bundle.slot_ids)
    if not bundle.readout.config.noncausal or bundle.readout.config.head:
        sys.exit("expected block-bidirectional attention and the slot_lm readout")
    calibration_file = Path(a.model) / "calibration.json"
    calibration = Calibration.load(calibration_file) if calibration_file.exists() else Calibration()
    media_scores(bundle, planned, found, "probabilities", "cpu", calibration)
    write(None)
    if not a.bf16:
        return None
    # qev's production dtypes (qev/engine.py:73-74, 127-141): bfloat16 weights, the LM head in float32
    bundle.model.to(device=a.bf16_device, dtype=torch.bfloat16)
    head = bundle.model.get_output_embeddings()
    head.weight = torch.nn.Parameter(head.weight.detach().clone().to(torch.float32), requires_grad=False)
    head.register_forward_pre_hook(lambda module, args: (args[0].to(torch.float32), *args[1:]))
    media_scores(bundle, planned, found, "probabilities_bf16", a.bf16_device, calibration)
    return max(
        abs(x - y)
        for name in planned
        for q, p in found[name]["probabilities"].items()
        for x, y in zip(p, found[name]["probabilities_bf16"][q], strict=True)
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--onejev", required=True, type=Path, help="OneJev checkout at 81ce62f with serve.patch applied")
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--requests", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--ippon", type=Path, help="ippon checkout, for its portable serializer")
    p.add_argument("--tokens-only", action="store_true", help="prompts and token counts only; no weights loaded")
    p.add_argument("--bf16", action="store_true", help="also run in the creator's production dtypes, for the bound")
    p.add_argument("--bf16-device", default="mps", help="where the bf16 run happens")
    p.add_argument(
        "--fork-mode",
        choices=["sequential", "batched"],
        default="sequential",
        help="the creator's two ways to run the branches (the same computation; sequential by default)",
    )
    p.add_argument("--media", type=Path, help="run the image requests, their files in this directory")
    p.add_argument("--place-images", choices=["refuse", "state-start"], default="refuse")
    p.add_argument("--only", nargs="*", help="request names to run")
    p.add_argument("--threads", type=int)
    a = p.parse_args()

    commit = onejev(a.onejev)
    dumps = serializer(a.ippon)
    cases = json.loads(Path(a.requests).read_text())
    if a.only:
        if missing := set(a.only) - {case["name"] for case in cases}:
            p.error(f"unknown request names: {sorted(missing)}")
        cases = [case for case in cases if case["name"] in a.only]

    import torch
    import transformers

    if a.threads:
        torch.set_num_threads(a.threads)
    found: dict[str, dict[str, Any]] = {}
    maker: dict[str, Any] = {
        "code": f"OneJev qev @{commit} + serve.patch",
        "device": "cpu",
        "dtype": "float32",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "flash-linear-attention": importlib.util.find_spec("fla") is not None,  # absent: transformers' torch path
        "causal-conv1d": importlib.util.find_spec("causal_conv1d") is not None,
    }
    worst = None

    def write(worst: float | None) -> None:
        reference = {
            "schema": SCHEMA,
            "adapter": ADAPTER,
            "model": {"repo": a.repo, "revision": a.revision},
            "maker": maker,
            "tokens_only": a.tokens_only,
            "requests": list(found.values()),
        }
        if a.media is not None:
            reference["media_format"] = {
                "directory": "references/media/images",
                "decode": "train/mm.py load_media: PIL.Image.open, convert('RGB'), no EXIF turn",
                "placement": "the state's <image:N> placeholders (qev/media.py); with none, --place-images "
                f"{a.place_images}",
                "image_token": json.loads((Path(a.model) / "config.json").read_text())["image_token_id"],
                "prompts": "the creator's input ids, each image's <|image_pad|> expanded to its rows (ippon's "
                "placeholders, arch/common.media_id, stand where these ids are image_token)",
                "images": "per request, its image_grid_thw and the sha256 of its pixel_values rows (float32)",
            }
        if worst is not None:
            reference["bound"] = round(max(0.01, float(worst) + 5e-4), 4)
            maker["bf16_distance"] = worst
        Path(a.out).write_bytes(gzip.compress(dumps(reference).encode()))
        bound = f", bound {reference['bound']} (bf16 from fp32: {worst})" if worst is not None else ""
        print(f"{a.out}: {len(found)} requests{bound}", flush=True)

    if a.media is not None:
        maker.update(
            path="train.common.batch_scores with the readout attached as train/eval_mm.py attaches it, one forward "
            "per question (Qwen3_5ForConditionalGeneration, the processor's own pixel range)",
            **({"bf16": f"bfloat16 weights, the LM head in float32, on {a.bf16_device}"} if a.bf16 else {}),
        )
        worst = media_pass(a, cases, found, write)
    elif a.tokens_only:
        engine = tokenizer_engine(a.model)
        text_pass(engine, cases, "probabilities", found, weights=False)
    else:
        from qev.engine import DecisionEngine

        engine = DecisionEngine(
            str(a.model), device="cpu", dtype="float32", head_dtype="float32", fork_mode=a.fork_mode
        )
        extra = engine.info.extra
        if extra.get("attention") != "noncausal" or extra.get("readout") != "slot_lm" or engine.max_slots != 256:
            sys.exit(f"unexpected engine setup: {extra}, {engine.max_slots} slots")
        maker.update(
            attention=extra["attention"],
            readout=extra["readout"],
            calibration=extra["calibration"],
            fork_mode=engine.info.fork_mode,
            max_branch_tokens=engine.max_branch_tokens,
            max_request_tokens=engine.max_request_tokens,
        )
        text_pass(engine, cases, "probabilities", found, weights=True)
        if a.bf16:  # qev's defaults: bfloat16 weights, the LM head upcast to float32 (qev/engine.py:73-74, 101-141)
            engine.model.to(device=a.bf16_device, dtype=torch.bfloat16)  # the CPU's bf16 is far slower than MPS's
            engine.device = torch.device(a.bf16_device)
            engine.torch_dtype = torch.bfloat16
            engine._upcast_head(torch.float32)
            text_pass(engine, cases, "probabilities_bf16", found, weights=True)
            worst = max(
                (
                    abs(x - y)
                    for e in found.values()
                    if "probabilities_bf16" in e
                    for q in e["probabilities"]
                    for x, y in zip(e["probabilities"][q], e["probabilities_bf16"][q], strict=True)
                ),
                default=0.0,
            )
    write(worst)
    if any(not math.isfinite(x) for e in found.values() for q in e.get("probabilities", {}).values() for x in q):
        sys.exit("non-finite probabilities")


if __name__ == "__main__":
    main()
