"""pplx-decider flash v1.3 on image requests, from the creator's own `pplx_decider` package (Apache-2.0, shipped in the
checkpoint's source/src), never inside ippon:

    python -I make_media_reference.py --model DIR --requests ../requests/media-qwen.json --media ../media/images \\
        --out pplx-decider-flash-v1.3-media.json.gz --repo ID --revision SHA [--tokens-only] [--bf16] [--threads N]

Run it in a venv with the creator's requirements.txt (torch, torchvision, transformers 5.17.0, pillow), on the CPU only
(CUDA is hidden). Each request's images are given as their paths, as the creator's `open_image` takes them
(PIL.Image.open, convert("RGB"), no EXIF turn), and its questions asked in the creator's default mode, as ippon asks
them: one sequence per question, each with the request's context and every image (`Encoder.plan`). Recorded for every
sequence, from the creator's Encoder (its processor set up as Decider.__init__ sets it: min 65,536 and
decision_config.json's max_pixels): the input ids, each `<|image_pad|>` expanded to its image's rows; the multimodal
RoPE positions Qwen3_5Model.get_rope_index gives them (and, when the model runs, the position ids its forward took,
which must be the same); each image's grid and the sha256 of its pixel_values rows. --tokens-only stops there (no
weights are loaded: NAME-media.tokens.json.gz). Otherwise the creator's Decider runs in fp32 (about 18 GB) and records
each question's probabilities, softmax(logits / temperature) in the request's option order (a noul as true, false);
with --bf16 it runs again in bf16, and the bound is the larger of 0.01 and that run's distance from fp32, else the
text reference's bound. The file is written after each run.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

TEXT_BOUND = 0.0169  # pplx-decider-flash-v1.3.json.gz: bf16's distance from fp32 on the text requests, plus 0.0005


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ordered(question: dict[str, Any], probabilities: list[float]) -> list[float]:
    return [probabilities[1], probabilities[0]] if question["type"] == "noul" else probabilities


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--requests", required=True, type=Path)
    p.add_argument("--media", required=True, type=Path, help="the directory the requests' image names are in")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--tokens-only", action="store_true")
    p.add_argument("--bf16", action="store_true", help="also run in bf16, for the bound")
    p.add_argument("--threads", type=int, default=8, help="CPU threads for torch (the creator's Decider sets 8)")
    p.add_argument("--only", action="append", help="run only the requests with this name (repeatable)")
    a = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # the CPU only: a GPU may be timing something else
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model, media = a.model.expanduser(), a.media.expanduser()
    sys.path.insert(0, str(model / "source" / "src"))
    import PIL
    import torch
    import torchvision
    import transformers
    from pplx_decider.model import DEFAULT_MAX_LENGTH, Decider
    from pplx_decider.protocol import LEGACY_MAX_PIXELS, MIN_PIXELS, Encoder, load_json
    from transformers import AutoProcessor
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    cases = [c for c in json.loads(a.requests.read_text()) if not a.only or c["name"] in a.only]
    decider = None
    if a.tokens_only:  # the processor and Encoder as Decider.__init__ sets them up, without the weights
        config = load_json(model / "decision_config.json")
        processor = AutoProcessor.from_pretrained(str(model))
        processor.tokenizer.padding_side = "left"
        ceiling = int(config.get("max_pixels") or LEGACY_MAX_PIXELS)
        processor.image_processor.size = {"shortest_edge": MIN_PIXELS, "longest_edge": ceiling}
        encoder = Encoder(processor, config["codes"])
        max_length: int | None = DEFAULT_MAX_LENGTH
        merge = json.loads((model / "config.json").read_text())["vision_config"]["spatial_merge_size"]
        stub = types.SimpleNamespace(
            config=types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=merge))
        )
        stub.get_vision_position_ids = types.MethodType(Qwen3_5Model.get_vision_position_ids, stub)
        rope_owner: Any = stub
    else:
        decider = Decider(model, device="cpu", cpu_threads=a.threads)
        assert all(t.device.type == "cpu" for t in [*decider.backbone.parameters(), *decider.readout.parameters()])
        encoder, processor, max_length, rope_owner = (
            decider.encoder,
            decider.processor,
            decider.max_length,
            decider.backbone,
        )
        ceiling = decider.max_pixels
    token = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    taken: list[list[list[int]]] = []  # the position ids each forward took, in order

    def taking(module: Any, args: Any, kwargs: dict[str, Any]) -> None:
        taken.append(kwargs["position_ids"][:, 0].tolist())

    if decider is not None:  # before the creator's noncausal hook, which needs keyword inputs too
        decider.backbone.language_model.register_forward_pre_hook(taking, with_kwargs=True)

    def sequences(case: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        request = case["request"]
        for m in case["media"]:
            assert sha256((media / m["name"]).read_bytes()) == m["sha256"], f"{m['name']} is not the listed file"
        given = {**request, "images": [str(media / name) for name in request["images"]]}
        out = []
        for chunk in encoder.plan(given, max_length):
            ids = chunk.input_ids[None]
            kinds = chunk.mm_token_type_ids[None]
            positions, delta = Qwen3_5Model.get_rope_index(
                rope_owner, ids, mm_token_type_ids=kinds, image_grid_thw=chunk.image_grid_thw
            )
            grids = chunk.image_grid_thw.tolist()
            pixels, at = [], 0
            for t, h, w in grids:  # each image's patches, in order
                rows = chunk.pixel_values[at : at + t * h * w]
                pixels.append(sha256(rows.contiguous().numpy().astype("<f4").tobytes()))
                at += t * h * w
            assert at == chunk.pixel_values.shape[0]
            assert int((chunk.input_ids == token).sum()) == sum(t * h * w // merge_of(processor) for t, h, w in grids)
            out.append(
                {
                    "questions": list(chunk.keys),
                    "ids": chunk.input_ids.tolist(),
                    "positions": positions[:, 0].tolist(),
                    "delta": int(delta[0, 0]),
                    "grids": grids,
                    "pixels": pixels,
                }
            )
        return out, given

    found: dict[str, dict[str, Any]] = {}
    for case in cases:
        prompts, _ = sequences(case)
        found[case["name"]] = {
            "name": case["name"],
            "request": case["request"],
            "media": case["media"],
            "prompts": [s["ids"] for s in prompts],
            "positions": [s["positions"] for s in prompts],
            "deltas": [s["delta"] for s in prompts],
            "images": [{"grids": s["grids"], "pixels": s["pixels"]} for s in prompts],
            "input_tokens": sum(len(s["ids"]) for s in prompts),
        }
        print(f"{case['name']}: {len(prompts)} sequences, {found[case['name']]['input_tokens']} tokens", flush=True)

    seconds: dict[str, float] = {}

    def write(bf16: bool) -> None:
        """The reference as far as it is made: after fp32 (the bound the text reference's), then after bf16."""
        distance, worst = 0.0, ""
        for e in found.values() if bf16 else ():
            for q, want in e["probabilities"].items():
                d = max(abs(x - y) for x, y in zip(want, e["probabilities_bf16"][q], strict=True))
                if d > distance:
                    distance, worst = d, f"{e['name']}/{q}"
        reference = {
            "schema": "ippon-reference/1",
            "adapter": "perplexity/pplx-decider",
            "model": {"repo": a.repo, "revision": a.revision},
            "maker": {
                "code": "pplx_decider (source/src): Encoder.plan, and Decider.logits (the reference path)",
                "device": "cpu",
                "dtype": "float32",
                "threads": torch.get_num_threads(),
                "versions": {
                    "torch": torch.__version__,
                    "torchvision": torchvision.__version__,
                    "transformers": transformers.__version__,
                    "pillow": PIL.__version__,
                },
                "questions": "one per sequence",
                "pixels": {"min": MIN_PIXELS, "max": ceiling},
                "seconds": seconds,
                **({"bf16_distance": distance, "bf16_worst": worst} if bf16 else {}),
            },
            "bound": round(max(0.01, distance + 5e-4), 4) if bf16 else TEXT_BOUND,
            "tokens_only": decider is None,
            "media_format": {
                "directory": "references/media/images",
                "decode": "protocol.open_image: PIL.Image.open, convert('RGB'), no EXIF turn",
                "image_token": token,
                "prompts": "the creator's input ids, each image's <|image_pad|> expanded to its rows (ippon's "
                "placeholders, arch/common.media_id, stand where these ids are image_token)",
                "positions": "per prompt [3][tokens]: Qwen3_5Model.get_rope_index's (t, h, w) position of every id",
                "images": "per prompt, each image's image_grid_thw and the sha256 of its pixel_values rows (float32, "
                "little-endian)",
            },
            "requests": list(found.values()),
        }
        data = json.dumps(reference, ensure_ascii=False).encode()
        a.out.write_bytes(gzip.compress(data) if a.out.suffix == ".gz" else data)
        measured = f"; bf16 {distance:.4f} at {worst}" if bf16 else ""
        print(f"{len(found)} requests -> {a.out} (bound {reference['bound']}{measured})", flush=True)

    runs = [(torch.float32, "probabilities")] + ([(torch.bfloat16, "probabilities_bf16")] if a.bf16 else [])
    for dtype, key in runs if decider is not None else ():
        decider.backbone.to(dtype)
        decider.readout.to(dtype)
        started = time.perf_counter()
        for case in cases:
            entry = found[case["name"]]
            _, given = sequences(case)
            taken.clear()
            # each question's sequence in a forward of its own, so the positions it took are that sequence's alone
            # (`separate` reads each question in its own sequence either way)
            logits = {
                key: v
                for q, body in given["questions"].items()
                for key, v in decider.logits([{**given, "questions": {q: body}}])[0].items()
            }
            assert taken == entry["positions"], f"{case['name']}: the forward's positions are not get_rope_index's"
            entry[key] = {
                q: ordered(
                    case["request"]["questions"][q], (torch.tensor(v) / decider.temperature).softmax(-1).tolist()
                )
                for q, v in logits.items()
            }
            print(f"{key} {case['name']}: {time.perf_counter() - started:.0f} s", flush=True)
        seconds[str(dtype).removeprefix("torch.")] = round(time.perf_counter() - started, 1)
        write(key == "probabilities_bf16")
    if decider is None:
        write(False)


def merge_of(processor: Any) -> int:
    return int(processor.image_processor.merge_size) ** 2


if __name__ == "__main__":
    main()
