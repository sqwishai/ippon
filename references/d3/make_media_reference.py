"""vllm-sr d3 on image and video requests, from vllm-sr's own runtime (d3_runtime.py, shipped in the model repository),
never inside ippon:

    python -I make_media_reference.py --model DIR --requests ../requests/media-qwen.json --media ../media/images \\
        --out d3-mini-media.json.gz --repo vllm-sr/d3-mini --revision SHA [--tokens-only] [--bf16 mps]

Run it in an environment with the checkpoint's requirements.txt (torch, torchvision, transformers 5.17.0, pillow,
opencv-python-headless). Each request's images and videos are given as their paths (`--media`, the directory their
names are under), as the creator's `load_image` and `load_video` take them, and the request goes through `D3.prepare`
(the creator's checks, its chat template, its image processor at 65,536 to 1,638,400 pixels, its OpenCV decoding and
frame sampling, its video processor at up to 200,704 pixels a frame). Recorded for every question's prompt: the ids
the processor gives it, each image's `<|image_pad|>` and each frame pair's `<|video_pad|>` expanded to its rows, with
the frames' timestamps; the multimodal RoPE positions Qwen3_5Model.get_rope_index gives them; each image's and video's
grid and the sha256 of its pixel rows; each video's sampled frames (source indices, frame rate, frame count). A
request the creator refuses is recorded with its error.

--tokens-only stops there (no weights are loaded: the runtime's D3 without its backbone). Otherwise D3 runs on the CPU
with the backbone in fp32, one question per forward pass (batch_size=1, so nothing is padded), and records each
question's probabilities in the request's option order (a noul as true, false), checking that the ids and positions
the forward took are the ones recorded. With --bf16 DEVICE it runs again with the backbone in bf16 (the runtime's own
precision) on DEVICE ("mps": the Mac's GPU), and the bound is the larger of 0.01 and that run's distance from fp32
plus 0.0005, rounded to four decimals (docs/accuracy.md). The file is written after each run.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rows_sha(pixels: Any, grids: list[list[int]]) -> list[str]:
    """The sha256 of each image's or video's pixel rows (float32, little-endian), split by its grid."""
    out, at = [], 0
    for t, h, w in grids:
        out.append(sha256(pixels[at : at + t * h * w].contiguous().float().numpy().astype("<f4").tobytes()))
        at += t * h * w
    assert at == pixels.shape[0]
    return out


def bare(runtime: Any, root: Path) -> Any:
    """The runtime's D3 as its __init__ sets it up for preparing requests, without the backbone's weights."""
    import torch
    from transformers import AutoTokenizer

    d3 = runtime.D3.__new__(runtime.D3)
    d3.torch, d3.root = torch, root
    d3.config = json.loads((root / "decision_config.json").read_text())
    d3.prompt = d3.config.get("prompt", "d3")
    limit = d3.config.get("max_length")
    d3.max_length = int(limit) if limit else None
    d3.permutation_average = False
    d3.tokenizer = AutoTokenizer.from_pretrained(str(root))
    d3.tokenizer.padding_side = "left"
    d3.codes, d3.token_ids = runtime.answer_codes(d3.tokenizer)
    d3.processor = runtime.load_processor(root)
    runtime.configure_video_processor(d3.processor)
    d3.image_unavailable = d3.video_unavailable = None
    return d3


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--requests", required=True, type=Path)
    p.add_argument("--media", required=True, type=Path, help="the directory the requests' file names are under")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--tokens-only", action="store_true")
    p.add_argument("--bf16", metavar="DEVICE", help="also run in bf16 on DEVICE (mps, cpu), for the bound")
    p.add_argument("--only", action="append", help="run only the requests with this name (repeatable)")
    a = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    root, media = a.model.expanduser().resolve(), a.media.expanduser().resolve()
    sys.path.insert(0, str(root))
    runtime = importlib.import_module("d3_runtime")
    import cv2
    import PIL
    import torch
    import torchvision
    import transformers
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    cases = [c for c in json.loads(a.requests.read_text()) if not a.only or c["name"] in a.only]
    d3 = bare(runtime, root) if a.tokens_only else runtime.D3(root, device="cpu", batch_size=1)
    processor = d3.processor
    config = json.loads((root / "config.json").read_text())
    merge = config["vision_config"]["spatial_merge_size"]
    stub = types.SimpleNamespace(
        config=types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=merge))
    )
    stub.get_vision_position_ids = types.MethodType(Qwen3_5Model.get_vision_position_ids, stub)
    image_token, video_token = config["image_token_id"], config["video_token_id"]

    def given(case: dict[str, Any]) -> tuple[list[str], list[str]]:
        for m in case["media"]:
            assert sha256((media / m["name"]).read_bytes()) == m["sha256"], f"{m['name']} is not the listed file"
        request = case["request"]
        return [str(media / n) for n in request.get("images") or ()], [
            str(media / n) for n in request.get("videos") or ()
        ]

    def prepare(case: dict[str, Any]) -> Any:
        request = case["request"]
        images, videos = given(case)
        prepared = d3.prepare(request.get("state"), request["questions"], images or None, videos or None)
        assert not prepared.errors, f"{case['name']}: {prepared.errors}"
        return prepared

    def prompt(prepared: Any, key: str) -> dict[str, Any]:
        """One question's prompt as the runtime's image_logits or video_logits builds it for a batch of one."""
        text = prepared.texts[key]
        found: dict[str, Any] = {}
        if prepared.videos:  # video_logits: the request's images and videos expanded, then tokenized
            m = prepared.media
            image_texts = [processor.replace_image_token(m, image_idx=i) for i in range(len(prepared.images))]
            expanded = processor.get_text_with_replacements([text], image_texts, prepared.video_texts)[0][0]
            encoded = processor.tokenizer([expanded], padding=True, return_token_type_ids=False, return_tensors="pt")
            kinds = torch.as_tensor(processor.create_mm_token_type_ids(encoded["input_ids"]))
            image_grid = m.get("image_grid_thw")
            video_grid = m["video_grid_thw"]
            if image_grid is not None:
                found["images"] = {
                    "grids": image_grid.tolist(),
                    "pixels": rows_sha(m["pixel_values"], image_grid.tolist()),
                }
            found["videos"] = {
                "grids": video_grid.tolist(),
                "pixels": rows_sha(m["pixel_values_videos"], video_grid.tolist()),
            }
        else:  # image_logits: the processor expands and tokenizes the prompt with the request's images
            encoded = processor(text=[text], images=list(prepared.images), padding=True, return_tensors="pt")
            kinds, image_grid, video_grid = encoded["mm_token_type_ids"], encoded["image_grid_thw"], None
            found["images"] = {
                "grids": image_grid.tolist(),
                "pixels": rows_sha(encoded["pixel_values"], image_grid.tolist()),
            }
        ids = encoded["input_ids"]
        assert bool(encoded["attention_mask"].all())
        positions, delta = Qwen3_5Model.get_rope_index(
            stub,
            ids,
            mm_token_type_ids=kinds,
            image_grid_thw=image_grid,
            video_grid_thw=None if video_grid is None else video_grid.clone(),
            attention_mask=encoded["attention_mask"],
        )
        assert ids.shape[1] == prepared.lengths[key], f"{key}: planned {prepared.lengths[key]}, got {ids.shape[1]}"
        rows = sum(t * h * w for t, h, w in found.get("images", {}).get("grids", [])) // merge**2
        assert int((ids == image_token).sum()) == rows
        if video_grid is not None:
            assert int((ids == video_token).sum()) == int(video_grid.prod(-1).sum()) // merge**2
        return {"ids": ids[0].tolist(), "positions": positions[:, 0].tolist(), "delta": int(delta[0, 0]), **found}

    found: dict[str, dict[str, Any]] = {}
    for case in cases:
        entry: dict[str, Any] = {"name": case["name"], "request": case["request"], "media": case["media"]}
        try:
            prepared = prepare(case)
        except ValueError as error:  # refused by the creator's code
            entry["error"] = str(error)
            found[case["name"]] = entry
            print(f"{case['name']}: refused ({error})", flush=True)
            continue
        prompts = [prompt(prepared, k) for k in prepared.keys]
        entry |= {
            "prompts": [s["ids"] for s in prompts],
            "positions": [s["positions"] for s in prompts],
            "deltas": [s["delta"] for s in prompts],
            "images": [s.get("images", {"grids": [], "pixels": []}) for s in prompts],
            "input_tokens": sum(prepared.lengths[k] for k in prepared.keys),
        }
        if prepared.videos:
            entry["videos"] = [s["videos"] for s in prompts]
            entry["frames"] = [
                {"indices": list(v.indices), "fps": v.fps, "total": v.total_frames, "size": list(v.size)}
                for v in prepared.videos
            ]
        found[case["name"]] = entry
        print(f"{case['name']}: {len(prompts)} prompts, {entry['input_tokens']} tokens", flush=True)

    seconds: dict[str, float] = {}
    devices: dict[str, str] = {}

    def write(bf16: bool) -> None:
        distance, worst = 0.0, ""
        for e in found.values() if bf16 else ():
            for q, want in e.get("probabilities", {}).items():
                d = max(abs(x - y) for x, y in zip(want, e["probabilities_bf16"][q], strict=True))
                if d > distance:
                    distance, worst = d, f"{e['name']}/{q}"
        directory = media.relative_to(REPOSITORY).as_posix()
        reference = {
            "schema": "ippon-reference/1",
            "adapter": "vllm-sr/d3",
            "model": {"repo": a.repo, "revision": a.revision, "directory": root.name},
            "maker": {
                "code": "d3_runtime.py: D3.prepare, and D3.run with batch_size=1 (image_logits, video_logits)",
                "device": "cpu",
                "dtype": "float32",
                "devices": devices,
                "threads": torch.get_num_threads(),
                "versions": {
                    "torch": torch.__version__,
                    "torchvision": torchvision.__version__,
                    "transformers": transformers.__version__,
                    "pillow": PIL.__version__,
                    "opencv": cv2.__version__,
                },
                "seconds": seconds,
                **({"bf16_distance": distance, "bf16_worst": worst} if bf16 else {}),
            },
            "bound": round(max(0.01, distance + 5e-4), 4) if bf16 else 0.01,
            "tokens_only": a.tokens_only,
            "media_format": {
                "directory": directory,
                "decode": "images: load_image (PIL.Image.open, convert('RGB'), no EXIF turn); videos: load_video "
                "(OpenCV's FFmpeg backend, 2 frames a second, 4 to 32 spread over the clip, BGR to RGB, an odd "
                "count's last frame twice)",
                "image_token": image_token,
                "video_token": video_token,
                "prompts": "the creator's input ids, each image's <|image_pad|> and each frame pair's <|video_pad|> "
                "expanded to its rows (ippon's placeholders, arch/common.media_id, stand where these ids are the "
                "image or video token)",
                "positions": "per prompt [3][tokens]: Qwen3_5Model.get_rope_index's (t, h, w) position of every id",
                "images": "per prompt, each image's image_grid_thw and the sha256 of its pixel_values rows (float32, "
                "little-endian)",
                "videos": "per prompt, each video's video_grid_thw and the sha256 of its pixel_values_videos rows",
                "frames": "per request, each video's sampled source frames, its frame rate and frame count, and "
                "the frames' (width, height)",
                "refused": "a request with `error` is one D3.prepare refuses",
            },
            "requests": list(found.values()),
        }
        if not a.tokens_only and not bf16:
            reference["maker"].pop("bf16_distance", None)
        data = json.dumps(reference, ensure_ascii=False).encode()
        a.out.write_bytes(gzip.compress(data) if a.out.suffix == ".gz" else data)
        measured = f"; bf16 {distance:.4f} at {worst}" if bf16 else ""
        print(f"{len(found)} requests -> {a.out} (bound {reference['bound']}{measured})", flush=True)

    if a.tokens_only:
        write(False)
        return

    taken_ids: list[list[int]] = []
    taken_positions: list[list[list[int]]] = []
    d3.backbone.get_input_embeddings().register_forward_pre_hook(
        lambda module, args: taken_ids.append(args[0][0].tolist())
    )
    original = d3.backbone.compute_3d_position_ids

    def positions(*args: Any, **kwargs: Any) -> Any:
        out = original(*args, **kwargs)
        taken_positions.append(out[:, 0].tolist())
        return out

    d3.backbone.compute_3d_position_ids = positions
    runs = [("probabilities", torch.float32, "cpu")]
    if a.bf16:
        runs.append(("probabilities_bf16", torch.bfloat16, a.bf16))
    for key, dtype, device in runs:
        d3.backbone.to(dtype)
        d3.to(device)
        devices[key] = f"{device} {str(dtype).removeprefix('torch.')}"
        started = time.perf_counter()
        for entry in found.values():
            if "error" in entry:
                continue
            case = next(c for c in cases if c["name"] == entry["name"])
            prepared = prepare(case)
            taken_ids.clear()
            taken_positions.clear()
            try:
                with torch.inference_mode():
                    probabilities, tokens = d3.run(prepared)
            except (RuntimeError, NotImplementedError) as error:
                if d3.device.type == "cpu":
                    raise
                print(f"{key} {entry['name']} on {d3.device}: {error}; the CPU instead", flush=True)
                d3.to("cpu")
                devices[key] = f"cpu {str(dtype).removeprefix('torch.')} (after {device} failed)"
                taken_ids.clear()
                taken_positions.clear()
                with torch.inference_mode():
                    probabilities, tokens = d3.run(prepared)
            assert taken_ids == entry["prompts"], f"{entry['name']}: the forward's ids are not the recorded ones"
            assert taken_positions == entry["positions"], f"{entry['name']}: the forward's positions differ"
            assert tokens == entry["input_tokens"], f"{entry['name']}: usage {tokens}, planned {entry['input_tokens']}"
            questions = case["request"]["questions"]
            entry[key] = {
                q: (lambda v: [v[1], v[0]] if questions[q]["type"] == "noul" else v)(
                    [float(x) for x in probabilities[q]]
                )
                for q in questions
            }
            print(f"{key} {entry['name']}: {time.perf_counter() - started:.0f} s", flush=True)
        seconds[devices[key]] = round(time.perf_counter() - started, 1)
        write(key == "probabilities_bf16")


if __name__ == "__main__":
    main()
