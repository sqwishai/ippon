"""Clef-Flash's prompts and multimodal RoPE positions on image requests, from the checkpoint's own
joint_schema_model.py, never inside ippon:

    python -I make_media_reference.py --model path/to/clef-flash --requests ../requests/media-qwen.json \\
        --media ../media/images --out clef-flash-media.tokens.json.gz [--repo ID --revision SHA]

Run it in a venv with the card's torch 2.11 and transformers 5.10.2 (plus torchvision and pillow), on the CPU only;
no weights are loaded (Clef-Flash in fp32 is 38 GB). Each request goes through the checks systemone makes, then
encode_record(processor.tokenizer, record, max_length=16384, processor=processor) with its images opened as the card
opens them (PIL.Image.open, handed to the processor as they are: no EXIF turn) and collate_records, as systemone
calls them. Recorded: the input ids (each image's `<|image_pad|>` expanded to its rows), each question's instruction
and option spans and option ids, the (t, h, w) positions Qwen3_5Model.get_rope_index gives every id from the
collated mm_token_type_ids and image_grid_thw (what Clef's forward passes Qwen3_5Model), each image's grid and the
sha256 of its pixel_values rows, and the usage (len(input_ids)). A request Clef's code refuses is recorded with its
error.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Any


def module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("clef_joint_schema_model", path)
    assert spec is not None
    assert spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded  # its dataclasses look their module up
    spec.loader.exec_module(loaded)
    return loaded


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--requests", required=True, type=Path)
    p.add_argument("--media", required=True, type=Path, help="the directory the requests' image names are in")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--repo", help="the Hugging Face repository the checkpoint's files are from")
    p.add_argument("--revision", help="its revision")
    a = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    root, media = a.model.expanduser(), a.media.expanduser()
    import PIL
    import torch
    import torchvision
    import transformers
    from PIL import Image
    from transformers import AutoProcessor
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    clef = module(root / "joint_schema_model.py")
    processor = AutoProcessor.from_pretrained(str(root))
    tokenizer = processor.tokenizer
    config = json.loads((root / "config.json").read_text())
    merge = config["vision_config"]["spatial_merge_size"]
    stub = types.SimpleNamespace(
        config=types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=merge))
    )
    stub.get_vision_position_ids = types.MethodType(Qwen3_5Model.get_vision_position_ids, stub)
    token = config["image_token_id"]
    out = []
    for case in json.loads(a.requests.read_text()):
        for m in case["media"]:
            assert sha256((media / m["name"]).read_bytes()) == m["sha256"], f"{m['name']} is not the listed file"
        request = {"model": "clef-flash", **case["request"]}
        request["images"] = [Image.open(media / name) for name in case["request"]["images"]]
        entry: dict[str, Any] = {"name": case["name"], "request": case["request"], "media": case["media"]}
        try:
            for qid, q in request["questions"].items():  # systemone's checks
                if q.get("type") not in clef.QUESTION_TYPES or (q["type"] != "noul" and not q.get("criteria")):
                    raise ValueError(qid)
            encoded = clef.encode_record(tokenizer, request, max_length=16384, processor=processor)
        except ValueError as error:  # refused by the creator's code
            entry["error"] = str(error)
            out.append(entry)
            print(f"{case['name']}: refused ({error})", flush=True)
            continue
        batch = clef.collate_records([encoded], tokenizer.pad_token_id, torch.device("cpu"))
        grid = batch["media"]["image_grid_thw"]
        positions, delta = Qwen3_5Model.get_rope_index(
            stub,
            batch["input_ids"],
            mm_token_type_ids=batch["media"]["mm_token_type_ids"],
            image_grid_thw=grid,
            attention_mask=batch["attention_mask"],
        )
        pixels, at = [], 0
        for t, h, w in grid.tolist():
            rows = batch["media"]["pixel_values"][at : at + t * h * w]
            pixels.append(sha256(rows.contiguous().numpy().astype("<f4").tobytes()))
            at += t * h * w
        ids = list(encoded.input_ids)
        assert sum(t == token for t in ids) == sum(t * h * w // merge**2 for t, h, w in grid.tolist())
        entry |= {
            "prompts": [ids],
            "readouts": [
                [list(q.question_span), [list(s) for s in q.option_spans], list(q.option_ids)]
                for q in encoded.questions
            ],
            "positions": [positions[:, 0].tolist()],
            "deltas": [int(delta[0, 0])],
            "images": [{"grids": grid.tolist(), "pixels": pixels}],
            "input_tokens": len(ids),
        }
        out.append(entry)
        print(f"{case['name']}: {len(ids)} tokens", flush=True)
    reference = {
        "schema": "ippon-reference/1",
        "adapter": "cloudflare/clef",
        "model": {"directory": root.name, **({"repo": a.repo, "revision": a.revision} if a.repo else {})},
        "maker": {
            "code": "joint_schema_model.py: encode_record with the processor, collate_records; "
            "Qwen3_5Model.get_rope_index on what they give",
            "device": "cpu",
            "versions": {
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "transformers": transformers.__version__,
                "pillow": PIL.__version__,
            },
        },
        "bound": 0.01,
        "tokens_only": True,
        "media_format": {
            "directory": "references/media/images",
            "decode": "PIL.Image.open, handed to the processor as it is (the card's): convert('RGB'), no EXIF turn",
            "image_token": token,
            "prompts": "the creator's input ids, each image's <|image_pad|> expanded to its rows (ippon's "
            "placeholders, arch/common.media_id, stand where these ids are image_token)",
            "positions": "per prompt [3][tokens]: Qwen3_5Model.get_rope_index's (t, h, w) position of every id",
            "images": "per prompt, each image's image_grid_thw and the sha256 of its pixel_values rows (float32, "
            "little-endian)",
            "refused": "a request with `error` is one encode_record refuses",
        },
        "requests": out,
    }
    data = json.dumps(reference, ensure_ascii=False).encode()
    a.out.write_bytes(gzip.compress(data) if a.out.suffix == ".gz" else data)
    print(f"{len(out)} requests -> {a.out}")


if __name__ == "__main__":
    main()
