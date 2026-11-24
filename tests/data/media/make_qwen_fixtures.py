# mypy: ignore-errors
"""Reference script, run once in a venv with Perplexity's pinned requirements for pplx-decider flash v1.2 (torch
2.14.0, torchvision 0.29.0, transformers 5.17.0, Pillow 12.3.0), never in ippon's: Qwen's image processor and
the Qwen3.5 model's position computations from transformers, which tests/test_media_qwen.py holds ippon's
media/qwen.py to exactly. It reads the two checkpoints' processor_config.json and config.json only; no weights are
loaded and no code from the model folders runs.

    python -I make_qwen_fixtures.py path/to/pplx-decider-flash-v1.2 path/to/Clef-Flash

Writes, next to this file:

- qwen_preprocess.json:
  - "sizes": smart_resize(height, width) under each checkpoint's pixel range, for edge sizes and seeded random ones
    (both rounding branches, the floor and ceiling, aspects up to and past 200), or the processor's error;
  - "images": for synthetic images (`pattern`, an integer hash, so the tests rebuild them in any numpy), what each
    checkpoint's image processor returns, loaded with AutoImageProcessor (its torchvision backend) and given a PIL
    image, as the creators give it: image_grid_thw and the sha256 of pixel_values (float32, little-endian); several
    images in one call are checked to give each image's own patches, in order;
  - "rope": Qwen3_5Model.get_rope_index on token-type sequences with images of several grids between text, with and
    without left padding: each real row's (t, h, w) positions and the delta;
- qwen_positions.npz: a seeded float32 48 x 48 x 8 position table resampled to patch grids as Qwen3_5VisionModel
  does it (`get_vision_interpolation_indices_and_weights`, then the forward's weighted embedding sum), and
  Qwen3_5VisionRotaryEmbedding's cos and sin for 64- and 72-wide heads (pplx's and Clef-Flash's towers).
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import types
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
import transformers
from PIL import Image
from torch.nn import functional
from transformers import AutoImageProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model, Qwen3_5VisionRotaryEmbedding
from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids

HERE = Path(__file__).resolve().parent
CHECKPOINTS = {"pplx": Path(sys.argv[1]).expanduser(), "clef": Path(sys.argv[2]).expanduser()}
PPLX_MIN_PIXELS = 65_536  # pplx_decider/protocol.py's MIN_PIXELS, which its model sets on the processor

# (width, height): the card photo, exact multiples, down past pplx's 1.6 MP ceiling, halves that round to even, the
# floor branch's clamp, tiny images scaled up, 1:50 either way, aspects at and past 200
IMAGES = [
    (640, 480),
    (1024, 768),
    (512, 512),
    (1300, 1200),
    (2000, 1500),
    (1500, 2000),
    (720, 720),
    (496, 336),
    (100, 5000),
    (5000, 100),
    (6000, 30),
    (31, 17),
    (3, 7),
    (1, 1),
    (64, 64),
    (8000, 40),
    (8100, 40),
]
# (token types, grids): 0 text, 1 image; each run of 1s is one image of the next grid (t, h, w) in patches
ROPE = [
    ([0] * 5 + [1] * 4 + [0] * 3, [(1, 4, 4)]),
    ([0] * 7 + [1] * 12 + [0] * 4 + [1] * 6 + [0] * 9, [(1, 6, 8), (1, 4, 6)]),
    ([0] * 3 + [1] * 15 + [0] * 2, [(1, 10, 6)]),  # taller than wide: the text resumes past the rows
    ([1] * 8 + [0] * 6, [(1, 4, 8)]),  # an image first
    ([0] * 2 + [1] * 1 + [0] * 1 + [1] * 2 + [0] * 1 + [1] * 30 + [0] * 5, [(1, 2, 2), (1, 2, 4), (1, 12, 10)]),
    ([0] * 9, []),
]
GRIDS = [(2, 2), (4, 6), (6, 4), (16, 16), (48, 48), (26, 36), (40, 30), (2, 400), (124, 20)]


def pattern(height: int, width: int, seed: int) -> np.ndarray:
    """A deterministic H x W x 3 uint8 image of hashed noise (uint32 arithmetic, the same in any numpy)."""
    y = np.arange(height, dtype=np.uint32)[:, None, None]
    x = np.arange(width, dtype=np.uint32)[None, :, None]
    c = np.arange(3, dtype=np.uint32)[None, None, :]
    h = (x * np.uint32(73856093)) ^ (y * np.uint32(19349663)) ^ (c * np.uint32(83492791)) ^ np.uint32(seed)
    h = (h ^ (h >> np.uint32(15))) * np.uint32(2246822519)
    return ((h >> np.uint32(24)) & np.uint32(255)).astype(np.uint8)


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def processors() -> dict:
    out = {}
    for name, path in CHECKPOINTS.items():
        processor = AutoImageProcessor.from_pretrained(str(path))
        assert type(processor).__name__ == "Qwen2VLImageProcessor"
        assert processor.backend == "torchvision"
        if name == "pplx":  # as pplx_decider/model.py sets it, with decision_config.json's max_pixels
            decision = json.loads((path / "decision_config.json").read_text())
            processor.size = {"shortest_edge": PPLX_MIN_PIXELS, "longest_edge": int(decision["max_pixels"])}
        out[name] = processor
    return out


def sizes(ranges: dict) -> list:
    edges = [1, 2, 15, 16, 17, 31, 32, 33, 47, 48, 49, 80, 255, 256, 257, 511, 512, 513, 720, 1023, 1024, 1025]
    edges += [1279, 1280, 1281, 2047, 2048, 4095, 4096, 4097, 8192, 16384]
    rng = np.random.default_rng(20261023)
    pairs = {(w, h) for w in edges for h in edges}
    pairs |= {(int(w), int(h)) for w, h in rng.integers(1, 9000, (3000, 2))}
    pairs |= {(int(w), int(h)) for w, h in rng.integers(1, 300, (500, 2))}
    pairs |= {
        (int(s), int(s * r))
        for s, r in zip(rng.integers(20, 6000, 600), rng.uniform(1 / 220, 220, 600), strict=True)
        if s * r >= 1
    }
    pairs |= {(16 * k + 8, 16 * j + 8) for k in range(1, 60, 3) for j in range(1, 60, 7)}  # halves of 32 to round
    out = []
    for w, h in sorted(pairs):
        row = [w, h]
        for least, most in ranges.values():
            try:
                row.append(list(smart_resize(h, w, factor=32, min_pixels=least, max_pixels=most)))
            except ValueError:
                row.append(None)
        out.append(row)
    return out


def rope_cases() -> list:
    config = Qwen3_5Config()
    stub = types.SimpleNamespace(config=config)
    stub.get_vision_position_ids = types.MethodType(Qwen3_5Model.get_vision_position_ids, stub)
    cases = []
    for types_, grids in ROPE:
        n = len(types_)
        for pad in (0, 3):  # left padding, as pplx's processor pads a batch
            ids = torch.zeros(1, n + pad, dtype=torch.long)
            kinds = torch.tensor([[0] * pad + types_], dtype=torch.int32)
            mask = torch.tensor([[0] * pad + [1] * n])
            grid = torch.tensor(grids, dtype=torch.long) if grids else None
            positions, delta = Qwen3_5Model.get_rope_index(
                stub, ids, mm_token_type_ids=kinds, image_grid_thw=grid, attention_mask=mask if pad else None
            )
            cases.append(
                {
                    "types": types_,
                    "grids": [list(g) for g in grids],
                    "pad": pad,
                    "positions": positions[:, 0, pad:].tolist(),
                    "padding": positions[:, 0, :pad].tolist(),
                    "delta": int(delta[0, 0]),
                }
            )
    return cases


def positions_npz(versions: str) -> None:
    rng = np.random.default_rng(20261023)
    table = torch.from_numpy(rng.standard_normal((48 * 48, 8)).astype(np.float32))
    arrays = {"table": table.numpy(), "versions": np.array(versions)}
    for h, w in GRIDS:
        grid = torch.tensor([[1, h, w]])
        indices, weights = get_vision_interpolation_indices_and_weights(
            grid, num_grid_per_side=48, mode="bilinear", align_corners=True, spatial_merge_size=2
        )
        # Qwen3_5VisionModel.forward: (self.pos_embed(interp_indices) * interp_weights[:, :, None]).sum(1)
        arrays[f"pos.{h}x{w}"] = (functional.embedding(indices, table) * weights[:, :, None]).sum(1).numpy()
        for hidden in (1024, 1152):
            rotary = Qwen3_5VisionRotaryEmbedding(Qwen3_5VisionConfig(hidden_size=hidden, num_heads=16))
            ids = get_vision_position_ids(grid, 2)
            cos, sin = rotary(torch.zeros(1), ids)
            arrays[f"cos.{hidden // 16}.{h}x{w}"] = cos.numpy()
            arrays[f"sin.{hidden // 16}.{h}x{w}"] = sin.numpy()
    np.savez_compressed(HERE / "qwen_positions.npz", **arrays)


def main() -> None:
    torch.set_num_threads(2)
    found = processors()
    ranges = {name: (p.size["shortest_edge"], p.size["longest_edge"]) for name, p in found.items()}
    images = []
    for i, (w, h) in enumerate(IMAGES):
        seed = 2000 + i
        pixels = pattern(h, w, seed)
        case = {"size": [w, h], "seed": seed, "image_sha256": sha(pixels)}
        for name, processor in found.items():
            try:
                out = processor(images=[Image.fromarray(pixels)], return_tensors="pt")
            except ValueError as error:
                case[name] = {"error": str(error)}
                continue
            values = out["pixel_values"].numpy()
            case[name] = {"grid": out["image_grid_thw"][0].tolist(), "pixel_sha256": sha(values.astype("<f4"))}
            print(name, w, h, case[name]["grid"], values.shape)
        images.append(case)
    # several images in one call: each image's own patches, in order (grouped by shape, then put back)
    for processor in found.values():
        sample = [pattern(h, w, 2000 + i) for i, (w, h) in enumerate(IMAGES[:6])]
        together = processor(images=[Image.fromarray(p) for p in sample], return_tensors="pt")["pixel_values"]
        apart = [processor(images=[Image.fromarray(p)], return_tensors="pt")["pixel_values"] for p in sample]
        assert torch.equal(together, torch.cat(apart))
    versions = (
        f"torch {torch.__version__}, torchvision {torchvision.__version__}, Pillow {PIL.__version__}, "
        f"transformers {transformers.__version__}; {platform.machine()}, Python {sys.version.split()[0]}"
    )
    out = {
        "source": "transformers' Qwen2VLImageProcessor (torchvision backend) and Qwen3_5Model",
        "versions": versions,
        "ranges": ranges,
        "processors": {
            name: json.loads((path / "processor_config.json").read_text()) for name, path in CHECKPOINTS.items()
        },
        "sizes": sizes(ranges),
        "images": images,
        "rope": rope_cases(),
    }
    (HERE / "qwen_preprocess.json").write_text(json.dumps(out, separators=(",", ":")) + "\n")
    positions_npz(versions)
    for file in ("qwen_preprocess.json", "qwen_positions.npz"):
        print(file, (HERE / file).stat().st_size, "bytes")


if __name__ == "__main__":
    main()
