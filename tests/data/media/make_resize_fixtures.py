# mypy: ignore-errors
"""Reference script, run once in a venv with torch, torchvision and Pillow, never in ippon's: torchvision's
own resizes, which tests/test_media_resize.py holds ippon's numpy ports to bit for bit.

    python -I make_resize_fixtures.py

Writes resize.npz next to this file: seeded uint8 RGB images and torchvision's `resize(..., BILINEAR | BICUBIC,
antialias=True)` of each, taken as the creators' vision code takes them (`pil_to_tensor` of a PIL image, so
channels-last strides), down, up, mixed, one axis only, odd sizes; also checked equal to the same resize of a
contiguous CHW tensor.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
from PIL import Image
from torchvision.transforms.v2 import functional as tvf

HERE = Path(__file__).resolve().parent

# (name, image, [(height, width), ...]): every target is resized from the image, bilinear and bicubic
SHAPES = {
    "odd": ((37, 53), [(16, 24), (64, 96), (80, 20), (37, 20), (12, 53), (1, 1), (41, 59)]),
    "wide": ((120, 97), [(3, 2), (119, 96), (31, 77)]),
    "thin": ((7, 300), [(32, 512), (2, 50)]),
    "smooth": ((96, 80), [(200, 136), (33, 29)]),
}


def image(name: str, height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    if name != "smooth":
        return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    y, x = np.mgrid[:height, :width]
    base = np.stack([x * 255 / width, y * 255 / height, (x + y) * 127 / (width + height)], -1)
    return np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)


def resized(img: np.ndarray, size: tuple[int, int], mode: tvf.InterpolationMode) -> np.ndarray:
    """As the creators' vision code: a PIL image through pil_to_tensor, then torchvision's resize with antialias."""
    x = tvf.pil_to_tensor(Image.fromarray(img))
    out = tvf.resize(x, list(size), interpolation=mode, antialias=True)
    contiguous = tvf.resize(torch.from_numpy(img).permute(2, 0, 1).contiguous(), list(size), mode, antialias=True)
    assert torch.equal(out, contiguous), "channels-last and contiguous inputs resize differently"
    return out.permute(1, 2, 0).contiguous().numpy()


def main() -> None:
    rng = np.random.default_rng(20261023)
    arrays: dict[str, np.ndarray] = {}
    for name, ((h, w), targets) in SHAPES.items():
        img = image(name, h, w, rng)
        arrays[f"{name}.input"] = img
        for th, tw in targets:
            for mode in (tvf.InterpolationMode.BILINEAR, tvf.InterpolationMode.BICUBIC):
                arrays[f"{name}.{mode.value}.{th}x{tw}"] = resized(img, (th, tw), mode)
    versions = f"torch {torch.__version__}, torchvision {torchvision.__version__}, Pillow {PIL.__version__}"
    machine = f"{platform.machine()}, {torch.backends.cpu.get_cpu_capability()}, Python {sys.version.split()[0]}"
    arrays["versions"] = np.array(f"{versions}; {machine}")
    np.savez_compressed(HERE / "resize.npz", **arrays)
    print("resize.npz", (HERE / "resize.npz").stat().st_size, "bytes")


if __name__ == "__main__":
    main()
