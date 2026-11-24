# mypy: ignore-errors
"""Reference script, run once in a venv with Pillow 12.3.0 and transformers 5.17.0, never in ippon's: small
images in every format ippon reads, and the RGB pixels the creators' decoders get from them, which
tests/test_media_decode.py holds media/decode.py's `load_rgb` to.

    python -I make_decode_fixtures.py

Writes images/*.{jpg,png,webp,gif,bmp,tiff} (seeded, a few kB each) and decode.json next to this file: for each
image its format, and the sha256 and shape of
- "exif": transformers.image_utils.load_image(path), as d1-omni's and Clef's cards decode (EXIF turned, then RGB);
- "plain": PIL.Image.open(path).convert("RGB"), as pplx's protocol.py decodes (no EXIF turn).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import PIL
import transformers
from PIL import Image
from transformers.image_utils import load_image

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images"


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def exif(orientation: int) -> Image.Exif:
    e = Image.Exif()
    e[0x0112] = orientation
    return e


def main() -> None:
    IMAGES.mkdir(exist_ok=True)
    rng = np.random.default_rng(20261023)
    y, x = np.mgrid[:20, :28]
    smooth = np.stack([x * 9, y * 12, (x + y) * 5], -1)
    rgb = np.clip(smooth + rng.normal(0, 10, smooth.shape), 0, 255).astype(np.uint8)  # 20 x 28
    rgba = np.concatenate([rgb, (x * 9).astype(np.uint8)[..., None]], -1)
    image = Image.fromarray(rgb)
    files = {
        "plain.jpg": lambda p: image.save(p, quality=92),
        "exif6.jpg": lambda p: image.save(p, quality=92, exif=exif(6)),
        "exif3.jpg": lambda p: image.save(p, quality=92, exif=exif(3)),
        "cmyk.jpg": lambda p: image.convert("CMYK").save(p, quality=92),
        "gray.jpg": lambda p: image.convert("L").save(p, quality=92),
        "plain.png": lambda p: image.save(p),
        "exif8.png": lambda p: image.save(p, exif=exif(8)),
        "rgba.png": lambda p: Image.fromarray(rgba).save(p),
        "palette.png": lambda p: image.quantize(16).save(p),
        "gray16.png": lambda p: Image.fromarray(rgb[..., 0].astype(np.uint16) * 257).save(p),
        "lossless.webp": lambda p: image.save(p, lossless=True),
        "two.gif": lambda p: image.quantize(32).save(p, save_all=True, append_images=[image.rotate(180).quantize(32)]),
        "plain.bmp": lambda p: image.save(p),
        "plain.tiff": lambda p: image.save(p),
    }
    out = {"versions": f"Pillow {PIL.__version__}, transformers {transformers.__version__}", "images": {}}
    for name, save in files.items():
        path = IMAGES / name
        save(path)
        with Image.open(path) as opened:
            fmt, mode = opened.format, opened.mode
            plain = np.asarray(opened.convert("RGB"))
        turned = np.asarray(load_image(str(path)))
        out["images"][name] = {
            "format": fmt.lower(),
            "mode": mode,
            "exif": {"sha256": sha(turned), "shape": list(turned.shape)},
            "plain": {"sha256": sha(plain), "shape": list(plain.shape)},
        }
        print(name, fmt, mode, turned.shape, plain.shape, path.stat().st_size, "bytes")
    (HERE / "decode.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
