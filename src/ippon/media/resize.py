"""Image resizes that give torchvision's numbers exactly: torchvision's `resize(uint8 tensor, BILINEAR | BICUBIC,
antialias=True)`, whose arithmetic native/src/lib.rs reproduces and whose output tests/test_media_resize.py holds it to
bit for bit. A clip's frames go to the native library in one call and resize in parallel: on an M5 Pro, 32 frames of
1280 x 720 to 576 x 320 (bicubic, d3-mini's video path) take 15 ms, where a numpy port takes 588 ms on one core.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .. import native

Mode = Literal["bilinear", "bicubic"]


def resize_frames(frames: np.ndarray, height: int, width: int, mode: Mode) -> np.ndarray:
    """torchvision's resize(uint8, mode, antialias=True) of each of [frames, H, W, C] uint8: [frames, height, width,
    C] uint8, a new array even when the size doesn't change."""
    if frames.dtype != np.uint8 or frames.ndim != 4:
        raise ValueError(f"expected [frames, H, W, C] uint8, not {frames.dtype} {frames.shape}")
    if height < 1 or width < 1:
        raise ValueError(f"cannot resize to {height} x {width}")
    out = np.empty((len(frames), height, width, frames.shape[3]), np.uint8)
    if out.size:
        native.resize(np.ascontiguousarray(frames), out, mode == "bicubic")
    return out
