"""A video as a model reads it, whatever decoded it: its sampled frames (`Clip`), which frames a creator samples
(`sample`) and each temporal patch's time (`timestamps`). Each creator's own decoder reads the frames: d3_video.py
(OpenCV, vllm-sr d3's) and lfm3_video.py (PyAV, Liquid's d1).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# a decoded frame [H, W, 3] of a clip of so many frames, resized as the model's processor will resize it
Shrink = Callable[[np.ndarray, int], np.ndarray]


@dataclass(frozen=True)
class Clip:
    """A decoded clip: its sampled frames [frames, height, width, 3] uint8 RGB (an even number), each one's source
    frame number (the last repeated for an odd count), the clip's frame rate and its frame count."""

    frames: np.ndarray
    indices: tuple[int, ...]
    fps: float
    total: int


def sample(total: int, fps: float, rate: float = 2.0, least: int = 4, most: int = 32) -> list[int]:
    """The source frames read from a clip of `total` frames at `fps`: `rate` a second, at least `least` and at most
    `most` (and never more than the clip has), spread evenly from the first to the last."""
    count = int(total / fps * rate)
    count = min(max(count, least), most, total)
    out: list[int] = np.linspace(0, total - 1, count).round().astype(int).tolist()
    return out


def timestamps(indices: tuple[int, ...] | list[int], fps: float, temporal: int = 2) -> list[float]:
    """Each temporal patch's time in seconds, as Qwen3VLProcessor._calculate_timestamps gives it: every frame at
    its source index / fps, the indices padded to a multiple of `temporal` with the last, each patch's time the mean
    of its first and last frame's."""
    padded = list(indices)
    if len(padded) % temporal:
        padded.extend(padded[-1] for _ in range(temporal - len(padded) % temporal))
    times = [i / fps for i in padded]
    return [(times[i] + times[i + temporal - 1]) / 2 for i in range(0, len(times), temporal)]
