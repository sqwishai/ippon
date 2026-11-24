"""vllm-sr d3's videos, decoded as its runtime decodes them (d3_runtime.py, `load_video`): with OpenCV's
FFmpeg backend, the decoder its code uses, so a clip decodes to the same frames as theirs.

- the clip's bytes are written to a temporary file and opened by path (cv2.VideoCapture with CAP_FFMPEG), as the
  creator reads encoded bytes;
- its frame rate is the container's (CAP_PROP_FPS), or 24 when that is missing or outside 0.1 to 1,000; its frame
  count the container's (CAP_PROP_FRAME_COUNT), or counted by grabbing every frame when the container gives none;
- `sample` frames are read: `fps` a second (2), at least `least` (4) and at most `most` (32), spread evenly over
  the whole clip (numpy's linspace from the first frame to the last, rounded half to even); the clip is read
  through in order and a sampled frame that can't be grabbed (a container that promises more frames than it holds)
  is left out;
- each kept frame is converted from OpenCV's BGR to RGB (cv2.cvtColor), and an odd count's last frame is read
  twice, so frames pair up for the temporal patches.

OpenCV is imported when a model first reads a video.
"""

from __future__ import annotations

import importlib
import math
import os
import tempfile
from typing import Any

import numpy as np

from ..errors import InvalidRequest
from .decode import Media
from .video import Clip, Shrink, sample

UNKNOWN_FPS = 24.0  # the creator's frame rate for a clip that names none, or a rate outside SOURCE_FPS
SOURCE_FPS = (0.1, 1000.0)


def opencv() -> Any:
    """OpenCV's cv2 module, imported when a model first reads a video: it takes 184 ms to import (M5 Pro), which a
    command or a text-only model shouldn't wait for."""
    return importlib.import_module("cv2")


def load_frames(media: Media, rate: float = 2.0, least: int = 4, most: int = 32, shrink: Shrink | None = None) -> Clip:
    """A clip's sampled frames, decoded as the creator decodes them (the module's docstring). InvalidRequest when
    OpenCV can't open or decode it. Any length and frame size, but the model reads `most` sampled frames (32) at
    most, evenly spread: past the creator's server's 300 s, events shorter than the gap between them can be missed
    (`Clip` says how many were read, over how long)."""
    if media.kind != "video":
        raise InvalidRequest(f"{'an image' if media.kind == 'image' else 'an audio clip'} is not a video")
    cv2 = opencv()
    try:
        api = cv2.CAP_FFMPEG if cv2.videoio_registry.hasBackend(cv2.CAP_FFMPEG) else cv2.CAP_ANY
    except (AttributeError, cv2.error):
        api = cv2.CAP_ANY
    handle = tempfile.NamedTemporaryFile(prefix="ippon-video-", delete=False)  # noqa: SIM115 (closed in the try)
    path = handle.name
    frames: list[np.ndarray] = []
    kept: list[int] = []
    reader = None
    try:
        with handle:
            handle.write(media.data)
        reader = cv2.VideoCapture(path, api)
        if not reader.isOpened():
            raise InvalidRequest("the video could not be opened")
        width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise InvalidRequest("the video has no frame size")
        fps = float(reader.get(cv2.CAP_PROP_FPS) or 0.0)
        if not (math.isfinite(fps) and SOURCE_FPS[0] <= fps <= SOURCE_FPS[1]):
            fps = UNKNOWN_FPS
        total = int(reader.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:  # the container does not say: count the frames, then read again
            while reader.grab():
                total += 1
            reader.release()
            reader = cv2.VideoCapture(path, api)
        if total <= 0:
            raise InvalidRequest("the video has no frames")
        indices = sample(total, fps, rate, least, most)
        wanted = set(indices)
        for number in range(indices[-1] + 1):
            if not reader.grab():  # the end of what decodes: a frame count read from a header can be anything
                break
            if number in wanted:
                ok, frame = reader.retrieve()
                if ok and frame is not None and frame.size:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(rgb if shrink is None else shrink(rgb, len(indices) + len(indices) % 2))
                    kept.append(number)
    except cv2.error as error:
        raise InvalidRequest(f"the video could not be decoded ({error.__class__.__name__})") from None
    finally:
        if reader is not None:
            reader.release()
        os.unlink(path)
    if not frames:
        raise InvalidRequest("no frame of the video could be decoded")
    if len(frames) % 2:
        frames.append(frames[-1])
        kept.append(kept[-1])
    if len({f.shape for f in frames}) != 1:
        raise InvalidRequest("the video's frames are not all the same size")
    return Clip(np.ascontiguousarray(np.stack(frames)), tuple(kept), fps, total)
