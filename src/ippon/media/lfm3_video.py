"""Video sampling with PyAV's FFmpeg decoder, as the creators' own code samples their models' videos.

Liquid's processor uses TorchCodec in exact-seek mode: the decoded frame count divided by the stream's actual
duration, then 2 fps uniformly over the whole clip, at least 4 and at most 768 (or the clip's length). Keep an odd
sample count until preprocessing: resize uses that count, then patchification repeats the last frame. The
reference fixtures compare the RGB frames and clock with the processor's TorchCodec output.
"""

from __future__ import annotations

import bisect
import importlib
import io
import itertools
import math
from typing import Any

import numpy as np

from ..errors import InvalidRequest
from .decode import Media
from .video import Clip, Shrink, sample

# FFmpeg's frame and slice threads: PyAV decodes on one thread unless asked, and its output is the same bits either
# way. A 20 s 1280x720 H.264 clip, read through twice (counted, then sampled), took 315 ms on one thread and 60 ms
# threaded on an M5 Pro, its 40 sampled frames identical; a 3 s 320x240 one 6.1 and 4.7 ms.
THREADS = "AUTO"


def pyav() -> Any:
    """PyAV, imported when a model first reads a video: it takes 138 ms to import (M5 Pro), which a command or a
    text-only model shouldn't wait for."""
    return importlib.import_module("av")


def load_frames(media: Media, rate: float = 2.0, least: int = 4, most: int = 768, shrink: Shrink | None = None) -> Clip:
    """Sampled RGB frames and their clock, counted over the video stream's actual presentation interval.
    No audio track is read: Liquid's decision Reader does not enable use_audio_in_video.

    The clock comes from the stream's packets, which carry each frame's timestamp without decoding it (a 30 s 1080p
    HEVC clip: 2 ms instead of 700). The pass that picks the sampled frames decodes to the end and counts them: where
    the frames decoded aren't one per packet, the clock is taken from the decoded frames, as Liquid's processor takes
    it, and the frames picked again. With `shrink`, each frame is resized as it is decoded."""
    if media.kind != "video":
        raise InvalidRequest("expected a video")
    av = pyav()
    try:
        timed = _packet_times(media, av)
        frames, decoded = _picked(media, av, timed, rate, least, most, shrink)
        if len(decoded) != len(timed):  # not one frame per packet: the decoded frames' clock, as Liquid's
            frames, again = _picked(media, av, decoded, rate, least, most, shrink)
            assert len(again) == len(decoded)
            timed = decoded
    except av.error.FFmpegError:
        raise InvalidRequest("the video could not be decoded by PyAV's FFmpeg decoder") from None
    total, fps = len(timed), _fps(timed)
    indices = sample(total, fps, rate, least, most)
    if len(frames) != len(indices) or len({f.shape for f in frames}) != 1:
        raise InvalidRequest("the video's sampled frames are missing or change size")
    return Clip(np.ascontiguousarray(np.stack(frames)), tuple(indices), fps, total)


def _packet_times(media: Media, av: Any) -> list[tuple[float, float]]:
    """Each frame's start and duration in seconds, read from the video stream's packets (none decoded)."""
    with av.open(io.BytesIO(media.data)) as container:
        if not container.streams.video:
            raise InvalidRequest("the video has no video stream")
        stream = container.streams.video[0]
        timed = []
        for packet in container.demux(stream):
            if packet.size == 0:  # the demuxer's closing empty packet
                continue
            if packet.pts is None:
                raise InvalidRequest("Liquid's video processor needs frame timestamps")
            timed.append((float(packet.pts * stream.time_base), float((packet.duration or 0) * stream.time_base)))
    return timed


def _fps(timed: list[tuple[float, float]]) -> float:
    """Frames a second over the presentation interval, from each frame's start and duration. A frame without a
    duration (a browser's WebM leaves them out) lasts until the next frame starts, the last as long as the gap before
    it."""
    starts = sorted(start for start, _ in timed)
    gaps = [b - a for a, b in itertools.pairwise(starts) if b > a]
    begin, end = math.inf, -math.inf
    for start, duration in timed:
        if not duration:
            at = bisect.bisect_right(starts, start)
            duration = starts[at] - start if at < len(starts) else (gaps[-1] if gaps else 0.0)
        begin, end = min(begin, start), max(end, start + duration)
    if not timed or not math.isfinite(end - begin) or end <= begin:
        raise InvalidRequest("the video has no timed frames")
    return len(timed) / (end - begin)


def _picked(
    media: Media, av: Any, timed: list[tuple[float, float]], rate: float, least: int, most: int, shrink: Shrink | None
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """The frames sampled by the clock `timed` gives, and every decoded frame's start and duration."""
    indices = set(sample(len(timed), _fps(timed), rate, least, most)) if timed else set()
    frames, decoded = [], []
    with av.open(io.BytesIO(media.data)) as container:
        stream = container.streams.video[0]
        stream.thread_type = THREADS
        for number, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise InvalidRequest("Liquid's video processor needs frame timestamps")
            decoded.append((float(frame.pts * frame.time_base), float((frame.duration or 0) * frame.time_base)))
            if number in indices:
                rgb = frame.to_ndarray(format="rgb24")
                if frame.rotation:
                    rgb = np.rot90(rgb, round(frame.rotation / 90))
                frames.append(rgb if shrink is None else shrink(np.ascontiguousarray(rgb), len(indices)))
    return frames, decoded
