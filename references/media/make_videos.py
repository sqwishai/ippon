"""Synthetic video clips for the media references: python -I make_videos.py [--out videos]

Each clip is drawn frame by frame with numpy (no randomness) and encoded by the ffmpeg on PATH from raw RGB frames,
so the clips are this repository's own work, dedicated CC0 1.0 (LICENSES.md). Each covers an edge case of vllm-sr
d3's video path (2 frames per second, 4 to 32 of them spread over the clip, each frame read at up to 200,704 pixels):

    ball-h264-320x240-24fps-3s.mp4       H.264 in MP4: 72 frames, 6 read; 240 rounds up to 256 (240 / 32 = 7.5)
    bar-h264-1280x720-29.97fps-20s.mp4   H.264 at 30000/1001 fps: 20 s would be 40 frames, 32 are read; 1280x720
                                         is over the per-frame budget, read at 576x320
    square-vp9-160x120-10fps-2.5s.webm   VP9 in WebM (no frame count in the container): 5 frames read, an odd
                                         number, so the last is read twice
    flash-h264-64x48-5fps-1s.mp4         five frames: 2 frames per second would be 2, the floor is 4
    tiny-h264-24x16-4fps-3s.mp4          frames under 32 pixels a side, scaled up before rounding
    portrait-mjpeg-180x320-15fps-4s.mov  Motion JPEG in a QuickTime file, portrait
    rings-mpeg4-200x200-12fps-6s.mkv     MPEG-4 Part 2 in Matroska

Byte-identical output needs the ffmpeg the committed files were made with (9.0.1, libx264 and libvpx single-threaded);
the committed files, with their sha256 in requests/media-video.json, are what the references are made on.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np

Frame = Callable[[int, int], np.ndarray]  # (frame number, frame count) -> H x W x 3 uint8


def canvas(width: int, height: int, colour: tuple[int, int, int]) -> np.ndarray:
    return np.broadcast_to(np.array(colour, np.uint8), (height, width, 3)).copy()


def disc(image: np.ndarray, cx: float, cy: float, r: float, colour: tuple[int, int, int]) -> None:
    h, w = image.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    image[(x - cx) ** 2 + (y - cy) ** 2 <= r * r] = colour


def box(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, colour: tuple[int, int, int]) -> None:
    image[max(y0, 0) : max(y1, 0), max(x0, 0) : max(x1, 0)] = colour


def ball(n: int, count: int) -> np.ndarray:
    """A red ball rolling from left to right over a white floor."""
    image = canvas(320, 240, (250, 250, 250))
    box(image, 0, 200, 320, 240, (120, 120, 120))
    disc(image, 30 + 260 * n / (count - 1), 170, 28, (220, 30, 30))
    return image


def bar(n: int, count: int) -> np.ndarray:
    """A bar growing across a dark screen, blue for the first half and green after it."""
    image = canvas(1280, 720, (24, 24, 32))
    colour = (40, 90, 230) if n < count // 2 else (40, 200, 80)
    box(image, 80, 300, 80 + round(1120 * (n + 1) / count), 420, colour)
    box(image, 80, 440, 1200, 450, (200, 200, 200))
    return image


def square(n: int, count: int) -> np.ndarray:
    """A square that stays put and turns from yellow to purple."""
    image = canvas(160, 120, (255, 255, 255))
    t = n / (count - 1)
    box(image, 50, 30, 110, 90, (round(240 - 110 * t), round(220 - 190 * t), round(30 + 170 * t)))
    return image


def flash(n: int, count: int) -> np.ndarray:
    """A black screen that flashes white on its middle frame."""
    return canvas(64, 48, (255, 255, 255) if n == count // 2 else (0, 0, 0))


def tiny(n: int, count: int) -> np.ndarray:
    """A 24 x 16 frame: a green dot moving down the middle."""
    image = canvas(24, 16, (0, 0, 0))
    box(image, 10, n * 16 // count, 14, n * 16 // count + 4, (0, 255, 0))
    return image


def portrait(n: int, count: int) -> np.ndarray:
    """An upright frame: an orange ball falling from the top to the bottom."""
    image = canvas(180, 320, (230, 240, 255))
    disc(image, 90, 30 + 260 * n / (count - 1), 22, (240, 140, 20))
    return image


def rings(n: int, count: int) -> np.ndarray:
    """Concentric rings spreading out from the centre."""
    y, x = np.mgrid[0:200, 0:200]
    r = np.sqrt((x - 100.0) ** 2 + (y - 100.0) ** 2)
    on = ((r - 4 * n) % 40) < 16
    image = canvas(200, 200, (255, 255, 255))
    image[on] = (30, 30, 160)
    return image


CODECS = {
    "h264": ["-c:v", "libx264", "-preset", "medium", "-crf", "23", "-threads", "1", "-pix_fmt", "yuv420p"],
    "vp9": ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32", "-threads", "1", "-pix_fmt", "yuv420p"],
    "mjpeg": ["-c:v", "mjpeg", "-q:v", "4", "-pix_fmt", "yuvj420p"],
    "mpeg4": ["-c:v", "mpeg4", "-q:v", "10", "-pix_fmt", "yuv420p"],
}
CLIPS: list[tuple[str, Frame, str, str, int]] = [  # file, frames, codec, rate, frame count
    ("ball-h264-320x240-24fps-3s.mp4", ball, "h264", "24", 72),
    ("bar-h264-1280x720-29.97fps-20s.mp4", bar, "h264", "30000/1001", 600),
    ("square-vp9-160x120-10fps-2.5s.webm", square, "vp9", "10", 25),
    ("flash-h264-64x48-5fps-1s.mp4", flash, "h264", "5", 5),
    ("tiny-h264-24x16-4fps-3s.mp4", tiny, "h264", "4", 12),
    ("portrait-mjpeg-180x320-15fps-4s.mov", portrait, "mjpeg", "15", 60),
    ("rings-mpeg4-200x200-12fps-6s.mkv", rings, "mpeg4", "12", 72),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "videos")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for name, draw, codec, rate, count in CLIPS:
        frames = [draw(n, count) for n in range(count)]
        h, w = frames[0].shape[:2]
        command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}"]
        command += ["-framerate", rate, "-i", "-", *CODECS[codec], "-an", "-map_metadata", "-1"]
        command += ["-fflags", "+bitexact", "-flags:v", "+bitexact", str(a.out / name)]
        subprocess.run(command, input=b"".join(f.tobytes() for f in frames), check=True)
        data = (a.out / name).read_bytes()
        print(f"{name}: {len(data):,} bytes, sha256 {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
