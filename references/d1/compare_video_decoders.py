"""Compare the actual ippon decoders with Liquid's recorded TorchCodec frames, without a tower.

uv run python references/d1/compare_video_decoders.py
Writes video-decoder-comparison.json. OpenCV uses the existing d3 decoder with Liquid's 768-frame ceiling;
PyAV uses Liquid's new path. The reference fixtures contain the creator's sampled frames before odd-frame padding.
"""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from ippon.media.d3_video import load_frames as opencv_frames
from ippon.media.decode import as_media
from ippon.media.lfm3_video import load_frames
from ippon.media.video import timestamps


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    directory = root / "references/d1"
    metadata = json.loads((directory / "video-processor.json").read_text())
    report: dict[str, Any] = {"versions": {k: version(k) for k in ("av", "opencv-python-headless")}, "clips": []}
    with np.load(directory / "video-processor.npz") as arrays:
        for i, case in enumerate(metadata["clips"]):
            path = root / "references/media" / case["name"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == case["sha256"]
            media = as_media(path, "video")
            wanted = arrays[f"frames_{i}"]
            record: dict[str, Any] = {"name": case["name"], "sampled_frames": len(wanted), "decoders": {}}
            for name, decode in (("pyav", load_frames), ("opencv", opencv_frames)):
                clip = decode(media, most=768)
                indices, frames = clip.indices, clip.frames
                if len(indices) > 1 and indices[-1] == indices[-2]:
                    indices, frames = indices[:-1], frames[:-1]
                same = indices == tuple(case["indices"])
                times = timestamps(indices, clip.fps)
                expected = timestamps(case["indices"], case["fps"])
                values: dict[str, Any] = {
                    "total": clip.total,
                    "fps": clip.fps,
                    "indices_equal": same,
                    "max_timestamp_difference": max(abs(a - b) for a, b in zip(times, expected, strict=True))
                    if len(times) == len(expected)
                    else None,
                    "timestamp_text_equal": [f"{t:.1f}" for t in times] == [f"{t:.1f}" for t in expected],
                }
                if same:
                    delta = np.abs(frames.astype(np.int16) - wanted.astype(np.int16))
                    values |= {
                        "max_pixel_difference": int(delta.max()),
                        "changed_channels": int(np.count_nonzero(delta)),
                        "channels": int(delta.size),
                    }
                record["decoders"][name] = values
            report["clips"].append(record)
    (directory / "video-decoder-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
