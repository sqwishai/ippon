"""Liquid video requests: python references/requests/make_media_d1_video.py > references/requests/media-d1-video.json.

Uses the existing CC0 synthetic and Commons clips (references/media/LICENSES.md). Includes the real clip's audio
track, which Liquid's decision API leaves unread; pictures precede videos, in each list's order. The processor-only
fixtures also exercise the full pixel budget and the 768-frame cap without a language-model forward.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from make_media_video import requests as d3_requests


def requests() -> list[dict[str, Any]]:
    cases = [c for c in d3_requests() if c["name"] not in ("example-square", "bar-32-frames", "too-many-video-tokens")]
    tiny = "videos/tiny-h264-24x16-4fps-3s.mp4"
    media = Path(__file__).resolve().parents[1] / "media"
    cases.append(
        {
            "name": "video-two-question-rows",
            "request": {
                "state": "A small test recording.",
                "videos": [tiny],
                "questions": {
                    f"q{i}": {"type": "noul", "instructions": "Is anything green visible?"} for i in range(17)
                },
            },
            "media": [{"name": tiny, "sha256": hashlib.sha256((media / tiny).read_bytes()).hexdigest()}],
        }
    )
    return cases


if __name__ == "__main__":
    print(json.dumps(requests(), indent=1, ensure_ascii=False))
