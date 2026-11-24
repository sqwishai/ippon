"""The image requests H2O-Lightning-4B's media reference is made on: python3 make_media_h2o.py > media-h2o.json

Deterministic: every case of make_media_qwen.py (H2O's server reads images ahead of the record, at most 1,048,576
pixels each, so the 12 MP photos are scaled down and fit), then five images in one request, which the server refuses
(serve_config.json's image.max_images, 4).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from make_media_d1 import noul
from make_media_qwen import requests as qwen_requests

IMAGES = Path(__file__).resolve().parents[1] / "media" / "images"


def requests() -> list[dict[str, Any]]:
    out = qwen_requests()
    images = ["bison.jpg", "squares-512x512.png", "solid-blue-24x24.png", "gradient-512x384.png", "circles-720x720.png"]
    request = {"state": "Five photos.", "images": images, "questions": {"animal": noul("Is there an animal?")}}
    media = [{"name": f, "sha256": hashlib.sha256((IMAGES / f).read_bytes()).hexdigest()} for f in images]
    out.append({"name": "five-images", "request": request, "media": media})
    return out


if __name__ == "__main__":
    print(json.dumps(requests(), indent=1, ensure_ascii=False))
