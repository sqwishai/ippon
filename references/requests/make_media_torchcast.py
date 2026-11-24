"""The image requests Torchcast Decision 4B's media reference is made on:
python3 make_media_torchcast.py > media-torchcast.json

Deterministic. Torchcast's serving package (torchcast_decision v1.3.1, model.py) puts each image where a text or JSON
state says "<image>", or, with no such mark, ahead of the state ("Image 1: ..." lines for more than one). So: every
case of make_media_qwen.py (no marks), then marks inside a text state, inside a JSON state and as the whole state,
and a state whose marks do not match its images, which the package refuses.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from make_media_d1 import choice, noul
from make_media_qwen import requests as qwen_requests

IMAGES = Path(__file__).resolve().parents[1] / "media" / "images"


def requests() -> list[dict[str, Any]]:
    cases: list[tuple[str, list[str], Any, dict[str, Any]]] = [
        (
            "marks-in-text",
            ["small-circle-100x80.jpg", "table-invoice-1024x512.png"],
            "Customer email: the sticker on the parcel looked like this <image> and the invoice I got is <image>. "
            "Was I charged twice?",
            {
                "invoice": noul("Does the customer include an invoice?"),
                "shape": choice(
                    "What shape is on the sticker?",
                    {"circle": "A circle", "square": "A square", "triangle": "A triangle"},
                ),
            },
        ),
        (
            "mark-in-json",
            ["table-invoice-1024x512.png"],
            {"ticket": 912, "customer": "Ana", "attachment": "<image>", "message": "Is this my invoice?"},
            {"kind": choice("What is attached?", {"invoice": None, "photo": None, "chart": None})},
        ),
        ("mark-alone", ["chart-bars-800x600.png"], "<image>", {"chart": noul("Is this a bar chart?")}),
        (
            "marks-too-few",
            ["bison.jpg", "squares-512x512.png"],
            "Only one is marked: <image>.",
            {"animal": noul("Is there an animal?")},
        ),
    ]
    out = qwen_requests()
    for name, images, state, questions in cases:
        request = {"state": state, "images": images, "questions": questions}
        media = [
            {"name": f, "sha256": hashlib.sha256((IMAGES / f).read_bytes()).hexdigest()} for f in dict.fromkeys(images)
        ]
        out.append({"name": name, "request": request, "media": media})
    return out


if __name__ == "__main__":
    print(json.dumps(requests(), indent=1, ensure_ascii=False))
