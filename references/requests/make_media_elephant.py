"""The image requests elephant-1's media reference is made on: python3 make_media_elephant.py > media-elephant.json

Deterministic. elephant-1's creator reads an image where its state names it, `<image:N>` (1-based, each image once, in
order: OneJev's qev/media.py); a request whose state names none has ippon's placement, the placeholders at the start of
the state. So: the cases of make_media_qwen.py up to about 3,000 image rows (elephant-1 reads an image at up to
16,777,216 pixels, its processor's own ceiling, so a 12 MP photo is 11,750 rows: too slow for its creator's code in
fp32 on a CPU), with the state at its start; then placeholders inside a text state, inside a JSON state, at a state's
end and a state's only content, and two that the creator refuses (out of order, one image not named).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from make_media_d1 import choice, noul, score
from make_media_qwen import requests as qwen_requests

IMAGES = Path(__file__).resolve().parents[1] / "media" / "images"
TAKEN = (
    "earth-noul",
    "aldrin-choice-3",
    "migrant-mother-score-5",
    "wright-three-questions",
    "exif-arrow",
    "gray-and-palette",
    "rgba-and-cmyk",
    "gray16",
    "below-the-floor",
    "banner-50-to-1",
    "receipt-1-to-8",
    "dashboard-score-10",
    "bison-twice",
)


def requests() -> list[dict[str, Any]]:
    by_name = {case["name"]: case for case in qwen_requests()}
    cases: list[tuple[str, list[str], Any, dict[str, Any]]] = [
        (
            "placeholders-in-text",
            ["small-circle-100x80.jpg", "table-invoice-1024x512.png"],
            "Customer email: the sticker on the parcel looked like this <image:1> and the invoice I got is "
            "<image:2>. Was I charged twice?",
            {
                "invoice": noul("Does the customer include an invoice?"),
                "shape": choice(
                    "What shape is on the sticker?",
                    {"circle": "A circle", "square": "A square", "triangle": "A triangle"},
                ),
            },
        ),
        (
            "placeholder-in-json",
            ["table-invoice-1024x512.png"],
            {"ticket": 912, "customer": "Ana", "attachment": "<image:1>", "message": "Is this my invoice?"},
            {"kind": choice("What is attached?", {"invoice": None, "photo": None, "chart": None})},
        ),
        (
            "placeholder-at-end",
            ["squares-512x512.png"],
            "Board photo, taken at the end of the shift:<image:1>",
            {"squares": score("How many squares are there?", ["none", "one or two", "three or four", "five or more"])},
        ),
        ("placeholder-alone", ["chart-bars-800x600.png"], "<image:1>", {"chart": noul("Is this a bar chart?")}),
        (
            "placeholders-out-of-order",
            ["bison.jpg", "squares-512x512.png"],
            "First <image:2>, then <image:1>.",
            {"animal": noul("Is there an animal?")},
        ),
        (
            "placeholder-missing",
            ["bison.jpg", "squares-512x512.png"],
            "Only one is named: <image:1>.",
            {"animal": noul("Is there an animal?")},
        ),
    ]
    out = [by_name[name] for name in TAKEN]
    for name, images, state, questions in cases:
        request = {"state": state, "images": images, "questions": questions}
        media = [
            {"name": f, "sha256": hashlib.sha256((IMAGES / f).read_bytes()).hexdigest()} for f in dict.fromkeys(images)
        ]
        out.append({"name": name, "request": request, "media": media})
    return out


if __name__ == "__main__":
    print(json.dumps(requests(), indent=1, ensure_ascii=False))
