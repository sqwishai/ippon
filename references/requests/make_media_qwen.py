"""The image requests pplx-decider flash v1.2's and Clef-Flash's media references are made on:
python3 make_media_qwen.py > media-qwen.json

Deterministic. Each case is a System One request whose `images` are file names in ../media/images/ (in order), with
each file's sha256 under `media`, as make_media_d1.py lists them; whoever runs a case opens those files and sends
their bytes. Both creators' code needs a state (null is one), and a choice's criteria as a map. Covered: every question
type (yes/no with and without criteria, choices of 3 to 8 options with and without descriptions, scores of 3, 5 and
10 levels), states null, empty, text, JSON and long, 1 to 3 questions, 1, 2 and 3 images and one image twice, the
decoders' edge cases (an EXIF-rotated JPEG, grayscale, palette, RGBA, CMYK, 16-bit), images below the processor's
floor, at its ceiling, far over pplx's (a 12 MP photo) and 50:1 and 1:8 aspects. On Clef-Flash's 16,384-token window:
a long state cut to what a 12 MP photo leaves, and two such photos, which Clef's code refuses.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from make_media_d1 import choice, long_text, noul, score

IMAGES = Path(__file__).resolve().parents[1] / "media" / "images"


def requests() -> list[dict[str, Any]]:
    rng = random.Random(23)
    long_state = long_text(rng, 400)  # about 6,000 tokens: with a 12 MP photo, more than Clef-Flash's window holds
    directions = {"up": None, "down": None, "left": None, "right": None}
    cases: list[tuple[str, list[str], Any, dict[str, Any]]] = [
        (
            "earth-noul",
            ["earth-apollo17.jpg"],
            "Photo from a NASA archive.",
            {"earth": noul("Is this a photograph of the Earth from space?")},
        ),
        (
            "aldrin-choice-3",
            ["aldrin-apollo11.jpg"],
            "",
            {
                "place": choice(
                    "Where was this photo taken?",
                    {"moon": "On the surface of the Moon", "desert": "In a desert on Earth", "studio": "In a studio"},
                )
            },
        ),
        (
            "migrant-mother-score-5",
            ["migrant-mother.jpg"],
            None,
            {
                "mood": score(
                    "How happy do the people in the photo look?",
                    ["very unhappy", "unhappy", "neutral", "happy", "very happy"],
                )
            },
        ),
        (
            "wright-three-questions",
            ["wright-first-flight.jpg"],
            "Archive photograph, December 1903.",
            {
                "aircraft": noul("Is an aircraft visible?"),
                "people": choice(
                    "How many people are in the photo?",
                    {"none": None, "one": None, "two": None, "three_or_more": "Three or more"},
                ),
                "sharpness": score("How sharp is the photo?", ["blurry", "acceptable", "sharp"]),
            },
        ),
        ("exif-arrow", ["arrow-exif6.jpg"], None, {"direction": choice("Which way does the arrow point?", directions)}),
        (
            "gray-and-palette",
            ["gray-gradient-640x480.png", "palette-stripes-600x400.png"],
            {"ticket": 4812, "channel": "email", "attachments": 2, "note": "Customer sent two images."},
            {
                "colour": choice(
                    "Which of the images is in colour?",
                    {"first": "The first image", "second": "The second image", "both": "Both", "neither": "Neither"},
                ),
                "text": noul("Does either image contain text?"),
            },
        ),
        (
            "rgba-and-cmyk",
            ["rgba-circle-512x384.png", "cmyk-blocks-640x480.jpg"],
            "Two files from the print shop.",
            {
                "background": choice(
                    "What colour is the first image's background?",
                    {"white": None, "black": None, "magenta": None, "green": None},
                ),
                "blocks": noul(
                    "Does the second image show coloured blocks?",
                    {"true": "Coloured blocks are visible", "false": "No coloured blocks"},
                ),
            },
        ),
        ("gray16", ["gray16-gradient-512x256.png"], None, {"white": noul("Is the image almost entirely white?")}),
        (
            "below-the-floor",
            ["solid-red-1x1.png", "small-circle-100x80.jpg"],
            None,
            {"colour": choice("What colour is the first image?", {"red": None, "green": None, "blue": None})},
        ),
        (
            "banner-50-to-1",
            ["banner-3000x60.png"],
            None,
            {
                "announces": choice(
                    "What does the banner announce?",
                    {
                        "sale": "A sale or discount",
                        "closure": "A shop closing",
                        "hiring": "A job opening",
                        "event": "A concert",
                    },
                )
            },
        ),
        (
            "receipt-1-to-8",
            ["receipt-400x3200.png"],
            "Expense claim attachment.",
            {
                "receipt": noul("Is this a shop receipt?"),
                "total": choice("What is the total?", {"a": "87.38", "b": "287.38", "c": "1,287.38"}),
            },
        ),
        (
            "dashboard-score-10",
            ["dashboard-1000x800.jpg"],
            "Screenshot from the on-call channel.",
            {
                "online": noul("Is the system shown as online?"),
                "health": score("How healthy does the system look, from 0 to 9?", [f"level {i}" for i in range(10)]),
            },
        ),
        (
            "three-photos",
            ["aldrin-apollo11.jpg", "earth-apollo17.jpg", "bison.jpg"],
            "Three photos from a museum's online collection.",
            {
                "animal": noul("Does any of the images show an animal?"),
                "first": choice(
                    "What does the first image show?",
                    {"astronaut": "An astronaut", "planet": "A planet", "animal": "An animal"},
                ),
                "variety": score(
                    "How different are the images from each other?",
                    ["all alike", "somewhat alike", "different", "very different"],
                ),
            },
        ),
        (
            "bison-twice",
            ["bison.jpg", "bison.jpg"],
            "The same photo, sent twice.",
            {
                "animal": choice(
                    "Which animal is shown?",
                    dict.fromkeys(["bison", "horse", "cow", "deer", "bear", "wolf", "elk", "sheep"]),
                )
            },
        ),
        (
            "shapes-12mp",
            ["shapes-4000x3000.jpg"],
            "Calibration board, lot photo.",
            {
                "red": choice(
                    "How many red circles are there?",
                    {"one": None, "two": None, "three": None, "four": None, "five": None, "six": None},
                ),
                "text": noul("Is there any text in the image?"),
            },
        ),
        (
            "shapes-12mp-long-state",
            ["shapes-4000x3000.jpg"],
            long_state,
            {"board": noul("Does the photo show a calibration board?")},
        ),
        (
            "shapes-12mp-twice",
            ["shapes-4000x3000.jpg", "chart-bars-800x600.png", "shapes-4000x3000.jpg"],
            "Lot photos and the week's chart.",
            {"same": noul("Are the first and the last image the same photo?")},
        ),
    ]
    out = []
    for name, images, state, questions in cases:
        request = {"state": state, "images": images, "questions": questions}
        media = [
            {"name": f, "sha256": hashlib.sha256((IMAGES / f).read_bytes()).hexdigest()} for f in dict.fromkeys(images)
        ]
        out.append({"name": name, "request": request, "media": media})
    return out


if __name__ == "__main__":
    print(json.dumps(requests(), indent=1, ensure_ascii=False))
