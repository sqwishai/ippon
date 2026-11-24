"""Image requests for Liquid's d1 media references: python3 make_media_d1.py > media-d1.json

Deterministic. Each case is a System One request whose `images` are file names in ../media/images/ (in order), with
each file's sha256 under `media`; whoever runs a case opens those files and sends their bytes. They were made for
the retired d1-omni-600M, whose rules hold throughout: images, never audio; at most 896 tokens of state and question
text, fewer once the images take more than 16,384 - 896 positions (`squeeze`, whose seven images take 15,679).
Covered: every question type (yes/no with and without criteria and with only one side given, choices of 2 to 8
options with and without descriptions, scores of 3, 4, 5 and 10 levels), states absent, empty, short, JSON and long
(cut to the budget), 1 to 5 questions, 1, 2, 3 and 7 images (one repeated), and the edge cases make_synthetic.py
lists.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

IMAGES = Path(__file__).resolve().parents[1] / "media" / "images"
WORDS = [
    "photo",
    "inspection",
    "site",
    "panel",
    "label",
    "damage",
    "corner",
    "crack",
    "colour",
    "light",
    "shadow",
    "frame",
    "report",
    "camera",
    "angle",
    "surface",
    "edge",
    "mark",
    "scale",
    "batch",
]


def long_text(rng: random.Random, sentences: int) -> str:
    return " ".join(
        " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 16))).capitalize() + "." for _ in range(sentences)
    )


def noul(instructions: str, criteria: dict[str, str] | None = None) -> dict[str, Any]:
    return {"type": "noul", "instructions": instructions, **({"criteria": criteria} if criteria is not None else {})}


def choice(instructions: str, criteria: dict[str, str | None]) -> dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: str, levels: list[str]) -> dict[str, Any]:
    return {"type": "score", "instructions": instructions, "criteria": levels}


def requests() -> list[dict[str, Any]]:
    rng = random.Random(11)
    long_state = long_text(rng, 90)  # about 1,400 tokens: cut to the text budget
    counts = {"two": None, "three": None, "four": None, "five": None, "six": None}
    colours = {"red": None, "green": None, "blue": None, "yellow": None}
    cases: list[tuple[str, list[str], Any, dict[str, Any]]] = [
        ("earth-noul", ["earth-apollo17.jpg"], None, {"earth": noul("Is this a photograph of the Earth from space?")}),
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
        (
            "lincoln-noul-criteria",
            ["lincoln-1863.jpg"],
            None,
            {
                "colour": noul(
                    "Is this a colour photograph?",
                    {"true": "The photo is in colour", "false": "The photo is black and white"},
                )
            },
        ),
        (
            "bison-choice-8",
            ["bison.jpg"],
            None,
            {
                "animal": choice(
                    "Which animal is shown?",
                    dict.fromkeys(["bison", "horse", "cow", "deer", "bear", "wolf", "elk", "sheep"]),
                )
            },
        ),
        (
            "vegetables-two-questions",
            ["vegetables.jpg"],
            "Produce photo for a grocery listing.",
            {
                "shown": choice(
                    "What is shown?",
                    {"vegetables": "Fresh vegetables", "fruit": "Fruit only", "bread": "Baked goods", "meat": "Meat"},
                ),
                "green": noul("Is any of the produce green?"),
            },
        ),
        (
            "constitution-scan",
            ["constitution-page1.jpg"],
            "Scanned page from the National Archives.",
            {
                "handwritten": noul("Is the text handwritten?"),
                "kind": choice(
                    "What kind of document is this?",
                    {
                        "letter": "A personal letter",
                        "charter": "A founding legal or government document",
                        "newspaper": "A newspaper page",
                        "map": "A map",
                        "receipt": "A shop receipt",
                    },
                ),
            },
        ),
        (
            "chart-three-types",
            ["chart-bars-800x600.png"],
            None,
            {
                "tallest": choice(
                    "Which region has the tallest bar?",
                    {"North": None, "South": None, "East": None, "West": None, "Central": None},
                ),
                "title": noul("Does the chart have a title?"),
                "bars": score("How many bars are there?", ["two", "three", "four", "five", "six"]),
            },
        ),
        (
            "table-total",
            ["table-invoice-1024x512.png"],
            "Invoice attached to an expense claim.",
            {
                "total": choice(
                    "What is the invoice total?",
                    {"a": "$152.84", "b": "$252.84", "c": "$225.48", "d": "$89.00"},
                ),
                "bulk": noul("Does any row have a quantity above 5?", {"false": "No row has a quantity above 5"}),
            },
        ),
        ("solid-1x1", ["solid-red-1x1.png"], None, {"colour": choice("What colour is the image?", colours)}),
        (
            "solid-24-long-state",
            ["solid-blue-24x24.png"],
            long_state,
            {
                "colour": choice(
                    "What colour is the image?",
                    {"red": "Mostly red", "green": "Mostly green", "blue": "Mostly blue", "yellow": "Mostly yellow"},
                ),
                "flat": noul("Is the image a single flat colour?"),
            },
        ),
        (
            "small-circle",
            ["small-circle-100x80.jpg"],
            None,
            {"circle": noul("Is there a circle?", {"true": "A round shape is visible", "false": "No round shape"})},
        ),
        (
            "gradient-exact-size",
            ["gradient-512x384.png"],
            None,
            {
                "ends": choice(
                    "Which colours does the gradient run between?",
                    {"red_blue": "Red to blue", "black_white": "Black to white", "green_yellow": "Green to yellow"},
                ),
                "smooth": noul("Is the image a smooth gradient?"),
            },
        ),
        (
            "squares-at-budget",
            ["squares-512x512.png"],
            None,
            {"count": choice("How many green squares are there?", {"two": None, "three": None, "four": None})},
        ),
        ("circles-720", ["circles-720x720.png"], None, {"count": choice("How many circles are there?", counts)}),
        ("circles-725-tiled", ["circles-725x725.png"], None, {"count": choice("How many circles are there?", counts)}),
        (
            "dashboard-grid-tie",
            ["dashboard-1000x800.jpg"],
            "Screenshot from the on-call channel.",
            {
                "online": noul("Is the system shown as online?"),
                "health": score("How healthy does the system look, from 0 to 9?", [f"level {i}" for i in range(10)]),
            },
        ),
        (
            "shapes-4000x3000",
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
            "receipt-1-to-8",
            ["receipt-400x3200.png"],
            None,
            {
                "receipt": noul("Is this a shop receipt?"),
                "total": choice("What is the total?", {"a": "87.38", "b": "287.38", "c": "1,287.38"}),
            },
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
            "rgba-background",
            ["rgba-circle-512x384.png"],
            None,
            {
                "background": choice(
                    "What colour is the background?",
                    {"white": None, "black": None, "magenta": None, "green": None},
                ),
                "circle": noul("Is there a green circle?"),
            },
        ),
        (
            "exif-arrow",
            ["arrow-exif6.jpg"],
            None,
            {
                "direction": choice(
                    "Which way does the arrow point?", {"up": None, "down": None, "left": None, "right": None}
                )
            },
        ),
        (
            "cmyk-and-16-bit",
            ["cmyk-blocks-640x480.jpg", "gray16-gradient-512x256.png"],
            "Two scans from the print shop.",
            {
                "blocks": choice(
                    "How many coloured blocks are in the first image?",
                    {"two": None, "three": None, "four": None, "five": None},
                ),
                "white": noul("Is the second image almost entirely white?"),
            },
        ),
        (
            "three-photos-five-questions",
            ["aldrin-apollo11.jpg", "earth-apollo17.jpg", "bison.jpg"],
            long_state,
            {
                "animal": noul("Does any of the images show an animal?"),
                "space": noul(
                    "Was any of the images taken in space?",
                    {"true": "At least one was taken in space", "false": "All were taken on the ground"},
                ),
                "first": choice(
                    "What does the first image show?",
                    {"astronaut": "An astronaut", "planet": "A planet", "animal": "An animal"},
                ),
                "era": choice("Are these photos historic or recent?", {"historic": None, "recent": None}),
                "variety": score(
                    "How different are the images from each other?",
                    ["all alike", "somewhat alike", "different", "very different"],
                ),
            },
        ),
        (
            "squeeze",
            ["receipt-400x3200.png"] * 5 + ["aldrin-apollo11.jpg", "shapes-4000x3000.jpg"],
            long_state,
            {"receipt": noul("Does any of the images show a receipt?")},
        ),
    ]
    out = []
    for name, images, state, questions in cases:
        request: dict[str, Any] = {} if state is None else {"state": state}
        request |= {"images": images, "questions": questions}
        media = [
            {"name": f, "sha256": hashlib.sha256((IMAGES / f).read_bytes()).hexdigest()} for f in dict.fromkeys(images)
        ]
        out.append({"name": name, "request": request, "media": media})
    return out


if __name__ == "__main__":
    print(json.dumps(requests(), indent=1, ensure_ascii=False))
