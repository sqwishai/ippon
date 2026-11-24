"""Ask a decision model about a photo, a document image and a short video, each by its file's path.

    uv init --bare && uv add ippon && uv run python multimodal.py photo.jpg receipt.jpg clip.mp4

`Decider.decide` accepts image, video, and audio files directly by path (or bytes / data: URIs).
"""

import sys

from ippon import Choice, Decider

photo, receipt, clip = sys.argv[1:4]


def choice(question: str, outcomes: list[str]) -> dict[str, Choice]:
    return {"answer": Choice(question, outcomes)}


with Decider.load("d1-2b:bf16") as model:  # Liquid's d1-2B reads text, pictures and video
    asks = {
        "photo": model.decide(
            "The photo.",
            choice("What does the photo show?", ["an aircraft", "a car", "a ship", "an animal"]),
            images=[photo],
        ),
        "image": model.decide(
            "The receipt image.",
            choice("What kind of document is this?", ["a law or charter", "a receipt", "a letter", "a menu"]),
            images=[receipt],
        ),
        "video": model.decide(
            "The clip.",
            choice("What happens in the video?", ["a ball rolls", "a person walks", "a car drives", "nothing moves"]),
            videos=[clip],
        ),
    }
for name, result in asks.items():
    answer = result["answer"]
    print(f"{name:7}{answer.top} {answer.confidence:.1%}")
