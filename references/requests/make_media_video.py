"""The video requests vllm-sr d3's media references are made on: python3 make_media_video.py > media-video.json

Deterministic. Each case is a System One request whose `videos` (and `images`) are file names under ../media/, in
order, with each file's sha256 under `media`, as make_media_qwen.py lists them; whoever runs a case opens those files
and sends their bytes. Covered: every question type, states null, empty, text, JSON and long, 1 to 3 questions;
H.264 in MP4, VP9 in WebM (with and without a frame count in the container, a real clip at 355/12 fps whose container
promises more frames than it holds), Motion JPEG in QuickTime and MPEG-4 Part 2 in Matroska; clips of 1 to 20 s (4 to
32 frames read, an odd number read twice at the end), frames under 32 pixels a side and over the 200,704-pixel
budget, portrait frames; one video, two, the same one twice, after one and two images; and six long clips, which
take more than the 16,384 input tokens a request may spend on videos, so d3's code refuses the request.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from make_media_d1 import choice, long_text, noul, score

MEDIA = Path(__file__).resolve().parents[1] / "media"


def requests() -> list[dict[str, Any]]:
    rng = random.Random(29)
    long_state = long_text(rng, 40)  # about 600 tokens
    ways = {"right": "From left to right", "left": "From right to left", "still": "It does not move"}
    cases: list[tuple[str, list[str], list[str], Any, dict[str, Any]]] = [
        (
            "example-square",  # the creator's own example, from its card
            [],
            ["d3-example-square.mp4"],
            "A short clip from a test camera.",
            {
                "direction": choice("Which way does the square move?", ways),
                "color_change": noul("Does the square change color?"),
            },
        ),
        (
            "ball-rolls",
            [],
            ["ball-h264-320x240-24fps-3s.mp4"],
            None,
            {
                "direction": choice("Which way does the ball roll?", ways),
                "red": noul("Is the ball red?", {"true": "The ball is red", "false": "The ball is another colour"}),
            },
        ),
        (
            "bar-32-frames",
            [],
            ["bar-h264-1280x720-29.97fps-20s.mp4"],
            "Screen recording of a progress bar.",
            {
                "colour": noul("Does the bar change colour during the clip?"),
                "progress": score(
                    "How far across does the bar reach at the end?",
                    ["a quarter", "half", "three quarters", "all the way"],
                ),
            },
        ),
        (
            "square-webm-odd-frames",
            [],
            ["square-vp9-160x120-10fps-2.5s.webm"],
            "",
            {"colour": noul("Does the square change colour?")},
        ),
        (
            "flash-four-frames",
            [],
            ["flash-h264-64x48-5fps-1s.mp4"],
            long_state,
            {
                "screen": choice(
                    "What happens on the screen?",
                    {"flash": "It flashes white once", "dark": "It stays black", "white": "It stays white"},
                )
            },
        ),
        ("tiny-frames", [], ["tiny-h264-24x16-4fps-3s.mp4"], None, {"green": noul("Is anything green visible?")}),
        (
            "portrait-quicktime",
            [],
            ["portrait-mjpeg-180x320-15fps-4s.mov"],
            {"camera": "lobby-3", "orientation": "portrait", "seconds": 4},
            {
                "direction": choice(
                    "Which way does the ball move?", {"down": None, "up": None, "left": None, "right": None}
                )
            },
        ),
        (
            "rings-matroska",
            [],
            ["rings-mpeg4-200x200-12fps-6s.mkv"],
            None,
            {"motion": score("How much does the pattern move?", ["not at all", "a little", "a lot"])},
        ),
        (
            "storefront",  # a real clip, VP9 at 355/12 fps whose container promises 262 frames and holds 253
            [],
            ["storefront-mazatlan-240p.webm"],
            "Clip sent by a mall's security team.",
            {
                "place": choice(
                    "What kind of place does the clip show?",
                    {
                        "clothing": "A clothing shop",
                        "restaurant": "A restaurant",
                        "pharmacy": "A pharmacy",
                        "supermarket": "A supermarket",
                    },
                ),
                "moving": noul("Does the camera move during the clip?"),
                "dog": noul("Is a live dog visible?"),
            },
        ),
        (
            "image-then-video",
            ["bison.jpg"],
            ["ball-h264-320x240-24fps-3s.mp4"],
            "An image and a clip from the same report.",
            {
                "animal": choice(
                    "Which of them shows an animal?",
                    {"image": "The image", "video": "The video", "both": "Both", "neither": "Neither"},
                )
            },
        ),
        (
            "two-videos",
            [],
            ["ball-h264-320x240-24fps-3s.mp4", "portrait-mjpeg-180x320-15fps-4s.mov"],
            "Two clips.",
            {"balls": noul("Does each clip show a ball?")},
        ),
        (
            "same-video-twice",
            [],
            ["square-vp9-160x120-10fps-2.5s.webm", "square-vp9-160x120-10fps-2.5s.webm"],
            "The same clip, sent twice.",
            {"same": noul("Are the two clips the same?")},
        ),
        (
            "two-images-two-videos",
            ["chart-bars-800x600.png", "solid-red-1x1.png"],
            ["flash-h264-64x48-5fps-1s.mp4", "square-vp9-160x120-10fps-2.5s.webm"],
            {"ticket": 5120, "attachments": 4, "note": "Two images, then two clips."},
            {
                "chart": noul("Is the first image a chart?"),
                "kinds": choice(
                    "What does the second clip show?",
                    dict.fromkeys(["square", "circle", "text", "person", "car", "tree", "flash", "nothing"]),
                ),
                "quality": score("How clear are the attachments, from 0 to 9?", [f"level {i}" for i in range(10)]),
            },
        ),
        (
            "too-many-video-tokens",  # six 32-frame clips: 17,280 video tokens, over the 16,384 a request may spend
            [],
            ["bar-h264-1280x720-29.97fps-20s.mp4"] * 6,
            None,
            {"same": noul("Are the clips the same?")},
        ),
    ]
    out = []
    for name, images, videos, state, questions in cases:
        files = [f"images/{f}" for f in images] + [f"videos/{f}" for f in videos]
        request: dict[str, Any] = {"state": state, "questions": questions}
        if images:
            request["images"] = [f"images/{f}" for f in images]
        request["videos"] = [f"videos/{f}" for f in videos]
        media = [
            {"name": f, "sha256": hashlib.sha256((MEDIA / f).read_bytes()).hexdigest()} for f in dict.fromkeys(files)
        ]
        out.append({"name": name, "request": request, "media": media})
    return out


if __name__ == "__main__":
    print(json.dumps(requests(), indent=1, ensure_ascii=False))
