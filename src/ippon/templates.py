"""Sample System One requests the tests use, and the sample request `ippon pull`
answers once a model is downloaded.

Each template is written the way a decision model reads a question best, and `ippon lint` finds nothing in it:
one thing asked per question, as a full question naming what to look at; a choice's options that don't overlap, each
with a description of what belongs to it, and a catch-all ("other") where a state may fit none of them; a score's
levels in order, lowest first. Their states are made-up examples, to be replaced by the caller's own.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image

from .media.decode import ARRAY_RATE, as_media

SAMPLE_TEXT: dict[str, Any] = {
    "state": "Order #48213 was charged twice. Please reverse the second charge before Friday.",
    "questions": {
        "refund": {"type": "noul", "instructions": "Is the customer asking for money back?"},
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this message?",
            "criteria": {
                "billing": "Charges, refunds and invoices.",
                "shipping": "Delivery, tracking and returns.",
                "other": "Anything else.",
            },
        },
    },
}
SAMPLE_COLOUR = (196, 32, 32)  # the sample image: one colour, so its right answer is plain


def sample_image() -> dict[str, Any]:
    """The sample request for a model that reads images: a 224 x 224 picture of one colour (SAMPLE_COLOUR, a red),
    asked which colour it is."""
    png = io.BytesIO()
    Image.new("RGB", (224, 224), SAMPLE_COLOUR).save(png, format="PNG")
    return {
        "state": "A picture.",
        "images": [base64.b64encode(png.getvalue()).decode()],
        "questions": {
            "colour": {
                "type": "choice",
                "instructions": "Which colour fills the picture?",
                "criteria": {"red": None, "green": None, "blue": None, "other": "Any other colour."},
            }
        },
    }


def sample_audio() -> dict[str, Any]:
    """The sample request for a model that hears audio: one second of a 440 Hz tone (16 kHz mono WAV), asked whether
    anyone speaks in it."""
    rate = ARRAY_RATE
    tone = (0.3 * 32767 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)).astype(np.int16)
    clip = as_media(tone, "audio")  # a WAV of the samples at ARRAY_RATE
    return {
        "state": "An audio clip.",
        "audio": base64.b64encode(clip.data).decode(),
        "questions": {"speech": {"type": "noul", "instructions": "Does anyone speak in this audio clip?"}},
    }
