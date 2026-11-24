"""What Qwen3.5 decision models share: the chat template their creators' prompts are written in, and a format that
reads images.

`chat` is the template's text for a system and a user message with the answer's turn opened, thinking off, as
transformers' `apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)` writes it: each
message's text trimmed, a user's images ahead of its text, then its videos. A creator that tokenizes the prompt in
pieces builds it from `chat_head` and `ANSWER_TURN`.

The chat template's image, `<|vision_start|><|image_pad|><|vision_end|>`, goes where the creator puts it, and its
`<|image_pad|>` becomes the image's placeholders (`QwenImageFormat`); the images and videos themselves are read by
media/qwen.py's `QwenImages` and `QwenVideos`, as transformers' Qwen3VLProcessor and Qwen3_5Model read them.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import numpy as np

from ..adapter import FormatBase, Rendering
from ..errors import InvalidRequest
from ..media.qwen import IMAGE_TOKEN, VIDEO_TOKEN, QwenImages
from ..schema import Request

VISION = f"<|vision_start|>{IMAGE_TOKEN}<|vision_end|>"  # one image, as the chat template writes it
VIDEO = f"<|vision_start|>{VIDEO_TOKEN}<|vision_end|>"  # one video, as the chat template writes it
ANSWER_TURN = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"  # the user's turn closed, the answer next
# pplx_decider/protocol.py's floor, which its Decider sets on the processor with its ceiling; the adapters that set a
# ceiling of their own pass it with theirs (`QwenImages.attach`'s `pixels`)
MIN_PIXELS = 65_536


def chat_head(system: str) -> str:
    """The chat template up to the user's text: the system message, then the user's turn opened."""
    return f"<|im_start|>system\n{system.strip()}<|im_end|>\n<|im_start|>user\n"


def chat(system: str, user: str, images: int = 0, videos: int = 0) -> str:
    """The prompt for a system and a user message with `images`, then `videos`, ahead of the user's text, up to where
    the answer starts."""
    return chat_head(system) + VISION * images + VIDEO * videos + user.strip() + ANSWER_TURN


class QwenImageFormat(FormatBase):
    """What a format whose model reads images through a Qwen vision tower (`images`, a dataclass field of the format;
    None for a build without one) takes from it: the media it takes, each image's placeholders, the prompt's image
    tokens expanded to them, and the images' rows encoded."""

    images: QwenImages | None

    @property
    def media(self) -> frozenset[str]:
        return frozenset({"images"}) if self.images is not None else frozenset()

    def encode_media(self, renderings: Sequence[Rendering]) -> dict[int, mx.array]:
        """Every image's rows through the checkpoint's vision tower, by digest (`QwenImages.encode`)."""
        return self.images.encode(renderings) if self.images is not None else {}

    def place(self, request: Request) -> tuple[list[tuple[int, ...]], dict[int, np.ndarray]]:
        """Each of the request's images' placeholders, in order (`QwenImages.rows`), and the pixels decoded to count
        them, the rendering's `pixels` for `encode_media`: none and {} for a request without images."""
        if not request.images:
            return [], {}
        assert self.images is not None, "the decider refuses media a format does not take"
        return self.images.rows(request.images)

    def expand_checked(self, ids: list[int], rows: Sequence[tuple[int, ...]], holder: str = "state") -> list[int]:
        """`ids` with each image token replaced by its image's placeholders (`place`), in order; a text request's as
        they are. InvalidRequest when the `holder`'s own text holds the image token too, which only an image may."""
        if not rows:
            return ids
        assert self.images is not None
        if ids.count(self.images.token) != len(rows):
            raise InvalidRequest(f"the {holder}'s text holds {IMAGE_TOKEN}, which only an image may")
        return self.images.expand(ids, rows)

    def images_detail(self) -> str:
        """What `ippon doctor` shows of the images the model reads."""
        return f"up to {self.images.config.max_pixels:,} pixels each" if self.images is not None else "none (text only)"
