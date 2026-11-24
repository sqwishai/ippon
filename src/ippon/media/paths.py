"""What every media path shares, whatever its model: the media token a checkpoint's config.json names, an encoder
built on first use, how a creator samples a clip's frames and how its prompts hold a video, and the last media's
encoded rows kept, so a medium sent again skips its encoder (`encode_kept`). A model's own processor and towers are
media/qwen.py's, media/lfm3.py's and media/lfm3_audio.py's.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import mlx.core as mx

from ..arch.common import Kept
from ..arch.ops import Ops
from ..errors import UnsupportedModel

if TYPE_CHECKING:
    from ..adapter import LoadContext
    from ..arch.base import Backbone
    from ..model_dir import ModelDir
    from ..tokenizer import Tokenizer
    from ..weights import Weights


class Encoder(Protocol):
    """A media path's encoder (a vision tower, a speech encoder), whose matmuls run as its `ops` say."""

    ops: Ops


E = TypeVar("E", bound=Encoder)
K = TypeVar("K", bound=Hashable)
S = TypeVar("S")


def config_token(model: ModelDir, tokenizer: Tokenizer, text: str, key: str) -> int:
    """The id config.json names under `key` (the image, video or audio token the processor expands), checked to be
    the tokenizer's `text`; UnsupportedModel otherwise."""
    token = model.config.get(key)
    if not isinstance(token, int) or tokenizer.token_id(text) != token:
        raise UnsupportedModel(f"{model.path}: the tokenizer's {text} is not config.json's {key}")
    return token


def encoder_loader(
    load: Callable[[Weights, Mapping[str, Any], mx.Dtype], E], model: ModelDir, ctx: LoadContext, backbone: Backbone
) -> Callable[[], E]:
    """A media path's encoder, built on its first use by `load` (its class's `load`) from the model's media weights
    (`LoadContext.media_weights`) in the load's dtype, so a text-only server never holds them. Each call hands it the
    backbone's ops as they are then: a warm-up or a profile swaps in measuring ops for a while."""
    built: list[E] = []

    def encoder() -> E:
        if not built:
            built.append(load(ctx.media_weights(model), model.config, ctx.dtype))
        built[0].ops = backbone.ops
        return built[0]

    return encoder


@dataclass(frozen=True)
class Sampling:
    """How a creator samples a clip's frames: `rate` a second, `least` to `most` of them."""

    rate: float = 2.0
    least: int = 4
    most: int = 32


@dataclass(frozen=True)
class Seen:
    """A video as its prompts hold it: each frame pair's time in seconds, the clip's grid (t, h, w) in patches, and
    the distinct frames read across the video's duration in seconds."""

    times: tuple[float, ...]
    grid: tuple[int, int, int]
    frames: int = 0
    duration_s: float = 0.0


def encode_kept(
    kept: Kept, needed: Mapping[K, S], encode: Callable[[dict[K, S]], dict[K, dict[int, mx.array]]]
) -> dict[int, mx.array]:
    """The rows of every medium `needed` (by key, with what encodes it), by digest: those `kept` from before, the rest
    encoded together (`encode`, each new medium's rows by digest: a video's under each frame pair's), evaluated and
    kept."""
    out: dict[int, mx.array] = {}
    new: dict[K, S] = {}
    for key, source in needed.items():
        found = kept.get(key)
        if found is not None:
            out |= found
        else:
            new[key] = source
    if new:
        fresh = encode(new)
        mx.eval([rows for medium in fresh.values() for rows in medium.values()])
        for key, medium in fresh.items():
            kept.put(key, medium)
            out |= medium
    return out
