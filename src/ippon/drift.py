"""How far each build below 8 bits drifts from its creator's answers, and the warning every such load gives.

ippon runs a decision model at full precision or 8-bit unless the caller asks for less: a build's own name
(kettei-core:q4, d3-mini:nvfp4), `precision="q4"` or `"q6"`, or the Hub id, file or Ollama
tag of a low-bit build. No bare name, recommendation or setup choice picks one (names.recommended). Fewer bits move
a model's probabilities further from its creator's, so each such load warns (`LowBitWarning`), logs it, puts it in
`Decider.details["drift"]` and in every response's `warnings`. The warning says in plain words how far that build moves
(MEASURED, from `ippon verify` against the creator's reference) and what that means for thresholds.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from . import names
from .quantization import form_of, low_bit
from .quantization import get as get_precision

log = logging.getLogger(__name__)


class LowBitWarning(UserWarning):
    """A model loaded below 8 bits, whose answers drift further from its creator's."""


Level = Literal["close", "noticeable", "a lot"]

# Each build below 8 bits against its model's reference answers (`ippon verify MODEL:q6 --reference R`, M5 Pro):
# "close" moved under 10 points with no top answer changed, "a lot" moved
# over 25, "noticeable" is between (docs/formats.md gives the points). The comment is the largest move and the
# requests whose top answer changed.
MEASURED: dict[str, Level] = {
    "d1-2b:q4": "noticeable",  # 22.7 points, 3 of 28
    "d1-2b:q6": "close",  # 4.6 points, 0 of 28
    "d1-omni-2b:q4": "noticeable",  # 20.5 points, 3 of 28
    "d1-omni-2b:q6": "close",  # 3.7 points, 0 of 28
    "d3-flash:q4": "a lot",  # 39.0 points, 1 of 20
    "d3-flash:q6": "close",  # 8.2 points, 0 of 20
    "d3-mini:q4": "a lot",  # 38.9 points, 1 of 20
    "d3-mini:q6": "close",  # 3.3 points, 0 of 20
    "elephant:q4": "a lot",  # 52.1 points, 2 of 27
    "elephant:q6": "noticeable",  # 14.1 points, 0 of 27
    "glide-2b:q4": "noticeable",  # 20.1 points, 0 of 20
    "glide-2b:q6": "noticeable",  # 12.9 points, 0 of 20
    "glide-9b:q4": "noticeable",  # 23.4 points, 4 of 20
    "glide-9b:q6": "noticeable",  # 10.8 points, 0 of 20
    "h2o:q4": "a lot",  # 26.9 points, 3 of 27
    "h2o:q6": "noticeable",  # 9.4 points, 1 of 27
    "jeff:q4": "a lot",  # 45.1 points, 4 of 18
    "jeff:q6": "noticeable",  # 21.4 points, 1 of 18
    "jeff-qwen-4b:q4": "a lot",  # 77.9 points, 6 of 18
    "jeff-qwen-4b:q6": "close",  # 7.7 points, 0 of 18
    "kettei-core:q4": "a lot",  # 74.5 points, 3 of 20
    "kettei-core:q6": "noticeable",  # 19.3 points, 0 of 20
    "kettei-dot:q4": "a lot",  # 48.0 points, 4 of 20
    "kettei-dot:q6": "noticeable",  # 23.5 points, 2 of 20
    "kev-0.8b:q4": "noticeable",  # 19.5 points, 3 of 20
    "kev-0.8b:q6": "close",  # 8.8 points, 0 of 20
    "kev-4b:q4": "noticeable",  # 21.6 points, 5 of 20
    "kev-4b:q6": "noticeable",  # 5.3 points, 1 of 20
    "kev-9b:q4": "a lot",  # 29.9 points, 1 of 20
    "kev-9b:q6": "close",  # 3.9 points, 0 of 20
    "pplx-flash:q4": "noticeable",  # 16.0 points, 0 of 15
    "pplx-flash:q6": "noticeable",  # 6.6 points, 1 of 15
    "torchcast:q4": "a lot",  # 43.1 points, 3 of 26
    "torchcast:q6": "close",  # 7.6 points, 0 of 26
}
# What a user sees, in one sentence: what moves, and what to do about it
SAYS: dict[Level, str] = {
    "close": "Its answers stay close to full precision.",
    "noticeable": "Its answers move a little from full precision. Check your thresholds on a few examples.",
    "a lot": "It sometimes gives a different answer than at full precision. Use 8-bit if this Mac has room.",
}
# The tag `ippon setup` puts on a model's row at that precision, and the brand accent it is drawn in. The mark and the
# words say it without the colour
TAG: dict[Level, tuple[str, str]] = {
    "close": ("✓ answers hold", "mint"),
    "noticeable": ("⚠ check thresholds", "amber"),
    "a lot": ("⚠ answers may change", "red"),
}


def measured(worst: float, changed: bool) -> Level:
    """The level a build measures at against its model's reference, as MEASURED's were found: "close" when its
    probabilities move under 10 points with no top answer changed, "a lot" over 25, "noticeable" between."""
    if worst > 0.25:
        return "a lot"
    return "close" if worst < 0.10 and not changed else "noticeable"


def level(name: str, form: str) -> Level:
    """How far `name` at `form` moves: as measured, else "noticeable" at 6 bits and "a lot" below (every 4-bit build
    measured moved 16 points or more)."""
    return MEASURED.get(name, "noticeable" if form == form_of("affine", 6) else "a lot")


def low_bit_format(matrices: Iterable[tuple[str, int]]) -> str | None:
    """The formats below 8 bits among a model's packed matrices, each (mode, bits), as a warning names them ("4-bit",
    "NVFP4", "4-bit and 6-bit"); None when every one is 8-bit or more."""
    found = {form_of(mode, bits) for mode, bits in matrices if low_bit(bits)}
    if not found:
        return None
    ordered = sorted(found)
    return ordered[0] if len(ordered) == 1 else ", ".join(ordered[:-1]) + " and " + ordered[-1]


def disclaimer(name: str, form: str) -> str:
    """The warning for `name` running at `form`: what moves, and what to do about it."""
    return f"{name} runs at {form}. {SAYS[level(name, form)]}"


def warn(text: str) -> None:
    """Give a `disclaimer` as a `LowBitWarning` and in ippon's log."""
    log.warning(text)
    warnings.warn(text, LowBitWarning, stacklevel=3)


def name_of(model: str | Path, precision: str) -> str:
    """The name a low-bit load's warning gives (MEASURED's key): its build's short name, FAMILY:q4 or FAMILY:q6 for a
    build quantized here, else the model as given."""
    text = str(model)
    named = names.find(text)[0] if names.is_name(text) else names.of_load(text)
    if not get_precision(precision).changes_weights:
        return named.name if named is not None else text
    return f"{named.family if named is not None else text}:{precision}"
