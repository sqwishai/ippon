"""Answers kept for requests the model has already answered: an identical request is answered again without reading
it.

Agents send the same request again and again (a command check for `git status`, the same file read twice, the same
question about a transcript that hasn't changed), and a model's answer to the same prompt is the same, so ippon keeps
the last answers it gave, keyed by what the model reads: each prompt's tokens and readouts as the format rendered them
(an image or a clip is in its placeholders' digest), each question's id, type and options with the request's
model name and `unsure_below`, which shape the response, and how the decider reads it (its chunk and strategy, which
round differently). A request answered in rounds (a choice wider than one pass) is read each time, and so is one
with a spoken question, since its answer holds the question and so the clip (about 320 KB for ten seconds). The
answers are
cleared when the model changes what it computes in, and when it unloads. On by default in `Decider.load` and every
command (`answer_cache=0`, `--answer-cache 0` keeps none): a decision model doesn't sample, so the kept answer is the
one it would give again. A caller that must read every request (a warm-up, a benchmark, a check) reads within
`Decider.uncached()`.
"""

from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict
from typing import TYPE_CHECKING

from .options import Strategy

if TYPE_CHECKING:  # MLX-free at run time: the command line reads DEFAULT_SIZE for its help
    from .adapter import Rendering
    from .schema import Result

DEFAULT_SIZE = 1024  # requests kept; an entry is a few kilobytes


class Answers:
    """The last `size` requests' results, by `key`, the oldest dropped first."""

    def __init__(self, size: int = DEFAULT_SIZE) -> None:
        if size <= 0:
            raise ValueError("keep at least one answer, or none at all (no Answers)")
        self.size = size
        self._kept: OrderedDict[bytes, Result] = OrderedDict()

    @staticmethod
    def key(rendering: Rendering, chunk: int, strategy: Strategy) -> bytes | None:
        """What identifies a rendered request's answer read in chunks of `chunk` tokens by `strategy`, or None for one
        that is not kept: answered in rounds, or asking a spoken question."""
        request = rendering.request
        if rendering.round is not None or any(q.spoken for q in request.questions):
            return None
        what = repr(
            (
                [(p.tokens, p.readouts, p.marks, p.groups) for p in rendering.prompts],
                [(q.id, q.type, q.instructions, q.criteria) for q in request.questions],
                request.model,
                request.unsure_below,
                rendering.cuts,
                chunk,
                strategy,
            )
        )
        return hashlib.blake2b(what.encode(), digest_size=16).digest()

    def get(self, key: bytes | None) -> Result | None:
        """A copy of the kept result, so a caller changing its response leaves the kept one as it was."""
        if key is None or key not in self._kept:
            return None
        self._kept.move_to_end(key)
        return copy.deepcopy(self._kept[key])

    def put(self, key: bytes | None, result: Result) -> None:
        if key is None:
            return
        self._kept[key] = copy.deepcopy(result)
        self._kept.move_to_end(key)
        while len(self._kept) > self.size:
            self._kept.popitem(last=False)

    def clear(self) -> None:
        self._kept.clear()
