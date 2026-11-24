"""A toy backbone and format with the real contracts, for testing the runtime without a real model.

The backbone is a linear recurrence, h[t] = h[t-1] / 2 + E[token[t]], so every output depends on every token
before it, as in a real model, and a wrong prefix, a shared cache or a lost chunk shows up in the answers. Its
branches run one after another from copies of the cache. The format renders the state's bytes, then one token
per question, then a readout token.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from ippon import Decider, Loaded
from ippon.adapter import FormatBase, Hidden, Rendering
from ippon.arch.ops import MLX, Ops
from ippon.plan import Prompt, Readout
from ippon.readout import LabelSet, label_logits, option_probabilities
from ippon.schema import Request

VOCABULARY, HIDDEN = 300, 8
READ = 299  # the readout token


def _embedding() -> mx.array:
    return mx.sin(mx.arange(VOCABULARY * HIDDEN, dtype=mx.float32).reshape(VOCABULARY, HIDDEN) * 0.37)


@dataclass
class ToyCache:
    state: mx.array
    length: int = 0
    kept: dict[int, ToySnapshot] = field(default_factory=dict)

    def fork(self) -> ToyCache:
        return ToyCache(mx.array(self.state), self.length)

    def checkpoint(self, at: int | None = None) -> ToySnapshot:
        if at is not None and at != self.length:
            return self.kept[at]
        return ToySnapshot(self.length, self.state)

    def remember(self) -> None:
        self.kept[self.length] = self.checkpoint()

    def keys(self, upto: int | None = None) -> ToySnapshot:
        return ToySnapshot(self.length if upto is None else upto, None)


@dataclass(frozen=True)
class ToySnapshot:
    length: int
    state: mx.array | None  # the recurrence's state at `length`; the toy has no keys

    @property
    def nbytes(self) -> int:
        return 4 * HIDDEN if self.state is not None else 0

    def arrays(self) -> list[mx.array]:
        return [self.state] if self.state is not None else []


class ToyBackbone:
    """A backbone without layers to tune: nothing to quantize, no GLU, its own arithmetic (not through `ops`)."""

    hidden_size = HIDDEN
    causal = True
    block = False
    dtype = mx.float32
    layers: tuple[dict[str, Any], ...] = ()
    linears: tuple[str, ...] = ()
    gated_mlp = None
    attention_share = 1.0
    max_tokens = None
    ops: Ops = MLX
    media: Mapping[int, mx.array] | None = None

    def __init__(self) -> None:
        self.embedding = _embedding()
        self.tokens_read = 0
        self.batches = 0  # batched passes

    @property
    def output_matrix(self) -> mx.array:
        return self.embedding

    def new_cache(self, capacity: int, start: tuple[Any, Any] | None = None) -> ToyCache:
        if start is None:
            return ToyCache(mx.zeros((HIDDEN,)))
        _, checkpoint = start
        return ToyCache(checkpoint.state, checkpoint.length)

    def prefill(self, tokens: Sequence[int], cache: Any, rows: Sequence[int], marks: Sequence[int] = ()) -> mx.array:
        assert isinstance(cache, ToyCache)
        outputs = []
        for i, token in enumerate(tokens):
            if i in marks:
                cache.kept[cache.length + i] = ToySnapshot(cache.length + i, cache.state)
            cache.state = cache.state * 0.5 + self.embedding[token]
            outputs.append(cache.state)
        mx.eval(cache.state)
        cache.length += len(tokens)
        self.tokens_read += len(tokens)
        return mx.stack([outputs[r] for r in rows]) if rows else mx.zeros((0, HIDDEN))

    def branch(
        self,
        tails: Sequence[Sequence[int]],
        cache: Any,
        rows: Sequence[Sequence[int]],
        head: Sequence[int] = (),
        marks: Sequence[int] = (),
    ) -> list[mx.array]:
        assert isinstance(cache, ToyCache)
        if head:
            self.prefill(head, cache, (), marks)
        return [self.prefill(tail, cache.fork(), r) for tail, r in zip(tails, rows, strict=True)]

    def batch(
        self,
        prompts: Sequence[Sequence[int]],
        rows: Sequence[Sequence[int]],
        groups: Sequence[Sequence[int]] | None = None,
    ) -> list[mx.array]:
        self.batches += 1
        return [self.prefill(p, self.new_cache(len(p)), r) for p, r in zip(prompts, rows, strict=True)]


class ToyFormat(FormatBase):
    def __init__(self, backbone: ToyBackbone) -> None:
        self.backbone = backbone
        self.tokenizer = None

    def render(self, request: Request) -> Rendering:
        state = str(request.state).encode()[:200]
        prompts = []
        for i, q in enumerate(request.questions):
            tokens = (*state, 256 + i % 40, READ)
            prompts.append(Prompt(tokens, (Readout(q.id, len(tokens) - 1),)))
        return Rendering(request, tuple(prompts))

    def scores(self, rendering: Rendering, hidden: Hidden) -> dict[str, mx.array]:
        out = {}
        for q in rendering.request.questions:
            labels = LabelSet(tuple((k,) for k in range(len(q.options))))
            logits = label_logits(hidden.rows(q.id)[-1:], self.backbone.embedding, labels.flat)
            out[q.id] = option_probabilities(logits, labels)[0]
        return out


def decider(**options: Any) -> Decider:
    backbone = ToyBackbone()
    return Decider(Loaded(backbone, ToyFormat(backbone)), **options)
