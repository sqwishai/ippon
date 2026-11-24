"""The cache of a hybrid model: keys and values for its attention layers, a fixed-size state for its recurrent
ones.

Keys and values are preallocated for the whole capacity and written in place, so reading a prompt in chunks
or adding branches never copies what was read before. A recurrent layer keeps two things: the Gated DeltaNet
state [1, value heads, value dim, key dim] in fp32 and the last kernel size - 1 inputs of its short
convolution. Both have the same size at any prompt length, so a copy of them for every branch costs the same
after 100 tokens as after 10,000.

A prefix kept across requests (`prefixes.py`) is two things taken from a cache: `keys()`, an exact-size copy of
the keys and values read so far, and a `checkpoint()` of every recurrent layer at a position, which costs nothing
to take (the cache replaces those arrays, never writes into them). A recurrent layer's state exists only at the
positions it was checkpointed at, so a new cache can start from a kept prefix only at one of them: `restore()`
copies the keys and values before it and continues the recurrent layers from its checkpoint.

A model whose image rows take multimodal RoPE positions (Qwen3.5) places the text after an image at its index plus
an offset, which each image changes once its last row is read. The cache keeps where the offset changes (`offsets`),
and each checkpoint the offset in effect at its position, so a pass that continues a cache, or a kept prefix, places
its rows where one pass over the whole prompt would.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx


@dataclass
class Attention:
    """Keys and values [1, kv heads, capacity, head dim] of one attention layer."""

    k: mx.array
    v: mx.array

    @classmethod
    def empty(cls, kv_heads: int, capacity: int, head_dim: int, dtype: mx.Dtype) -> Attention:
        shape = (1, kv_heads, capacity, head_dim)
        return cls(mx.zeros(shape, dtype), mx.zeros(shape, dtype))

    def write(self, at: int, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        """Write [1, kv heads, n, head dim] at positions at..at+n; the keys and values of 0..at+n."""
        end = at + k.shape[2]
        if end > self.k.shape[2]:
            raise ValueError(f"the cache holds {self.k.shape[2]} tokens; {end} were written")
        self.k[..., at:end, :] = k
        self.v[..., at:end, :] = v
        return self.k[..., :end, :], self.v[..., :end, :]


@dataclass
class Recurrent:
    """The state of one recurrent layer: None before the first token."""

    state: mx.array | None = None  # Gated DeltaNet [1, Hv, Dv, Dk] fp32
    window: mx.array | None = None  # the convolution's last inputs [1, kernel - 1, channels]


@dataclass(frozen=True)
class Checkpoint:
    """Every recurrent layer's state after `length` tokens (None for an attention layer), and the position offset of
    the rows after them (`HybridCache.offset`)."""

    length: int
    layers: tuple[Recurrent | None, ...]
    offset: int = 0

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.arrays())

    def arrays(self) -> list[mx.array]:
        return [a for r in self.layers if r is not None for a in (r.state, r.window) if a is not None]


@dataclass(frozen=True)
class Keys:
    """The keys and values of the first `length` tokens of every attention layer (None for a recurrent one), compact
    copies: what they count is what they hold."""

    length: int
    layers: tuple[tuple[mx.array, mx.array] | None, ...]

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.arrays())

    def arrays(self) -> list[mx.array]:
        return [a for kv in self.layers if kv is not None for a in kv]


@dataclass
class HybridCache:
    layers: list[Attention | Recurrent]
    length: int = 0
    kept: dict[int, Checkpoint] = field(default_factory=dict)  # checkpoints a pass took at its marks
    pending: list[mx.array] = field(default_factory=list)  # what the last pass left the GPU to evaluate
    offsets: dict[int, int] = field(default_factory=dict)  # position -> the offset of the rows from there on

    @classmethod
    def of(cls, blocks: Sequence[Attention | Recurrent], start: tuple[Any, Any] | None = None) -> HybridCache:
        """A cache of these layer blocks, empty, or continuing a kept prefix (`start`: its keys and a checkpoint at a
        position they reach)."""
        cache = cls(list(blocks))
        if start is not None:
            keys, checkpoint = start
            assert isinstance(keys, Keys)
            assert isinstance(checkpoint, Checkpoint)
            cache.restore(keys, checkpoint)
        return cache

    def offset(self, at: int | None = None) -> int:
        """How far the positions of the rows from `at` on (the cache's end by default) run ahead of their index: 0
        before any image, else what the last change at or before `at` set."""
        at = self.length if at is None else at
        changed = [p for p in self.offsets if p <= at]
        return self.offsets[max(changed)] if changed else 0

    def checkpoint(self, at: int | None = None) -> Checkpoint:
        """The recurrent state after `at` tokens: now (the default), or at a mark a pass kept it at."""
        if at is not None and at != self.length:
            if at not in self.kept:
                raise KeyError(f"no checkpoint at {at}: the passes kept {sorted(self.kept)}")
            return self.kept[at]
        return Checkpoint(
            self.length,
            tuple(Recurrent(x.state, x.window) if isinstance(x, Recurrent) else None for x in self.layers),
            self.offset(),
        )

    def remember(self) -> None:
        self.kept[self.length] = self.checkpoint()

    def keep(self, marks: dict[int, list[tuple[mx.array | None, mx.array | None]]]) -> None:
        """Record the recurrent layers' (state, window), in layer order, at absolute positions."""
        for at, states in marks.items():
            it = iter(states)
            self.kept[at] = Checkpoint(
                at,
                tuple(Recurrent(*next(it)) if isinstance(x, Recurrent) else None for x in self.layers),
                self.offset(at),
            )

    def keys(self, upto: int | None = None) -> Keys:
        """The keys and values of the first `upto` tokens (all read so far by default): a compact copy, evaluated as
        soon as the GPU gets to it (scheduled now), so what the cache writes next never reaches it. Never a view: a
        view of the cache's arrays, even of most of them, kept the memory of the request that wrote them alive (about
        three times what the store counted, d1-3B at 4,000 words), for a copy that costs about 0.3 ms."""
        n = self.length if upto is None else min(upto, self.length)
        layers = tuple(
            (mx.contiguous(x.k[..., :n, :]), mx.contiguous(x.v[..., :n, :])) if isinstance(x, Attention) else None
            for x in self.layers
        )
        mx.async_eval([a for kv in layers if kv is not None for a in kv])  # type: ignore[no-untyped-call]
        return Keys(n, layers)

    def restore(self, keys: Keys, checkpoint: Checkpoint) -> None:
        """Continue an empty cache from a kept prefix, at the checkpoint's position: the keys and values before it
        copied in, every recurrent layer's state taken from the checkpoint."""
        n = checkpoint.length
        if self.length or n > keys.length or len(keys.layers) != len(self.layers):
            raise ValueError("a cache is restored empty, from keys of the same model that reach the checkpoint")
        for i, layer in enumerate(self.layers):
            kv, r = keys.layers[i], checkpoint.layers[i]
            if isinstance(layer, Attention):
                assert kv is not None
                layer.write(0, kv[0][..., :n, :], kv[1][..., :n, :])
            else:
                assert r is not None
                layer.state, layer.window = r.state, r.window
        self.length = n
        self.offsets = {n: checkpoint.offset} if checkpoint.offset else {}

    def schedule(self) -> None:
        """After a pass: start evaluating what the cache holds without waiting for it (`mx.async_eval`), then wait
        for the pass before, so a prompt read in chunks never builds one graph over all of them, at most one chunk
        waits on the GPU, and the GPU never waits for Python to build the next one. The last chunk of a request is
        waited for by the request's answer."""
        previous, self.pending = self.pending, self.arrays()
        mx.async_eval(self.pending)  # type: ignore[no-untyped-call]
        if previous:
            mx.eval(previous)

    def arrays(self) -> list[mx.array]:
        """Everything the cache holds, for one evaluation after a chunk: its layers and the checkpoints kept at marks.
        A checkpoint left unevaluated would keep the whole pass it was taken from alive (its window is a copy of a
        slice of a layer's input projection), which nothing that answers the request would ever evaluate."""
        out: list[mx.array] = []
        for layer in self.layers:
            if isinstance(layer, Attention):
                out += [layer.k, layer.v]
            else:
                out += [a for a in (layer.state, layer.window) if a is not None]
        for kept in self.kept.values():
            out += kept.arrays()
        return out
