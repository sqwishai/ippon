"""What a model architecture gives the runtime.

An architecture (Qwen3.5's mix of Gated DeltaNet and attention layers, LFM2's mix of short convolutions and
attention, a dense transformer, an encoder) turns tokens into hidden states and keeps a cache of what it has
read: keys and values for attention layers, a fixed-size state for recurrent ones. The runtime needs three
passes from it, and nothing about prompts or answers, which are the adapter's:

- `prefill` reads tokens after what a cache holds, causally, and adds them to it: a whole prompt, a shared
  prefix, or one chunk of either;
- `branch` reads several sequences that all continue the cache (each question's own tokens after a shared
  state) in one pass, each seeing the cache and its own earlier tokens only, and leaves the cache as it was
  for the prefix. Reading them together reads the weights once for all of them;
- `batch` reads several whole prompts together, each from scratch: for models whose prompts can't share a prefix.

All return the final hidden states, after the model's final norm, of only the rows asked for: the last
layer runs its attention queries and its MLP for those rows only, and none at all when no row is asked for.

A causal backbone's cache can also be kept across requests: `keys()` and `checkpoint()` take what a later cache
needs to continue from a position, and `new_cache(capacity, start)` continues from it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol

import mlx.core as mx

from ..quant import Table

if TYPE_CHECKING:
    from .ops import Ops


class Snapshot(Protocol):
    """Part of a cache, kept: its keys and values, or its recurrent state at one position."""

    @property
    def length(self) -> int: ...

    @property
    def nbytes(self) -> int: ...

    def arrays(self) -> list[mx.array]:
        """The arrays it holds, for evaluating them before it is kept."""
        ...


class Cache(Protocol):
    @property
    def length(self) -> int:
        """Tokens read into the cache so far: the position the next token takes."""
        ...

    def checkpoint(self, at: int | None = None) -> Snapshot:
        """What reading continues from at `length` (or at a kept position) besides the keys and values: the
        recurrent state. Cheap to take, and unchanged by what the cache reads next."""
        ...

    def keys(self, upto: int | None = None) -> Snapshot:
        """A copy of the keys and values of the first `upto` tokens (all read so far by default), unchanged by what
        the cache reads next."""
        ...

    @property
    def kept(self) -> Mapping[int, Snapshot]:
        """Checkpoints kept so far (at marks inside passes, or by `remember`), by position."""
        ...

    def remember(self) -> None:
        """Keep the checkpoint at the current length."""
        ...


class Layered(Protocol):
    """A model's layers as kernel choices see them: their weights, and which inputs multiply."""

    @property
    def dtype(self) -> mx.Dtype:
        """What it computes in."""
        ...

    @property
    def layers(self) -> Sequence[dict[str, Any]]:
        """Each layer's weights by key, which kernel choices read and replace (its `linears`)."""
        ...

    @property
    def linears(self) -> tuple[str, ...]:
        """The keys of each layer's weight matrices that inputs multiply (`quant.mm`)."""
        ...

    @property
    def gated_mlp(self) -> tuple[str, str] | None:
        """The key of each layer's [gate; up] matrix and its GLU's activation (one fused matmul), or None."""
        ...


class Backbone(Layered, Protocol):
    @property
    def hidden_size(self) -> int: ...

    ops: Ops  # how its norms, GLUs, matmuls and attention run: the decider sets it from the chip profile
    # the encoded media a pass's placeholder tokens read (`arch.common.embed`), set by the decider around that pass
    media: Mapping[int, mx.array] | None

    @property
    def max_tokens(self) -> int | None:
        """The longest prompt it reads (learned positions; a sliding window it doesn't implement), or None: the
        decider refuses a longer one (ContextTooLong) before reading anything."""
        ...

    @property
    def attention_share(self) -> float:
        """The share of its layers that are attention (1 for a transformer; a hybrid's fraction): attention's cost
        grows with the keys, so the runtime weighs reading a prefix once against batching by it (`runtime.shares`)."""
        ...

    @property
    def causal(self) -> bool:
        """True when every layer is causal, so prompts that share a prefix can share its reading."""
        ...

    @property
    def block(self) -> bool:
        """True for block attention (elephant-1's): the prompts' shared prefix, up to the boundary their format marks,
        is read once and both ways within itself, and each tail reads that prefix and itself both ways, never another
        tail; recurrent layers stay causal. Such a prefix is kept and reused only whole."""
        ...

    @property
    def output_matrix(self) -> Table:
        """The language-model head [vocabulary, hidden] (the embedding when tied), for label readouts; it may be
        quantized, so read its rows with `quant.rows`."""
        ...

    def new_cache(self, capacity: int, start: tuple[Snapshot, Snapshot] | None = None) -> Cache:
        """A cache for up to `capacity` tokens: a prompt, or a prefix and every branch after it. Empty, or with
        `start` (kept keys, and a checkpoint at a position they reach) holding that position's tokens already."""
        ...

    def prefill(self, tokens: Sequence[int], cache: Cache, rows: Sequence[int], marks: Sequence[int] = ()) -> mx.array:
        """Read `tokens` after what `cache` holds and add them to it. The final hidden states [len(rows), hidden]
        of `rows` (indices into `tokens`, in that order); [0, hidden] for none. At each of `marks` (indices into
        `tokens`, inside them) the recurrent state is kept in `cache.kept`. What it adds to the cache is being
        evaluated when it returns, and the pass before it has been (`HybridCache.schedule`), so a prompt read in chunks
        never builds one graph over all of them and the GPU doesn't wait for Python between chunks."""
        ...

    def branch(
        self,
        tails: Sequence[Sequence[int]],
        cache: Cache,
        rows: Sequence[Sequence[int]],
        head: Sequence[int] = (),
        marks: Sequence[int] = (),
    ) -> list[mx.array]:
        """Read `head` after what `cache` holds and add it to it, then every tail after that, each seeing the cache,
        the head and its own earlier tokens only, all in one pass: the end of a shared prefix and every question.
        The final hidden states [len(rows[i]), hidden] of each tail's rows. The recurrent state is left at the end
        of the head (kept at `marks` inside it too); the attention buffers may hold the tails' keys past `length`."""
        ...

    def batch(
        self,
        prompts: Sequence[Sequence[int]],
        rows: Sequence[Sequence[int]],
        groups: Sequence[Sequence[int]] | None = None,
    ) -> list[mx.array]:
        """Read whole prompts together, each from an empty cache, in one pass: the weights are read once for all of
        them. Each prompt's rows' final hidden states [len(rows[i]), hidden]. The way to read several prompts on a
        non-causal backbone, whose prompts can't share a prefix. `groups`: set prompts' tokens' groups
        (`plan.Prompt.groups`), which a set backbone's non-causal layers read by."""
        ...
