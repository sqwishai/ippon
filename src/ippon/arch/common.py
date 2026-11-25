"""What every architecture's passes share: how a pass's rows sit in their sequences, RoPE at those rows'
positions, the attention mask each pass needs, and attention over the cache for only the rows that are read.

A pass is one of three layouts (`Rows`):

- prefill: one sequence of `n` rows continuing the cache at `start`;
- branch: `head` rows that continue the cache causally (the last part of a shared prefix), then several tails one
  after another, each continuing the cache and the head: `positions` are each row's position, `tree` the rows
  each row may attend to within the pass (a tail's earlier rows, or all of them for a block model's tails), and
  `gather`/`valid`/`scatter` turn the tails' flat rows into [tails, longest tail] for recurrent layers and back. One
  pass reads the end of the prefix and every question;
- batch: several whole prompts as a batch [prompts, n], right-padded; `key_valid` marks the real tokens (None when
  every prompt has the same length: no padding, no mask). Set prompts (`plan.Prompt.groups`) carry each token's
  group in `groups`: a non-causal layer's state tokens read the whole prompt, its question tokens the state and
  their own question.

`marks` are positions within the pass's causal rows (a prefill's, or a branch's head) where recurrent layers keep
their state for `prefixes.py`: they split the recurrence there, not the pass.
"""

from __future__ import annotations

import hmac
import secrets
from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import numpy as np

from ..checkpoint import HEAD
from ..errors import UnsupportedModel
from ..kernels import attention as attention_kernels
from ..quant import JoinedStacked, Matrix, Stacked, Table, arrays, concat, quantize, rows
from ..weights import Weights
from .cache import Attention, HybridCache, Recurrent
from .ops import MLX, Ops


@dataclass
class Rows:
    n: int  # rows per sequence
    start: int  # the cache length the rows continue
    positions: mx.array | None = None  # [n] int32, branch passes only
    tree: mx.array | None = None  # [n, n] bool
    gather: mx.array | None = None  # [tails, longest] int32 indices into the rows
    valid: mx.array | None = None  # [tails, longest] bool
    scatter: mx.array | None = None  # [tail rows] indices into tails * longest
    key_valid: mx.array | None = None  # [prompts, n] bool, batch passes with padding only
    groups: mx.array | None = None  # [prompts, n] int32, -1 on the state (and padding): set prompts of 2+ questions
    head: int = 0  # a branch's causal rows before its tails
    marks: tuple[int, ...] = ()  # where recurrent layers keep their state: 0 < mark < the causal rows' count
    masks: dict[tuple[bool, int | None], mx.array | str | None] = field(default_factory=dict, repr=False)  # per pass

    @classmethod
    def prefill(cls, n: int, start: int, marks: Sequence[int] = ()) -> Rows:
        if any(not 0 < m < n for m in marks):
            raise ValueError("a mark is inside the pass")
        return cls(n, start, marks=tuple(sorted(set(marks))))

    @classmethod
    def branch(
        cls, lengths: Sequence[int], start: int, head: int = 0, marks: Sequence[int] = (), both_ways: bool = False
    ) -> Rows:
        """`both_ways`: each tail's rows see all of that tail's rows (a block model's questions), not its earlier
        ones only."""
        if any(not 0 < m < head for m in marks):
            raise ValueError("a mark is inside the branch's head")
        sizes = np.asarray(lengths, dtype=np.int64)
        n, longest, count = head + int(sizes.sum()), int(np.max(sizes)), len(sizes)
        owner = np.concatenate([np.full(head, -1), np.repeat(np.arange(count), sizes)])  # -1: the head
        offsets = head + np.concatenate([[0], np.cumsum(sizes)[:-1]])  # where each tail starts in the pass
        within = np.concatenate([np.arange(head), head + np.arange(n - head) - np.repeat(offsets - head, sizes)])
        o, w = mx.array(owner), mx.array(within)
        same = o[:, None] == o[None, :]
        tree = (same | (o[None, :] == -1)) & (w[None, :] <= w[:, None])
        if both_ways:
            tree = tree | (same & (o[:, None] >= 0))
        t = np.arange(longest)
        gather = offsets[:, None] + np.minimum(t[None, :], sizes[:, None] - 1)  # a tail's padding repeats its last row
        tails = np.repeat(np.arange(count), sizes)
        position = np.arange(n - head) - np.repeat(offsets - head, sizes)
        return cls(
            n,
            start,
            positions=mx.array((start + within).astype(np.int32)),
            tree=tree,
            gather=mx.array(gather.astype(np.int32)),
            valid=mx.array(t[None, :] < sizes[:, None]),
            scatter=mx.array((tails * longest + position).astype(np.int32)),
            head=head,
            marks=tuple(sorted(set(marks))),
        )

    @classmethod
    def batch(cls, lengths: Sequence[int], groups: Sequence[Sequence[int]] | None = None) -> Rows:
        """`groups`: each prompt's tokens' groups (`plan.Prompt.groups`), kept when a prompt has two questions or more
        (with one, every token reads every other)."""
        longest = max(lengths)
        valid = None
        if min(lengths) != longest:
            valid = mx.arange(longest)[None, :] < mx.array(list(lengths))[:, None]
        kept = None
        if groups is not None and any(len({g for g in gs if g >= 0}) > 1 for gs in groups):
            kept = mx.array([list(gs) + [-1] * (longest - len(gs)) for gs in groups], dtype=mx.int32)
        return cls(longest, 0, key_valid=valid, groups=kept)

    @property
    def is_branch(self) -> bool:
        return self.tree is not None

    @property
    def padded(self) -> bool:
        """A batch whose prompts differ in length, so some of its rows are padding."""
        return self.key_valid is not None

    def to_tails(self, x: mx.array) -> mx.array:
        """A branch pass's tail rows of [1, n, ...] as [tails, longest tail, ...] (padding repeats a tail's last
        row)."""
        assert self.gather is not None
        return mx.take(x[0], self.gather.reshape(-1), axis=0).reshape(*self.gather.shape, *x.shape[2:])

    def from_tails(self, x: mx.array) -> mx.array:
        """[tails, longest tail, ...] back to the tails' flat rows [1, tail rows, ...]."""
        assert self.scatter is not None
        return mx.take(x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]), self.scatter, axis=0)[None]


@dataclass(frozen=True)
class Read:
    """The rows of a pass whose final hidden states are asked for, which the last layer alone computes: `index`
    ([sequences, rows] indices into each sequence's rows), or, when every sequence reads the same run of rows, that
    run `start`..`start + count` (`index` None). A run is taken as a slice (all the rows: as they are), and a run
    that ends at the pass's last row keeps causal attention's fast path; heads that read every position read one."""

    index: mx.array | None
    start: int = 0
    count: int = 0

    @classmethod
    def of(cls, rows: Sequence[Sequence[int]]) -> Read:
        """From each sequence's rows, in order."""
        first = list(rows[0]) if rows else []
        if first and all(list(r) == first for r in rows[1:]) and first == list(range(first[0], first[0] + len(first))):
            return cls(None, first[0], len(first))
        if not first and all(not r for r in rows):
            return cls(None, 0, 0)
        return cls(mx.array([list(r) for r in rows], dtype=mx.int32))

    @property
    def size(self) -> int:
        """Rows read per sequence."""
        return self.count if self.index is None else self.index.shape[1]

    def suffix(self, n: int) -> bool:
        """True when the rows are the last ones of a pass of `n` rows, in order."""
        return self.index is None and self.start + self.count == n

    def rows(self) -> mx.array:
        """The first sequence's rows, [rows] int32."""
        return mx.arange(self.start, self.start + self.count) if self.index is None else self.index[0]

    def take(self, x: mx.array, axis: int = 1) -> mx.array:
        """The read rows of x, whose axis 0 is the sequences and `axis` the pass's rows."""
        if self.index is None:
            if self.start == 0 and self.count == x.shape[axis]:
                return x
            at = [slice(None)] * x.ndim
            at[axis] = slice(self.start, self.start + self.count)
            return x[tuple(at)]
        shape = [1] * x.ndim
        shape[0], shape[axis] = self.index.shape
        return mx.take_along_axis(x, self.index.reshape(shape), axis=axis)


def mask(r: Rows, causal: bool, window: int | None = None) -> mx.array | str | None:
    """Which keys each query sees: for a branch, the cache and its own tail's earlier rows; for a batch, its
    prompt's real tokens (and only earlier ones when causal); for a prefill, the earlier tokens when causal; with
    `window`, only keys at most that many positions away too (a pass continuing a cache or a branch's: [rows,
    cache + rows], by the rows' and the keys' positions). Built once per pass and shared by its attention layers (a
    branch's is [rows, cache + rows]). A window no two of the pass's positions are farther apart than is none."""
    if window is not None and r.start + r.n - 1 <= window:
        window = None
    key = (causal, window)
    if key not in r.masks:
        m = _mask(r, causal)
        if window is not None and (r.start or r.is_branch):
            rows_at = r.positions if r.positions is not None else r.start + mx.arange(r.n)
            keys_at = mx.concatenate([mx.arange(r.start), rows_at])
            gap = rows_at[:, None] - keys_at[None, :]
            near = (gap <= window) & (gap >= (0 if causal else -window))
            m = near if not isinstance(m, mx.array) else m & near
        elif window is not None:
            n = r.n
            near = mx.abs(mx.arange(n)[:, None] - mx.arange(n)[None, :]) <= window
            if causal:
                near = near & mx.tril(mx.ones((n, n), dtype=mx.bool_))
            m = near[None, None] if not isinstance(m, mx.array) else m & near[None, None]
            if r.padded:  # a padding row far past its prompt sees no real key: let it see itself, never nothing
                m = m | mx.eye(n, dtype=mx.bool_)[None, None]
        r.masks[key] = m
    return r.masks[key]


def _mask(r: Rows, causal: bool) -> mx.array | str | None:
    if r.groups is not None and not causal:  # set prompts: the state reads every token, a question the state, itself
        state = r.groups < 0
        m = (state[:, :, None] | state[:, None, :] | (r.groups[:, :, None] == r.groups[:, None, :]))[:, None]
        return m & r.key_valid[:, None, None, :] if r.key_valid is not None else m
    if r.is_branch:
        assert r.tree is not None
        return mx.concatenate([mx.ones((r.n, r.start), dtype=mx.bool_), r.tree], axis=1)
    if r.padded and not causal:
        assert r.key_valid is not None
        return r.key_valid[:, None, None, :]
    # a causal batch needs no padding mask: each prompt's padding comes after its real tokens, which never see it
    return "causal" if causal else None


def attend(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    block: Attention | None,
    r: Rows,
    read: Read | None,
    causal: bool,
    scale: float,
    window: int | None = None,
    ops: Ops = MLX,
) -> mx.array:
    """Attention for one layer: q [b, heads, n, d], k and v [b, kv heads, n, d] after RoPE. Writes k and v into the
    cache block, then attends for the `read` rows only (every row when None). With `window`, a query sees only keys
    at most `window` positions away (`mask`). The output is [b, rows, heads, d]; [b, 0, heads, d] when no row is
    read (the keys and values are still written)."""
    b, n = q.shape[0], r.n
    keys, values = block.write(r.start, k, v) if block is not None else (k, v)
    if window is not None and keys.shape[2] == n and banded_pass(r, window):
        o = banded(q, keys, values, r, causal, window, scale, ops)
        return (o if read is None else read.take(o, axis=2)).transpose(0, 2, 1, 3)
    m = mask(r, causal, window)
    if read is None and r.is_branch and r.head and causal and (window is None or r.start + n - 1 <= window):
        # a branch's head (the prefix's last rows) reads causally, on MLX's fused attention (its rows are the last
        # of the keys up to them), and only the tails' rows by the tree mask: the whole pass under one explicit mask
        # cost a dense model about three times as much per question on a 4,000-token state
        h, seen = r.head, r.start + r.head
        before = ops.attention(q[:, :, :h], keys[:, :, :seen], values[:, :, :seen], scale, "causal")
        assert isinstance(m, mx.array)
        after = ops.attention(q[:, :, h:], keys, values, scale, m[h:])
        return mx.concatenate([before, after], axis=2).transpose(0, 2, 1, 3)
    if read is not None:
        if read.size == 0:
            return mx.zeros((b, 0, q.shape[1], q.shape[3]), dtype=q.dtype)
        q = read.take(q, axis=2)
        if m == "causal" and not read.suffix(n):  # a suffix stays causal: MLX aligns the queries with the last keys
            if read.index is None:
                m = mx.arange(r.start + n)[None, :] <= (r.start + read.rows())[:, None]
            else:  # each sequence its own rows (a batch of prompts of different lengths)
                m = mx.arange(r.start + n)[None, None, None, :] <= (r.start + read.index)[:, None, :, None]
        elif isinstance(m, mx.array) and m.ndim == 2:  # [rows, cache + rows]: a branch's, or a windowed pass's
            m = m[read.rows()]
        elif isinstance(m, mx.array) and m.shape[2] != 1:  # a batch's causal mask: the read rows' lines
            m = read.take(mx.broadcast_to(m, (b, 1, n, n)), axis=2)
    o = ops.attention(q, keys, values, scale, m)
    return o.transpose(0, 2, 1, 3)


# A sliding layer's whole pass from an empty cache reads its queries in blocks of BAND rows, each against only the
# keys its window can reach (`banded`), from BAND_WINDOWS windows' worth of rows: under one dense mask, MLX's
# attention visits every key block of every query block, the masked ones too, but gathering each block's keys costs
# more than that until the pass is long (M5 Pro, ModernBERT's 64-row window, the pass with the band against the
# dense mask: a ModernBERT-large encoder 1.079 and 1.023 times as long at 512 and 1,024 rows, 0.944 at 2,048; an
# mmBERT-base one 1.127, 1.047, 0.903): a pass shorter than BAND_WINDOWS windows (2,048 rows) keeps the dense mask.
BAND, BAND_WINDOWS = 128, 32


def banded_pass(r: Rows, window: int) -> bool:
    """Whether a layer of `window` reads pass `r` in bands (`banded`, no mask): a whole pass from an empty cache of at
    least BAND_WINDOWS windows' rows."""
    return r.start == 0 and not r.is_branch and r.n >= BAND_WINDOWS * window


def banded(
    q: mx.array, k: mx.array, v: mx.array, r: Rows, causal: bool, window: int, scale: float, ops: Ops = MLX
) -> mx.array:
    """Windowed attention for a whole pass, q [b, heads, n, d] and k, v [b, kv heads, n, d] with no cache before
    them: each block of BAND queries attends the BAND + 2 * window keys around it (+ window before it when causal),
    gathered into one batch of blocks for one call, under the same window (and a batch's padding) as `mask`. The
    same keys as the dense mask's, so the output differs only in the order of the fp32 sums. [b, heads, n, d]."""
    b, heads, n, d = q.shape
    kv = k.shape[1]
    blocks = -(-n // BAND)
    left, right = window, 0 if causal else window
    span = BAND + left + right
    tail = blocks * BAND - n
    qb = mx.pad(q, [(0, 0), (0, 0), (0, tail), (0, 0)]).reshape(b, heads, blocks, BAND, d)
    qb = qb.transpose(0, 2, 1, 3, 4).reshape(b * blocks, heads, BAND, d)
    at = (mx.arange(blocks)[:, None] * BAND + mx.arange(span)[None, :]).reshape(-1)  # into keys padded by `left`

    def gather(x: mx.array) -> mx.array:
        padded = mx.pad(x, [(0, 0), (0, 0), (left, right + tail), (0, 0)])
        out = mx.take(padded, at, axis=2).reshape(b, kv, blocks, span, d)
        return out.transpose(0, 2, 1, 3, 4).reshape(b * blocks, kv, span, d)

    query = mx.arange(blocks)[:, None, None] * BAND + mx.arange(BAND)[None, :, None]  # [blocks, BAND, 1] positions
    key = mx.arange(blocks)[:, None, None] * BAND + mx.arange(span)[None, None, :] - left  # [blocks, 1, span]
    near = (key >= 0) & (key < n) & (query - key <= window) & (key - query <= (0 if causal else window))
    m = mx.broadcast_to(near[None], (b, blocks, BAND, span))
    if r.padded:  # a batch: only its prompt's real keys, and a padding row past them sees itself, never nothing
        assert r.key_valid is not None
        valid = mx.pad(r.key_valid, [(0, 0), (left, right + tail)])
        m = (m & mx.take(valid, at, axis=1).reshape(b, blocks, 1, span)) | (query == key)[None]
    o = ops.attention(qb, gather(k), gather(v), scale, m.reshape(b * blocks, 1, BAND, span))
    o = o.reshape(b, blocks, heads, BAND, d).transpose(0, 2, 1, 3, 4).reshape(b, heads, blocks * BAND, d)
    return o[:, :, :n]


Snapshots = dict[int, tuple[mx.array | None, mx.array | None]]  # mark -> one recurrent layer's (state, window)

# A medium (an image, a clip) enters a prompt as placeholder tokens, one per row its encoder gives: negative ids
# made of 48 bits of the medium's digest and the row, so prompts with different media differ token for token and
# everything keyed by tokens (plans, kept prefixes) tells them apart unchanged. The pass reads each placeholder's row
# from the media encoded for it (`embed`). The 48 bits are of an HMAC of the medium's sha256 under a key made fresh
# in each process: 48 bits of the sha256 itself let anyone make two images with the same placeholders in seconds,
# and the second was answered as the first.
MEDIA_ROW_BITS = 15  # rows of one medium a placeholder addresses: 32,768
_MEDIA_KEY = secrets.token_bytes(32)


def media_digest(sha256: str) -> int:
    """48 bits of the keyed hash of a medium's sha256: what its placeholder tokens (`media_id`) name it by."""
    return int.from_bytes(hmac.digest(_MEDIA_KEY, bytes.fromhex(sha256), "sha256")[:6], "big")


def media_id(digest: int, row: int) -> int:
    """The placeholder token for row `row` of the medium whose digest's first 48 bits are `digest`."""
    if not 0 <= row < 1 << MEDIA_ROW_BITS:
        raise ValueError(f"a medium has at most {1 << MEDIA_ROW_BITS} rows")
    return -(1 + (((digest & ((1 << 48) - 1)) << MEDIA_ROW_BITS) | row))


def media_of(token: int) -> tuple[int, int]:
    """The (digest, row) of a placeholder token (`media_id`)."""
    v = -token - 1
    return v >> MEDIA_ROW_BITS, v & ((1 << MEDIA_ROW_BITS) - 1)


def is_placeholder(token: int) -> bool:
    """Whether a prompt's token is a medium's placeholder (`media_id`) rather than one of the tokenizer's."""
    return token < 0


def placeholders(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where `tokens` (int64, flattened) holds placeholders, and each one's digest and row (`media_of`, over the
    array)."""
    flat = tokens.reshape(-1)
    at = np.flatnonzero(flat < 0)
    v = -flat[at] - 1
    return at, v >> MEDIA_ROW_BITS, v & ((1 << MEDIA_ROW_BITS) - 1)


def expand(ids: Sequence[int], token: int, media: Sequence[Sequence[int]]) -> list[int]:
    """`ids` with each `token` replaced by its medium's placeholders (`media`, in order), as a processor expands a
    template's one media token per medium. RuntimeError when the prompt does not hold one token per medium."""
    at = [i for i, t in enumerate(ids) if t == token]
    if len(at) != len(media):
        raise RuntimeError(f"the prompt holds {len(at)} media tokens for {len(media)} media")
    out: list[int] = []
    last = 0
    for i, placeholders in zip(at, media, strict=True):
        out += ids[last:i]
        out += placeholders
        last = i + 1
    return out + list(ids[last:])


KEPT_BYTES = 256 << 20  # a few dozen pictures, or a 20 s 720p video's 12,072 rows (49 MB) five times over


class Kept:
    """The encoded rows of the media read last, up to `budget` bytes, by medium: a medium sent again (a state re-sent
    with its pictures, a new question about the same video) skips its encoder. A medium is kept and dropped whole,
    with all its digests (a video's frame pairs), so a long video never pushes out its own first pairs; one larger
    than the budget is not kept."""

    def __init__(self, budget: int = KEPT_BYTES) -> None:
        self.budget = budget
        self.media: OrderedDict[Hashable, dict[int, mx.array]] = OrderedDict()
        self.bytes = 0

    def get(self, medium: Hashable) -> dict[int, mx.array] | None:
        found = self.media.get(medium)
        if found is not None:
            self.media.move_to_end(medium)
        return found

    def put(self, medium: Hashable, rows: dict[int, mx.array]) -> None:
        size = sum(r.nbytes for r in rows.values())
        if size > self.budget:
            return
        old = self.media.pop(medium, None)
        self.bytes += size - (sum(r.nbytes for r in old.values()) if old else 0)
        self.media[medium] = rows
        while self.bytes > self.budget:
            self.bytes -= sum(r.nbytes for r in self.media.popitem(last=False)[1].values())


class MediaRows(dict[int, mx.array]):
    """A pass's encoded media by digest (`Backbone.media`): those encoded before it, and media `encode` gives the rows
    of (by digest) encoded only once a pass reads one of their placeholders (`rows_of`), so a medium inside a kept
    prefix (the pictures of a state sent again, or grown by one more) is never encoded at all."""

    def __init__(
        self, rows: Mapping[int, mx.array], encode: Callable[[set[int]], Mapping[int, mx.array]] | None = None
    ) -> None:
        super().__init__(rows)
        self.encode = encode

    def take(self, digests: Sequence[int]) -> dict[int, mx.array]:
        missing = {d for d in digests if d not in self}
        if missing and self.encode is not None:
            self.update(self.encode(missing))
        return {d: self[d] for d in digests if d in self}


def rows_of(media: Mapping[int, mx.array], digests: Sequence[int]) -> dict[int, mx.array]:
    """The encoded rows of the media `digests` name, of those `media` has or encodes now (`MediaRows`)."""
    if isinstance(media, MediaRows):
        return media.take(digests)
    return {d: media[d] for d in digests if d in media}


def embed(table: Table, ids: mx.array, media: Mapping[int, mx.array] | None = None) -> mx.array:
    """The embedding rows of `ids` ([sequences, rows]), a placeholder's (a negative id) from `media`: each medium's
    encoded rows by its digest's first 48 bits, [rows, hidden], or shaped by its grid, [frames, rows, columns,
    hidden], for a model that places them by it (Qwen3.5's multimodal RoPE)."""
    if media is None:
        return rows(table, ids)
    flat = np.array(ids).reshape(-1)
    at, digests, media_rows = placeholders(flat)
    if at.size == 0:
        return rows(table, ids)
    plain = flat.copy()
    plain[at] = 0  # a placeholder's place in the table's lookup, replaced below
    x = rows(table, mx.array(plain.reshape(ids.shape)))
    starts, parts, total = {}, [], 0
    for digest, encoded in rows_of(media, list(dict.fromkeys(digests.tolist()))).items():
        starts[digest] = total
        parts.append(encoded.reshape(-1, encoded.shape[-1]))
        total += parts[-1].shape[0]
    found = list(zip(digests.tolist(), media_rows.tolist(), strict=True))
    missing = {d for d, _ in found if d not in starts}
    if missing:
        raise UnsupportedModel(f"a prompt holds {len(missing)} medium(s) the pass was given no rows for")
    index = mx.array([starts[d] + row for d, row in found])
    out = x.reshape(-1, x.shape[-1])
    out[mx.array(at)] = mx.concatenate(parts)[index].astype(out.dtype)
    return out.reshape(x.shape)


PIPELINE_MIN_LAYERS = 12  # a model with fewer layers builds its pass faster than a hand-over saves


def hand_over(ops: Ops, layer: int, last: int, *arrays: mx.array | None) -> None:
    """With the pipeline switch, every `ops.pipeline` layers: start the GPU on the pass so far (`mx.async_eval`) while
    Python builds the layers after it. MLX runs nothing until something is evaluated, so a short request otherwise
    waits for its whole graph to be built (about 2.5 ms for a 2B model) before its first kernel. Not for a model of
    fewer than PIPELINE_MIN_LAYERS layers. "early" (ops.pipeline_first) also hands over after the first layer, so
    the GPU starts sooner."""
    if not ops.pipeline or last + 1 < PIPELINE_MIN_LAYERS or layer >= last:
        return
    if layer % ops.pipeline == ops.pipeline - 1 or layer + 1 == ops.pipeline_first:
        mx.async_eval([a for a in arrays if a is not None])  # type: ignore[no-untyped-call]


@dataclass(frozen=True)
class Recurrence:
    """Recurrent depth: blocks of layers read more than once within one pass, with no weights added (Fastino's GLiDE
    9B, its decision_config.json "midloop"). Each block [a, b] (inclusive, in order, apart) is a function G of the
    residual x entering layer a. Its `passes` P reads move x by damped Euler steps, x <- x + (G(x) - x) / P, each in
    fp32 and rounded back to the residual's dtype, and the last x goes on to layer b + 1. P = 1 is the plain pass.
    Only for a pass with no cache, since a block read again would overwrite what its first read kept."""

    blocks: tuple[tuple[int, int], ...]
    passes: int

    def __post_init__(self) -> None:
        if self.passes < 1 or any(a > b for a, b in self.blocks):
            raise UnsupportedModel(f"recurrent depth {self}: a block is [first, last] and passes are 1 or more")
        if any(earlier[1] >= later[0] for earlier, later in zip(self.blocks, self.blocks[1:], strict=False)):
            raise UnsupportedModel(f"recurrent depth {self}: blocks go in order and must not overlap")

    def step(self, entering: mx.array, out: mx.array) -> mx.array:
        """One damped Euler step from `entering` toward the block's output `out`."""
        moved = entering.astype(mx.float32)
        return (moved + (out.astype(mx.float32) - moved) / self.passes).astype(entering.dtype)


def remember(cache: HybridCache | None, r: Rows, snapshots: Sequence[Snapshots]) -> None:
    """Keep, in the cache, the recurrent layers' state at the pass's marks (`snapshots` in layer order)."""
    if cache is not None and r.marks:
        cache.keep({r.start + m: [s[m] for s in snapshots] for m in r.marks})


# -- compiled layers ----------------------------------------------------------------------------------------------
#
# With the compile switch, each layer runs as one call of an `mx.compile`d function whose inputs are the layer's
# weights, the pass's arrays (its masks, positions, padding) and the layer's cache block, and whose constants are
# everything else the layer's graph depends on (its kind, the pass's shape and positions). Every layer of a kind then
# replays one trace: Python builds a layer once per kind and pass shape, not once per layer, and the replay creates
# the same kernels' arrays without running the layer's Python or rebuilding each kernel's source. The arithmetic is
# the eager layer's, kernel for kernel, except that MLX fuses the GLU's chain of elementwise ops, whose last bits can
# move: a test holds the outputs equal to fp32 rounding (4e-7 on Gemma 3's outputs of about 3). d1-omni-600M
# (M5 Pro), 143 tokens: the
# pass's graph built in 0.06 ms of CPU replayed against 0.40 ms eagerly (0.79 against 1.18 ms with its hand-overs'
# encoding); a length seen for the first time costs about what building it eagerly does.

# Pass shapes whose traces a backbone keeps, in each of two generations: a trace holds its kernels' sources, about
# 0.36 MB of traces a pass shape for d1-omni-600M (200 lengths: 75 MB), so a third generation's first shape drops the
# oldest. A shape seen for the first time costs its trace, about what building the pass eagerly costs.
COMPILED_SHAPES = 32
ROW_ARRAYS = ("positions", "tree", "gather", "valid", "scatter", "key_valid", "groups")


class Compiled:
    """`fn` through mx.compile, in two generations of at most COMPILED_SHAPES pass shapes each: a pass of a shape
    either has seen replays its traces; a new one is traced in the newer, which once full becomes the older, the
    older's traces dropped."""

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn
        self.generations: list[tuple[Callable[..., Any], set[Hashable]]] = [(mx.compile(fn), set())]

    def start(self, shape: Hashable) -> Callable[..., Any]:
        """The compiled function for a pass of this shape."""
        for call, seen in self.generations:
            if shape in seen:
                return call
        if len(self.generations[0][1]) >= COMPILED_SHAPES:
            self.generations = [(mx.compile(self.fn), set()), self.generations[0]]
        call, seen = self.generations[0]
        seen.add(shape)
        return call


def split_rows(r: Rows) -> tuple[tuple[Any, ...], dict[str, mx.array]]:
    """A pass's layout as a compiled layer takes it: the constants its trace is keyed by, and its arrays (the masks
    built so far among them) as inputs."""
    arrays = {name: a for name in ROW_ARRAYS if (a := getattr(r, name)) is not None}
    masks = []
    for (causal, window), m in r.masks.items():
        if isinstance(m, mx.array):
            arrays[f"mask {causal} {window}"] = m
        masks.append((causal, window, "array" if isinstance(m, mx.array) else m))
    names = tuple(name for name in ROW_ARRAYS if name in arrays)
    return (r.n, r.start, r.head, r.marks, names, tuple(masks)), arrays


def joined_rows(spec: tuple[Any, ...], arrays: Mapping[str, mx.array]) -> Rows:
    """The Rows `split_rows` took apart, its arrays from `arrays` (a trace's inputs)."""
    n, start, head, marks, names, masks = spec
    r = Rows(n, start, head=head, marks=marks)
    for name in names:
        setattr(r, name, arrays[name])
    for causal, window, m in masks:
        r.masks[(causal, window)] = arrays[f"mask {causal} {window}"] if m == "array" else m
    return r


def split_block(block: Attention | Recurrent | None) -> tuple[tuple[Any, ...], list[mx.array | None]]:
    """A cache block as a compiled layer takes it: its kind (and which of its arrays exist), and its arrays."""
    if block is None:
        return ("none",), []
    if isinstance(block, Attention):
        return ("attention",), [block.k, block.v]
    return ("recurrent", block.state is None, block.window is None), [block.state, block.window]


def joined_block(spec: tuple[Any, ...], state: Sequence[mx.array | None]) -> Attention | Recurrent | None:
    if spec[0] == "attention":
        k, v = state
        assert k is not None
        assert v is not None
        return Attention(k, v)
    if spec[0] == "recurrent":
        return Recurrent(state[0], state[1])
    return None


def block_arrays(block: Attention | Recurrent | None) -> list[mx.array | None]:
    if isinstance(block, Attention):
        return [block.k, block.v]
    if isinstance(block, Recurrent):
        return [block.state, block.window]
    return []


class Passes:
    """prefill, branch and batch for a causal architecture over its `_forward(ids, cache, rows, keep)`, and the layer
    loop every architecture's `_forward` runs (`_layers`): each layer's `_layer`, eagerly or compiled. The pre-norm
    decoders share `_decoder`, whose layer is the input norm, the architecture's `_mixer` and `_feed_forward`."""

    config: Any  # the architecture's settings: `heads` among them
    ops: Ops = MLX  # how norms and GLUs are computed: set by the decider from the chip profile
    block: bool = False  # Backbone.block: a shared prefix read both ways within itself, its tails after it
    layers: list[dict[str, Any]]
    media: Mapping[int, mx.array] | None = None  # the encoded media a pass's placeholders read (`embed`), while set
    recurrence: Recurrence | None = None  # blocks of layers read more than once a pass (qwen35.load's `recurrence`)
    _compiled: tuple[Ops, Compiled | None] | None = None  # the compiled layer, for the ops it was traced with

    @property
    def max_tokens(self) -> int | None:
        """No limit of its own: positions are rotary (`Backbone.max_tokens`)."""
        return None

    @property
    def causal(self) -> bool:
        """Every layer reads only earlier positions: an architecture whose layers read both ways says otherwise."""
        return True

    @property
    def layer_kinds(self) -> Sequence[Hashable]:
        """What sets each layer's graph apart besides its weights (its type, for a hybrid): layers of one kind replay
        one compiled trace."""
        return [None] * len(self.layers)

    def mask_keys(self, r: Rows) -> Sequence[tuple[bool, int | None]]:
        """The (causal, window) masks the layers of pass `r` ask `mask` for: built once a pass, before the layers, so
        a compiled layer takes them as inputs instead of building them again."""
        return [(self.causal, None)]

    def _forward(self, ids: mx.array, cache: HybridCache | None, r: Rows, keep: Read) -> mx.array:
        raise NotImplementedError

    def _mixer(
        self,
        kind: Hashable,
        layer: dict[str, Any],
        h: mx.array,
        block: Any,
        r: Rows,
        read: Read | None,
        kept: list[Snapshots],
        at: Any,
    ) -> mx.array:
        """A pre-norm decoder layer's token mixer over its normed input `h` (attention or a recurrent layer), for the
        `read` rows only in the last layer; a recurrent one adds its state at the pass's marks to `kept`."""
        raise NotImplementedError

    def _feed_forward(self, kind: Hashable, layer: dict[str, Any], h: mx.array) -> mx.array:
        """A pre-norm decoder layer's MLP over its normed input."""
        raise NotImplementedError

    def _layer(
        self,
        kind: Hashable,
        layer: dict[str, Any],
        x: mx.array,
        pending: mx.array | None,
        block: Any,
        r: Rows,
        read: Read | None,
        kept: list[Snapshots],
        constants: tuple[Any, ...],
        at: Any,
    ) -> tuple[mx.array, mx.array | None]:
        """One layer of a pre-norm decoder: (the residual, the MLP's output not yet added to it). Its input norm
        `in_norm` (added to the previous layer's MLP output `pending` in one `ops.add_norm`), `_mixer` (the read rows
        only in the last layer, which stops once its keys are kept when none is read: (no rows, None)), the
        residual and the norm `post`, and `_feed_forward`. `constants` are (eps, norm groups, post)."""
        eps, groups, post = constants
        ops = self.ops
        if pending is None:
            h = ops.norm(x, layer["in_norm"], eps, groups=groups)
        else:
            x, h = ops.add_norm(x, pending, layer["in_norm"], eps, groups=groups)
        mixed = self._mixer(kind, layer, h, block, r, read, kept, at)
        if read is not None:
            if read.size == 0:
                return mx.zeros((x.shape[0], 0, x.shape[-1]), dtype=x.dtype), None
            x = read.take(x)
        x, h = ops.add_norm(x, mixed, layer[post], eps, groups=groups)
        return x, self._feed_forward(kind, layer, h)

    def _decoder(
        self,
        x: mx.array,
        cache: HybridCache | None,
        r: Rows,
        keep: Read,
        final_norm: mx.array,
        eps: float,
        groups: int = 1,
        post: str = "post_norm",
        at: Any = None,
    ) -> mx.array:
        """Final-normed hidden states [sequences, rows, hidden] of the `keep` rows from embeddings `x`, through the
        layers of a pre-norm decoder (`_layer`). Recurrent mixers add their state at the pass's marks to `kept`,
        which goes into the cache once the layers have run. `at`: what the architecture's mixers place the rows by
        when not the pass's own positions (None: by `r`; a pass with one runs eagerly)."""
        x, pending, kept = self._layers(x, cache, r, keep, (eps, groups, post), at)
        remember(cache, r, kept)
        if pending is None:  # the last layer read no row
            return x
        return self.ops.add_norm(x, pending, final_norm, eps, groups=groups)[1]

    def _layers(
        self,
        x: mx.array,
        cache: HybridCache | None,
        r: Rows,
        keep: Read,
        constants: tuple[Any, ...],
        at: Any = None,
    ) -> tuple[mx.array, mx.array | None, list[Snapshots]]:
        """Every layer's `_layer` in turn, handing the GPU the pass so far every few (`hand_over`): (the residual,
        the last MLP output not yet added to it, None when the last layer read no row; the recurrent layers' states
        at the marks). Compiled (see above) with the ops' compile switch on the GPU, when every weight is an array
        and the rows are placed by the pass itself (`at` None); eagerly otherwise, and for the measuring Ops. With
        `recurrence`, its blocks of layers are read again as it says."""
        ops, last, kinds = self.ops, len(self.layers) - 1, self.layer_kinds
        compiled = self._compiled_layer() if at is None and keep.size and not self._oversized(x, r) else None
        kept: list[Snapshots] = []
        blocks = cache.layers if cache is not None else [None] * len(self.layers)
        if compiled is None:

            def step(i: int, x: mx.array, pending: mx.array | None) -> tuple[mx.array, mx.array | None]:
                read = keep if i == last else None
                return self._layer(kinds[i], self.layers[i], x, pending, blocks[i], r, read, kept, constants, at)

        else:
            for key in self.mask_keys(r):
                mask(r, *key)
            rows, arrays = split_rows(r)
            reading = (keep.start, keep.count, keep.index is not None)
            if keep.index is not None:
                arrays = arrays | {"read": keep.index}
            shapes = [split_block(b)[0] for b in blocks]
            capacity = next((b.k.shape[2] for b in blocks if isinstance(b, Attention)), 0)
            call = compiled.start((x.shape, rows, reading, constants, capacity, tuple(shapes)))

            def step(i: int, x: mx.array, pending: mx.array | None) -> tuple[mx.array, mx.array | None]:
                block = blocks[i]
                spec = (kinds[i], rows, reading if i == last else None, shapes[i], constants)
                x, pending, state, marks = call(x, pending, self.layers[i], block_arrays(block), arrays, spec)
                if isinstance(block, Attention):
                    block.k, block.v = state
                elif isinstance(block, Recurrent):
                    block.state, block.window = state
                if marks is not None:
                    kept.append({m: (s, w) for m, (s, w) in zip(r.marks, marks, strict=True)})
                return x, pending

        recurrence = self.recurrence
        ends = {b: a for a, b in recurrence.blocks} if recurrence is not None and recurrence.passes > 1 else {}
        if ends and cache is not None:
            raise ValueError("a model with recurrent depth reads every pass whole, with no cache")
        starts = set(ends.values())
        pending: mx.array | None = None
        entering = x  # the residual entering the current block (`Recurrence`)
        for i in range(len(self.layers)):
            if i in starts:
                entering, pending = (x if pending is None else x + pending), None
                x = entering
            x, pending = step(i, x, pending)
            if i in ends:
                assert recurrence is not None
                for again in range(recurrence.passes):
                    if again:  # G read again from the last step's residual
                        x, pending = entering, None
                        for j in range(ends[i], i + 1):
                            x, pending = step(j, x, pending)
                    entering = recurrence.step(entering, x if pending is None else x + pending)
                x, pending = entering, None
            hand_over(ops, i, last, x, pending)
        return x, pending, kept

    def _traced(
        self,
        x: mx.array,
        pending: mx.array | None,
        layer: dict[str, Any],
        state: list[mx.array | None],
        arrays: dict[str, mx.array],
        spec: tuple[Any, ...],
    ) -> tuple[mx.array, mx.array | None, list[mx.array | None], list[list[mx.array | None]] | None]:
        """`_layer` as the compiled function runs it: the pass's layout, the read rows and the cache block rebuilt
        from its inputs; its block's arrays and the recurrent state at the marks (None: not a recurrent layer)
        returned with its output."""
        kind, rows, reading, shape, constants = spec
        r = joined_rows(rows, arrays)
        read = None
        if reading is not None:
            start, count, indexed = reading
            read = Read(arrays["read"], start, count) if indexed else Read(None, start, count)
        block = joined_block(shape, state)
        kept: list[Snapshots] = []
        x, pending = self._layer(kind, layer, x, pending, block, r, read, kept, constants, None)
        marks = [list(kept[0][m]) for m in r.marks] if kept else None
        return x, pending, block_arrays(block), marks

    def _oversized(self, x: mx.array, r: Rows) -> bool:
        """Whether a both-ways pass's attention scores exceed what MLX can allocate (`score_limit`): then its layers
        run eagerly, so attention can read it in evaluated query blocks (`Ops.attention`)."""
        if self.causal or mx.default_device() != mx.gpu:
            return False
        heads: int = self.config.heads
        scores = x.shape[0] * heads * r.n * (r.start + r.n) * x.dtype.size
        return scores > attention_kernels.score_limit()

    def _compiled_layer(self) -> Compiled | None:
        """The compiled `_traced` for the current ops (made at its first use), or None where layers run eagerly: the
        compile switch off, explicit bounded-attention memory mode (which evaluates each query block), Ops that
        measure (a subclass), off the GPU, or a weight that isn't an array (a quantized matrix)."""
        ops = self.ops
        if not ops.compiled or ops.attention_bounded or type(ops) is not Ops or mx.default_device() != mx.gpu:
            return None
        if self._compiled is None or self._compiled[0] is not ops:
            plain = all(isinstance(v, mx.array) for layer in self.layers for v in layer.values())
            self._compiled = (ops, Compiled(self._traced) if plain else None)
        return self._compiled[1]

    def prefill(self, tokens: Sequence[int], cache: Any, rows: Sequence[int], marks: Sequence[int] = ()) -> mx.array:
        assert isinstance(cache, HybridCache)
        keep = Read.of([rows])
        out = self._forward(mx.array([list(tokens)]), cache, Rows.prefill(len(tokens), cache.length, marks), keep)
        cache.length += len(tokens)
        cache.schedule()
        return out[0]

    def branch(
        self,
        tails: Sequence[Sequence[int]],
        cache: Any,
        rows: Sequence[Sequence[int]],
        head: Sequence[int] = (),
        marks: Sequence[int] = (),
    ) -> list[mx.array]:
        if not self.causal and not self.block:  # no prefix to continue: every token reads them all
            raise ValueError("a model whose attention reads both ways reads every prompt whole")
        if self.block and head:
            raise ValueError("a block model reads its whole prefix before its tails")
        assert isinstance(cache, HybridCache)
        lengths = [len(t) for t in tails]
        if min(lengths) < 1:
            raise ValueError("every tail has at least one token")
        offsets = [len(head) + sum(lengths[:i]) for i in range(len(lengths))]
        flat_rows = [o + r for o, rs in zip(offsets, rows, strict=True) for r in rs]
        ids = mx.array([[*head, *(t for tail in tails for t in tail)]])
        r = Rows.branch(lengths, cache.length, len(head), marks, both_ways=self.block)
        out = self._forward(ids, cache, r, Read.of([flat_rows]))[0]
        cache.length += len(head)
        result, at = [], 0
        for rs in rows:
            result.append(out[at : at + len(rs)])
            at += len(rs)
        return result

    def batch(
        self,
        prompts: Sequence[Sequence[int]],
        rows: Sequence[Sequence[int]],
        groups: Sequence[Sequence[int]] | None = None,
    ) -> list[mx.array]:
        """Whole prompts as one right-padded batch, each from an empty cache: the weights are read once for all
        of them. Every prompt's rows, [len(rows[i]), hidden]."""
        lengths = [len(p) for p in prompts]
        if min(lengths) < 1:
            raise ValueError("every prompt has at least one token")
        longest, width = max(lengths), max(len(r) for r in rows)
        ids = mx.array([list(p) + [0] * (longest - len(p)) for p in prompts])
        keep = Read.of([list(r) + [r[-1]] * (width - len(r)) for r in rows])  # padded with a repeat
        out = self._forward(ids, None, Rows.batch(lengths, groups), keep)
        return [out[i, : len(r)] for i, r in enumerate(rows)]


def finish_layer(layer: dict[str, Any], linears: Sequence[str], bits: int | None) -> dict[str, Any]:
    """A layer's weights ready to run: its `linears` quantized to `bits` (None: as stored), and all of it evaluated
    now, one layer at a time, so the lazy casts of a whole model never pile up in memory."""
    for key in linears:
        if key in layer:
            layer[key] = quantize(layer[key], bits)
    mx.eval(arrays(list(layer.values())))
    return layer


def output_head(weights: Weights, root: str, tied: bool, dtype: mx.Dtype, *names: str) -> Table | None:
    """The language-model head stored apart from the embedding (`lm_head.weight`, at the top or under the text root,
    or under `names`), None when the model ties it to the embedding or stores none (a sequence classifier's)."""
    if tied:
        return None
    found = next((k for k in (HEAD, root + HEAD, *names) if k in weights), None)
    return weights.table(found, dtype) if found else None


def stacked_experts(
    weights: Weights, prefix: str, dtype: mx.Dtype, experts: int, mlx_lm: str
) -> tuple[Stacked | JoinedStacked, Stacked]:
    """A mixture of experts' [gate; up] and down, every expert's stacked: from mlx-lm's conversion (`mlx_lm`, its
    switch module's name), transformers 5's fused [experts, rows, columns] tensors, or one tensor per expert."""

    def m(name: str) -> Matrix:
        return weights.matrix(prefix + name, dtype)

    if prefix + mlx_lm + ".gate_proj.weight" in weights:
        gate, up, down = (Stacked.stored(m(f"{mlx_lm}.{x}_proj.weight")) for x in ("gate", "up", "down"))
        gate_up = Stacked.join([gate, up])
    elif prefix + "experts.gate_up_proj" in weights:
        gate_up, down = Stacked.stored(m("experts.gate_up_proj")), Stacked.stored(m("experts.down_proj"))
    else:
        gate_up = Stacked.of(
            [concat([m(f"experts.{e}.gate_proj.weight"), m(f"experts.{e}.up_proj.weight")]) for e in range(experts)]
        )
        down = Stacked.of([m(f"experts.{e}.down_proj.weight") for e in range(experts)])
    if gate_up.shape[0] != experts:
        raise UnsupportedModel(f"{prefix}: {gate_up.shape[0]} experts stored, the config says {experts}")
    return gate_up, down


class AttentionDecoder(Passes):
    """A causal decoder whose every layer is attention (the dense ones, Gemma 3 and 4): a cache of keys and values per
    layer, and the language-model head, lm_head or the tied embedding. Its config has hidden, kv_heads, head_dim,
    layers and tied; a family whose layers differ in their key and value shapes gives its own `_blocks`."""

    dtype: mx.Dtype
    embed: Table
    lm_head: Table | None = None
    attention_share = 1.0  # every layer is attention

    @property
    def hidden_size(self) -> int:
        return int(self.config.hidden)

    @property
    def output_matrix(self) -> Table:
        if self.lm_head is not None:
            return self.lm_head
        if self.config.tied:
            return self.embed
        raise UnsupportedModel("this checkpoint has no language-model head (lm_head.weight) and doesn't tie it")

    def _blocks(self, capacity: int) -> list[Attention | Recurrent]:
        c = self.config
        return [Attention.empty(c.kv_heads, capacity, c.head_dim, self.dtype) for _ in range(c.layers)]

    def new_cache(self, capacity: int, start: tuple[Any, Any] | None = None) -> HybridCache:
        return HybridCache.of(self._blocks(capacity), start)


Held = tuple[mx.array | None, mx.array | None]  # a recurrent layer's (state, window), as `Recurrent` keeps them


class Hybrid(Passes):
    """A decoder whose "full_attention" layers attend and whose other layers carry a recurrent state (LFM2's short
    convolutions, Qwen3.5's Gated DeltaNet): their cache, and the order a recurrent layer runs a pass's rows in."""

    dtype: mx.Dtype

    @property
    def hidden_size(self) -> int:
        return int(self.config.hidden)

    @property
    def attention_share(self) -> float:
        """The share of layers that are attention (the rest are recurrent): how much a row's attention costs."""
        kinds: Sequence[str] = self.config.layer_types
        return sum(t == "full_attention" for t in kinds) / len(kinds)

    def new_cache(self, capacity: int, start: tuple[Any, Any] | None = None) -> HybridCache:
        c = self.config
        blocks = [
            Attention.empty(c.kv_heads, capacity, c.head_dim, self.dtype) if kind == "full_attention" else Recurrent()
            for kind in c.layer_types
        ]
        return HybridCache.of(blocks, start)

    def _recurrent(
        self,
        proj: mx.array,
        block: Recurrent | None,
        r: Rows,
        snapshots: list[Snapshots],
        step: Callable[[mx.array, Held, Sequence[int]], tuple[mx.array, Held, Snapshots]],
    ) -> mx.array:
        """A recurrent layer's output over the pass's rows, `step` running it over rows that continue a held state
        (keeping it at the given marks): a prefill continues the cache. A branch pass's head continues the cache,
        then its tails run as one batch from where the head ends. The tails are right-padded and their final state
        is not kept, so their padding reaches no row that is read: they run unmasked (MLX's masked recurrence is a
        slower kernel, by 30 to 100 times)."""
        held: Held = (block.state, block.window) if block is not None else (None, None)
        if not r.is_branch:
            y, held, kept = step(proj, held, r.marks)
            if block is not None:
                block.state, block.window = held
            snapshots.append(kept)
            return y
        parts, kept = [], {}
        if r.head:
            assert block is not None
            y_head, held, kept = step(proj[:, : r.head], held, r.marks)
            block.state, block.window = held
            parts.append(y_head)
        snapshots.append(kept)
        y_tails, _, _ = step(r.to_tails(proj), held, ())
        return mx.concatenate([*parts, r.from_tails(y_tails)], axis=1)


class Encoder(Passes):
    """The passes of a bidirectional encoder: every prompt read whole, in one pass (each token reads the ones after
    it), alone or with others in a padded batch (`Passes.batch`); nothing is kept between prompts, so its cache is
    empty and it has no branch pass. Its `_forward(ids, cache, rows, keep)` never gets a cache."""

    attention_share = 1.0  # every layer is attention

    @property
    def causal(self) -> bool:
        return False

    def new_cache(self, capacity: int, start: tuple[Any, Any] | None = None) -> HybridCache:
        if start is not None:
            raise ValueError("an encoder keeps nothing between prompts")
        return HybridCache([])  # nothing to keep: every prompt is read whole

    def prefill(self, tokens: Sequence[int], cache: Any, rows: Sequence[int], marks: Sequence[int] = ()) -> mx.array:
        assert isinstance(cache, HybridCache)
        if cache.length:
            raise ValueError("an encoder reads a prompt whole, in one pass")
        out = self._forward(mx.array([list(tokens)]), None, Rows.prefill(len(tokens), 0), Read.of([rows]))
        cache.length += len(tokens)
        return out[0]

    def branch(
        self,
        tails: Sequence[Sequence[int]],
        cache: Any,
        rows: Sequence[Sequence[int]],
        head: Sequence[int] = (),
        marks: Sequence[int] = (),
    ) -> list[mx.array]:
        raise ValueError("an encoder reads every prompt whole")


def window_before(window: mx.array | None, x: mx.array, k1: int, at: int) -> mx.array:
    """The `k1` inputs before position `at` of x's rows ([sequences, length, channels]), those before x coming from
    `window` ([1 or sequences, k1, channels]; zeros when None): a short convolution's state at `at`. A compact copy:
    a slice would keep all of x (a layer's whole input projection) alive for as long as the state is kept."""
    if at >= k1:
        return mx.contiguous(x[:, at - k1 : at])
    if window is None:
        window = mx.zeros((1, k1, x.shape[2]), x.dtype)
    head = mx.broadcast_to(window.astype(x.dtype), (x.shape[0], k1, x.shape[2]))[:, at:]
    return mx.concatenate([head, x[:, :at]], axis=1)
