"""Prefixes kept across requests, so that a state sent again is not read again.

Agents ask about the same state many times: a request per step, each with the state so far, or the same state
with other questions. After a causal backbone reads a request's shared prefix, the store keeps a copy of its keys
and values and the recurrent state at a few positions: each chunk boundary, each of the prompts' marks (where the
state ends) and the end of the prefix. A later request whose prefix starts with the same tokens up to one of those
positions copies the keys and values before it and reads on from its recurrent state.

Only those positions can be resumed from, because a recurrent layer's state exists only where it was saved; an
attention-only model could resume anywhere, a hybrid one cannot. Entries are dropped least recently used first
beyond a byte budget, and an entry that a newer one extends, with every position it can resume from, is dropped
for it. A non-causal backbone keeps nothing: its first tokens read the ones after them.

A head that reads every position (Clef's) needs the prefix's final hidden states too, so a request that read all
of them keeps them with the prefix (`rows`), and a request that needs rows inside the prefix resumes only from an
entry that has them.

"""

from __future__ import annotations

import sys
from collections import Counter, OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property

import mlx.core as mx
import numpy as np

from .arch.base import Backbone, Cache, Snapshot


@dataclass(eq=False)
class _Entry:
    tokens: np.ndarray  # the prefix read, int64
    keys: Snapshot  # its keys and values
    marks: dict[int, Snapshot]  # position -> the recurrent state there
    rows: mx.array | None = None  # the final hidden state of every prefix position, when a request read them all
    hashes: dict[int, int] = field(default_factory=dict)  # mark -> the hash of the tokens before it (`prefix_hashes`)

    @cached_property
    def nbytes(self) -> int:  # once: an entry never changes
        return (
            self.keys.nbytes
            + sum(s.nbytes for s in self.marks.values())
            + (self.rows.nbytes if self.rows is not None else 0)
        )

    def common(self, tokens: np.ndarray) -> int:
        """How many leading tokens this prefix and `tokens` share."""
        n = min(len(self.tokens), len(tokens))
        differ = np.flatnonzero(self.tokens[:n] != tokens[:n])
        return int(differ[0]) if differ.size else n


@dataclass
class Stats:
    hits: int = 0
    misses: int = 0
    tokens_reused: int = 0


_WEIGHTS = np.random.default_rng(0x5EED).integers(1, 2**63, size=1 << 16, dtype=np.uint64) | np.uint64(1)


def prefix_hashes(tokens: np.ndarray) -> np.ndarray:
    """[len + 1] uint64: a hash of every prefix of `tokens` (the sum of (token + 1) x a random odd weight per
    position, wrapping), in one cumulative sum; equal prefixes hash alike, and a hit is checked token by token."""
    global _WEIGHTS
    if len(tokens) > len(_WEIGHTS):
        more = np.random.default_rng(len(_WEIGHTS)).integers(1, 2**63, size=len(tokens), dtype=np.uint64)
        _WEIGHTS = np.concatenate([_WEIGHTS, more | np.uint64(1)])
    with np.errstate(over="ignore"):
        terms = (tokens.astype(np.uint64) + np.uint64(1)) * _WEIGHTS[: len(tokens)]
        return np.concatenate([np.zeros(1, np.uint64), np.cumsum(terms, dtype=np.uint64)])


def spaced(marks: dict[int, Snapshot], inherited: set[int], end: int) -> dict[int, Snapshot]:
    """The marks an entry keeps: its request's own, and of those it inherits from the entry it resumed from, every
    one holding no state and of those holding a recurrent state (tens of MB each on a hybrid model) the latest in each
    doubling distance back from `end` (1, 2, 4, ... tokens). An entry grown a request at a time otherwise holds a
    mark per request and outgrows the budget (H2O-Lightning-4B on an M5 Pro: 50 MB a mark, refused past 5k tokens
    under a server's share), where these stay dense near the end, where an appended or edited state resumes."""
    kept, seen = {}, set()
    for m in sorted(marks, reverse=True):
        band = (end - m).bit_length()
        if m not in inherited or not marks[m].nbytes or band not in seen:
            kept[m] = marks[m]
            if m in inherited and marks[m].nbytes:
                seen.add(band)
    return kept


class Prefixes:
    """The kept prefixes of one backbone, within `budget` bytes, least recently used first out. Each entry is found
    by its marks: (mark, the hash of the tokens before it) -> entries, so a lookup hashes the request's prefix once
    and checks only the positions some entry can resume from (scanning every entry was 0.6 ms a request with a few
    hundred small ones).

    With `grows` (a budget nobody asked for: ippon's default), the newest entry stays past the budget, by itself,
    whatever its size: an agent's state outgrows a small Mac's default within a session (27,000 tokens of
    H2O-Lightning-4B pass 1 GiB), and reading it again would build the same keys and values, so dropping it keeps
    no memory free and costs seconds a check (a session of 160 checks, M5 Pro at a 16 GB Mac's budget: 158 s, 38 s
    kept, the same peak footprint). A server takes the allowance back once requests stop (`serve.Models.rest`)."""

    def __init__(self, budget: int, grows: bool = False):
        if budget < 0:
            raise ValueError("the budget is a number of bytes")
        self.budget, self.grows = budget, grows
        self.default = budget  # the budget it was made with: a share of a server's memory never exceeds it
        self.alone = sys.maxsize if grows else 0  # one entry of at most this many bytes stays past the budget, alone
        # never more than this, whatever `alone` says: half the room the GPU has for readings (`fit`), so that a kept
        # reading leaves room to read another as long (three 130,000-token readings of Kettei Core kept in turn took a
        # 36 GB Mac to its memory warning)
        self.most_alone = sys.maxsize
        self.stats = Stats()
        self._lru: OrderedDict[int, _Entry] = OrderedDict()  # least recently used first
        self._index: dict[tuple[int, int], list[_Entry]] = {}
        self._positions: Counter[int] = Counter()  # every mark some entry has
        self._bytes = 0  # the entries' bytes, kept as they come and go (summing them was 2 ms a request at 40)

    @property
    def nbytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._lru)

    def _at(self, tokens: np.ndarray, hashes: np.ndarray, upto: int) -> list[tuple[int, _Entry]]:
        """(mark, entry) for every entry resumable from a mark at most `upto` that `tokens` begin with, the latest
        marks first."""
        found = []
        for m in sorted((p for p in self._positions if p <= upto), reverse=True):
            for entry in self._index.get((m, int(hashes[m])), ()):
                if np.array_equal(entry.tokens[:m], tokens[:m]):
                    found.append((m, entry))
        return found

    def _add(self, entry: _Entry) -> None:
        self._lru[id(entry)] = entry
        for m, h in entry.hashes.items():
            self._index.setdefault((m, h), []).append(entry)
            self._positions[m] += 1
        self._bytes += entry.nbytes

    def _drop(self, entry: _Entry) -> None:
        del self._lru[id(entry)]
        for m, h in entry.hashes.items():
            self._index[(m, h)].remove(entry)
            if not self._index[(m, h)]:
                del self._index[(m, h)]
            self._positions[m] -= 1
            if not self._positions[m]:
                del self._positions[m]
        self._bytes -= entry.nbytes

    def start(
        self, backbone: Backbone, prefix: Sequence[int], capacity: int, rows: bool = False, whole: bool = False
    ) -> tuple[Cache, dict[int, Snapshot], mx.array | None]:
        """A cache of `capacity` tokens for reading `prefix`, already holding as many of its first tokens as a kept
        prefix can give (its `length`), the recurrent states kept at or before that position, and with `rows` (the
        request reads positions inside the prefix) the kept rows of those first tokens, from an entry that has them.
        With `whole` (a block model's prefix, read both ways), all of it or none."""
        best, at = self._best(prefix, rows)
        if whole and at != len(prefix):
            best = None
        if best is None:
            self.stats.misses += 1
            return backbone.new_cache(capacity), {}, None
        self._lru.move_to_end(id(best))
        self.stats.hits += 1
        self.stats.tokens_reused += at
        cache = backbone.new_cache(capacity, (best.keys, best.marks[at]))
        kept = best.rows[:at] if rows and best.rows is not None else None
        return cache, {m: s for m, s in best.marks.items() if m <= at}, kept

    def resumable(self, prefix: Sequence[int], rows: bool = False) -> int:
        """How many of `prefix`'s first tokens a kept prefix could give, without using it."""
        return self._best(prefix, rows)[1]

    def _best(self, prefix: Sequence[int], rows: bool = False) -> tuple[_Entry | None, int]:
        """The entry the most of `prefix` resumes from (one with rows, with `rows`), and how many tokens it gives."""
        if not self._positions:
            return None, 0
        tokens = np.asarray(prefix, dtype=np.int64)
        hashes = prefix_hashes(tokens)
        best: tuple[_Entry | None, int] = (None, 0)
        for m, entry in self._at(tokens, hashes, len(tokens)):
            if not rows or entry.rows is not None:
                best = (entry, m)
                break
        return best

    def keep(
        self,
        prefix: Sequence[int],
        cache: Cache,
        marks: dict[int, Snapshot],
        rows: mx.array | None = None,
        inherited: Sequence[int] = (),
        reach: int | None = None,
    ) -> None:
        """Keep a prefix just read into `cache` (the cache may hold more after it), resumable at `marks` (those
        `inherited` from the entry it resumed from spaced out, `spaced`), with the final hidden state of each of its
        positions when the request read them all (`rows`, [len(prefix), hidden]).

        Nothing is kept when an entry already holds the prefix's first `reach` tokens (all of them by default) and
        every position this one has up to there. A request's prefix may run past its last mark into its questions,
        which the next request changes: that stretch made a new copy of the whole prefix each time, evicting the one
        it repeats (Kettei Core, a 26k-token state asked a new 42-token question: 279 ms a request, 188 ms without)."""
        if self.budget == 0 or not marks:
            return
        tokens = np.asarray(prefix, dtype=np.int64)
        hashes = prefix_hashes(tokens)
        reach = len(tokens) if reach is None else min(reach, len(tokens))
        near = self._at(tokens, hashes, len(tokens)) if self._positions else []
        for _, entry in near:  # already kept, with every position this one has up to `reach`
            covered = entry.common(tokens) >= reach and {m for m in marks if m <= reach} <= set(entry.marks)
            if covered and (rows is None or entry.rows is not None):
                self._lru.move_to_end(id(entry))
                return
        new = _Entry(tokens, cache.keys(len(tokens)), spaced(marks, set(inherited), len(tokens)), rows)
        new.hashes = {m: int(hashes[m]) for m in new.marks if 0 < m <= len(tokens)}
        if new.nbytes > max(self.budget, self.alone):
            return
        # evaluated (scheduled now, without waiting): an unevaluated state or row keeps the whole pass it came from
        # alive, uncounted by `nbytes`
        held = [a for m in new.marks.values() for a in m.arrays()]
        mx.async_eval(held + ([rows] if rows is not None else []))  # type: ignore[no-untyped-call]
        # an entry the new one extends, with every position it can resume from, goes: one with marks shares a mark
        extended = {id(e): e for _, e in near if new.common(e.tokens) == len(e.tokens)}
        for e in extended.values():
            if set(e.marks) <= set(marks) and (rows is not None or e.rows is None):
                self._drop(e)
        self._add(new)
        while self._bytes > self.budget and len(self._lru) > 1:
            self._drop(next(iter(self._lru.values())))

    def fit(self, room: int) -> None:
        """Hold a kept reading past the budget to half of `room`, the bytes the GPU has for readings beside the model's
        weights and this store's budget (`Decider`, whenever it makes a store), from now on."""
        self.most_alone = max(room // 2, 0)
        self.limit(self.budget, self.alone)

    def limit(self, budget: int, alone: int = 0) -> None:
        """Keep at most `budget` bytes from now on, the least recently used entries dropped now to fit (a server
        sharing its memory between models: `serve.Models`), except one entry of at most `alone` bytes kept by
        itself: a server short of memory keeps an agent's growing state, whose reading again would build the same
        keys and values (a 35,000-token state on H2O-Lightning-4B, 1.4 GB kept against a quarter share of 1 GiB:
        0.15 s a check kept, 9 s read again, M5 Pro)."""
        if budget < 0 or alone < 0:
            raise ValueError("the budget is a number of bytes")
        self.budget, self.alone = budget, min(alone, self.most_alone)
        alone = self.alone
        while self._lru and self._bytes > budget and not (len(self._lru) == 1 and self._bytes <= alone):
            self._drop(next(iter(self._lru.values())))

    def share(self, room: int | None, scale: float = 1.0, resting: bool = False) -> int:
        """Keep at most `room` bytes (None: the default budget, which a room never raises it past) times `scale`, the
        part a server short of memory lets it hold (`limit`); the budget set. While requests come, a growing state
        stays past it alone: at full scale with `grows`, and up to the room while memory is short (above 0). Once
        `resting` (no request for a while), nothing stays past it."""
        share = self.default if room is None else min(self.default, room)
        budget = int(share * scale)
        if resting:
            alone = 0
        elif scale < 1:
            alone = share if scale > 0 else 0
        else:
            alone = sys.maxsize if self.grows else 0
        self.limit(budget, alone)
        return budget

    def clear(self) -> None:
        """Drop every kept prefix and start the counts again."""
        self._lru.clear()
        self._index.clear()
        self._positions.clear()
        self._bytes = 0
        self.stats = Stats()
