"""How a request's prompts run: their shared prefix once, then each prompt's own tail.

A decision format renders a request as one or more prompts: one per question for most models, a single
prompt for every question for models that ask them jointly. Prompts of one request usually share a long
prefix (the template and the state); the questions differ only at the end. That prefix runs through the
model once. Each prompt's tail then runs from a copy of the model's state at the end of the prefix, which
for attention is the key/value cache and for recurrent layers (short convolutions, Gated DeltaNet) is a
fixed-size state, so the copy is cheap either way.

A readout is a position whose output scores a question's options. Every readout falls in a tail, so the
prefix pass keeps no outputs and the tail pass keeps only the readout rows.

A mark is a position where the prompt moves from what later requests are likely to send again (the template and
the state) to what they change (the questions); the prefix is kept there across requests (`prefixes.py`).

A prompt that packs several questions after its state for a set model (Liquid's d1-2B) gives each token its group:
-1 on the state, then one group per question. Its state reads every question, so such prompts share no prefix: each
runs whole.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

DEFAULT_CHUNK = 4096  # tokens a pass reads at most of a long prompt (`chunks`)


class Readout(NamedTuple):
    """The output at `position` (an index into its prompt's tokens) scores question `question`. A named tuple:
    a head that reads every position makes one per token."""

    question: str
    position: int


@dataclass(frozen=True)
class Prompt:
    tokens: tuple[int, ...]
    readouts: tuple[Readout, ...]
    marks: tuple[int, ...] = ()  # positions to keep the prefix at, such as the end of the state
    groups: tuple[int, ...] = ()  # each token's group for a set model (-1: the state); none for others

    def __post_init__(self) -> None:
        if not self.tokens:
            raise ValueError("a prompt has at least one token")
        if not self.readouts:
            raise ValueError("a prompt has at least one readout")
        if any(not 0 <= r.position < len(self.tokens) for r in self.readouts):
            raise ValueError("a readout position is outside its prompt")
        if any(not 0 < m <= len(self.tokens) for m in self.marks):
            raise ValueError("a mark is outside its prompt")
        if self.groups and len(self.groups) != len(self.tokens):
            raise ValueError("a set prompt gives every token a group")


@dataclass(frozen=True)
class Tail:
    """A prompt's tokens after the shared prefix; readout positions count from the tail's first token, and are
    negative for rows inside the prefix (a head that reads every position, in a plan whose prefix ends at a mark)."""

    tokens: tuple[int, ...]
    readouts: tuple[Readout, ...]
    groups: tuple[int, ...] = ()  # the prompt's (`Prompt.groups`), when it runs whole


@dataclass(frozen=True)
class Plan:
    prefix: tuple[int, ...]
    tails: tuple[Tail, ...]
    marks: tuple[int, ...] = ()  # the prompts' marks within the prefix, in order


CLOSERS = "]}) \t\r\n"
BEFORE_END = 2  # token boundaries between a state's mark and the last one before its end


def state_end_marks(text: str, offsets: Sequence[tuple[int, int]], end: int | None = None) -> tuple[int, ...]:
    """The mark where a state that grows at its end parts from its next version: the token boundary BEFORE_END
    boundaries before the last one before the end of the state's own text (character `end` of the prompt `text`;
    its end by default) without its closing brackets and whitespace, from the tokens' character `offsets`; none
    when that is the start.

    An agent re-sends its state with something appended. The prompt around the state (a closing tag, the
    question) then follows the old state's text no longer, a JSON state's brackets close later, and a tokenizer
    may merge the old text's last characters with what now follows them, so the next prompt shares this one's
    tokens up to the last boundary or one of the two before it: a request resumes from this mark and reads the
    few tokens after it again. One mark, not one at each of those boundaries: every mark splits each recurrent
    layer's pass and keeps a copy of its state (18 MB for a 2B hybrid), which cost a fresh 20-word request 3 to
    4% (a 2B Qwen3.5 hybrid, M5 Pro)."""
    i = len(text) if end is None else end
    while i > 0 and text[i - 1] in CLOSERS:
        i -= 1
    n = sum(1 for _, stop in offsets if stop <= i) - BEFORE_END
    return (n,) if n > 0 else ()


def one_mark(ends: Sequence[int], fallback: int) -> tuple[int, ...]:
    """A prompt's marks: the state-end mark (`ends`, from `state_end_marks`) when it lies within the prompt's state,
    else `fallback`, the end of the part of the state the prompt holds (a state cut short, or no state-end mark)."""
    inside = [m for m in ends if 0 < m <= fallback]
    if inside:
        return (inside[0],)
    return (fallback,) if fallback > 0 else ()


def shared_prefix(prompts: Sequence[Prompt]) -> int:
    """How many leading tokens every prompt shares, limited so that every readout and at least one token
    of every prompt remain in its tail."""
    if not prompts:
        raise ValueError("no prompts")
    first = prompts[0].tokens
    n = min(len(p.tokens) for p in prompts) - 1
    n = min(n, *(r.position for p in prompts for r in p.readouts))
    for p in prompts[1:]:
        i = 0
        while i < n and p.tokens[i] == first[i]:
            i += 1
        n = i
    return max(n, 0)


def plan(prompts: Sequence[Prompt], *, share: bool = True, block: bool = False) -> Plan:
    """The prefix and tails for `prompts`; with share=False, every prompt runs whole (the reference the
    shared plan must match). A lone prompt's prefix reaches its last mark even past readouts, which then count back
    from the tail (negative): a head that reads every position can still keep the state's reading for the next
    request, with its rows (`prefixes.py`). For a block model (`block`, Backbone.block) the prefix is the block every
    prompt marks, whatever more they share: its reading is the same for each prompt only up to there. Set prompts
    (with groups) run whole."""
    grouped = any(p.groups for p in prompts)
    n = 0 if grouped else block_prefix(prompts) if block else shared_prefix(prompts) if share else 0
    if share and not block and not grouped and len(prompts) == 1:
        n = max(n, max((m for m in prompts[0].marks if m < len(prompts[0].tokens)), default=0))
    return Plan(
        prefix=prompts[0].tokens[:n],
        tails=tuple(
            Tail(p.tokens[n:], p.readouts if n == 0 else tuple(Readout(q, at - n) for q, at in p.readouts), p.groups)
            for p in prompts
        ),
        marks=tuple(sorted({m for p in prompts for m in p.marks if m <= n})),
    )


def block_prefix(prompts: Sequence[Prompt]) -> int:
    """The length of the block every prompt of a block model starts with: its one mark, the same tokens in each, and
    every readout after it."""
    if not prompts:
        raise ValueError("no prompts")
    first = prompts[0]
    if any(len(p.marks) != 1 for p in prompts) or len({p.marks[0] for p in prompts}) != 1:
        raise ValueError("a block model's prompts mark one boundary, the same in each")
    n = first.marks[0]
    if any(p.tokens[:n] != first.tokens[:n] for p in prompts):
        raise ValueError("a block model's prompts start with the same block")
    if any(r.position < n for p in prompts for r in p.readouts) or any(len(p.tokens) <= n for p in prompts):
        raise ValueError("a block model's readouts are after its block")
    return n


def chunks(length: int, size: int) -> list[tuple[int, int]]:
    """[start, end) bounds that read `length` tokens `size` at a time (a long prefix in pieces, so the
    activations of one piece fit in memory). A last piece shorter than a quarter of `size` joins the one before:
    every pass reads all the weights, which for a few tokens costs as much as their reading (4,156 tokens in chunks
    of 4,096 take one pass, not two), and no piece is longer than 1.25 * size."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    out = [(start, min(start + size, length)) for start in range(0, length, size)]
    if len(out) > 1 and out[-1][1] - out[-1][0] < size // 4:
        out[-2:] = [(out[-2][0], length)]
    return out
