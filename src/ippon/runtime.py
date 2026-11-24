"""Running a plan on a backbone.

The strategy follows from the plan and the backbone; nobody chooses it:

- **one prompt**: read it causally in chunks, keeping the readout rows of the last chunk only;
- **several prompts, causal backbone ("fork")**: read their shared prefix in chunks but the last, then that last
  chunk and the prompts' own tails in one pass (`Backbone.branch`): the state is read once, and a request whose
  prefix fits one chunk reads the weights once for everything. A pass takes at most a chunk of the tails' rows
  (`branch_groups`): its attention mask grows with its rows squared, so more questions than that are read in
  further passes, each continuing from the prefix's end. When the prompts share little (a template that puts
  the question before the state) and each fits a chunk, a padded batch of whole prompts costs less: the branch
  pass's attention spans every tail's rows (its mask hides the other tails but computes them), and `shares`
  chooses by the work each would do;
- **non-causal backbone**: every prompt is read whole, since each token reads the ones after it, and several are
  read together in batched passes of up to DEFAULT_BATCH_TOKENS padded tokens (`Backbone.batch`), with their
  tokens' groups (set prompts); a lone prompt is a batch of one, which fills no cache (nothing could reuse it) and
  pads nothing;
- **block backbone** (`Backbone.block`): the prefix the prompts mark, once and in one pass (it reads itself both
  ways), then the tails as branches from it, each reading itself both ways (`blocks`);
- **"reference"**: every prompt whole, one after another, each from an empty cache (a block model's: its prefix,
  then its tail; a non-causal one's, a batch of one). It is the simple path the others are tested against, and
  what `verify` can use to separate a format's errors from the runtime's.

Every pass reads all of the model's weights (about 4 GB for a 2B model in bf16), which for a short request costs
as much as its tokens, so a request takes one pass per chunk and no more: the positions kept for `prefixes.py`
split the recurrence inside a pass, not the pass.

With a `Prefixes` store, a causal prefix starts from the longest kept prefix it begins with and is kept in turn
(`prefixes.py`); the reference path never uses one.

Several requests at once (`read_requests`) are read as few plans as they can be: requests whose prompts start alike
are read as one plan, and single prompts about different states that each fit a chunk in one padded batch, which reads
the weights once between them. Each request is planned once.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextvars import ContextVar

import mlx.core as mx

from .arch.base import Backbone, Cache
from .errors import DecisionError, out_of_memory
from .options import Strategy
from .plan import DEFAULT_CHUNK, Plan, Prompt, Tail, chunks
from .plan import plan as plan_of
from .prefixes import Prefixes

DEFAULT_BATCH_TOKENS = 16384  # padded tokens per batched pass (prompts x longest prompt)
# What one more pass costs, in tokens of a pass's work: reading the weights again (a 2B model's 4 GB in bf16), about
# what 50 to 200 tokens cost on M1 to M5 (their bandwidth against their matmul rate). Padding a prompt by more than
# this to save a pass costs more than the pass: d1-2B's 50 questions are 4 set prompts (16 questions a prompt) of
# about 2,600, 2,600, 2,600 and 1,200 tokens, and one batch padded the last by 1,400 tokens, 15% of the request: ippon
# took 2.95 s against Ollama's 2.65 s on an M4 Pro (0.28 ms a token, Ollama's rate over the real tokens). Batched by
# cost, the request takes 4.7% less time on an M5 Pro (619.7 to 590.7 ms, 1,000 words, 4 rounds), one question as
# before.
PASS_TOKENS = 128


# Asked between passes on the model thread: the error to stop this reading with, if any. Everyone waiting for it has
# gone (each request past its deadline, or its caller hung up), so that the next request isn't kept waiting behind an
# answer nobody reads; or the Mac is critically short of memory, which a long reading would take further, and which
# the server otherwise relieves only between batches (serve.Batcher sets it)
STOP: ContextVar[Callable[[], DecisionError | None] | None] = ContextVar("stop", default=None)


def go_on() -> None:
    """The error STOP gives, raised before the next pass, if it gives one."""
    asked = STOP.get()
    if asked is not None and (error := asked()) is not None:
        raise error


# Told the end of each step of an answer, by name (profile.PHASES), while `ippon profile --phases` times them: the
# decider's own answer path, with a hook. Unset, nothing waits.
PHASE: ContextVar[Callable[[str], None] | None] = ContextVar("phase", default=None)


def phase(name: str, wait: Sequence[mx.array] = ()) -> None:
    """The step `name` done, told to PHASE when it is set, once `wait` is evaluated (its GPU work, timed apart)."""
    told = PHASE.get()
    if told is not None:
        if wait:
            mx.eval(list(wait))
        told(name)


def read(
    backbone: Backbone,
    tokens: Sequence[int],
    cache: Cache,
    rows: Sequence[int],
    chunk: int = DEFAULT_CHUNK,
    marks: Sequence[int] | None = None,
) -> mx.array:
    """Read `tokens` into `cache` `chunk` tokens at a time; the hidden states of `rows` (indices into `tokens`, in
    that order) as [len(rows), hidden]. With `marks` (indices into `tokens`), the recurrent state is kept in the
    cache at each of them and at every chunk boundary, for `prefixes.py`."""
    if any(not 0 <= r < len(tokens) for r in rows):
        raise ValueError("a row is outside the tokens")
    if not backbone.causal:
        chunk = max(len(tokens), 1)  # a non-causal prompt is one pass: every token reads all of it
    order: list[int] = []  # the rows in the order the chunks give them
    pieces = []
    for start, end in chunks(len(tokens), chunk):
        go_on()
        inside = [r for r in rows if start <= r < end]
        within = [m - start for m in marks or () if start < m < end]
        hidden = backbone.prefill(tokens[start:end], cache, [r - start for r in inside], within)
        if marks is not None and end < len(tokens):
            cache.remember()
        if inside:
            order += inside
            pieces.append(hidden)
    if not rows:
        return mx.zeros((0, backbone.hidden_size))
    every = pieces[0] if len(pieces) == 1 else mx.concatenate(pieces)
    if order == list(rows):  # rows in order (a head reading every position): the chunks' rows as they are
        return every
    where = {r: i for i, r in enumerate(order)}
    return mx.take(every, mx.array([where[r] for r in rows]), axis=0)


def batched(
    backbone: Backbone,
    prompts: Sequence[Sequence[int]],
    rows: Sequence[Sequence[int]],
    budget: int = DEFAULT_BATCH_TOKENS,
    groups: Sequence[Sequence[int]] | None = None,
) -> list[mx.array]:
    """Whole prompts in batches of at most `budget` padded tokens, the hidden states in the prompts' order. A batch
    pads every prompt to its longest, so the prompts go into batches by length, shortest first: prompts of mixed
    lengths (requests about different states) pad far less than in the order they came. `groups`: set prompts'
    (`plan.Prompt.groups`)."""
    order = sorted(range(len(prompts)), key=lambda j: len(prompts[j]))
    out: list[mx.array | None] = [None] * len(prompts)
    waiting = [[order[g] for g in group] for group in pass_groups([len(prompts[j]) for j in order], budget)]
    while waiting:
        go_on()
        picked = waiting.pop(0)
        grouped = None if groups is None else [groups[j] for j in picked]
        try:
            hidden = backbone.batch([prompts[j] for j in picked], [rows[j] for j in picked], grouped)
            mx.eval(hidden)
        except (RuntimeError, MemoryError) as error:
            # A pass the GPU's memory can't hold (several long prompts on a small Mac: Torchcast-4B's three
            # 4,335-token prompts peak at 13.2 GB, past a 16 GB M2 Pro's 12 GB working set) is read in halves.
            if len(picked) == 1 or not out_of_memory(error):
                raise
            mx.clear_cache()
            waiting[:0] = [picked[: len(picked) // 2], picked[len(picked) // 2 :]]
            continue
        for j, h in zip(picked, hidden, strict=True):
            out[j] = h
    return [h for h in out if h is not None]


def pass_groups(lengths: Sequence[int], budget: int, pass_tokens: int = PASS_TOKENS) -> list[list[int]]:
    """Consecutive prompts (their indices; `batched` sorts them by length) in the batches that cost least: a batch
    costs its padded tokens (its prompts times its longest) and `pass_tokens` more, and holds at most `budget` padded
    tokens; a prompt longer than that alone. Equal prompts fill a batch to the budget; a much longer one starts its
    own."""
    n = len(lengths)
    best = [0] + [-1] * n  # the least cost of the first i prompts, and where its last batch starts
    start = [0] * (n + 1)
    for i in range(1, n + 1):
        longest = 0
        for j in range(i - 1, -1, -1):
            longest = max(longest, lengths[j])
            padded = longest * (i - j)
            if j < i - 1 and padded > budget:
                break
            cost = best[j] + padded + pass_tokens
            if best[i] < 0 or cost < best[i]:
                best[i], start[i] = cost, j
    out: list[list[int]] = []
    i = n
    while i:
        out.append(list(range(start[i], i)))
        i = start[i]
    return out[::-1]


def branch_groups(lengths: Sequence[int], budget: int) -> list[list[int]]:
    """Consecutive tails (their indices) in groups of at most `budget` rows; a tail longer than that is a group of
    its own."""
    groups: list[list[int]] = []
    rows = 0
    for i, n in enumerate(lengths):
        if groups and rows + n <= budget:
            groups[-1].append(i)
            rows += n
        else:
            groups.append([i])
            rows = n
    return groups


def run(
    backbone: Backbone,
    plan: Plan,
    chunk: int = DEFAULT_CHUNK,
    strategy: Strategy = "auto",
    prefixes: Prefixes | None = None,
) -> list[mx.array]:
    """The hidden states of every tail's readouts, [readouts, hidden] per tail, in the plan's order, by the strategy
    the module docstring describes."""
    go_on()
    if backbone.block:
        return blocks(backbone, plan, chunk, None if strategy == "reference" else prefixes, strategy == "reference")
    if not backbone.causal or (strategy != "reference" and whole_in_batches(plan, backbone, chunk, prefixes)):
        prompts = [plan.prefix + t.tokens for t in plan.tails]
        rows = [[len(plan.prefix) + r.position for r in t.readouts] for t in plan.tails]
        groups = [t.groups for t in plan.tails] if any(t.groups for t in plan.tails) else None
        return batched(backbone, prompts, rows, 0 if strategy == "reference" else DEFAULT_BATCH_TOKENS, groups)
    if strategy == "reference" or (plan.prefix == () and len(plan.tails) > 1):
        return [each_whole(backbone, plan.prefix + t.tokens, len(plan.prefix), t, chunk) for t in plan.tails]
    return from_prefix(backbone, plan, chunk, prefixes)


def blocks(
    backbone: Backbone, plan: Plan, chunk: int, prefixes: Prefixes | None, apart: bool = False
) -> list[mx.array]:
    """A block model's plan: its prefix in one pass (from a kept prefix only when one is all of it), kept whole, then
    its tails as branches from it, at most a chunk of their rows a pass; `apart`, each tail from a prefix read again
    (the reference)."""
    tails, prefix = plan.tails, plan.prefix
    groups = [[i] for i in range(len(tails))] if apart else branch_groups([len(t.tokens) for t in tails], chunk)
    capacity = len(prefix) + max(sum(len(tails[i].tokens) for i in g) for g in groups)
    cache: Cache | None = None
    hidden: list[mx.array] = []
    for group in groups:
        go_on()
        if cache is None or apart:
            if prefixes is not None:
                cache, _, _ = prefixes.start(backbone, prefix, capacity, whole=True)
            else:
                cache = backbone.new_cache(capacity)
            if cache.length < len(prefix):
                backbone.prefill(prefix, cache, ())
                if prefixes is not None and prefix:
                    prefixes.keep(prefix, cache, {len(prefix): cache.checkpoint(len(prefix))})
        hidden += backbone.branch(
            [tails[i].tokens for i in group], cache, [[r.position for r in tails[i].readouts] for i in group]
        )
    return hidden


def whole_in_batches(plan: Plan, backbone: Backbone, chunk: int, prefixes: Prefixes | None) -> bool:
    """Whether a causal backbone reads the plan's prompts whole in padded batches: several tails that each fit a
    chunk whole, when reading them so costs less than branching from their prefix (`shares`), counting what a kept
    prefix already holds."""
    if len(plan.tails) < 2 or needs_prefix_rows(plan):
        return False
    done = prefixes.resumable(plan.prefix) if prefixes is not None else 0
    longest = len(plan.prefix) + max(len(t.tokens) for t in plan.tails)
    return longest <= chunk and not shares(plan, backbone, done)


def each_whole(backbone: Backbone, tokens: Sequence[int], start: int, tail: Tail, chunk: int) -> mx.array:
    """One prompt read whole from an empty cache, its tail's readout rows (from `start`, the prefix's length)."""
    return read(backbone, tokens, backbone.new_cache(len(tokens)), [start + r.position for r in tail.readouts], chunk)


def from_prefix(backbone: Backbone, plan: Plan, chunk: int, prefixes: Prefixes | None) -> list[mx.array]:
    """A causal plan read from its prefix: the prefix once (from the longest kept prefix it begins with), then its
    lone tail in the same pass, or its tails as branches; the prefix kept in turn, at its marks and its end."""
    tails, prefix = plan.tails, plan.prefix
    groups = branch_groups([len(t.tokens) for t in tails], chunk)
    capacity = len(prefix) + max(
        sum(len(tails[i].tokens) for i in g) for g in groups
    )  # a group's tails, then the next's
    inside = needs_prefix_rows(plan)  # a lone prompt reading positions inside its prefix (an every-row head)
    kept_rows: mx.array | None = None
    if prefixes is not None:
        cache, inherited, kept_rows = prefixes.start(backbone, prefix, capacity, rows=inside)
    else:
        cache, inherited = backbone.new_cache(capacity), {}
    done = cache.length
    rest = prefix[done:]
    # positions to keep the prefix at, relative to `rest`: the marks still ahead and the prefix's end
    ahead = sorted({m - done for m in plan.marks if m > done} | {len(rest)}) if prefixes is not None else None
    prefix_rows: mx.array | None = None
    if len(tails) == 1:  # the rest of the prefix and the question in one pass (per chunk)
        tail = tails[0]
        tokens = rest + tail.tokens
        wanted = [len(rest) + r.position for r in tail.readouts]  # into `tokens`; negative: before `done`, kept
        fresh = sorted({w for w in wanted if w >= 0})
        got = read(backbone, tokens, cache, fresh, chunk, ahead)
        if inside:
            assert kept_rows is not None or done == 0
            every = got if kept_rows is None else mx.concatenate([kept_rows, got])  # from position 0, if all read
            index = {w: i for i, w in enumerate(fresh)}
            picks = [done + w if w < 0 else done + index[w] for w in wanted]
            hidden = [every if picks == list(range(every.shape[0])) else mx.take(every, mx.array(picks), axis=0)]
            if fresh == list(range(len(fresh))) and len(fresh) >= len(rest):  # every row of the prefix was read
                prefix_rows = mx.contiguous(every[: len(prefix)])  # a copy: a slice would keep every row alive
        else:
            hidden = [got if wanted == fresh else mx.take(got, mx.array([fresh.index(w) for w in wanted]), axis=0)]
    else:  # the rest of the prefix but its last chunk, then that chunk and every question in one pass
        bounds = chunks(len(rest), chunk)
        head = bounds[-1][1] - bounds[-1][0] if bounds else 0
        earlier = len(rest) - head
        if earlier:
            read(backbone, rest[:earlier], cache, (), chunk, [] if ahead is None else [m for m in ahead if m < earlier])
            if ahead is not None:
                cache.remember()
        within = [] if ahead is None else [m - earlier for m in ahead if earlier < m < len(rest)]
        hidden = []
        for k, group in enumerate(groups):  # the first pass reads the head; a branch leaves the cache at its end
            go_on()
            hidden += backbone.branch(
                [tails[i].tokens for i in group],
                cache,
                [[r.position for r in tails[i].readouts] for i in group],
                rest[earlier:] if k == 0 else (),
                within if k == 0 else (),
            )
    if prefixes is not None and prefix:
        marks = dict(inherited)
        for at in sorted({done + m for m in ahead or () if m} | {p for p in cache.kept if done < p <= len(prefix)}):
            marks[at] = cache.checkpoint(at)
        # a later request is likely to share the prefix up to its last mark (the state's end), not past it
        prefixes.keep(prefix, cache, marks, prefix_rows, list(inherited), max(plan.marks, default=None))
    return hidden


def batchable(
    plan: Plan,
    backbone: Backbone,
    chunk: int = DEFAULT_CHUNK,
    strategy: Strategy = "auto",
    prefixes: Prefixes | None = None,
) -> bool:
    """True when a request may be read in a padded batch with other requests' (`batched`): a causal backbone, one
    prompt that fits one chunk (a batch reads it whole), and no kept prefix to resume from (a batch reads from
    position 0 and keeps nothing). Several such requests about different states then read the weights once
    between them."""
    if strategy == "reference" or not backbone.causal or len(plan.tails) != 1:
        return False
    if len(plan.prefix) + len(plan.tails[0].tokens) > chunk:
        return False
    return prefixes is None or not prefixes.resumable(plan.prefix, needs_prefix_rows(plan))


def shares(plan: Plan, backbone: Backbone, done: int = 0) -> bool:
    """True when reading the prompts' shared prefix once (all but the `done` tokens a kept prefix gives) and their
    tails in one branch pass costs less than reading every prompt whole in a padded batch. The work of a row is its
    matmuls (about 24 hidden^2 multiply-adds a layer) plus its attention over `keys` (4 hidden keys a layer, in the
    attention layers' share of them): rows x (1 + share x keys / 6 hidden). A branch row attends to the cache and
    every row of the pass (the mask hides other tails but they are computed); a batched one, causally, to half of
    its prompt on average."""
    hidden = backbone.hidden_size
    share = backbone.attention_share
    tails = [len(t.tokens) for t in plan.tails]
    rows = len(plan.prefix) - done + sum(tails)
    branch = rows * (1 + share * (done + rows) / (6 * hidden))
    longest = len(plan.prefix) + max(tails)
    batch = len(tails) * longest * (1 + share * longest / (12 * hidden))
    return branch <= batch


def needs_prefix_rows(plan: Plan) -> bool:
    """True when a readout lies inside the plan's prefix (negative tail positions): only a lone prompt's."""
    return any(r.position < 0 for t in plan.tails for r in t.readouts)


def read_requests(
    backbone: Backbone,
    requests: Sequence[Sequence[Prompt]],
    chunk: int = DEFAULT_CHUNK,
    strategy: Strategy = "auto",
    prefixes: Prefixes | None = None,
) -> Iterator[tuple[int, list[mx.array], int]]:
    """(index, the hidden states of each of `requests[index]`'s prompts' readouts, [readouts, hidden] per prompt, the
    tokens its reading took from a kept prefix), for every request, read as the module docstring says: alike requests
    together (`together`), single prompts about different states in one padded batch. A request is yielded as soon as
    it is read, so its caller can finish it before the next plan is read."""
    planned: dict[int, Plan] = {}

    def alone(i: int) -> Plan:
        if i not in planned:
            planned[i] = plan_of(requests[i], block=backbone.block)
        return planned[i]

    groups = together(requests, backbone, strategy, alone)
    lone: list[int] = []  # none to look for in a request alone, which would only be planned for nothing
    if len(requests) > 1:
        lone = [
            g[0]
            for g in groups
            if len(g) == 1 and len(requests[g[0]]) == 1 and batchable(alone(g[0]), backbone, chunk, strategy, prefixes)
        ]
    if len(lone) > 1:  # single prompts about different states: one batched pass reads the weights once
        prompts = [requests[i][0] for i in lone]
        hidden = batched(backbone, [p.tokens for p in prompts], [[r.position for r in p.readouts] for p in prompts])
        yield from ((i, [h], 0) for i, h in zip(lone, hidden, strict=True))  # a batch reads from position 0
        groups = [g for g in groups if not (len(g) == 1 and g[0] in lone)]
    for group in groups:
        if len(group) == 1:
            planned_group = alone(group[0])
        else:
            planned_group = plan_of([prompt for i in group for prompt in requests[i]], block=backbone.block)
        phase("plan")
        before = prefixes.stats.tokens_reused if prefixes is not None else 0
        hidden = run(backbone, planned_group, chunk, strategy, prefixes)
        kept = prefixes.stats.tokens_reused - before if prefixes is not None else 0
        phase("graph")
        phase("gpu", hidden)
        at = 0
        for i in group:
            n = len(requests[i])
            yield i, hidden[at : at + n], kept
            at += n


def together(
    requests: Sequence[Sequence[Prompt]], backbone: Backbone, strategy: Strategy, alone: Callable[[int], Plan]
) -> list[list[int]]:
    """Which requests to read as one plan: on a causal or block backbone, those whose prompts start the same up to the
    first mark (the end of the state, or of the block), or that share the whole prefix (`alone(i)`, request i's own
    plan) when unmarked; on a non-causal one, all of them (in batches); with the reference strategy, none."""
    if strategy == "reference":
        return [[i] for i in range(len(requests))]
    if not backbone.causal and not backbone.block:
        return [list(range(len(requests)))] if requests else []
    groups: dict[tuple[int, ...], list[int]] = {}
    for i, prompts in enumerate(requests):
        first = prompts[0]
        start = min(first.marks) if first.marks else len(alone(i).prefix)
        groups.setdefault(first.tokens[:start], []).append(i)
    return list(groups.values())
