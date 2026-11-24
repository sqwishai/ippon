"""H2O-Lightning's choice of a record's passages when a question's prompt is over its server's context, reimplemented
from selector_v2.py beside h2o_lightning_shim.py (h2oai/h2o-lightning-4b v1.4.1, sha256 bd518a2c...): the record
split into units of at most 400 tokens (a text by blank lines, then lines, then sentences; JSON by its keys, each
unit under its path), the units ranked by BM25 over the question's words (its options' at half weight) with a bonus
for exact codes, dates and numbers, then again with 30 terms of pseudo-relevance feedback from the top four; the
first and last unit kept, then the best units within 75% of the budget, their neighbours, the gaps between them up
to 40 tokens, and the best again to the budget; kept runs joined by "\\n[…]\\n", each under its heading.
`EvidenceSelectionUnsupported` when no unit matches or the record can't be split to fit (the server then keeps the
head and tail). On 300 generated records (text with headings, JSON, CJK, codes and dates; budgets of 30 to 4,000
tokens) it returned what the creator's file returns, character for character, with the checkpoint's tokenizer.
"""

from __future__ import annotations

import collections
import itertools
import json
import math
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any

UNIT_TOKENS = 400
GAP_TOKENS = 40
OPTION_WEIGHT = 0.5
PRF_TERMS = 30
PRF_WEIGHT = 0.3
PRF_UNITS = 4
EXACT_BONUS = 2.0
PACK_TOKENS = 150
SAMPLE_CHARS = 16000
PHASE_A = 0.75
SIZE_DIV = 6
MARK = "\n[…]\n"
_WORD = re.compile(r"[^\W_]+", re.U)
_CJK = re.compile("[぀-ヿ㐀-䶿一-鿿가-힯฀-๿豈-﫿]")
_EXACT = re.compile(r"\b(?:[A-Za-z]{1,6}-?\d[\w-]*|\d{4}-\d{2}-\d{2}|\d+(?:[.,]\d+)*)\b")
_SENT = re.compile("(?<=[.!?。！？])\\s+(?=[A-Z0-9\\\"'(\\[぀-鿿])")  # noqa: RUF001  # the full-width marks end CJK sentences
_HEADING = re.compile(
    r"^\s*(#{1,6}\s+\S.*|={2,}.*={2,}|[A-Z][A-Za-z0-9 ,&()/'-]{0,60}:|(?:section|part|chapter|article)\s+\S+.*"
    r"|\d+(?:\.\d+)*[.)]\s+[A-Z].{0,80})\s*$",
    re.I,
)
_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "is",
        "are",
        "be",
        "was",
        "were",
        "by",
        "with",
        "as",
        "at",
        "from",
        "that",
        "this",
        "it",
        "its",
        "what",
        "which",
        "should",
        "would",
        "will",
        "can",
        "could",
        "do",
        "does",
        "did",
        "how",
        "why",
        "when",
        "who",
        "whom",
        "whose",
    ]
)
_SPLITTERS = (re.compile(r"\n[ \t]*\n"), re.compile(r"\n"), _SENT)

Count = Callable[[str], int]
Encode = Callable[[str], Sequence[int]]


class EvidenceSelectionUnsupported(ValueError):
    """No unit matches the question, or the record can't be split to fit the budget."""


class Unit:
    __slots__ = ("a", "b", "head", "ntok", "src", "terms", "text")

    def __init__(self, text: str, head: str, a: int | None = None, b: int | None = None, src: str | None = None):
        self.text, self.head, self.a, self.b, self.src = text, head, a, b, src
        self.terms: list[str] = []
        self.ntok = 0


def terms(text: str) -> list[str]:
    """Lower-case words but stop words; a CJK run as its character pairs (a single character alone)."""
    out: list[str] = []
    for w in _WORD.findall(text.lower()):
        if _CJK.search(w):
            run: list[str] = []
            for c in [*w, " "]:
                if _CJK.match(c):
                    run.append(c)
                    continue
                if run:
                    out.extend(run[i] + run[i + 1] for i in range(len(run) - 1)) if len(run) > 1 else out.append(run[0])
                    run = []
                if c != " ":
                    out.append(c)
        elif w not in _STOP:
            out.append(w)
    return out


def spans(text: str, a: int, b: int, size: int, ntok: Count, level: int = 0) -> list[tuple[int, int]]:
    if b - a <= 0:
        return []
    if ntok(text[a:b]) <= size or level >= len(_SPLITTERS):
        if level >= len(_SPLITTERS) and ntok(text[a:b]) > size:
            step = max(64, size * 3)
            return [(i, min(i + step, b)) for i in range(a, b, step)]
        return [(a, b)]
    cuts = [m.end() for m in _SPLITTERS[level].finditer(text, a, b) if a < m.end() < b]
    if not cuts:
        return spans(text, a, b, size, ntok, level + 1)
    pieces, prev = [], a
    for c in [*cuts, b]:
        pieces.append((prev, c))
        prev = c
    packed: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = None
    for pa, pb in pieces:
        if cur and ntok(text[cur[0] : pb]) <= min(size, PACK_TOKENS):
            cur = (cur[0], pb)
        else:
            if cur:
                packed.append(cur)
            cur = (pa, pb)
    assert cur is not None
    packed.append(cur)
    out: list[tuple[int, int]] = []
    for pa, pb in packed:
        out.extend(spans(text, pa, pb, size, ntok, level + 1) if ntok(text[pa:pb]) > size else [(pa, pb)])
    return out


def text_units(text: str, ntok: Count, head: str = "", size: int = UNIT_TOKENS) -> list[Unit]:
    units, current = [], head
    for a, b in spans(text, 0, len(text), size, ntok):
        piece = text[a:b]
        if not piece.strip():
            continue
        first = piece.strip().split("\n", 1)[0]
        if len(first) < 120 and _HEADING.match(first):  # length first: the pattern backtracks cubically on long lines
            current = (head + " > " if head else "") + first.strip()
        units.append(Unit(piece, current, a, b, text))
    return units


def json_units(obj: Any, ntok: Count, path: str = "", size: int = UNIT_TOKENS) -> list[Unit]:
    def render(v: Any) -> str:
        return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(",", ":"))

    if isinstance(obj, dict):
        items = [(f"{path}.{k}" if path else str(k), v) for k, v in obj.items()]
    elif isinstance(obj, list):
        items = [(f"{path}[{i}]", v) for i, v in enumerate(obj)]
    else:
        return text_units(render(obj), ntok, path, size)
    units: list[Unit] = []
    group: list[str] = []
    for p, v in items:
        line = f"{p}: {render(v)}"
        if ntok(line) <= size:
            if group and ntok("\n".join([*group, line])) > size:
                units.append(Unit("\n".join(group), path))
                group = []
            group.append(line)
            continue
        if group:
            units.append(Unit("\n".join(group), path))
            group = []
        units.extend(
            json_units(v, ntok, p, size) if isinstance(v, (dict, list)) else text_units(render(v), ntok, p, size)
        )
    if group:
        units.append(Unit("\n".join(group), path))
    return units


def bm25(
    units: Sequence[Unit], weights: dict[str, float], k1: float = 1.5, b: float = 0.75
) -> tuple[list[float], collections.Counter[str]]:
    n = len(units)
    avg = sum(len(u.terms) for u in units) / max(n, 1) or 1.0
    df = collections.Counter(t for u in units for t in set(u.terms))
    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in weights if df.get(t)}
    out = []
    for u in units:
        tf = collections.Counter(u.terms)
        s, length = 0.0, len(u.terms)
        for t, w in weights.items():
            f = tf.get(t)
            if f:
                s += w * idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * length / avg))
        out.append(s)
    return out, df


def query(instructions: str, options: Any) -> tuple[dict[str, float], set[str]]:
    weights: dict[str, float] = {}
    for t in terms(instructions):
        weights[t] = 1.0
    texts = [f"{k} {v}" for k, v in options.items()] if isinstance(options, dict) else [str(o) for o in options or []]
    for t in terms(" ".join(texts)):
        weights[t] = max(weights.get(t, 0.0), OPTION_WEIGHT)
    return weights, {m.group(0).lower() for m in _EXACT.finditer(instructions)}


def runs(selected: Iterable[int]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    prev = None
    for i in sorted(selected):
        if prev is not None and i == prev + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
        prev = i
    return [(a, b) for a, b in out]


def assemble(units: Sequence[Unit], keep: set[int]) -> str:
    """The kept units' text, each run of them joined under its headings, the runs by MARK."""
    out = []
    for a1, b1 in runs(keep):
        segments: list[Any] = []
        cur: tuple[Any, Any, Any] | None = None
        last_head = None
        for i in range(a1, b1 + 1):
            u = units[i]
            if u.head and u.head != last_head and not u.text.lstrip().startswith(u.head):
                if cur:
                    segments.append(cur)
                    cur = None
                segments.append(u.head)
            last_head = u.head
            if cur and u.src is not None and cur[0] is u.src and cur[2] == u.a:
                cur = (cur[0], cur[1], u.b)
            else:
                if cur:
                    segments.append(cur)
                cur = (u.src, u.a, u.b) if u.src is not None else (None, u.text, None)
        if cur:
            segments.append(cur)
        out.append(
            "\n".join(
                x if isinstance(x, str) else x[0][x[1] : x[2]].strip("\n") if x[0] is not None else x[1]
                for x in segments
            )
        )
    return MARK.join(out)


def whole_or_unsupported(full: str, budget: int, encode: Encode) -> str:
    """`full` when it fits after all (counted exactly), else EvidenceSelectionUnsupported."""
    tokens = len(encode(full))
    if tokens <= budget:
        return full
    raise EvidenceSelectionUnsupported(f"the record's {tokens} tokens can't be cut to {budget} by passage")


def select_state(state: Any, instructions: str, options: Any, budget: int, encode: Encode) -> str:
    """The record (`state` as sent: text, or JSON) cut to `budget` tokens for the question."""
    text = state if isinstance(state, str) else None
    if text is not None and len(text.encode("utf-8", "ignore")) <= budget:
        return text
    full = text if text is not None else json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    if not budget or budget <= 0:
        return full
    nbytes = len(full.encode("utf-8", "ignore"))
    sample = full[:SAMPLE_CHARS]
    ratio = len(encode(sample)) / max(1, len(sample.encode("utf-8", "ignore")))
    if nbytes * ratio <= 1.25 * budget and len(encode(full)) <= budget:
        return full

    def ntok(s: str) -> int:
        return int(len(s.encode("utf-8", "ignore")) * ratio * 1.05) + 1

    size = max(48, min(UNIT_TOKENS, budget // SIZE_DIV))
    units = text_units(text, ntok, size=size) if text is not None else json_units(state, ntok, size=size)
    if len(units) <= 2:
        return whole_or_unsupported(full, budget, encode)
    for u in units:
        u.terms = terms(u.text)
        u.ntok = ntok(u.text) + 1
    n = len(units)
    weights, exact = query(instructions, options)
    first, df = bm25(units, weights)
    for i, u in enumerate(units):
        low = u.text.lower()
        first[i] += EXACT_BONUS * sum(1 for e in exact if e in low)
    top = sorted(range(n), key=lambda i: (-first[i], i))[:PRF_UNITS]
    candidates: dict[str, float] = {}
    for i in top:
        if first[i] <= 0:
            continue
        for t in set(units[i].terms):
            if t not in weights:
                candidates[t] = math.log(1 + n / (df[t] + 0.5))
    feedback = dict(weights)
    for t, _ in sorted(candidates.items(), key=lambda kv: (-kv[1], kv[0]))[:PRF_TERMS]:
        feedback[t] = PRF_WEIGHT
    second, _ = bm25(units, feedback)
    score = [second[i] + EXACT_BONUS * sum(1 for e in exact if e in units[i].text.lower()) for i in range(n)]
    if not any(value > 0 for value in score):
        raise EvidenceSelectionUnsupported("no passage of the record matches the question")
    sep = ntok(MARK) + 2
    keep = {0, n - 1}

    def cost(selected: set[int]) -> int:
        found, used = runs(selected), 0
        for a, b in found:
            used += sum(units[i].ntok for i in range(a, b + 1))
            h = units[a].head
            if h and not units[a].text.startswith(h):
                used += ntok(h) + 1
        return used + sep * (len(found) - 1)

    if cost(keep) > budget:
        keep = {n - 1} if cost({n - 1}) <= budget else set()
        if not keep:
            return whole_or_unsupported(full, budget, encode)
    order = [i for i in sorted(range(n), key=lambda i: (-score[i], i)) if score[i] > 0]
    parent: list[tuple[Any, ...]] = []
    for u in units:
        if u.src is not None and u.a is not None and u.b is not None:
            start = u.src.rfind("\n", 0, u.a) + 1
            end = u.src.find("\n", u.a)
            end = len(u.src) if end < 0 else end
            parent.append((id(u.src), start) if u.b <= end + 1 else (id(u.src), u.a, "own"))
        else:
            parent.append(("unit", id(u)))
    members: dict[tuple[Any, ...], list[int]] = collections.defaultdict(list)
    for i, p in enumerate(parent):
        members[p].append(i)
    least = min(u.ntok for u in units)

    def fill(limit: int) -> None:
        nonlocal keep
        used = cost(keep)
        for i in order:
            if limit - used < least:
                break
            if i in keep:
                continue
            for candidate in (set(members[parent[i]]) - keep, {i}):
                if not candidate or (
                    len(candidate) > 1 and sum(units[j].ntok for j in candidate) > max(size, budget // 3)
                ):
                    continue
                if used + sum(units[j].ntok for j in candidate) > limit:
                    continue
                new_cost = cost(keep | candidate)
                if new_cost <= limit:
                    keep, used = keep | candidate, new_cost
                    break

    fill(int(budget * PHASE_A))
    for i in [i for i in order if i in keep]:
        for j in sorted((i - 1, i + 1), key=lambda j: -score[j] if 0 <= j < n else 0):
            if 0 <= j < n and j not in keep and cost(keep | {j}) <= budget:
                keep = keep | {j}
    found = runs(keep)
    for (_, b1), (a2, _) in itertools.pairwise(found):
        gap = set(range(b1 + 1, a2))
        if gap and (len(gap) == 1 or sum(units[j].ntok for j in gap) <= GAP_TOKENS) and cost(keep | gap) <= budget:
            keep = keep | gap
    fill(budget)
    result = assemble(units, keep)
    while len(encode(result)) > budget:
        removable = [i for i in keep if i not in (0, n - 1)]
        if not removable:
            return whole_or_unsupported(full, budget, encode)
        keep = keep - {min(removable, key=lambda i: (score[i], -i))}
        result = assemble(units, keep)
    return result
