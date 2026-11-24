"""The answer formulas several creators' servers share, each named for what it computes: how sure an answer is (a
`Confidence`, which `schema.systemone_answer` takes), a score's median level, and a weak yes/no committed to a floor.
A formula one creator alone uses lives in its adapter."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

Confidence = Callable[[Sequence[float]], float]  # how sure an answer is, from its probabilities


def top_probability(p: Sequence[float]) -> float:
    """The most likely option's probability."""
    return max(p)


def above_chance(p: Sequence[float]) -> float:
    """How far the most likely option stands above an even spread, (max - 1/n) / (1 - 1/n) within [0, 1]: the System One
    choice confidence, which Perplexity's, Torchcast's, H2O's and Elephant's servers return."""
    n = len(p)
    if n < 2:
        return 1.0
    return max(0.0, min(1.0, (max(p) - 1 / n) / (1 - 1 / n)))


def near_best(p: Sequence[float]) -> float:
    """1 - E|level - most likely level| / E|level - middle level| under an even spread, within [0, 1]: the System One
    score confidence, which Perplexity's, Torchcast's and Elephant's servers return."""
    best = max(range(len(p)), key=p.__getitem__)
    midpoint = (len(p) - 1) / 2
    even = sum(abs(i - midpoint) for i in range(len(p))) / len(p)
    if not even:
        return 1.0
    return max(0.0, 1.0 - sum(x * abs(i - best) for i, x in enumerate(p)) / even)


def certainty(p: Sequence[float]) -> float:
    """One minus the normalized entropy, within [0, 1]: vllm-sr d3's confidence for a choice and a score."""
    if len(p) < 2:
        return 1.0
    entropy = -sum(x * math.log(x) for x in p if x > 0)
    return max(0.0, min(1.0, 1 - entropy / math.log(len(p))))


def chance_rescaled(p: Sequence[float]) -> float:
    """The top probability rescaled for chance, (n * max - 1) / (n - 1), unclipped (2 options at least): Kettei's
    choice confidence, as its SDK computes it. `above_chance` is the same measure in
    another order of operations, whose last bit can differ."""
    n = len(p)
    return (n * max(p) - 1) / (n - 1)


def median_level(probabilities: Sequence[float]) -> int:
    """The first level whose cumulative probability reaches one half: Liquid's d1 score (api.py's score_level),
    and the level Kettei's score confidence is measured from."""
    total, mass = sum(probabilities) or 1.0, 0.0
    for i, p in enumerate(probabilities):
        mass += p / total
        if mass >= 0.5:
            return i
    return len(probabilities) - 1


def floored(p_yes: float, floor: float) -> float:
    """A yes/no probability whose max(P, 1 - P) is under `floor` moved to the floor on its own side (a tie to yes), so
    the answer stays and its confidence reaches the floor, as Torchcast's and H2O's servers commit a weak answer
    (JevBench counts 0.2 < P(yes) < 0.8 as no answer); a floor of 0 is off."""
    if not floor or max(p_yes, 1.0 - p_yes) >= floor:
        return p_yes
    return floor if p_yes >= 0.5 else 1.0 - floor
