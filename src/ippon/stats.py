"""The statistics every timing reports (bench's cells, ab's and compare's ratios): quantiles by linear interpolation,
and bootstrap 95% intervals from CI_RESAMPLES resamples, seeded so a report is the same each time it is made."""

from __future__ import annotations

import random
import statistics
from collections.abc import Callable, Sequence

CI_RESAMPLES = 2000


def quantile(values: Sequence[float], q: float) -> float:
    """The q-quantile by linear interpolation between the sorted values (q=0.5: the median)."""
    s = sorted(values)
    at = q * (len(s) - 1)
    lo = int(at)
    return s[lo] if lo + 1 >= len(s) else s[lo] + (s[lo + 1] - s[lo]) * (at - lo)


def _interval(resampled: Callable[[random.Random], float], seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    values = sorted(resampled(rng) for _ in range(CI_RESAMPLES))
    return values[int(0.025 * CI_RESAMPLES)], values[int(0.975 * CI_RESAMPLES) - 1]


def median_interval(values: Sequence[float], seed: int = 0) -> tuple[float, float]:
    """A bootstrap 95% interval of the median."""
    return _interval(lambda rng: statistics.median(rng.choices(values, k=len(values))), seed)


def ratio_interval(a: Sequence[float], b: Sequence[float], q: float = 0.5, seed: int = 0) -> tuple[float, float]:
    """A bootstrap 95% interval of quantile q of b over quantile q of a."""
    return _interval(lambda rng: quantile(rng.choices(b, k=len(b)), q) / quantile(rng.choices(a, k=len(a)), q), seed)
