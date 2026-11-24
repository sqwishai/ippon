"""Whether a change makes a model faster: two configurations timed in alternating fresh processes.

    ippon bench MODEL --ab "--switch glu=mlx" "--switch glu=kernel" [--rounds 4] [bench's options]

An arm is extra `bench` options, after which any NAME=value words set environment variables for its process
(`"MLX_METAL_FAST_SYNCH=1"`, `"--chunk 8192"`). Each arm runs `rounds` times in a new process (load and warm-up
untimed), in the order A B, B A, A B... so drift (heat, other processes) falls on both; in a round both arms time
the same requests (the same seed), so their answers pair up too. Per cell it reports both medians, B's median over
A's with a bootstrap 95% interval, the same for p95, and the largest difference between the arms' probabilities.
The table displays signed percentage time changes; negative means B is faster. JSON retains the ratios.

The verdict follows the promotion rule (docs/benchmarks.md): B is faster where its median is at least 3% below
A's with the interval below 1; slower where its median is more than 1% above A's with the interval above 1 (or its
p95 more than 1% above, with the interval above 1). "promote" needs a faster cell and no slower one.
"""

from __future__ import annotations

import os
import re
import shlex
import statistics
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import bench, processes
from ._portable import private_report, record
from .compare_arms import difference
from .processes import run_json
from .stats import quantile, ratio_interval

ENV = re.compile(r"^[A-Z][A-Z0-9_]*=")
GAIN, LOSS = 0.97, 1.01
COOLDOWN = 3.0  # seconds between processes, for the GPU to settle


@dataclass
class Arm:
    spec: str
    args: list[str]
    env: dict[str, str]
    cells: dict[bench.Key, dict[str, Any]] = field(default_factory=dict)  # merged over rounds
    reports: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def parse(cls, spec: str) -> Arm:
        words = shlex.split(spec)
        env = dict(w.split("=", 1) for w in words if ENV.match(w))
        return cls(spec, [w for w in words if not ENV.match(w)], env)

    def add(self, report: dict[str, Any]) -> None:
        self.reports.append(report)
        for c in report["cells"]:
            merged = self.cells.setdefault(bench.key(c), {"ms": [], "answers": [], "input_tokens": c["input_tokens"]})
            merged["ms"] += c["ms"]
            merged["answers"] += c["answers"]


def largest_difference(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float | None:
    """The largest difference between paired requests' probabilities (`compare_arms.difference`); None when they
    don't pair up."""
    try:
        return difference(a, b)[0]
    except ValueError:
        return None


def judge(r50: float, ci50: tuple[float, float], r95: float, ci95: tuple[float, float]) -> str:
    """One cell's verdict by the promotion rule: "faster", "slower" or "same" for B, from B's median over A's and
    its p95 over A's, each with its bootstrap 95% interval."""
    if (r50 > LOSS and ci50[0] > 1.0) or (r95 > LOSS and ci95[0] > 1.0):
        return "slower"
    if r50 <= GAIN and ci50[1] < 1.0:
        return "faster"
    return "same"


def compare(a: Arm, b: Arm) -> list[dict[str, Any]]:
    rows = []
    for key, ca in a.cells.items():
        cb = b.cells.get(key)
        if cb is None:
            continue
        r50 = statistics.median(cb["ms"]) / statistics.median(ca["ms"])
        r95 = quantile(cb["ms"], 0.95) / quantile(ca["ms"], 0.95)
        lo50, hi50 = ratio_interval(ca["ms"], cb["ms"])
        lo95, hi95 = ratio_interval(ca["ms"], cb["ms"], 0.95)
        verdict = judge(r50, (lo50, hi50), r95, (lo95, hi95))
        rows.append(
            {
                **bench.fields(key),
                "input_tokens": ca["input_tokens"],
                "a_median_ms": round(statistics.median(ca["ms"]), 2),
                "b_median_ms": round(statistics.median(cb["ms"]), 2),
                "ratio": round(r50, 4),
                "ratio_ci": [round(lo50, 4), round(hi50, 4)],
                "p95_ratio": round(r95, 4),
                "p95_ratio_ci": [round(lo95, 4), round(hi95, 4)],
                "max_probability_difference": largest_difference(ca["answers"], cb["answers"]),
                "verdict": verdict,
            }
        )
    return rows


def verdict(rows: Sequence[dict[str, Any]]) -> str:
    kinds = {r["verdict"] for r in rows}
    if "slower" in kinds:
        return "reject" if "faster" not in kinds else "mixed"
    return "promote" if "faster" in kinds else "no difference"


def table(a: Arm, b: Arm, rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        f"A: {a.spec or '(defaults)'}",
        f"B: {b.spec or '(defaults)'}",
        "",
        "Time change: B relative to A; negative means B is faster.",
        "",
        "| scenario | words | q | tokens | A median | B median | time change | 95% CI | p95 time change | max Δp | |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        dp = r["max_probability_difference"]
        lines.append(
            f"| {bench.label(r)} | {r['words']:,} | {r['questions']} | {r['input_tokens']:,} | "
            f"{r['a_median_ms']:,.1f} | {r['b_median_ms']:,.1f} | {r['ratio'] - 1:+.1%} | "
            f"{r['ratio_ci'][0] - 1:+.1%} to {r['ratio_ci'][1] - 1:+.1%} | "
            f"{r['p95_ratio'] - 1:+.1%} | {'-' if dp is None else f'{dp:.1e}'} | {r['verdict']} |"
        )
    lines.append(f"\nverdict for B: {verdict(rows)}")
    busy = sorted({x for arm in (a, b) for rep in arm.reports for c in rep["cells"] for x in c["busy"]})
    if busy:
        lines.append("busy while timing: " + ", ".join(busy))
    return "\n".join(lines)


def run(
    model: str,
    specs: Sequence[str],
    shared: Sequence[str],
    n: int,
    rounds: int,
    seed: int,
    log: Any = print,
) -> tuple[str, dict[str, Any]]:
    """Time arms `specs` (two) of `model` with `shared` bench options, n requests per cell each, over `rounds`
    rounds; the table and the JSON report."""
    arms = [Arm.parse(s) for s in specs]
    with tempfile.TemporaryDirectory(prefix="ippon-ab-") as tmp:
        for r, (count, round_seed) in enumerate(bench.split_rounds(n, rounds, seed)):
            for which in (0, 1) if r % 2 == 0 else (1, 0):
                arm, out = arms[which], Path(tmp) / f"{r}-{which}.json"
                command = [*processes.command(), "bench", model, *shared]
                command += ["--n", str(count), "--seed", str(round_seed), "--out", str(out), *arm.args]
                log(f"round {r + 1}/{rounds}, arm {'AB'[which]}")
                arm.add(run_json(command, out, f"arm {'AB'[which]}", {**os.environ, **arm.env}))
                time.sleep(COOLDOWN)
    rows = compare(*arms)
    data = {
        "model": model,
        "a": {"spec": arms[0].spec, "reports": arms[0].reports},
        "b": {"spec": arms[1].spec, "reports": arms[1].reports},
        "rounds": rounds,
        "n": n,
        "cells": rows,
        "verdict": verdict(rows),
    }
    return table(arms[0], arms[1], rows), record(data, process_names=private_report())
