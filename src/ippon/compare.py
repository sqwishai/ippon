"""ippon against the other ways to run the same model on this Mac, on the same requests.

    ippon bench MODEL_DIR --compare mlx-vlm-model=~/venvs/mlxvlm/bin/python [bench's options] [--rounds 2]

An arm is a runtime and the Python that has it installed (`compare_arms/`: a community port, another runtime's
layers, transformers). Every arm, ippon included, runs as its own process, from its own environment (ippon
never imports another runtime), on a file of requests that is the same for every arm in a round, and times each
call in its own process. Rounds run the arms in turn, starting from another arm each round, so drift falls on
all of them; loading and warm-up are untimed.

Per cell it records each arm's median and a speedup ratio (the arm's median over ippon's), with a bootstrap
95% interval and the same for p95. Tables show these as signed percentage speed changes; positive means ippon is
faster. Answer agreement records the largest probability difference and whether the most likely option is the
same. A cell is "faster" or "slower" for ippon by the promotion rule
(`ab.py`): at least 3% with the interval clear of 1, or more than 1% the other way.
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import ab, bench, compare_arms, machine
from .asks import Asker, asker, standard
from .compare_arms import lookup
from .errors import InvalidRequest
from .model_dir import ModelDir
from .processes import run_json
from .registry import select
from .stats import median_interval, quantile, ratio_interval

OURS = "ippon"


def requests(spec: bench.Spec, ask: Asker = standard) -> list[dict[str, Any]]:
    """One round's requests: per cell, `warmup` untimed then `n` timed, as bench generates them (`ask`: the questions
    the model can answer)."""
    items = []
    for words, count, gap, seed in spec.cells():
        for i, body in enumerate(bench.bodies(spec.scenario, words, count, seed, spec.warmup + spec.n, ask)):
            items.append(
                {
                    "cell": bench.fields((spec.scenario, words, count, gap)),
                    "warm": i < spec.warmup,
                    "gap": gap,
                    "body": body,
                }
            )
    return items


def asker_for(model: str) -> Asker:
    """The questions `model` can answer, as its loaded Decider asks them (`asks.asker`)."""
    directory = ModelDir.open(model)
    return asker(select(directory), directory)


def run(
    model: str,
    wanted: Sequence[str],
    ours_options: Sequence[str],
    spec: bench.Spec,
    rounds: int,
    log: Any = print,
) -> tuple[str, dict[str, Any]]:
    """Time ippon (with `ours_options`) and the `wanted` arms on `model` over `rounds` rounds of the spec's `n`
    timed requests per cell in all; the table and the JSON report."""
    if not Path(model).expanduser().is_dir():
        raise InvalidRequest("--compare runs every arm on the same files: give the model's downloaded directory")
    if spec.scenario == "concurrent":  # an arm answers its requests one after another: nothing would be concurrent
        raise InvalidRequest("--compare times requests one after another: give another --scenario than concurrent")
    model = str(Path(model).expanduser())
    others = lookup.arms_for(model, wanted)
    ask = asker_for(model)

    names = [OURS, *others]
    commands = {OURS: [sys.executable, "-P", "-m", "ippon.compare_arms.ours"]}
    commands |= {name: [python, "-I", script] for name, (python, script) in others.items()}
    times: dict[str, dict[bench.Key, list[float]]] = {a: {} for a in names}
    answers: dict[str, dict[bench.Key, list[Any]]] = {a: {} for a in names}
    tokens: dict[str, dict[bench.Key, Any]] = {a: {} for a in names}
    runtimes: dict[str, str] = {}
    seen: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="ippon-compare-") as tmp:
        for r, (count, seed) in enumerate(bench.split_rounds(spec.n, rounds, spec.seed)):
            items = requests(replace(spec, n=count, seed=seed), ask)
            path = Path(tmp) / f"requests-{r}.json"
            path.write_text(json.dumps(items))
            for name in names[r % len(names) :] + names[: r % len(names)]:
                out = Path(tmp) / f"{name}-{r}.json"
                log(f"round {r + 1}/{rounds}: {name}")
                tail = [*ours_options] if name == OURS else []
                result = run_json([*commands[name], model, str(path), str(out), *tail], out, f"arm {name}")
                runtimes[name] = result["runtime"]
                for item, got in zip(items, result["results"], strict=True):
                    if item["warm"]:
                        continue
                    key = bench.key(item["cell"])
                    times[name].setdefault(key, []).append(got["ms"])
                    answers[name].setdefault(key, []).append(got["answers"])
                    tokens[name][key] = got["input_tokens"]
                seen.update(machine.busy())
                time.sleep(ab.COOLDOWN)
    rows = []
    for key, mine in times[OURS].items():
        row: dict[str, Any] = bench.fields(key)
        row |= {"input_tokens": tokens[OURS][key], "ippon": _stats(mine), "arms": {}}
        for name in others:
            theirs = times[name][key]
            speedup = statistics.median(theirs) / statistics.median(mine)
            lo, hi = ratio_interval(mine, theirs)
            p95 = quantile(theirs, 0.95) / quantile(mine, 0.95)
            lo95, hi95 = ratio_interval(mine, theirs, 0.95)
            # ippon is the rule's B and the arm its A: B over A is the reciprocal of each speedup
            verdict = ab.judge(1 / speedup, (1 / hi, 1 / lo), 1 / p95, (1 / hi95, 1 / lo95))
            row["arms"][name] = {
                **_stats(theirs),
                "input_tokens": tokens[name][key],
                "speedup": round(speedup, 4),
                "speedup_ci": [round(lo, 4), round(hi, 4)],
                "p95_speedup": round(p95, 4),
                **compare_arms.agreement(answers[OURS][key], answers[name][key]),
                "verdict": verdict,
            }
        rows.append(row)
    data = {
        "model": model,
        "runtimes": runtimes,
        "rounds": rounds,
        **spec.to_json(),
        "busy": sorted(seen),
        "cells": rows,
    }
    return table(data), data


def _stats(ms: Sequence[float]) -> dict[str, Any]:
    lo, hi = median_interval(ms)
    return {
        "median_ms": round(statistics.median(ms), 2),
        "median_ci_ms": [round(lo, 2), round(hi, 2)],
        "p95_ms": round(quantile(ms, 0.95), 2),
        "n": len(ms),
    }


def table(data: dict[str, Any]) -> str:
    lines = [f"{name}: {runtime}" for name, runtime in data["runtimes"].items()]
    for name in [a for a in data["runtimes"] if a != OURS]:
        lines += [
            "",
            f"ippon against {name} (speed change: positive means ippon is faster)",
            "",
            "| scenario | words | q | tokens | ippon | " + name + " | speed change | 95% CI | "
            "p95 speed change | max Δp | top differ | |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for r in data["cells"]:
            a = r["arms"][name]
            toks = f"{r['input_tokens']:,}" + (
                "" if a["input_tokens"] in (None, r["input_tokens"]) else f" ({a['input_tokens']:,})"
            )
            lines.append(
                f"| {bench.label(r)} | {r['words']:,} | {r['questions']} | {toks} | {r['ippon']['median_ms']:,.1f} | "
                f"{a['median_ms']:,.1f} | {a['speedup'] - 1:+.1%} | "
                f"{a['speedup_ci'][0] - 1:+.1%} to {a['speedup_ci'][1] - 1:+.1%} | "
                f"{a['p95_speedup'] - 1:+.1%} | {a['max_probability_difference']:.1e} | {a['top_answers_differ']} | "
                f"{a['verdict']} |"
            )
        verdicts = [r["arms"][name]["verdict"] for r in data["cells"]]
        counts = ", ".join(f"{verdicts.count(v)} {v}" for v in ("faster", "same", "slower"))
        lines.append(f"\nippon against {name}: {counts}")
    if data["busy"]:
        lines.append("busy while timing: " + ", ".join(data["busy"]))
    return "\n".join(lines)
