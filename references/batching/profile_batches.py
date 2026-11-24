"""Compare one batch with consecutive single passes, using the kit's request tokens.

Run on an idle Mac on High Power and AC. Wall times
use the ordinary compiled path; TimedOps separately attributes eager operations.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

from ippon import Decider, parse_request
from ippon._portable import dumps
from ippon.choices import RULE
from ippon.machine import describe
from ippon.profile import TimedOps, round_trip

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/kit"))
import gen_requests as requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--tokens", type=int, default=1344)
    parser.add_argument("--count", type=int, default=7)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--wall-only", action="store_true")
    parser.add_argument("--last-row", action="store_true", help="retain only the last hidden row for attribution")
    parser.add_argument("--layout", choices=("both", "singles", "batch"), default="both")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model = Decider.load(args.model, precision=args.precision, prefix_budget=0)
    prompts, rows, groups = [], [], []
    if args.tokens == 1344:
        bodies = [
            requests.to_systemone(req["state"], requests.questions(1)) for req in requests.plan([1000], [1], 100, 0)
        ]
    else:
        bodies = [
            {
                "state": f"Ticket {i}. " + "a " * n,
                "questions": {"q": {"type": "noul", "instructions": "Is this a ticket?"}},
            }
            for i in range(args.count)
            for n in range(args.tokens)
        ]
    for body in bodies:
        prompt = model.render(parse_request(body)).prompts[0]
        if len(prompt.tokens) == args.tokens:
            prompts.append(prompt.tokens)
            rows.append([len(prompt.tokens) - 1] if args.last_row else [r.position for r in prompt.readouts])
            groups.append(prompt.groups)
        if len(prompts) == args.count:
            break
    assert len(prompts) == args.count, "not enough prompts of the requested length"
    assert all(len(p) == args.tokens for p in prompts)
    backbone = model.backbone

    def run(batch: bool) -> tuple[float, list[mx.array]]:
        start = time.perf_counter()
        if batch:
            out = backbone.batch(prompts, rows, groups if groups[0] else None)
            mx.eval(out)
        else:
            out = []
            for p, r, g in zip(prompts, rows, groups, strict=True):
                result = backbone.batch([p], [r], [g] if g else None)
                mx.eval(result)
                out.extend(result)
        return (time.perf_counter() - start) * 1000, out

    report = {
        "machine": describe(),
        "model": args.model,
        "precision": model.precision,
        "kernel_rule": RULE,
        "readout": "last-row" if args.last_row else "maker",
        "tokens": args.tokens,
        "count": args.count,
        "repeat": args.repeat,
        "layout": args.layout,
    }
    print(json.dumps(report), flush=True)
    plain = backbone.ops
    outputs = {}
    modes = (False, True) if args.layout == "both" else (args.layout == "batch",)
    for batch in modes:
        name = "batch" if batch else "singles"
        run(batch)
        samples = [run(batch)[0] for _ in range(args.repeat)]
        outputs[name] = run(batch)[1]
        print(name, samples, flush=True)
        report[name] = {"ms": samples, "median_ms": statistics.median(samples)}
    if args.layout == "both":
        report["hidden_max_difference"] = max(
            float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())
            for a, b in zip(outputs["singles"], outputs["batch"], strict=True)
        )
    report["round_trip_ms"] = floor = round_trip()
    for batch in modes:
        if args.wall_only:
            break
        name = "batch" if batch else "singles"
        timed = TimedOps.of(plain)
        backbone.ops = timed
        run(batch)
        per_op = defaultdict(list)
        for _ in range(args.repeat):
            timed.clock.clear()
            timed.calls.clear()
            run(batch)
            for key, value in timed.clock.items():
                per_op[key].append(max(value * 1000 - timed.calls[key] * floor, 0))
        operations = {k: statistics.median(v) for k, v in per_op.items()}
        report[name]["operations_ms"] = operations
        print(name, json.dumps(operations), flush=True)
    backbone.ops = plain
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dumps(report, indent=2) + "\n")
    model.close()


if __name__ == "__main__":
    main()
