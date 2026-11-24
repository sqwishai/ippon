"""Recompute the complete paired rounds without recorded competing CPU activity; never select on latency."""

import argparse
import json
import re
from pathlib import Path

from ippon._portable import dumps
from ippon.ab import Arm, compare, verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    a, b = Arm.parse(report["a"]["spec"]), Arm.parse(report["b"]["spec"])
    kept, excluded = [], []
    for index, (ra, rb) in enumerate(zip(report["a"]["reports"], report["b"]["reports"], strict=True), 1):
        activity = sorted(
            {v for rep in (ra, rb) for c in rep["cells"] for v in c["busy"] if re.search(r" \(\d+%\)$", v)}
        )
        if activity:
            excluded.append({"round": index, "activity": activity})
        else:
            kept.append(index)
            a.add(ra)
            b.add(rb)
    if not kept:
        parser.error("no uncontaminated complete pairs")
    rows = compare(a, b)
    args.out.write_text(
        dumps(
            {
                "selection": (
                    "Exclude both arms of a round when either records a process above 50% CPU; independent of latency"
                ),
                "included_rounds": kept,
                "excluded_rounds": excluded,
                "cells": rows,
                "verdict": verdict(rows),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
