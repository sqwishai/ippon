"""Profile singles and a batch in alternating fresh processes under the caller's heavy lock."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    folder = args.out.with_suffix("")
    folder.mkdir(parents=True, exist_ok=True)
    runs = []
    for i, layout in enumerate(("singles", "batch", "batch", "singles")):
        output = folder / f"{i}-{layout}.json"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("profile_batches.py")),
                args.model,
                "--precision",
                args.precision,
                "--layout",
                layout,
                "--repeat",
                "1",
                "--out",
                str(output),
            ],
            check=True,
        )
        runs.append(json.loads(output.read_text()))
        time.sleep(3)
    result = {"runs": runs, "order": [r["layout"] for r in runs]}
    for layout in ("singles", "batch"):
        reports = [r[layout] for r in runs if r["layout"] == layout]
        samples = [ms for r in reports for ms in r["ms"]]
        result[layout] = {
            "ms": samples,
            "median_ms": statistics.median(samples),
            "operations_ms": {
                key: statistics.median(r["operations_ms"][key] for r in reports) for key in reports[0]["operations_ms"]
            },
        }
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
