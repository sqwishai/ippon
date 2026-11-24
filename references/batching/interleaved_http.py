"""Run the kit's eight-client workload in alternating fresh servers under the caller's lock."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ippon._portable import dumps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="arm A's source directory")
    parser.add_argument("--b", required=True, help="arm B's source directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    folder = args.out.with_suffix("")
    folder.mkdir(parents=True, exist_ok=True)
    runs = []
    for i, arm in enumerate(("a", "b", "b", "a")):
        output = folder / f"{i}-{arm}.json"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("serve_workload.py")),
                "--ippon", str(Path(sys.executable).with_name("ippon")),
                "--model", "ollama://clef-flash:9b-mxfp8",
                "--engine", f"clef-{arm}-{i}",
                "--sizes", "1000", "--questions", "1", "--clients", "8",
                "--n", "10", "--warmup", "2", "--max-input", "16384",
                "--out", str(output),
            ],
            env={**os.environ, "PYTHONPATH": getattr(args, arm)},
            check=True,
        )  # fmt: skip
        runs.append({"arm": arm, "report": json.loads(output.read_text())})
        time.sleep(3)
    args.out.write_text(dumps({"a": args.a, "b": args.b, "runs": runs}, indent=2) + "\n")


if __name__ == "__main__":
    main()
