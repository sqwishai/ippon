"""ippon itself as a `ippon bench --compare` arm, under the same contract as the others:

python -m ippon.compare_arms.ours MODEL_DIR REQUESTS.json OUT.json [--precision ... --switch ... ...]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from importlib.metadata import version

from .. import __version__
from ..cli import common_options, load
from .harness import time_requests, write


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(parents=[common_options()])
    parser.add_argument("model")
    parser.add_argument("requests")
    parser.add_argument("out")
    args = parser.parse_args(argv)
    with load(args) as decider:
        results = time_requests(args.requests, decider.systemone)
        runtime = f"ippon {__version__} + mlx {version('mlx')} ({decider.precision}, profile {decider.profile.name})"
    write(args.out, runtime, results)


if __name__ == "__main__":
    main(sys.argv[1:])
