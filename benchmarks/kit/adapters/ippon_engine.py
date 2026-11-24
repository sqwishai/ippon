"""A decision model through ippon, in-process:
Decider.load(path).systemone(body), on this Mac's chip profile unless --profile forces one.

    envs/ippon/bin/python -I adapters/ippon_engine.py --engine d1-2B-ippon \
        --model ~/models/d1-2B --max-input 32768 [--profile m4] [--ippon-precision auto]

Precision is ippon's default, auto (fp16 where the model's activations fit, else bf16); the result records
the one it chose.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common, gen_requests
import common


def main() -> None:
    p = common.add_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    p.add_argument("--model", required=True)
    p.add_argument("--profile")
    p.add_argument("--ippon-precision", default="auto", help="ippon's --precision")
    args = p.parse_args()

    import mlx.core as mx

    import ippon
    from ippon import Decider
    from ippon.machine import revision

    model_dir = args.model if args.model.startswith("ollama://") else str(Path(args.model).expanduser())
    decider = Decider.load(model_dir, profile=args.profile, precision=args.ippon_precision)

    def decide(req):
        out = decider.systemone(req["systemone"])
        return common.systemone_answers(out), out["usage"]["input_tokens"], None

    def memory():
        return {"mlx_peak": mx.get_peak_memory(), "mlx_active": mx.get_active_memory()}

    meta = {
        "runtime": f"ippon {ippon.__version__} + mlx {mx.__version__}",
        "adapter": decider.adapter,
        "profile": decider.profile.name,
        "forced_profile": decider.profile.forced,
        "precision": decider.precision,
        "precision_asked": args.ippon_precision,
        "peak_activation": decider.details.get("peak_activation"),
        "model_dir": model_dir,
        "revision": revision(),
    }
    common.run(args, decide, meta, memory=memory)


if __name__ == "__main__":
    main()
