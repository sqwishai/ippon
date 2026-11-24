"""Perplexity's pplx-decider through its creator's own package (the checkpoint's source/src/pplx_decider, Apache-2.0),
its reference path on the Mac's GPU: Decider(path, device="mps").decide(...). Off CUDA the package computes in
fp32; --dtype bf16 casts the backbone and readout to bf16, its dtype on CUDA (the reference script does the same).
Its fast path needs CUDA (FlashAttention, CUDA graphs) and stock llama.cpp doesn't implement its protocol, so this
is the creator's only way to run it on a Mac. A request goes to `decide` whole, which asks its questions in the
package's default mode, one sequence per question after the request's context, as ippon asks them.

    envs/pplx/bin/python -I adapters/pplx_decider.py --engine pplx-flash-maker-mps \
        --model ~/models/pplx-decider-flash-v1.3 --max-input 32768 [--dtype bf16]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common
import common


def main() -> None:
    p = common.add_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    p.add_argument("--model", required=True)
    p.add_argument("--dtype", default="fp32", choices=("fp32", "bf16"))
    args = p.parse_args()

    import importlib.metadata as md

    import torch

    model = Path(args.model).expanduser()
    sys.path.insert(0, str(model / "source" / "src"))
    from pplx_decider.model import Decider  # the creator's package, from the checkpoint

    decider = Decider(model, device="mps")
    if args.dtype == "bf16":
        decider.backbone.to(torch.bfloat16)
        decider.readout.to(torch.bfloat16)

    def decide(req):
        body = dict(req["systemone"])
        if "files" in body:  # the package takes images as data URLs, among other forms (protocol.open_image)
            body["images"] = body.pop("files")
        response = decider.decide(body)
        torch.mps.synchronize()
        tokens = int((response.get("usage") or {}).get("input_tokens") or 0)
        return common.systemone_answers(response), tokens or None, None

    meta = {
        "runtime": f"pplx_decider reference path on MPS ({args.dtype}), torch {torch.__version__}, "
        f"transformers {md.version('transformers')}",
        "questions": "one per sequence",
        "question_method": "one decide call, the package's default mode: a separate sequence per question",
    }
    common.run(args, decide, meta)


if __name__ == "__main__":
    main()
