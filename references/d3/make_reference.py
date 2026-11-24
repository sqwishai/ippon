"""vllm-sr d3's reference answers, from vllm-sr's own runtime (d3_runtime.py, shipped in the model repository).

Run it in an environment with torch and transformers>=5.17, never inside ippon:

    python -I make_reference.py --model path/to/d3-mini --requests ../requests/standard.json \
        --out d3-mini.json.gz [--tokens-only]

It loads the checkpoint with `d3_runtime.D3(root, device="cpu")`, takes every request's token ids from
`D3.prepare` (the creator's chat template and format), and scores them with `D3.run` with the backbone in fp32
(and again in bf16, the runtime's own precision, for the bound). Probabilities are written in the request's
option order (a noul as true, false).

For additional cases at an already established bound, use --fp32-only --bound VALUE. This does not fit a new
bound to the new cases. --only selects case names without changing their requests.
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
import math
import sys
from pathlib import Path
from runpy import run_path
from typing import Any

# Load just the stdlib serializer without importing ippon or changing runtime module discovery.
dumps = run_path(str(Path(__file__).resolve().parents[2] / "src/ippon/_portable.py"))["dumps"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, type=Path, help="a downloaded d3 checkpoint (its directory)")
    p.add_argument("--requests", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tokens-only", action="store_true")
    p.add_argument("--fp32-only", action="store_true", help="keep the established bound; omit a new bf16 run")
    p.add_argument("--bound", type=float, default=0.01, help="established bound with --fp32-only")
    p.add_argument("--only", nargs="*", help="request names to run")
    args = p.parse_args()
    if not math.isfinite(args.bound) or args.bound <= 0:
        p.error("--bound must be finite and positive")
    root = args.model.expanduser()
    sys.path.insert(0, str(root))
    runtime = importlib.import_module("d3_runtime")

    import torch

    d3 = runtime.D3(root, device="cpu", batch_size=1)
    cases = json.loads(args.requests.read_text())
    if args.only:
        if missing := set(args.only) - {c["name"] for c in cases}:
            p.error(f"unknown request names: {sorted(missing)}")
        cases = [c for c in cases if c["name"] in args.only]
    if not cases:
        p.error("no matching requests")

    def order(question: dict[str, Any], probs: list[float]) -> list[float]:
        return [probs[1], probs[0]] if question["type"] == "noul" else probs

    out = []
    for case in cases:
        request = case["request"]
        prepared = d3.prepare(request.get("state"), request["questions"])
        if prepared.errors:
            continue  # a request the model refuses is not a reference case
        out.append(
            {
                "name": case["name"],
                "request": request,
                "prompts": [prepared.sequences[k] for k in prepared.keys],
                "input_tokens": sum(len(prepared.sequences[k]) for k in prepared.keys),
            }
        )
    if not out:
        p.error("the creator refused every request")
    precisions = [] if args.tokens_only else [("probabilities", torch.float32), ("probabilities_bf16", torch.bfloat16)]
    if args.fp32_only and not args.tokens_only:
        precisions = [("probabilities", torch.float32)]
    for key, dtype in precisions:
        d3.backbone.to(dtype)
        with torch.inference_mode():
            for entry in out:
                request = entry["request"]
                probabilities, _ = d3.run(d3.prepare(request.get("state"), request["questions"]))
                entry[key] = {
                    q: order(request["questions"][q], [float(x) for x in probabilities[q]])
                    for q in request["questions"]
                }
                print(f"{key} {entry['name']}: {entry['input_tokens']} tokens", flush=True)
    bound = args.bound if args.fp32_only else 0.01
    if precisions and not args.fp32_only:
        distance = max(
            abs(a - b)
            for e in out
            for q in e["probabilities"]
            for a, b in zip(e["probabilities"][q], e["probabilities_bf16"][q], strict=True)
        )
        bound = max(0.01, round(distance + 5e-4, 4))
    config = json.loads((root / "decision_config.json").read_text())
    reference = {
        "schema": "ippon-reference/1",
        "adapter": "vllm-sr/d3",
        "model": {"directory": root.name, "base_model": config.get("base_model")},
        "maker": {"code": "d3_runtime.py", "device": "cpu", "dtype": "float32"},
        "bound": bound,
        "tokens_only": not precisions,
        "requests": out,
    }
    data = dumps(reference, ensure_ascii=False).encode()
    args.out.write_bytes(gzip.compress(data) if args.out.suffix == ".gz" else data)
    print(f"{len(out)} requests -> {args.out} (bound {bound}{', tokens only' if not precisions else ''})")


if __name__ == "__main__":
    main()
