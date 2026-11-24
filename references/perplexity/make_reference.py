"""Reference answers for Perplexity's pplx-decider (chunked joint protocol, version 4), from the creator's own code: the
file `ippon verify` checks.

The creator's package ships in the checkpoint (source/src/pplx_decider, Apache-2.0); this script runs its reference
path (`Decider`, the unmodified transformers backbone with its noncausal hook and the readout, one request one
sequence per question) in an environment with the creator's requirements.txt, never inside ippon:

    python -I make_reference.py --model DIR --requests REQUESTS.json --out NAME.json.gz --repo ID --revision SHA

With --tokens-only, only the prompts (the creator's encoder, no weights loaded): NAME.tokens.json.gz.

fp32 on the CPU (the creator's dtype off CUDA) for the reference, then bf16 (its dtype on CUDA) for the bound: the
larger of 0.01 and bf16's distance from fp32 plus 0.0005, rounded to four decimals, on these requests.
Prompts are the creator's encoder's token ids;
probabilities are softmax(logits / temperature) in the request's option order (a noul as true, false).
For additional cases use --fp32-only --bound VALUE to retain an established bound instead of fitting a new one.

A request's questions are asked in the creator's default mode, "separate": one sequence per question, the request's
context then that question (`Encoder.plan`), as ippon asks them.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from runpy import run_path
from typing import Any

import torch

# Load just the stdlib serializer without importing ippon or changing runtime module discovery.
dumps = run_path(str(Path(__file__).resolve().parents[2] / "src/ippon/_portable.py"))["dumps"]


def ordered(question: dict[str, Any], probabilities: list[float]) -> list[float]:
    return [probabilities[1], probabilities[0]] if question["type"] == "noul" else probabilities


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--requests", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--tokens-only", action="store_true")
    p.add_argument("--fp32-only", action="store_true", help="reuse an established bound without a new bf16 run")
    p.add_argument("--bound", type=float, default=0.01, help="established bound with --fp32-only")
    p.add_argument("--only", nargs="*", help="request names to run")
    p.add_argument("--max-length", type=int, help="the creator's per-sequence limit; omitted: its default")
    a = p.parse_args()
    if not math.isfinite(a.bound) or a.bound <= 0:
        p.error("--bound must be finite and positive")
    if a.max_length is not None and a.max_length <= 0:
        p.error("--max-length must be positive")
    model = Path(a.model)
    sys.path.insert(0, str(model / "source" / "src"))
    cases = json.loads(Path(a.requests).read_text())
    if a.only:
        if missing := set(a.only) - {case["name"] for case in cases}:
            p.error(f"unknown request names: {sorted(missing)}")
        cases = [case for case in cases if case["name"] in a.only]
    if not cases:
        p.error("no matching requests")
    if a.tokens_only:
        tokens_only(model, cases, a)
        return
    from pplx_decider.model import DEFAULT_MAX_LENGTH, Decider  # the creator's package, from the checkpoint
    from pplx_decider.protocol import LengthError

    decider = Decider(model, device="cpu", max_length=a.max_length or DEFAULT_MAX_LENGTH)
    found: dict[str, dict[str, Any]] = {}
    precisions = [(torch.float32, "probabilities")]
    if not a.fp32_only:
        precisions.append((torch.bfloat16, "probabilities_bf16"))
    for dtype, key in precisions:
        decider.backbone.to(dtype)
        decider.readout.to(dtype)
        for case in cases:
            name, request = case["name"], case["request"]
            try:
                plan = decider.encoder.plan(request, decider.max_length)
            except LengthError as error:
                found[name] = {"name": name, "request": request, "error": str(error)}
                print(f"{name}: refused ({error})", flush=True)
                continue
            logits = decider.logits([request])[0]
            entry = found.setdefault(name, {"name": name, "request": request})
            entry["prompts"] = [chunk.input_ids.tolist() for chunk in plan]
            entry["input_tokens"] = sum(chunk.length for chunk in plan)
            entry[key] = {
                q: ordered(request["questions"][q], (torch.tensor(v) / decider.temperature).softmax(-1).tolist())
                for q, v in logits.items()
            }
            print(f"{key} {name}: {entry['input_tokens']} tokens", flush=True)
    worst = (
        max(
            (
                abs(x - y)
                for e in found.values()
                if "probabilities" in e
                for q in e["probabilities"]
                for x, y in zip(e["probabilities"][q], e["probabilities_bf16"][q], strict=True)
            ),
            default=0.0,
        )
        if not a.fp32_only
        else None
    )
    reference = {
        "schema": "ippon-reference/1",
        "adapter": "perplexity/pplx-decider",
        "model": {"repo": a.repo, "revision": a.revision},
        "maker": {
            "code": "pplx_decider (source/src) reference path",
            "tokenizer": "transformers AutoProcessor",
            "device": "cpu",
            "dtype": "float32",
            "transformers": __import__("transformers").__version__,
            "escapes": list(decider.encoder._escapes),
            "questions": "one per sequence",
            "max_length": decider.max_length,
        },
        "bound": a.bound if a.fp32_only else round(max(0.01, float(worst) + 5e-4), 4),
        "tokens_only": False,
        "requests": list(found.values()),
    }
    Path(a.out).write_bytes(gzip.compress(dumps(reference).encode()))
    print(f"{a.out}: {len(found)} requests, bound {reference['bound']} (bf16 from fp32: {worst})")


def tokens_only(model: Path, cases: list[dict[str, Any]], a: argparse.Namespace) -> None:
    """The prompts as the creator's Decider builds them (its processor, as Decider.__init__ sets it up, and Encoder)."""
    from pplx_decider.model import DEFAULT_MAX_LENGTH
    from pplx_decider.protocol import Encoder, LengthError, load_json
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(model))
    processor.tokenizer.padding_side = "left"
    encoder = Encoder(processor, load_json(model / "decision_config.json")["codes"])
    limit = a.max_length or DEFAULT_MAX_LENGTH
    found = []
    for case in cases:
        try:
            plan = encoder.plan(case["request"], limit)
        except LengthError as error:
            found.append({"name": case["name"], "request": case["request"], "error": str(error)})
            print(f"{case['name']}: refused ({error})", flush=True)
            continue
        found.append(
            {
                "name": case["name"],
                "request": case["request"],
                "prompts": [chunk.input_ids.tolist() for chunk in plan],
                "input_tokens": sum(chunk.length for chunk in plan),
            }
        )
        print(f"{case['name']}: {found[-1]['input_tokens']} tokens", flush=True)
    reference = {
        "schema": "ippon-reference/1",
        "adapter": "perplexity/pplx-decider",
        "model": {"repo": a.repo, "revision": a.revision},
        "maker": {
            "code": "pplx_decider (source/src) Encoder",
            "tokenizer": "transformers AutoProcessor",
            "transformers": __import__("transformers").__version__,
            "escapes": list(encoder._escapes),
            "questions": "one per sequence",
            "max_length": limit,
        },
        "tokens_only": True,
        "requests": found,
    }
    Path(a.out).write_bytes(gzip.compress(dumps(reference).encode()))
    print(f"{a.out}: {len(found)} requests, prompts only")


if __name__ == "__main__":
    main()
