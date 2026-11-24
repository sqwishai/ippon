"""Jeff reference answers, from the creator's own serving code (github.com/j-geuter/Jeff v1.3.0), never inside ippon:

    python -I make_reference.py --model ~/models/jeff-1.1-medium --requests ../requests/standard.json \\
        --out jeff-1.1-medium.json.gz

Run it in a venv with the release's requirements.txt pins (torch 2.13.0, transformers 5.17.0, safetensors 0.8.0,
huggingface_hub 1.32.0, accelerate 1.15.0) and the jeff package installed from the v1.3.0 checkout. The model loads as
jeff.model.JeffModel loads it (bf16 weights, the fp32 head, bidirectional full layers, the serving rule committed), on
the Mac's GPU ("mps"), its production dtype: at 12B its fp32 weights would need the whole of a 48 GB Mac, so the
reference is the creator's bf16 (the user's choice, 4 November 2026). Each question is a row as the server builds it
(JeffModel.row), its prompt tokens from Decider.prepare, its probabilities from Decider.decide, unrounded, in the
request's option order (a noul as true, false: the row's own order), and the served answer beside them. A request the
creator's code refuses, or one with a choice past 16 options (answered in rounds, which verify's one-pass prompts don't
cover), is left out with its reason.

The bound: the same code again in bf16 on the CPU, the creator's run-to-run distance across devices; the larger of 0.01
and that distance plus 0.0005, rounded to four decimals.

With --dtype float32 (a model whose fp32 weights fit, Jeff-1.1-Qwen-4B) the reference is the creator's code in fp32 on
the CPU, jeff.model.load_causal_model loading the weights in fp32, and the bound the distance of its production bf16 on
the Mac's GPU from it, the way the other references are made:

    python -I make_reference.py --model ~/models/jgeuter-jeff-1.1-qwen-4b \\
        --requests ../requests/standard.json --out jeff-1.1-qwen-4b.json.gz --dtype float32
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any


def run(model: Any, requests: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out, refused = [], []
    for case in requests:
        request = case["request"]
        if any(len(q.get("criteria") or {}) > 16 for q in request["questions"].values()):
            refused.append({"name": case["name"], "reason": "a choice past 16 options: answered in rounds"})
            continue  # verify compares one pass's prompts; rounds.py's product is tested in tests/test_jeff.py
        try:
            entry: dict[str, Any] = {"name": case["name"], "request": request, "prompts": [], "probabilities": {}}
            answers, tokens = {}, 0
            model.decider.start_request()
            for key, question in request["questions"].items():
                row = model.row(request["state"], question, key)
                entry["prompts"].append(model.decider.prepare(row)[1])
                probs, _, n, _ = model.decider.decide(row)
                entry["probabilities"][key] = [float(p) for p in probs]
                answers[key] = model.answer(question["type"], [o["id"] for o in row["options"]], probs)
                tokens += n
        except (KeyError, TypeError, ValueError) as error:
            refused.append({"name": case["name"], "reason": f"{type(error).__name__}: {error}"})
            continue
        entry["input_tokens"] = tokens
        entry["response"] = {"answers": answers}
        out.append(entry)
    return out, refused


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--requests", required=True, type=Path, nargs="+")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16", help="the reference's precision")
    args = p.parse_args()

    import functools

    import jeff
    import jeff.model
    import torch
    from jeff.model import JeffModel

    cases = [c for path in args.requests for c in json.loads(path.read_text())]
    root = args.model.expanduser()
    if args.dtype == "float32":
        production = jeff.model.load_causal_model
        jeff.model.load_causal_model = functools.partial(production, dtype="float32")
        out, refused = run(JeffModel(str(root), device="cpu"), cases)
        jeff.model.load_causal_model = production
        cpu, _ = run(JeffModel(str(root), device="mps"), [{"name": e["name"], "request": e["request"]} for e in out])
    else:
        model = JeffModel(str(root), device="mps")
        out, refused = run(model, cases)
        del model
        torch.mps.empty_cache()
        cpu, _ = run(JeffModel(str(root), device="cpu"), [{"name": e["name"], "request": e["request"]} for e in out])
    distance = 0.0
    for a, b in zip(out, cpu, strict=True):
        for qid, got in b["probabilities"].items():
            a.setdefault("probabilities_cpu", {})[qid] = got
            distance = max(distance, *(abs(x - y) for x, y in zip(got, a["probabilities"][qid], strict=True)))
    bound = max(0.01, round(distance + 5e-4, 4))
    reference = {
        "schema": "ippon-reference/1",
        "adapter": "jgeuter/jeff",
        "model": {"directory": root.name},
        "maker": {
            "code": f"jeff {getattr(jeff, '__version__', 'v1.4.0')}",
            "device": "cpu" if args.dtype == "float32" else "mps",
            "dtype": args.dtype,
        },
        "bound": bound,
        "tokens_only": False,
        "requests": out,
        "refused": refused,
    }
    data = json.dumps(reference, ensure_ascii=False).encode()
    args.out.write_bytes(gzip.compress(data) if args.out.suffix == ".gz" else data)
    print(f"{len(out)} requests ({len(refused)} refused) -> {args.out} (bound {bound}, cpu distance {distance:.5f})")


if __name__ == "__main__":
    main()
