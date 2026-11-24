"""Kev reference answers, from the creator's own code (github.com/jaredpalmer/kev), never inside ippon:

    python -I make_reference.py --model ~/models/jaredpalmer-kev-4b \\
        --requests ../requests/standard.json --out kev-4b.json.gz

Run it in a venv synced from Kev's uv.lock (`uv sync --frozen --extra serve`: torch 2.8.0, transformers 5.17.0, peft
0.21.0). The checkpoint loads as kev.checkpoint.load loads it (the LoRA merged into the base at the revision head.pt
pins, the head with the temperature it carries), in fp32 on the CPU: the exact path Kev's reported numbers use
(LoadOptions' default). Each request is mapped as the server maps it (api.SystemOneRequest, api.to_record, model.admit)
and scored by model.probs; its prompt tokens are each question's causal row (model.rows_of: the state, then the
question's branch), its probabilities unrounded in ippon's option order (a noul as true, false). A request the server
refuses is left out with its reason.

The bound: the same code in bf16 on the Mac's GPU (torch, "mps"), the creator's run-to-run distance across devices and
precisions; the larger of 0.01 and that distance plus 0.0005, rounded to four decimals.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any


def run(tok: Any, model: Any, requests: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from kev.api import SystemOneRequest, to_record
    from kev.model import admit, rows_of

    out, refused = [], []
    for case in requests:
        request = case["request"]
        try:
            rec, meta = to_record(SystemOneRequest(state=request.get("state"), questions=request["questions"]))
            enc = admit(model, tok, rec)
        except ValueError as error:
            refused.append({"name": case["name"], "reason": f"{type(error).__name__}: {error}"})
            continue
        state_ids, _, rows = rows_of(enc)
        probabilities = model.probs(enc)
        entry: dict[str, Any] = {"name": case["name"], "request": request, "prompts": [], "probabilities": {}}
        for m, row, p in zip(meta, rows, probabilities, strict=True):
            entry["prompts"].append([int(t) for t in [*state_ids, *row["ids"]]])
            p = [float(x) for x in (p.tolist() if hasattr(p, "tolist") else p)]
            entry["probabilities"][m["id"]] = [p[1], p[0]] if m["type"] == "noul" else p
        entry["input_tokens"] = len(enc["ids"])
        out.append(entry)
    return out, refused


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--requests", required=True, type=Path, nargs="+")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    import torch
    from kev.checkpoint import LoadOptions, load

    cases = [c for path in args.requests for c in json.loads(path.read_text())]
    root = args.model.expanduser()
    with torch.no_grad():
        tok, model = load(str(root), "cpu", LoadOptions(dtype=torch.float32, backend="torch"))
        model.eval()
        out, refused = run(tok, model, cases)
        del model
        tok, model = load(str(root), "mps", LoadOptions(dtype=torch.bfloat16, backend="torch"))
        model.eval()
        mps, _ = run(tok, model, [{"name": e["name"], "request": e["request"]} for e in out])
    distance = 0.0
    for a, b in zip(out, mps, strict=True):
        for qid, got in b["probabilities"].items():
            a.setdefault("probabilities_mps_bf16", {})[qid] = got
            distance = max(distance, *(abs(x - y) for x, y in zip(got, a["probabilities"][qid], strict=True)))
    bound = max(0.01, round(distance + 5e-4, 4))
    reference = {
        "schema": "ippon-reference/1",
        "adapter": "jaredpalmer/kev",
        "model": {"directory": root.name},
        "maker": {"code": "kev (github.com/jaredpalmer/kev)", "device": "cpu", "dtype": "float32"},
        "bound": bound,
        "tokens_only": False,
        "requests": out,
        "refused": refused,
    }
    data = json.dumps(reference, ensure_ascii=False).encode()
    args.out.write_bytes(gzip.compress(data) if args.out.suffix == ".gz" else data)
    print(
        f"{len(out)} requests ({len(refused)} refused) -> {args.out} (bound {bound}, mps bf16 distance {distance:.5f})"
    )


if __name__ == "__main__":
    main()
