"""Clef-Flash reference answers, from the checkpoint's own joint_schema_model.py, never inside ippon:

    python -I make_reference.py --model path/to/clef-flash --requests ../requests/standard.json \
        --out clef-flash.tokens.json.gz [--tokens-only]
    python -I make_reference.py --head-fixture ../../tests/fixtures/clef_head    # the head on a tiny random input

Run it in a venv with torch and transformers. Prompt tokens come from encode_record (with the checkpoint's
tokenizer), after the checks systemone makes; the readouts are each question's instruction span and option
spans. Without --tokens-only the model is load_release_model(path, device="cpu", dtype=float32) and each
question's probabilities are the softmax of its own logits, unrounded, in the request's option order (a noul as
true, false). --head-fixture writes a tiny JointSchemaHead with random weights, a random input and its logits.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("clef_joint_schema_model", path)
    assert spec is not None
    assert spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded  # its dataclasses look their module up
    spec.loader.exec_module(loaded)
    return loaded


def head_fixture(clef: Any, out: Path) -> None:
    import torch
    from safetensors.torch import save_file

    torch.manual_seed(0)
    config = {"hidden_size": 24, "width": 16, "routing_layers": 2, "layers": 2, "heads": 2, "feedforward": 32}
    head = clef.JointSchemaHead(**config).eval()
    with torch.no_grad():
        for p in head.parameters():
            p.copy_(torch.randn_like(p) * 0.3 + (1.0 if p.ndim == 1 and p.shape[0] in (16, 24) else 0.0))
        head.prior_logit_scale.fill_(0.7)
        head.joint_logit_scale.fill_(1.3)
        head.residual_gate.fill_(0.4)
    length, vocab = 30, 50
    hidden = torch.randn(1, length, 24)
    ids = torch.randint(0, vocab, (1, length))
    output = torch.randn(vocab, 24)
    questions = [
        clef.EncodedQuestion("a", 0, (2, 5), ((6, 8), (9, 12)), ("true", "false")),
        clef.EncodedQuestion("b", 1, (13, 15), ((16, 18), (19, 20), (21, 25)), ("x", "y", "z")),
    ]
    record = clef.EncodedRecord(tuple(ids[0].tolist()), tuple(questions), "fixture")
    with torch.no_grad():
        logits = head(hidden, ids, torch.ones(1, length, dtype=torch.long), [record], output)[0]
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in head.state_dict().items()}, str(out / "head.safetensors"))
    (out / "case.json").write_text(
        json.dumps(
            {
                "config": config,
                "hidden": hidden[0].tolist(),
                "ids": ids[0].tolist(),
                "output": output.tolist(),
                "spans": [
                    {"question": q.question_span, "options": q.option_spans, "type": q.question_type} for q in questions
                ],
                "logits": [z.tolist() for z in logits],
            }
        )
    )
    print(f"head fixture -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", type=Path, default=Path("~/models/clef-flash"))
    p.add_argument("--requests", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--tokens-only", action="store_true")
    p.add_argument("--head-fixture", type=Path)
    args = p.parse_args()
    root = args.model.expanduser()
    clef = module(root / "joint_schema_model.py")
    if args.head_fixture:
        head_fixture(clef, args.head_fixture)
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(root))
    model = None
    if not args.tokens_only:
        import torch

        model, processor = clef.load_release_model(root, device="cpu", dtype=torch.float32)
        tokenizer = processor.tokenizer
    out = []
    for case in json.loads(args.requests.read_text()):
        request = {"model": "clef-flash", **case["request"]}
        request.setdefault("state", None)
        try:
            for qid, q in request["questions"].items():  # systemone's checks
                if q.get("type") not in clef.QUESTION_TYPES or (q["type"] != "noul" and not q.get("criteria")):
                    raise ValueError(qid)
            encoded = clef.encode_record(tokenizer, request, max_length=16384)
        except (ValueError, AttributeError, KeyError, TypeError):  # refused by the creator's code
            continue
        entry: dict[str, Any] = {
            "name": case["name"],
            "request": case["request"],
            "prompts": [list(encoded.input_ids)],
            "readouts": [
                [list(q.question_span), [list(s) for s in q.option_spans], list(q.option_ids)]
                for q in encoded.questions
            ],
            "input_tokens": len(encoded.input_ids),
        }
        if model is not None:
            import torch

            with torch.inference_mode():
                logits = model(clef.collate_records([encoded], tokenizer.pad_token_id, torch.device("cpu")))[0]
            probabilities = {}
            for q, z in zip(encoded.questions, logits, strict=True):
                p_by = dict(zip(q.option_ids, z.float().softmax(-1).tolist(), strict=True))
                given = request["questions"][q.question_id]
                order = (
                    ["true", "false"]
                    if given["type"] == "noul"
                    else (
                        [str(k) for k in given["criteria"]]
                        if given["type"] == "choice"
                        else [str(i) for i in range(len(given["criteria"]))]
                    )
                )
                probabilities[q.question_id] = [p_by[o] for o in order]
            entry["probabilities"] = probabilities
        out.append(entry)
    reference = {
        "schema": "ippon-reference/1",
        "adapter": "cloudflare/clef",
        "model": {"directory": root.name},
        "maker": {"code": "joint_schema_model.py", "device": "cpu", "dtype": "float32"},
        "bound": 0.01,
        "tokens_only": args.tokens_only,
        "requests": out,
    }
    data = json.dumps(reference, ensure_ascii=False).encode()
    args.out.write_bytes(gzip.compress(data) if args.out.suffix == ".gz" else data)
    print(f"{len(out)} requests -> {args.out}{' (tokens only)' if args.tokens_only else ''}")


if __name__ == "__main__":
    main()
