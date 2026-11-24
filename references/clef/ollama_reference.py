"""Clef's creator renders prompts; an isolated Ollama server supplies same-build probabilities.

Runs the explicitly supplied creator's joint_schema_model.py, never during library loading or tests. Use an
environment with torch, transformers and ippon's source on PYTHONPATH. No checkpoint conversion occurs.

    python references/clef/ollama_reference.py --model /path/to/clef-flash \
        --build ollama://clef-flash:9b-mxfp8 --endpoint http://127.0.0.1:11439 --out reference.json.gz
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

from make_reference import module
from transformers import AutoTokenizer

from ippon.ollama import open_build


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--build", required=True)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    build = open_build(args.build)
    maker_file = args.model / "joint_schema_model.py"
    maker = module(maker_file)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    questions = {
        "refund": {"type": "noul", "instructions": "Was the customer charged twice?"},
        "route": {
            "type": "choice",
            "instructions": "Choose a team.",
            "criteria": {"billing": "Payments", "support": "Faults"},
        },
        "urgency": {"type": "score", "instructions": "Rate urgency.", "criteria": ["Routine", "Soon", "Immediate"]},
    }

    def encode(body: dict[str, Any]) -> list[int]:
        return list(maker.encode_record(tokenizer, {"model": "clef-flash", **body}, max_length=16384).input_ids)

    cases = []
    for length in (128, 512, 2048, 4096):
        for group in (*questions, "multiple"):
            body = {
                "state": "A customer reports two completed charges for one order.",
                "questions": questions if group == "multiple" else {group: questions[group]},
            }
            minimum = len(encode(body))
            body["state"] += " detail" * max(0, length - minimum)
            tokens = encode(body)
            assert len(tokens) == max(length, minimum)
            cases.append(
                {"name": f"{group}-{length}", "request": body, "prompts": [tokens], "input_tokens": len(tokens)}
            )
    for name, state in (
        ("negative", "Only one charge is completed; the other is a receipt."),
        ("unicode", "Élodie: J'ai été débitée deux fois. 返金をお願いします。"),
    ):
        body = {"state": state, "questions": questions}
        tokens = encode(body)
        cases.append({"name": name, "request": body, "prompts": [tokens], "input_tokens": len(tokens)})
    with urllib.request.urlopen(args.endpoint + "/api/version", timeout=10) as response:
        runtime = json.load(response)
    for case in cases:
        request = urllib.request.Request(
            args.endpoint + "/v1/systemone",
            json.dumps({"model": args.build.removeprefix("ollama://"), **case["request"]}).encode(),
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            answer = json.load(response)
        probabilities = {}
        for qid, q in case["request"]["questions"].items():
            a = answer["answers"][qid]
            if q["type"] == "noul":
                probabilities[qid] = [a["noul"], 1 - a["noul"]]
            else:
                keys = q["criteria"] if q["type"] == "choice" else map(str, range(len(q["criteria"])))
                probabilities[qid] = [a["probabilities"][key] for key in keys]
        assert case["input_tokens"] == answer["usage"]["input_tokens"], (case["name"], answer["usage"])
        case["probabilities"] = probabilities
        print(case["name"], case["input_tokens"], probabilities, flush=True)
    result = {
        "schema": "ippon-reference/1",
        "reference_kind": "serving",
        "model": {"repo": args.build, "revision": build.identity},
        "maker": {
            "repo": "Cloudflare/clef-flash",
            "revision": "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3",
            "code_sha256": hashlib.sha256(maker_file.read_bytes()).hexdigest(),
        },
        "runtime": runtime,
        "bound": 0.01,
        "thresholds": [0.5, 0.7, 0.9],
        "notes": [
            "Requests whose schema alone exceeds the requested 128 tokens retain the full creator prompt; "
            "input_tokens records their actual length."
        ],
        "requests": cases,
    }
    args.out.write_bytes(gzip.compress(json.dumps(result, ensure_ascii=False).encode(), mtime=0))


if __name__ == "__main__":
    main()
