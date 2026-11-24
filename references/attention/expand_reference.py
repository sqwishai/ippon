"""Exercise single, equal-length batched, and mixed-length batched execution of independent questions.

D3's creator runs each question independently (the reference runner uses batch_size=1). Flash v1.2's reference
explicitly records one question per sequence. Their per-question prompts and probabilities can therefore also
check single-question requests and two copies of the same question. No probabilities are recomputed or changed.
This does not apply to joint-question models. Preserve the original mixed requests, and record the source hash.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from runpy import run_path

# Load just the stdlib serializer without importing ippon or changing runtime module discovery.
dumps = run_path(str(Path(__file__).resolve().parents[2] / "src/ippon/_portable.py"))["dumps"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.reference.read_bytes()
    reference = json.loads(gzip.decompress(raw) if args.reference.suffix == ".gz" else raw)
    count = len(reference.get("requests", []))
    if reference.get("complete") is False or any(
        reference.get(key, count) != count for key in ("planned_requests", "completed_requests")
    ):
        parser.error("the creator reference is incomplete")
    if reference.get("adapter") != "vllm-sr/d3" and not (
        reference.get("adapter") == "perplexity/pplx-decider"
        and reference.get("maker", {}).get("questions") == "one per sequence"
    ):
        parser.error("only independent-question D3 or explicitly one-per-sequence Perplexity references qualify")
    if args.out.exists():
        parser.error("use a new output path")
    expanded = []
    for case in reference["requests"]:
        body, prompts, probabilities = case["request"], case["prompts"], case["probabilities"]
        questions = body["questions"]
        if set(body) - {"state", "questions"} or len(prompts) != len(questions) or set(questions) != set(probabilities):
            parser.error(f"{case['name']}: expected one text prompt and probability vector per question")
        if case["input_tokens"] != sum(map(len, prompts)):
            parser.error(f"{case['name']}: additional passes or tokens need their own references")
        expanded.append(case)
        for (key, question), prompt in zip(questions.items(), prompts, strict=True):
            expanded.append(
                {
                    "name": f"{case['name']}/single-{key}",
                    "request": {**body, "questions": {key: question}},
                    "prompts": [prompt],
                    "probabilities": {key: probabilities[key]},
                    "input_tokens": len(prompt),
                }
            )
        first = next(iter(questions))
        repeated = {f"copy-{i}": questions[first] for i in range(2)}
        expanded.append(
            {
                "name": f"{case['name']}/equal-pair",
                "request": {**body, "questions": repeated},
                "prompts": [prompts[0], prompts[0]],
                "probabilities": dict.fromkeys(repeated, probabilities[first]),
                "input_tokens": 2 * len(prompts[0]),
            }
        )
    if not expanded or len({c["name"] for c in expanded}) != len(expanded):
        parser.error("the expanded requests must be nonempty with unique names")
    reference.update(
        requests=expanded,
        planned_requests=len(expanded),
        completed_requests=len(expanded),
        complete=True,
        derived_from={
            "file": args.reference.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "operation": "independent per-question creator outputs regrouped; original requests retained",
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = dumps(reference, ensure_ascii=False, allow_nan=False).encode()
    args.out.write_bytes(gzip.compress(result, mtime=0) if args.out.suffix == ".gz" else result)
    print(f"{len(expanded)} cases -> {args.out}")


if __name__ == "__main__":
    main()
