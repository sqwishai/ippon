"""A `ippon bench --compare` arm: Fastino's GLiDE through its creator's own `glide` package, shipped in each
checkpoint folder, as its card documents: model = Glide.from_pretrained(path); model.decide(state, questions).
It runs on the Mac's GPU (MPS) in the package's default bf16. This runs the glide/ code the model directory came
with, so read it before running it.

    PYTHON -I glide_package.py MODEL_DIR REQUESTS.json OUT.json    (PYTHON: an environment with the checkpoint's
                                                                    requirements.txt, triton and its kernels left out)

IPPON_ARM_DTYPE=float32 loads the checkpoint in fp32, as references/glide/make_reference.py does on the CPU.
"""

import json
import os
import sys
import time
from importlib.metadata import version

import torch


def main() -> None:
    model_dir, requests, out = sys.argv[1:4]

    sys.path.insert(0, model_dir)
    from glide import Glide  # the model repository's own package

    dtype = os.environ.get("IPPON_ARM_DTYPE")
    model = Glide.from_pretrained(model_dir, device="mps", dtype=getattr(torch, dtype) if dtype else None)
    results = []
    with open(requests, encoding="utf-8") as f:
        items = json.load(f)
    for item in items:
        if item["gap"]:
            time.sleep(item["gap"])
        body = item["body"]
        # the package takes a choice's criteria only as {label: description}; a list's labels have none, which
        # ippon shows the same way
        questions = {
            k: {**q, "criteria": dict.fromkeys(map(str, q["criteria"]))}
            if q.get("type") == "choice" and isinstance(q.get("criteria"), list)
            else q
            for k, q in body["questions"].items()
        }
        start = time.perf_counter()
        response = model.decide(body["state"], questions)
        ms = (time.perf_counter() - start) * 1000
        answers = {}
        for name, a in response["answers"].items():
            if a.get("type") == "noul" or "noul" in a:
                answers[name] = {"yes": float(a["noul"]), "no": 1 - float(a["noul"])}
            else:
                answers[name] = {str(k): float(v) for k, v in a["probabilities"].items()}
        results.append({"ms": ms, "answers": answers, "input_tokens": response.get("usage", {}).get("input_tokens")})
    with open(out, "w", encoding="utf-8") as f:
        runtime = f"the repository's glide package + torch {version('torch')} (mps, {dtype or 'bfloat16'})"
        json.dump({"runtime": runtime, "dtype": dtype, "results": results}, f)


if __name__ == "__main__":
    main()
