"""Fastino GLiDE's reference answers, from the creator's own `glide` package (shipped in each model repository), never
inside ippon:

    python -I make_reference.py --model ~/models/glide-2b --requests ../requests/standard.json \\
        --out glide-2b.json.gz [--tokens-only]

Run it in an environment with the creator's pins (requirements.txt: torch 2.14.0, transformers 5.17.0, tokenizers
0.23.2, safetensors 0.8.0, jinja2 3.1.6, numpy 2.5.3). It loads the checkpoint with `Glide.from_pretrained(root,
device="cpu", dtype=torch.float32, vision=False)`, takes every question's prompt from the package's `_prepare` (its
Prompter, which checks itself against a full chat-template render) tokenized as `decide_batch` tokenizes it, and scores
each prompt alone through `_logits` (the 255-code readout at the last token, with the 9B's recurrent depth) and
`_softmax` at the checkpoint's temperature, in fp32. Then again with the checkpoint loaded in bf16 on `--bf16-device`,
for the bound, the larger of 0.01 and the bf16 run's largest distance from fp32 plus 0.0005. A request the package
refuses (a choice given as a list of labels, which ippon accepts) is left out, with its reason under "maker"."left_out".
"""

from __future__ import annotations

import argparse
import gc
import gzip
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
    p.add_argument("--model", required=True, type=Path, help="a downloaded GLiDE checkpoint (its directory)")
    p.add_argument("--requests", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tokens-only", action="store_true")
    p.add_argument("--bf16-device", default="mps", help="where the bf16 run for the bound happens")
    p.add_argument("--only", nargs="*", help="request names to run")
    args = p.parse_args()
    root = args.model.expanduser().resolve()
    sys.path.insert(0, str(root))
    import torch
    import transformers
    from glide import Glide, RequestError
    from glide.model import _softmax

    model = Glide.from_pretrained(str(root), device="cpu", dtype=torch.float32, vision=False)
    cases = json.loads(args.requests.read_text())
    if args.only:
        if missing := set(args.only) - {c["name"] for c in cases}:
            p.error(f"unknown request names: {sorted(missing)}")
        cases = [c for c in cases if c["name"] in args.only]

    out, left_out = [], []
    for case in cases:
        try:
            entries, _ = model._prepare(case["request"])
        except RequestError as refusal:
            left_out.append({"name": case["name"], "why": str(refusal)})
            continue
        prompts = model.tokenizer([text for *_, text in entries], add_special_tokens=False)["input_ids"]
        if any(len(ids) + 8 > model.max_len for ids in prompts):
            left_out.append({"name": case["name"], "why": "prompt exceeds the maximum context length"})
            continue
        out.append(
            {
                "name": case["name"],
                "request": case["request"],
                "prompts": [list(ids) for ids in prompts],
                "input_tokens": sum(len(ids) for ids in prompts),
            }
        )
    if not out:
        p.error("the package refused every request")

    def score(key: str) -> None:
        for entry in out:
            entries, _ = model._prepare(entry["request"])
            got = {}
            for (question, _, keys, _), ids in zip(entries, entry["prompts"], strict=True):
                row = model._logits([ids], [None])[0]  # one prompt a pass, so no padding
                got[question] = [float(x) for x in _softmax(row, len(keys), model.temperature)]
            entry[key] = got
            print(f"{key} {entry['name']}: {entry['input_tokens']} tokens", flush=True)

    bound = 0.01
    if not args.tokens_only:
        with torch.inference_mode():
            score("probabilities")
        del model
        gc.collect()
        model = Glide.from_pretrained(str(root), device=args.bf16_device, dtype=torch.bfloat16, vision=False)
        with torch.inference_mode():
            score("probabilities_bf16")
        distance = max(
            abs(a - b)
            for e in out
            for q in e["probabilities"]
            for a, b in zip(e["probabilities"][q], e["probabilities_bf16"][q], strict=True)
        )
        if not math.isfinite(distance):
            raise SystemExit("non-finite probabilities")
        bound = max(0.01, round(distance + 5e-4, 4))
    config = json.loads((root / "decision_config.json").read_text())
    reference: dict[str, Any] = {
        "schema": "ippon-reference/1",
        "adapter": "fastino/glide",
        "model": {"directory": root.name, "repo": repo(root), "revision": revision(root)},
        "maker": {
            "code": "glide (Glide._prepare, Glide._logits, _softmax)",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "temperature": float(config.get("temperature", 1.0)),
            "midloop": config.get("midloop"),
            "left_out": left_out,
            "device": "cpu",
            "dtype": "float32",
            "bf16_device": None if args.tokens_only else args.bf16_device,
        },
        "bound": bound,
        "tokens_only": args.tokens_only,
        "requests": out,
    }
    data = dumps(reference, ensure_ascii=False).encode()
    args.out.write_bytes(gzip.compress(data) if args.out.suffix == ".gz" else data)
    print(f"{len(out)} requests -> {args.out} (bound {bound}{', tokens only' if args.tokens_only else ''})")


def repo(root: Path) -> str | None:
    """The Hub repository the model card links its licence to."""
    card = root / "README.md"
    for line in card.read_text().splitlines() if card.is_file() else ():
        if line.startswith("license_link: https://huggingface.co/"):
            return "/".join(line.split("huggingface.co/")[1].split("/")[:2])
    return None


def revision(root: Path) -> str | None:
    """The Hub commit a `hf download --local-dir` folder holds (its download metadata's first line)."""
    found = root / ".cache/huggingface/download/config.json.metadata"
    return found.read_text().splitlines()[0] if found.is_file() else None


if __name__ == "__main__":
    main()
