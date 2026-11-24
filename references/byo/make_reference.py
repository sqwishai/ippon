"""Reference answers for a bring-your-own model (a `*ForSequenceClassification` checkpoint, or a causal LM with a
label-token `ippon.json`), from transformers itself: the file `ippon verify` checks against.

Run it in an environment with torch and transformers, never inside ippon:

    python -I make_reference.py --model DIR --requests REQUESTS.json --out NAME.json.gz [--repo ID --revision SHA]

The model directory's `ippon.json` (if any) says how a request becomes the model's input; this script reads it
on its own, independently of ippon's code: the input text (and a text pair) filled from {state} and
{instructions}, tokenized as `tokenizer(text, pair)` does; a classifier's logits from transformers'
AutoModelForSequenceClassification, softmaxed (sigmoid for a multi-label or one-logit model) over the labels a
question asks about; a label-token model's last-position logits from AutoModelForCausalLM at each option's answer
token. Weights from safetensors only, no remote code; fp32 on the CPU, then bf16 for the bound (the larger of 0.01
and bf16's distance from fp32 on these requests).
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import torch
import transformers


def text_of(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def options(q: dict[str, Any]) -> list[str]:
    criteria = q.get("criteria")
    if q["type"] == "noul":
        return ["true", "false"]
    if isinstance(criteria, list):
        return [str(i) for i in range(len(criteria))] if q["type"] == "score" else list(criteria)
    return list(criteria)


class Classifier:
    def __init__(self, path: Path, settings: dict[str, Any], dtype: torch.dtype) -> None:
        self.tok = transformers.AutoTokenizer.from_pretrained(path)
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            path, dtype=dtype, use_safetensors=True
        ).eval()
        config = self.model.config
        self.labels = [config.id2label[i] for i in range(config.num_labels)]
        self.independent = config.problem_type == "multi_label_classification" or config.num_labels == 1
        self.settings = settings

    def prompts(self, request: dict[str, Any]) -> tuple[list[list[int]], dict[str, int]]:
        """Each distinct input's ids, in the questions' order, and each question's input."""
        s, state = self.settings, text_of(request.get("state"))
        seen: dict[tuple[str, str | None], int] = {}
        self.encodings: list[Any] = []
        out, which = [], {}
        for qid, q in request["questions"].items():
            fields = {"state": state, "instructions": text_of(q.get("instructions"))}
            key = (s.get("text", "{state}").format(**fields), s["pair"].format(**fields) if s.get("pair") else None)
            if key not in seen:
                seen[key] = len(out)
                encoded = self.tok(key[0], key[1], return_tensors="pt")  # token types included, as tokenizers give
                self.encodings.append(encoded)
                out.append([int(t) for t in encoded["input_ids"][0]])
            which[qid] = seen[key]
        return out, which

    def answer(self, request: dict[str, Any]) -> tuple[list[list[int]], dict[str, list[float]]]:
        prompts, which = self.prompts(request)
        with torch.no_grad():
            logits = [self.model(**encoded).logits[0].float() for encoded in self.encodings]
        out = {}
        for qid, q in request["questions"].items():
            z = logits[which[qid]]
            if q["type"] == "noul":
                if len(self.labels) == 1:
                    label = 0
                else:
                    named = q.get("instructions") if q.get("instructions") in self.labels else None
                    label = self.labels.index(named or self.settings["true_label"])
                p = torch.sigmoid(z[label]) if self.independent else torch.softmax(z, -1)[label]
                out[qid] = [float(p), float(1 - p)]
            else:
                if q["type"] == "score":
                    rows = [self.labels.index(level) for level in self.settings.get("levels", self.labels)]
                else:
                    rows = [self.labels.index(label) for label in options(q)]
                out[qid] = [float(x) for x in torch.softmax(z[rows], -1)]
        return prompts, out


class LabelTokens:
    def __init__(self, path: Path, settings: dict[str, Any], dtype: torch.dtype) -> None:
        self.tok = transformers.AutoTokenizer.from_pretrained(path)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=dtype, use_safetensors=True).eval()
        self.settings = settings

    def answer(self, request: dict[str, Any]) -> tuple[list[list[int]], dict[str, list[float]]]:
        s, state = self.settings, text_of(request.get("state"))
        prompts, out = [], {}
        for qid, q in request["questions"].items():
            shown = "\n".join(
                s.get("option", "{label}: {description}").format(label=k, description=text_of(v), index=i)
                for i, (k, v) in enumerate(
                    (q.get("criteria") or {}).items() if isinstance(q.get("criteria"), dict) else []
                )
            )
            text = s["prompt"].format(state=state, instructions=text_of(q.get("instructions")), options=shown)
            ids = list(self.tok(text)["input_ids"])
            prompts.append(ids)
            if q["type"] != "noul":
                raise SystemExit("this reference script covers yes/no questions for label-token models")
            answers = [s["noul"]["true"], s["noul"]["false"]]
            first = [self.tok(a, add_special_tokens=False)["input_ids"][0] for a in answers]
            with torch.no_grad():
                z = self.model(input_ids=torch.tensor([ids])).logits[0, -1].float()
            p = torch.softmax(z[first], -1)
            out[qid] = [float(x) for x in p]
        return prompts, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo")
    ap.add_argument("--revision")
    args = ap.parse_args()
    path = Path(args.model)
    settings_file = path / "ippon.json"
    settings = json.loads(settings_file.read_text()) if settings_file.exists() else {}
    kind = LabelTokens if settings.get("format") == "label_tokens" else Classifier
    cases = json.loads(Path(args.requests).read_text())
    fp32, bf16 = kind(path, settings, torch.float32), kind(path, settings, torch.bfloat16)
    requests, worst = [], 0.0
    for case in cases:
        prompts, probabilities = fp32.answer(case["request"])
        _, half = bf16.answer(case["request"])
        worst = max(
            [worst] + [abs(a - b) for q in probabilities for a, b in zip(probabilities[q], half[q], strict=True)]
        )
        requests.append(
            {
                "name": case["name"],
                "request": case["request"],
                "prompts": prompts,
                "input_tokens": sum(len(p) for p in prompts),
                "probabilities": probabilities,
                "probabilities_bf16": half,
            }
        )
    reference = {
        "schema": "ippon-reference/1",
        "adapter": "generic/label-tokens" if kind is LabelTokens else "generic/classifier",
        "model": {"repo": args.repo, "revision": args.revision},
        "maker": {
            "code": f"transformers {transformers.__version__}",
            "tokenizer": "transformers",
            "device": "cpu",
            "dtype": "float32",
        },
        "bound": round(max(0.01, worst), 4),
        "tokens_only": False,
        "requests": requests,
    }
    Path(args.out).write_bytes(gzip.compress(json.dumps(reference).encode()))
    print(f"{args.out}: {len(requests)} requests, bound {reference['bound']}")


if __name__ == "__main__":
    main()
