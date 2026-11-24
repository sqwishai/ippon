"""A `ippon bench --compare` arm for bring-your-own models: transformers' own model and head on the Mac's GPU
(PyTorch's MPS), the way most people run a fine-tuned classifier or reranker on a Mac. It reads the model's
`ippon.json` on its own (as references/byo/make_reference.py does, independently of ippon's code): the
input texts from the templates, tokenized as `tokenizer(text, pair)` gives them; a classifier's logits from
AutoModelForSequenceClassification, a label-token model's last-position logits from AutoModelForCausalLM; each
request's inputs in one padded batch. Weights from safetensors only, no remote code, fp16.

    PYTHON -I transformers_byo.py MODEL_DIR REQUESTS.json OUT.json   (an environment with torch and transformers)
"""

from __future__ import annotations

import json
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
import transformers

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16


def text_of(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


class Arm:
    def __init__(self, path: Path) -> None:
        file = path / "ippon.json"
        self.settings = json.loads(file.read_text()) if file.exists() else {}
        self.label_tokens = self.settings.get("format") == "label_tokens"
        self.tok = transformers.AutoTokenizer.from_pretrained(path)
        kind = (
            transformers.AutoModelForCausalLM if self.label_tokens else transformers.AutoModelForSequenceClassification
        )
        self.model = kind.from_pretrained(path, dtype=DTYPE, use_safetensors=True).to(DEVICE).eval()
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        if not self.label_tokens:
            config = self.model.config
            self.labels = [config.id2label[i] for i in range(config.num_labels)]
            self.independent = config.problem_type == "multi_label_classification" or config.num_labels == 1

    def inputs(self, request: dict[str, Any]) -> tuple[list[tuple[str, str | None]], dict[str, int]]:
        s, state = self.settings, text_of(request.get("state"))
        texts: list[tuple[str, str | None]] = []
        which = {}
        for qid, q in request["questions"].items():
            fields = {"state": state, "instructions": text_of(q.get("instructions")), "options": ""}
            if self.label_tokens:
                key: tuple[str, str | None] = (s["prompt"].format(**fields), None)
            else:
                key = (s.get("text", "{state}").format(**fields), s["pair"].format(**fields) if s.get("pair") else None)
            if key not in texts:
                texts.append(key)
            which[qid] = texts.index(key)
        return texts, which

    def answer(self, request: dict[str, Any]) -> dict[str, dict[str, float]]:
        texts, which = self.inputs(request)
        firsts = [t for t, _ in texts]
        pairs = [p for _, p in texts]
        batch = self.tok(
            firsts, pairs if any(p is not None for p in pairs) else None, padding=True, return_tensors="pt"
        )
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            out = self.model(**batch).logits.float()
            if self.label_tokens:
                last = batch["attention_mask"].sum(-1) - 1
                out = out[torch.arange(out.shape[0], device=out.device), last]
            logits = out.cpu()
        answers = {}
        for qid, q in request["questions"].items():
            z = logits[which[qid]]
            if self.label_tokens:
                if q["type"] != "noul":
                    raise SystemExit("this arm answers yes/no questions for label-token models")
                ids = [
                    self.tok(t, add_special_tokens=False)["input_ids"][0]
                    for t in (self.settings["noul"]["true"], self.settings["noul"]["false"])
                ]
                p = float(torch.softmax(z[ids], -1)[0])
                answers[qid] = {"yes": p, "no": 1 - p}
            elif q["type"] == "noul":
                if len(self.labels) == 1:
                    label = 0
                else:
                    named = q.get("instructions") if q.get("instructions") in self.labels else None
                    label = self.labels.index(named or self.settings["true_label"])
                p = float(torch.sigmoid(z[label]) if self.independent else torch.softmax(z, -1)[label])
                answers[qid] = {"yes": p, "no": 1 - p}
            else:
                criteria = q["criteria"]
                options = list(criteria) if isinstance(criteria, (list, dict)) else []
                rows = (
                    [self.labels.index(o) for o in options] if q["type"] == "choice" else list(range(len(self.labels)))
                )
                probs = torch.softmax(z[rows], -1)
                keys = options if q["type"] == "choice" else [str(i) for i in range(len(rows))]
                answers[qid] = {k: float(v) for k, v in zip(keys, probs, strict=True)}
        return answers


def main() -> None:
    model_dir, requests, out = sys.argv[1:4]
    arm = Arm(Path(model_dir))
    with open(requests, encoding="utf-8") as f:
        items = json.load(f)
    results = []
    for item in items:
        if item["gap"]:
            time.sleep(item["gap"])
        start = time.perf_counter()
        answered = arm.answer(item["body"])
        ms = (time.perf_counter() - start) * 1000
        results.append({"ms": ms, "answers": answered, "input_tokens": None})
    runtime = f"transformers {version('transformers')} on {DEVICE} (torch {version('torch')}), fp16"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"runtime": runtime, "results": results}, f)


if __name__ == "__main__":
    main()
