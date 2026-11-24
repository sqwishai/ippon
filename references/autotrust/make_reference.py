"""AutoTrust-Decider-Mini-9B reference answers, from the creator's own System 1 code (the model card's "Plain
transformers + peft" example, at the revision below), never inside ippon:

    python -I make_reference.py \\
        --model ~/.cache/huggingface/hub/models--autotrust--AutoTrust-Decider-Mini-9B/snapshots/REVISION \\
        --requests ../requests/autotrust.json --out decider-mini-9b.json.gz

Run it in a venv with torch 2.8.0, transformers 5.17.0 and peft 0.21.0. The checkpoint loads as the card loads it
(Qwen3.5-9B's weights in the repository, the LoRA in adapter/ merged in, head.safetensors, judge_config.json's slot
ranges and calibration.json's per-kind temperatures), in fp32 on the CPU, and `decide` below is the card's, with its
device and dtype the run's. A System One request becomes the card's (kind, state, question, options) calls:

- a state as the creator's server writes it (vl/serve_decide.py, `_parts`): text as it is, a JSON object as
  json.dumps(ensure_ascii=False), a list's items one after the other (text as it is, anything else as JSON);
- a noul: kind "noul", options ["false", "true"]; a score: kind "score", options "0" to "5", for a score of six levels
  only (the card's validation); a choice: kind "choice", each option its name or "name: description" (a description
  that isn't text as JSON), 2 to 16 of them (the card's validation);
- every question its own call, its probabilities in ippon's option order (a noul as true, false).

A request the card's code refuses (a score not on six levels, a choice of fewer than 2 or more than 16 options) is
recorded with its reason (`error`), which ippon must refuse too. The bound: the same code as the card runs it, bf16
weights under bf16 autocast, on the CPU; the larger of 0.01 and that distance plus 0.0005. Not on the Mac's GPU
("mps"), where transformers falls back to its reference recurrence for Qwen3.5's linear-attention layers and moved
the answers up to 0.39 from fp32 (40 questions about one state), where the card says bf16 moves them in the third
decimal. `--reuse FILE` keeps the fp32 answers of a reference made before and measures the bound again.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

REPO, REVISION = "autotrust/AutoTrust-Decider-Mini-9B", "7ac669b8fb476ba926491e3d1c39f9330d68dbed"
LETTERS = "ABCDEFGHIJKLMNOP"


def state_text(state: Any) -> str:
    """vl/serve_decide.py `_parts`, text only."""
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    if isinstance(state, list):
        return "".join(p if isinstance(p, str) else json.dumps(p, ensure_ascii=False) for p in state)
    return json.dumps(state, ensure_ascii=False)


def calls(request: dict[str, Any]) -> list[tuple[str, str, str, str, list[str]]]:
    """The card's (question id, kind, state, question, options) calls a System One request makes; ValueError for one
    the card refuses."""
    state, out = state_text(request.get("state")), []
    for qid, q in request["questions"].items():
        kind, criteria, question = q["type"], q.get("criteria"), q.get("instructions")
        if not isinstance(question, str):
            raise ValueError(f"{qid}: a question needs its instructions as text")
        if kind == "noul":
            options = ["false", "true"]
        elif kind == "score":
            if not isinstance(criteria, list) or len(criteria) != 6:
                raise ValueError(f"{qid}: score must be ['0'..'5']")
            options = [str(i) for i in range(6)]
        else:
            items = list(criteria.items()) if isinstance(criteria, dict) else [(c, None) for c in criteria or []]
            options = [name if d in (None, "") else f"{name}: {state_text(d)}" for name, d in items]
            if not 2 <= len(options) <= 16:
                raise ValueError(f"{qid}: choice takes 2-16 free-text options")
        out.append((qid, kind, state, question, options))
    return out


def run(
    model: Any,
    tok: Any,
    head: dict[str, Any],
    cfg: dict[str, Any],
    temp: dict[str, float],
    cases: list[dict[str, Any]],
    device: str,
    autocast: Any,
) -> list[dict[str, Any]]:
    import torch

    weight, bias = head["proj.weight"].to(device), head["proj.bias"].to(device)  # the card's W and b

    def decide(kind: str, state: str, question: str, options: list[str]) -> tuple[list[float], list[int]]:
        # the card's decide, its token ids returned too
        lines = options if kind != "choice" else [f"{LETTERS[i]}) {o}" for i, o in enumerate(options)]
        text = (
            f"[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"
        )
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad(), autocast():
            h = model.model(**ids).last_hidden_state[0, -1].float()  # backbone only, last token
        z = (weight @ h + bias) / temp[kind]
        s, _ = cfg["slots"]["ranges"][kind]
        p = torch.softmax(z[s : s + len(options)], 0)
        return [float(x) for x in p.tolist()], [int(t) for t in ids["input_ids"][0].tolist()]

    out = []
    for case in cases:
        request = case["request"]
        try:
            asked = calls(request)
        except ValueError as error:
            out.append({"name": case["name"], "request": request, "error": f"ValueError: {error}"})
            continue
        entry: dict[str, Any] = {"name": case["name"], "request": request, "prompts": [], "probabilities": {}}
        for qid, kind, state, question, options in asked:
            p, ids = decide(kind, state, question, options)
            entry["prompts"].append(ids)
            entry["probabilities"][qid] = [p[1], p[0]] if kind == "noul" else p
        entry["input_tokens"] = sum(len(ids) for ids in entry["prompts"])
        out.append(entry)
    return out


def load(root: Path, dtype: Any, device: str) -> Any:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(str(root), dtype=dtype).to(device)
    return PeftModel.from_pretrained(base, str(root), subfolder="adapter").merge_and_unload().eval()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--requests", required=True, type=Path, nargs="+")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--reuse", type=Path, help="a reference made before: keep its fp32 answers, measure the bound again")
    args = p.parse_args()

    import contextlib
    import functools

    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    root = args.model.expanduser()
    cases = [c for path in args.requests for c in json.loads(path.read_text())]
    tok = AutoTokenizer.from_pretrained(str(root))
    head = load_file(str(root / "head.safetensors"))
    cfg = json.loads((root / "judge_config.json").read_text())
    temp = json.loads((root / "calibration.json").read_text())["per_kind"]

    if args.reuse:
        out = json.loads(gzip.decompress(args.reuse.read_bytes()))["requests"]
        for entry in out:
            entry.pop("probabilities_bf16", None)
    else:
        model = load(root, torch.float32, "cpu")
        out = run(model, tok, head, cfg, temp, cases, "cpu", contextlib.nullcontext)
        del model
    model = load(root, torch.bfloat16, "cpu")
    bf16 = run(
        model, tok, head, cfg, temp, cases, "cpu", functools.partial(torch.autocast, "cpu", dtype=torch.bfloat16)
    )
    distance = 0.0
    for a, c in zip(out, bf16, strict=True):
        for qid, got in c.get("probabilities", {}).items():
            a.setdefault("probabilities_bf16", {})[qid] = got
            distance = max(distance, *(abs(x - y) for x, y in zip(got, a["probabilities"][qid], strict=True)))
    bound = max(0.01, round(distance + 5e-4, 4))
    reference = {
        "schema": "ippon-reference/1",
        "adapter": "autotrust/decider",
        "model": {"repo": REPO, "revision": REVISION},
        "maker": {
            "code": "the AutoTrust-Decider-Mini-9B model card's System 1 example (transformers + peft)",
            "device": "cpu",
            "dtype": "float32",
            "torch": torch.__version__,
        },
        "bound": bound,
        "tokens_only": False,
        "requests": out,
    }
    data = json.dumps(reference, ensure_ascii=False).encode()
    args.out.write_bytes(gzip.compress(data, mtime=0) if args.out.suffix == ".gz" else data)
    refused = sum("error" in e for e in out)
    print(f"{len(out)} requests ({refused} refused) -> {args.out} (bound {bound}, bf16 distance {distance:.5f})")


if __name__ == "__main__":
    main()
