"""Reference answers for Sqwish's Kettei Core and Dot from the sqwishai SDK's own local code (sqwishai.kettei 0.5.0).

Run it in an environment with `sqwishai[local]==0.5.0` and torch, never inside ippon:

    python -I make_reference.py --model ~/models/kettei-dot --requests ../requests/standard.json \\
        --out kettei-dot.json.gz [--tokens-only]

A System One request becomes the SDK's decide request as System One maps onto decide: a noul is a binary decision
(its "true" and "false" descriptions as "yes" and "no"), a choice a single one (a list of labels as labels without
descriptions), a score an ordinal one over "0".."n-1" with its levels as the rubric; a request without a state has an
empty context. The SDK checks it (`_request.check`, a refusal is recorded as the case's error) and renders each
decision (`_prompt.render_decision`); its PyTorch runtime (`_torch.Runtime`, which on the CPU runs the checkpoint's
own code in fp32) gives each decision's label logits, softmaxed as `_request.probabilities` does. The SDK's MLX
runtime, its Mac path, gives them again for the bound: the larger of 0.01 and its largest distance from fp32 plus
0.0005.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from runpy import run_path
from typing import Any

# Load just the stdlib serializer without importing ippon or changing runtime module discovery.
dumps = run_path(str(Path(__file__).resolve().parents[2] / "src/ippon/_portable.py"))["dumps"]


def decide_request(request: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    """The context and decisions a System One request asks (see the module docstring)."""
    decisions = []
    for key, q in request["questions"].items():
        decision: dict[str, Any] = {"id": key, "question": q.get("instructions")}
        criteria = q.get("criteria")
        if q["type"] == "noul":
            given = criteria if isinstance(criteria, dict) else {}
            described = {new: given[old] for old, new in (("true", "yes"), ("false", "no")) if old in given}
            decision.update(kind="binary", **({"outcomes": described} if described else {}))
        elif q["type"] == "choice":
            outcomes = dict(criteria) if isinstance(criteria, dict) else dict.fromkeys(criteria)
            decision.update(kind="single", outcomes=outcomes)
        else:
            levels = list(criteria.values()) if isinstance(criteria, dict) else list(criteria)
            decision.update(kind="ordinal", outcomes=[str(i) for i in range(len(levels))], rubric=levels)
        decisions.append(decision)
    state = request.get("state")
    return ("" if state is None else state), decisions


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, type=Path, help="a downloaded Kettei Core or Dot checkpoint")
    p.add_argument("--requests", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tokens-only", action="store_true")
    args = p.parse_args()
    root = args.model.expanduser()

    from sqwishai.kettei import _catalogue, _prompt, _request, _tokenizer

    vocabulary = json.loads((root / "config.json").read_text())["vocab_size"]
    size = _catalogue.BY_VOCABULARY[vocabulary]
    tokenizer = _tokenizer.load_tokenizer(root, local_files_only=True)
    out, asked = [], {}
    for case in json.loads(args.requests.read_text()):
        request = case["request"]
        context, decisions = decide_request(request)
        try:
            checked = _request.check(context, decisions)
            rendered = [
                _prompt.render_decision(tokenizer, q, variant=size.variant, max_input_tokens=sys.maxsize)
                for q in _request.prompts(context, checked)
            ]
        except ValueError as refusal:  # KetteiError, or the renderer's own
            out.append({"name": case["name"], "request": request, "error": str(refusal)})
            continue
        ids = [list(r.input_ids) for r in rendered]
        shared = _shared(ids)
        out.append(
            {
                "name": case["name"],
                "request": request,
                "prompts": ids,
                "input_tokens": shared + sum(len(i) - shared for i in ids),
            }
        )
        asked[case["name"]] = (ids, [list(r.label_ids) for r in rendered], list(request["questions"]))
    if not args.tokens_only:
        from sqwishai.kettei._mlx import Runtime as MlxRuntime
        from sqwishai.kettei._torch import Runtime as TorchRuntime

        for key, runtime in (
            ("probabilities", lambda: TorchRuntime(root, "cpu")),
            ("probabilities_mlx", lambda: MlxRuntime(root, "mlx")),
        ):
            loaded = runtime()
            for entry in out:
                if entry["name"] not in asked:
                    continue
                ids, labels, keys = asked[entry["name"]]
                logits = loaded.score(ids, labels)
                entry[key] = {k: _request.probabilities(row) for k, row in zip(keys, logits, strict=True)}
                print(f"{key} {entry['name']}: {entry['input_tokens']} tokens", flush=True)
            loaded.close()
    bound = 0.01
    if not args.tokens_only:
        distance = max(
            abs(a - b)
            for e in out
            if "probabilities" in e
            for q in e["probabilities"]
            for a, b in zip(e["probabilities"][q], e["probabilities_mlx"][q], strict=True)
        )
        bound = max(0.01, round(distance + 5e-4, 4))
    import sqwishai

    reference = {
        "schema": "ippon-reference/1",
        "adapter": "sqwish/kettei",
        "model": {"directory": root.name, "repo": size.repo, "revision": revision(root), "size": size.id},
        "maker": {"code": f"sqwishai {sqwishai.__version__} (sqwishai.kettei)", "device": "cpu", "dtype": "float32"},
        "bound": bound,
        "tokens_only": args.tokens_only,
        "requests": out,
    }
    data = dumps(reference, ensure_ascii=False).encode()
    args.out.write_bytes(gzip.compress(data) if args.out.suffix == ".gz" else data)
    print(f"{len(out)} requests -> {args.out} (bound {bound}{', tokens only' if args.tokens_only else ''})")


def revision(root: Path) -> str | None:
    """The Hub commit a `hf download --local-dir` folder holds (its download metadata's first line)."""
    found = root / ".cache/huggingface/download/config.json.metadata"
    return found.read_text().splitlines()[0] if found.is_file() else None


def _shared(ids: list[list[int]]) -> int:
    """The tokens every prompt starts with, short of the shortest prompt's last (the SDK's usage count)."""
    first, last, most = min(ids), max(ids), min(map(len, ids)) - 1
    return next((n for n, (a, b) in enumerate(zip(first[:most], last[:most], strict=True)) if a != b), most)


if __name__ == "__main__":
    main()
