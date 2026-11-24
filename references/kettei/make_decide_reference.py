"""Kettei's prompts for Sqwish's decide API, as the sqwishai SDK's own local engine writes them (sqwishai.kettei 0.5.0),
never inside ippon:

    python -I make_decide_reference.py --model ~/models/kettei-dot --requests ../requests/decide.json \\
        --out kettei-dot-decide.json.gz

Each request is checked by the SDK (`_request.check`) and every decision rendered (`_request.prompts` and
`_prompt.render_decision`, the model's variant and tokenizer), as `Kettei.decide` renders them; the reference keeps each
decision's prompt tokens and label tokens, and the input tokens the SDK's MLX runtime reports (the shared start once).
Tokens only: the probabilities on this path are the System One references' (kettei-core.json.gz, kettei-dot.json.gz).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--requests", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    from sqwishai.kettei import _model, _request
    from sqwishai.kettei._prompt import render_decision
    from sqwishai.kettei._tokenizer import load_tokenizer

    root = args.model.expanduser()
    size, _ = _model._resolve(str(root), None)
    tokenizer = load_tokenizer(root, local_files_only=True)  # as Kettei.__init__ loads it
    out = []
    for case in json.loads(args.requests.read_text()):
        body = case["request"]
        checked = _request.check(body["context"], body["decisions"], body.get("temperature"))
        rendered = [
            render_decision(tokenizer, q, variant=size.variant, max_input_tokens=sys.maxsize)
            for q in _request.prompts(body["context"], checked)
        ]
        ids = [list(r.input_ids) for r in rendered]
        shared = _model._shared([tuple(i) for i in ids])
        out.append(
            {
                "name": case["name"],
                "request": body,
                "prompts": ids,
                "labels": [list(r.label_ids) for r in rendered],
                "input_tokens": shared + sum(len(i) - shared for i in ids),
            }
        )
    reference = {"schema": "ippon-decide-reference/1", "model": {"directory": root.name}, "requests": out}
    args.out.write_bytes(gzip.compress(json.dumps(reference, ensure_ascii=False).encode(), mtime=0))
    print(f"{len(out)} requests -> {args.out}")


if __name__ == "__main__":
    main()
