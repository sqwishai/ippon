"""Cloudflare Clef (clef-flash) through the model card's own loader, on MPS: load_release_model, then
encode_record -> collate_records -> model(batch), one logit per allowed option, softmax per question.

    envs/clef/bin/python -I adapters/clef.py --engine clef-flash-tf --repo Cloudflare/clef-flash \
        --model ~/models/clef-flash --max-input 16384

The card documents device="cuda" (tested on one H200); here the same call is made with device="mps".
The loader file (joint_schema_model.py) is imported from the downloaded official Cloudflare repo only.
Clef's request schema is the System One one (noul / choice / score), so a request's questions pass through
unchanged; a noul answer's options come back as true/false and are recorded as yes/no.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common, gen_requests
import common

OFFICIAL = ("Cloudflare/",)


def main() -> None:
    p = common.add_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    p.add_argument("--model", required=True, help="a local download of an official Cloudflare Clef repo")
    p.add_argument("--repo", required=True, help="its Hub id (Cloudflare/...), recorded and checked")
    p.add_argument("--device", default="mps")
    args = p.parse_args()
    if not args.repo.startswith(OFFICIAL):
        sys.exit(f"the repo's loader is imported only for {OFFICIAL}")

    import torch
    import transformers

    model_dir = str(Path(args.model).expanduser())
    sys.path.insert(0, model_dir)
    from joint_schema_model import collate_records, encode_record, load_release_model

    model, processor = load_release_model(model_dir, device=args.device)
    model.eval()
    tok = processor.tokenizer
    device = torch.device(args.device)

    def decide(req):
        body = req["systemone"]
        encoded = encode_record(
            tok,
            {"state": body["state"], "questions": body["questions"]},
            processor=processor,
            max_length=args.max_input,
        )
        batch = collate_records([encoded], tok.pad_token_id, device)
        logits = model(batch)[0]
        answers = {}
        for q, ql in zip(encoded.questions, logits, strict=False):
            probs = dict(zip(q.option_ids, ql.float().softmax(-1).tolist(), strict=False))
            if body["questions"][q.question_id]["type"] == "noul":
                probs = {"yes": probs["true"], "no": probs["false"]}
            answers[q.question_id] = {str(k): v for k, v in probs.items()}
        return answers, len(encoded.input_ids), None

    def memory():
        if args.device == "mps":
            return {
                "mps_driver_allocated": torch.mps.driver_allocated_memory(),
                "mps_current_allocated": torch.mps.current_allocated_memory(),
            }
        return None

    meta = {
        "runtime": f"transformers {transformers.__version__} + torch {torch.__version__} ({args.device}), "
        "card loader joint_schema_model.load_release_model",
        "family": "clef",
        "repo": args.repo,
        "model_dir": model_dir,
        "revision": common.repo_revision(model_dir),
        "device": args.device,
        "dtype": str(next(model.parameters()).dtype),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "mps_high_watermark_ratio": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
    }
    meta["question_method"] = "all questions in one encoded sequence, one model call"
    with torch.inference_mode():
        common.run(args, decide, meta, memory=memory)


if __name__ == "__main__":
    main()
