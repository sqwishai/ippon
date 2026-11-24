"""A `ippon bench --compare` arm for bring-your-own encoder classifiers (BERT, XLM-RoBERTa, RoBERTa,
ModernBERT): the plain MLX way to run them, mlx-embeddings' own model code. The prompts and the head are
ippon's (`generic/classifier`, checked against transformers), so both arms read the same tokens and score them
alike; the layers are mlx-embeddings', and each request's prompts run whole in one right-padded batch with its
attention mask (a BERT pair's second text with token type 1, as its tokenizer marks it). Nothing of ippon's
execution is used.

    PYTHON -I mlx_embeddings_model.py MODEL_DIR REQUESTS.json OUT.json

(PYTHON: an environment with mlx-embeddings and ippon.)

Compute dtype: fp16. IPPON_ARM_DTYPE=float32 (`verify --against`): fp32.
"""

from __future__ import annotations

import importlib
import sys
from importlib.metadata import version
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten

from ippon.adapter import Hidden
from ippon.compare_arms.harness import ASKED, COMPUTE, ippon_format, time_requests, write
from ippon.schema import parse_request
from ippon.weights import Weights

MODULES = {"bert": "bert", "xlm-roberta": "xlm_roberta", "roberta": "xlm_roberta", "modernbert": "modernbert"}
ROOTS = {"bert": "bert.", "xlm-roberta": "roberta.", "roberta": "roberta.", "modernbert": ""}


def load(model_dir: str) -> tuple[Any, Any, str]:
    """(ippon's format, mlx-embeddings' encoder, its model type)."""
    md, adapter, fmt, files = ippon_format(model_dir, "mlx-embeddings-model")
    kind = md.config.get("model_type")
    if kind not in MODULES:
        raise SystemExit(f"{adapter} ({kind}): this arm runs BERT, RoBERTa, XLM-RoBERTa and ModernBERT classifiers")
    module = importlib.import_module(f"mlx_embeddings.models.{MODULES[kind]}")
    model = module.Model(module.ModelArgs.from_dict(dict(md.config)))
    wanted = {str(pair[0]) for pair in tree_flatten(model.parameters())}
    root = ROOTS[kind]
    found = {
        k[len(root) :] if k.startswith(root) else k: v for k, v in Weights.from_dir(md.path, files).tensors.items()
    }
    found = {k.replace(".gamma", ".weight").replace(".beta", ".bias"): v for k, v in found.items()}
    missing = sorted(k for k in wanted - set(found) if not k.startswith("pooler."))
    if missing:
        raise SystemExit(f"mlx-embeddings' {kind} wants tensors the checkpoint lacks: {missing[:5]}")
    model.load_weights([(k, v) for k, v in found.items() if k in wanted], strict=False)
    model.eval()  # MLX modules start in training mode: BERT's dropout (0.1) would be on
    model.set_dtype(COMPUTE)
    mx.eval(model.parameters())
    return fmt, model, kind


def answer(fmt: Any, model: Any, kind: str, sep: int | None, body: dict[str, Any]) -> dict[str, Any]:
    """The System One response for `body`: ippon's prompts and head, mlx-embeddings' layers."""
    request = parse_request(body)
    rendering = fmt.render(request)
    prompts = [list(p.tokens) for p in rendering.prompts]
    longest = max(len(p) for p in prompts)
    ids = mx.array([p + [0] * (longest - len(p)) for p in prompts])
    real = (mx.arange(longest)[None, :] < mx.array([len(p) for p in prompts])[:, None]).astype(COMPUTE)
    if kind == "modernbert":  # its own pooling (cls or mean, as the checkpoint's): one pooled row per prompt
        pooled = model(ids, attention_mask=real).last_hidden_state
        hidden = Hidden({r.question: pooled[i][None] for i, p in enumerate(rendering.prompts) for r in p.readouts})
    else:
        types = None
        if sep is not None:  # a pair: type 1 after the first [SEP]
            s = (ids == sep).astype(mx.int32)
            types = mx.minimum(mx.cumsum(s, axis=1) - s, 1)
        h = model(ids, attention_mask=real, token_type_ids=types).last_hidden_state
        picked = [h[i][mx.array([r.position for r in p.readouts])] for i, p in enumerate(rendering.prompts)]
        hidden = Hidden.gather([p.readouts for p in rendering.prompts], picked)
    scores = fmt.scores(rendering, hidden)
    mx.eval(list(scores.values()))
    shaped = {q.id: fmt.answer(q, scores[q.id].astype(COMPUTE).tolist())[1] for q in request.questions}
    return {"answers": shaped, "usage": fmt.usage(rendering)}


def main() -> None:
    model_dir, requests, out = sys.argv[1:4]
    fmt, model, kind = load(model_dir)
    sep = None
    if fmt.settings.pair is not None and kind == "bert":
        sep = fmt.tokenizer.token_id("[SEP]")
    results = time_requests(requests, lambda body: answer(fmt, model, kind, sep, body))
    runtime = f"mlx-embeddings {version('mlx-embeddings')} layers, ippon's prompts and head, mlx {version('mlx')}"
    write(out, runtime, results, ASKED)


if __name__ == "__main__":
    main()
