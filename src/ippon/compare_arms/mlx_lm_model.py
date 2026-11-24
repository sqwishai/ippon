"""A `ippon bench --compare` arm for bring-your-own decoders (a Qwen3, Qwen2 or Llama classifier, or a
label-token model on one): the plain MLX way to run them, mlx-lm's own model code. The prompts and the head are
ippon's (`generic/classifier`, `generic/label-tokens`: checked against transformers), so both arms read the
same tokens and score them alike; the layers are mlx-lm's, and each request's prompts run whole in right-padded
batches of at most 16,384 tokens, one forward each (causal attention keeps padding from reaching a real token).
Nothing of ippon's execution is used: no shared prefix, no kept state, none of its kernels.

    PYTHON -I mlx_lm_model.py MODEL_DIR REQUESTS.json OUT.json   (PYTHON: an environment with mlx-lm and ippon)

Compute dtype: fp16, as ippon's precision="auto" chooses for these models.
IPPON_ARM_DTYPE=float32 (`verify --against`): fp32.
"""

from __future__ import annotations

import sys
from importlib.metadata import version
from typing import Any

import mlx.core as mx
from mlx_lm.utils import _get_classes

from ippon.adapter import Hidden
from ippon.compare_arms.harness import ASKED, COMPUTE, budget_groups, ippon_format, time_requests, write
from ippon.schema import parse_request
from ippon.weights import Weights

BUDGET = 16384  # padded tokens per batched forward


def load(model_dir: str) -> tuple[Any, Any]:
    """(ippon's format, mlx-lm's decoder) for a checkpoint: the decoder without its LM head (ippon's
    head reads the final hidden states)."""
    md, _, fmt, files = ippon_format(model_dir, "mlx-lm-model")
    config = dict(md.config.get("text_config", md.config))
    model_class, args_class = _get_classes(config)
    model = model_class(args_class.from_dict(config))
    tensors = Weights.from_dir(md.path, files).tensors
    inner = {k[len("model.") :]: v for k, v in tensors.items() if k.startswith("model.")}
    model.model.load_weights(list(inner.items()), strict=True)
    model.eval()  # MLX modules start in training mode
    model.model.set_dtype(COMPUTE)
    mx.eval(model.model.parameters())
    return fmt, model.model


def answer(fmt: Any, decoder: Any, body: dict[str, Any]) -> dict[str, Any]:
    """The System One response for `body`: ippon's prompts and head, mlx-lm's layers."""
    request = parse_request(body)
    rendering = fmt.render(request)
    scores = _scores(fmt, decoder, rendering)
    shaped = {q.id: fmt.answer(q, scores[q.id].astype(mx.float32).tolist())[1] for q in request.questions}
    return {"answers": shaped, "usage": fmt.usage(rendering)}


def _scores(fmt: Any, decoder: Any, rendering: Any) -> Any:
    prompts = [list(p.tokens) for p in rendering.prompts]
    hidden: list[mx.array] = []
    for group in budget_groups([len(p) for p in prompts], BUDGET):
        longest = max(len(prompts[i]) for i in group)
        h = decoder(mx.array([prompts[i] + [0] * (longest - len(prompts[i])) for i in group]))
        hidden += [h[k][mx.array([r.position for r in rendering.prompts[i].readouts])] for k, i in enumerate(group)]
    scores = fmt.scores(rendering, Hidden.gather([p.readouts for p in rendering.prompts], hidden))
    mx.eval(list(scores.values()))
    return scores


def main() -> None:
    model_dir, requests, out = sys.argv[1:4]
    fmt, decoder = load(model_dir)
    results = time_requests(requests, lambda body: answer(fmt, decoder, body))
    write(out, f"mlx-lm {version('mlx-lm')} layers, ippon's prompts and head, mlx {version('mlx')}", results, ASKED)


if __name__ == "__main__":
    main()
