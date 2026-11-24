"""A `ippon bench --compare` arm for decision models no MLX runtime runs as published (vllm-sr d3): the plain MLX way to
run them. The prompts and the head are ippon's (checked token for token, and against the creators' fp32 answers), so
both arms read the same tokens and score them alike; the layers are mlx-vlm's own model code (Qwen3.5), and each
request's prompts run whole in right-padded batches of at most 16,384 tokens, one forward each, as mlx-vlm's own
decision ports run theirs. Nothing of ippon's execution is used: no shared prefix, no kept state, none of its kernels.
The two arms differ only in how the model's layers are executed.

    PYTHON -I mlx_vlm_model.py MODEL_DIR REQUESTS.json OUT.json   (PYTHON: an environment with mlx-vlm and ippon)

Compute dtype: fp16, as ippon's precision="auto" chooses for these models.
IPPON_ARM_DTYPE=float32 (`verify --against`): fp32.
"""

from __future__ import annotations

import sys
from importlib.metadata import version
from typing import Any

import mlx.core as mx
from mlx_vlm.models.qwen3_5_text import Model, ModelConfig

from ippon.adapter import Hidden
from ippon.compare_arms.harness import ASKED, COMPUTE, budget_groups, ippon_format, time_requests, write
from ippon.schema import parse_request
from ippon.weights import Weights

BUDGET = 16384  # padded tokens per batched forward


def qwen(weights: Any, config: dict[str, Any], dtype: mx.Dtype) -> Any:
    """mlx-vlm's Qwen3.5 text model with a checkpoint's text tensors (ippon's `Weights`; renamed from their
    root, which `text_root` finds, to mlx-vlm's names; vision and multi-token-prediction tensors left out)."""
    text = config.get("text_config", config)
    model = Model(ModelConfig.from_dict({**text, "model_type": "qwen3_5_text"}))
    root = weights.text_root()
    renamed = {}
    for k, v in weights.tensors.items():
        if k.startswith(root):
            renamed["model." + k[len(root) :]] = v
        elif k.endswith("lm_head.weight"):
            renamed["lm_head.weight"] = v
    model.load_weights(list(model.sanitize(renamed).items()), strict=True)
    model.eval()  # MLX modules start in training mode (dropout on)
    model.set_dtype(dtype)
    mx.eval(model.parameters())
    return model


def load(model_dir: str) -> tuple[Any, Any, str]:
    """(ippon's format, mlx-vlm's model, its kind) for a checkpoint."""
    md, adapter, fmt, files = ippon_format(model_dir, "mlx-vlm-model")
    if adapter != "vllm-sr/d3":
        raise SystemExit(f"{adapter}: this arm runs d3")
    causal = md.json("decision_config.json").get("attention_mode") == "causal"
    model = qwen(Weights.from_dir(md.path, files), dict(md.config), COMPUTE)
    return fmt, model, "causal" if causal else "bidirectional"


def forward(model: Any, kind: str, prompts: list[list[int]]) -> mx.array:
    """Final hidden states [prompts, longest, hidden] of right-padded prompts, in one forward."""
    lengths = [len(p) for p in prompts]
    longest = max(lengths)
    ids = mx.array([p + [0] * (longest - len(p)) for p in prompts])
    real = mx.arange(longest)[None, :] < mx.array(lengths)[:, None]
    inner = model.language_model.model
    positions = mx.broadcast_to(mx.arange(longest)[None, None, :], (3, len(prompts), longest))
    if kind == "causal":
        causal: mx.array = inner(ids, position_ids=positions)
        return causal
    # bidirectional full attention (vllm-sr d3): every layer, its attention unmasked but for padding
    keys = None if min(lengths) == longest else real[:, None, None, :]
    h = inner.embed_tokens(ids)
    for layer in inner.layers:
        h = layer(h, mask=None if layer.is_linear else keys, cache=None, position_ids=positions)
    normed: mx.array = inner.norm(h)
    return normed


def answer(fmt: Any, model: Any, kind: str, body: dict[str, Any]) -> dict[str, Any]:
    """The System One response for `body`: ippon's prompts and head, mlx-vlm's layers."""
    request = parse_request(body)
    rendering = fmt.render(request)
    prompts = [list(p.tokens) for p in rendering.prompts]
    hidden: list[mx.array] = []
    for group in budget_groups([len(p) for p in prompts], BUDGET):  # batches of whole prompts within the budget
        h = forward(model, kind, [prompts[i] for i in group])
        hidden += [h[k][mx.array([r.position for r in rendering.prompts[i].readouts])] for k, i in enumerate(group)]
    scores = fmt.scores(rendering, Hidden.gather([p.readouts for p in rendering.prompts], hidden))
    mx.eval(list(scores.values()))
    shaped = {q.id: fmt.answer(q, scores[q.id].astype(mx.float32).tolist())[1] for q in request.questions}
    return {"answers": shaped, "usage": fmt.usage(rendering)}


def main() -> None:
    model_dir, requests, out = sys.argv[1:4]
    fmt, model, kind = load(model_dir)
    results = time_requests(requests, lambda body: answer(fmt, model, kind, body))
    write(
        out,
        f"mlx-vlm {version('mlx-vlm')} layers ({kind}), ippon's prompts and head, mlx {version('mlx')}",
        results,
        ASKED,
    )


if __name__ == "__main__":
    main()
