"""ModernBERT in plain numpy (float64), written from Hugging Face's modeling_modernbert, for checking the MLX
implementation on tiny random checkpoints."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def ln(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return np.asarray((x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps) * w)


def rope(x: np.ndarray, theta: float) -> np.ndarray:
    """x [H, T, D] at positions 0..T-1."""
    _, t, d = x.shape
    angles = np.arange(t)[:, None] * (1.0 / theta ** (np.arange(0, d, 2) / d))[None]
    cos, sin = np.cos(np.concatenate([angles, angles], -1)), np.sin(np.concatenate([angles, angles], -1))
    return np.asarray(x * cos + np.concatenate([-x[..., d // 2 :], x[..., : d // 2]], -1) * sin)


def forward(c: dict[str, Any], w: dict[str, np.ndarray], ids: list[int]) -> np.ndarray:
    d, heads, eps, t = c["hidden_size"], c["num_attention_heads"], c["norm_eps"], len(ids)
    hd = d // heads
    erf = np.vectorize(math.erf)
    x = ln(w["embeddings.tok_embeddings.weight"][ids].astype(np.float64), w["embeddings.norm.weight"], eps)
    for i, kind in enumerate(c["layer_types"]):
        p = f"layers.{i}."
        h = x if i == 0 else ln(x, w[p + "attn_norm.weight"], eps)
        qkv = (h @ w[p + "attn.Wqkv.weight"].T).reshape(t, 3, heads, hd)
        theta = c["rope_parameters"][kind]["rope_theta"]
        q, k, v = (qkv[:, j].transpose(1, 0, 2) for j in range(3))
        q, k = rope(q, theta), rope(k, theta)
        s = q @ k.transpose(0, 2, 1) * hd**-0.5
        if kind == "sliding_attention":
            near = np.abs(np.arange(t)[:, None] - np.arange(t)[None]) <= c["local_attention"] // 2
            s = np.where(near[None], s, -np.inf)
        a = np.exp(s - s.max(-1, keepdims=True))
        a /= a.sum(-1, keepdims=True)
        x = x + (a @ v).transpose(1, 0, 2).reshape(t, d) @ w[p + "attn.Wo.weight"].T
        h = ln(x, w[p + "mlp_norm.weight"], eps) @ w[p + "mlp.Wi.weight"].T
        g, u = np.split(h, 2, axis=-1)
        x = x + (0.5 * g * (1 + erf(g / math.sqrt(2))) * u) @ w[p + "mlp.Wo.weight"].T
    return ln(x, w["final_norm.weight"], eps)


def tiny_config(layer_types: list[str], local_attention: int = 6) -> dict[str, Any]:
    return {
        "model_type": "modernbert",
        "hidden_size": 32,
        "intermediate_size": 40,
        "num_attention_heads": 4,
        "num_hidden_layers": len(layer_types),
        "layer_types": layer_types,
        "local_attention": local_attention,
        "norm_eps": 1e-5,
        "norm_bias": False,
        "attention_bias": False,
        "mlp_bias": False,
        "hidden_activation": "gelu",
        "rope_parameters": {
            "full_attention": {"rope_theta": 160_000.0, "rope_type": "default"},
            "sliding_attention": {"rope_theta": 10_000.0, "rope_type": "default"},
        },
        "vocab_size": 64,
    }


def tiny_weights(c: dict[str, Any], seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    d, f = c["hidden_size"], c["intermediate_size"]

    def m(*shape: int, scale: float = 0.2) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def ones(n: int) -> np.ndarray:
        return (1 + rng.standard_normal(n) * 0.1).astype(np.float32)

    w = {
        "embeddings.tok_embeddings.weight": m(c["vocab_size"], d, scale=1.0),
        "embeddings.norm.weight": ones(d),
        "final_norm.weight": ones(d),
    }
    for i in range(len(c["layer_types"])):
        p = f"layers.{i}."
        w |= {
            p + "attn.Wqkv.weight": m(3 * d, d),
            p + "attn.Wo.weight": m(d, d),
            p + "mlp_norm.weight": ones(d),
            p + "mlp.Wi.weight": m(2 * f, d),
            p + "mlp.Wo.weight": m(d, f),
        }
        if i:
            w[p + "attn_norm.weight"] = ones(d)
    return w
