"""Gemma 3 text in plain numpy (float64), written from Hugging Face's modeling_gemma3 (Gemma3RMSNorm's 1 + w, the
scaled embedding, the four norms per layer, q/k norms, query_pre_attn_scalar, per-layer-type RoPE, and
masking_utils' sliding window: a key is seen when kv_idx > q_idx - sliding_window) rather than from ippon's
code, for checking the MLX decoder on tiny random checkpoints."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def rms(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return np.asarray(x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * (1.0 + w))


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    return np.asarray(0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3))))


def rope(x: np.ndarray, theta: float, factor: float = 1.0) -> np.ndarray:
    t, _, d = x.shape
    inv = 1.0 / theta ** (np.arange(0, d, 2) / d)
    angles = (np.arange(t) / factor)[:, None] * inv[None]
    emb = np.concatenate([angles, angles], -1)
    cos, sin = np.cos(emb)[:, None], np.sin(emb)[:, None]
    a, b = x[..., : d // 2], x[..., d // 2 :]
    return np.asarray(x * cos + np.concatenate([-b, a], -1) * sin)


def forward(c: dict[str, Any], w: dict[str, np.ndarray], ids: list[int]) -> np.ndarray:
    """Final-normed hidden states [T, hidden] of one prompt."""
    d, heads, kv, eps, hd = (
        c["hidden_size"],
        c["num_attention_heads"],
        c["num_key_value_heads"],
        c["rms_norm_eps"],
        c["head_dim"],
    )
    t = len(ids)
    x = w["model.embed_tokens.weight"][ids].astype(np.float64) * math.sqrt(d)
    q_i, k_i = np.arange(t)[:, None], np.arange(t)[None, :]
    for i in range(c["num_hidden_layers"]):
        p, a = f"model.layers.{i}.", f"model.layers.{i}.self_attn."
        kind = c["layer_types"][i]
        theta = c["rope_local_base_freq"] if kind == "sliding_attention" else c["rope_theta"]
        factor = 1.0 if kind == "sliding_attention" else (c.get("rope_scaling") or {}).get("factor", 1.0)
        h = rms(x, w[p + "input_layernorm.weight"], eps)
        q = rms((h @ w[a + "q_proj.weight"].T).reshape(t, heads, hd), w[a + "q_norm.weight"], eps)
        k = rms((h @ w[a + "k_proj.weight"].T).reshape(t, kv, hd), w[a + "k_norm.weight"], eps)
        v = (h @ w[a + "v_proj.weight"].T).reshape(t, kv, hd)
        q, k = rope(q, theta, factor), rope(k, theta, factor)
        k, v = np.repeat(k, heads // kv, axis=1), np.repeat(v, heads // kv, axis=1)
        scores = np.einsum("thd,shd->hts", q, k) * c["query_pre_attn_scalar"] ** -0.5
        seen = k_i <= q_i
        if kind == "sliding_attention":
            seen &= k_i > q_i - c["sliding_window"]
        scores = np.where(seen[None], scores, -np.inf)
        probs = np.exp(scores - scores.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        o = np.einsum("hts,shd->thd", probs, v).reshape(t, heads * hd) @ w[a + "o_proj.weight"].T
        x = x + rms(o, w[p + "post_attention_layernorm.weight"], eps)
        h = rms(x, w[p + "pre_feedforward_layernorm.weight"], eps)
        m = p + "mlp."
        f = (gelu_tanh(h @ w[m + "gate_proj.weight"].T) * (h @ w[m + "up_proj.weight"].T)) @ w[m + "down_proj.weight"].T
        x = x + rms(f, w[p + "post_feedforward_layernorm.weight"], eps)
    return rms(x, w["model.norm.weight"], eps)


def tiny_config(**extra: Any) -> dict[str, Any]:
    return {
        "model_type": "gemma3_text",
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "rope_local_base_freq": 10_000.0,
        "query_pre_attn_scalar": 16,
        "sliding_window": 5,  # small: a 19-token prompt crosses it
        "layer_types": ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"],
        "hidden_activation": "gelu_pytorch_tanh",
        "vocab_size": 64,
        "tie_word_embeddings": True,
    } | extra


def tiny_weights(c: dict[str, Any], seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    d, f, heads, kv, hd = (
        c["hidden_size"],
        c["intermediate_size"],
        c["num_attention_heads"],
        c["num_key_value_heads"],
        c["head_dim"],
    )

    def m(*shape: int, scale: float = 0.2) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def norm(n: int) -> np.ndarray:  # Gemma stores w, applies 1 + w
        return (rng.standard_normal(n) * 0.1).astype(np.float32)

    w = {"model.embed_tokens.weight": m(c["vocab_size"], d, scale=0.3), "model.norm.weight": norm(d)}
    for i in range(c["num_hidden_layers"]):
        p, a = f"model.layers.{i}.", f"model.layers.{i}.self_attn."
        w |= {
            p + "input_layernorm.weight": norm(d),
            p + "post_attention_layernorm.weight": norm(d),
            p + "pre_feedforward_layernorm.weight": norm(d),
            p + "post_feedforward_layernorm.weight": norm(d),
            a + "q_proj.weight": m(heads * hd, d),
            a + "k_proj.weight": m(kv * hd, d),
            a + "v_proj.weight": m(kv * hd, d),
            a + "o_proj.weight": m(d, heads * hd),
            a + "q_norm.weight": norm(hd),
            a + "k_norm.weight": norm(hd),
            p + "mlp.gate_proj.weight": m(f, d),
            p + "mlp.up_proj.weight": m(f, d),
            p + "mlp.down_proj.weight": m(d, f),
        }
    return w
