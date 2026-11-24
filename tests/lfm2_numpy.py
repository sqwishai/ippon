"""LFM2 in plain numpy (float64), written from Liquid's d1 code (hybrid.py, lfm2_vl.py) rather than from ippon's,
for checking the MLX implementation on tiny random checkpoints."""

from __future__ import annotations

from typing import Any

import numpy as np


def rms(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return np.asarray(x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w)


def silu(x: np.ndarray) -> np.ndarray:
    return np.asarray(x / (1 + np.exp(-x)))


def rope(x: np.ndarray, theta: float) -> np.ndarray:
    """rotate_half RoPE over the whole head; x [T, H, D] at positions 0..T-1."""
    t, _, d = x.shape
    inv = 1.0 / theta ** (np.arange(0, d, 2) / d)
    angles = np.arange(t)[:, None] * inv[None]
    cos, sin = (
        np.cos(np.concatenate([angles, angles], -1))[:, None],
        np.sin(np.concatenate([angles, angles], -1))[:, None],
    )
    a, b = x[..., : d // 2], x[..., d // 2 :]
    return np.asarray(x * cos + np.concatenate([-b, a], -1) * sin)


def forward(c: dict[str, Any], w: dict[str, np.ndarray], ids: list[int]) -> np.ndarray:
    """Final-normed hidden states [T, hidden] of one prompt, read causally."""
    d, heads, kv, eps = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["norm_eps"]
    x = w["embed_tokens.weight"][ids].astype(np.float64)
    hd, t = d // heads, len(x)
    for i, kind in enumerate(c["layer_types"]):
        p = f"layers.{i}."
        h = rms(x, w[p + "operator_norm.weight"], eps)
        if kind == "full_attention":
            a = p + "self_attn."
            q = (h @ w[a + "q_proj.weight"].T).reshape(t, heads, hd)
            k = (h @ w[a + "k_proj.weight"].T).reshape(t, kv, hd)
            v = (h @ w[a + "v_proj.weight"].T).reshape(t, kv, hd)
            q, k = (
                rope(rms(q, w[a + "q_layernorm.weight"], eps), c["rope_theta"]),
                rope(rms(k, w[a + "k_layernorm.weight"], eps), c["rope_theta"]),
            )
            k, v = np.repeat(k, heads // kv, axis=1), np.repeat(v, heads // kv, axis=1)
            scores = np.einsum("thd,shd->hts", q, k) * hd**-0.5
            scores = np.where(np.tril(np.ones((t, t), bool))[None], scores, -np.inf)
            p_ = np.exp(scores - scores.max(-1, keepdims=True))
            p_ /= p_.sum(-1, keepdims=True)
            y = np.einsum("hts,shd->thd", p_, v).reshape(t, d) @ w[a + "out_proj.weight"].T
        else:
            proj = h @ w[p + "conv.in_proj.weight"].T
            b, cc, u = proj[:, :d], proj[:, d : 2 * d], proj[:, 2 * d :]
            bx = b * u
            taps = w[p + "conv.conv.weight"][:, 0, :]  # [d, kernel]
            kernel = taps.shape[1]
            padded = np.concatenate([np.zeros((kernel - 1, d)), bx])
            z = sum(padded[j : j + t] * taps[:, j] for j in range(kernel))
            y = (cc * z) @ w[p + "conv.out_proj.weight"].T
        x = x + y
        f = p + "feed_forward."
        h = rms(x, w[p + "ffn_norm.weight"], eps)
        x = x + (silu(h @ w[f + "w1.weight"].T) * (h @ w[f + "w3.weight"].T)) @ w[f + "w2.weight"].T
    return rms(x, w["embedding_norm.weight"], eps)


def tiny_config(layer_types: list[str]) -> dict[str, Any]:
    return {
        "model_type": "lfm2",
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "layer_types": layer_types,
        "conv_L_cache": 3,
        "conv_bias": False,
        "norm_eps": 1e-5,
        "rope_theta": 10_000.0,
        "vocab_size": 64,
    }


def tiny_weights(c: dict[str, Any], seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    d, f, hd = c["hidden_size"], c["intermediate_size"], c["hidden_size"] // c["num_attention_heads"]

    def m(*shape: int, scale: float = 0.2) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def ones(n: int) -> np.ndarray:
        return (1 + rng.standard_normal(n) * 0.1).astype(np.float32)

    w = {"embed_tokens.weight": m(c["vocab_size"], d, scale=1.0), "embedding_norm.weight": ones(d)}
    for i, kind in enumerate(c["layer_types"]):
        p = f"layers.{i}."
        w |= {
            p + "operator_norm.weight": ones(d),
            p + "ffn_norm.weight": ones(d),
            p + "feed_forward.w1.weight": m(f, d),
            p + "feed_forward.w3.weight": m(f, d),
            p + "feed_forward.w2.weight": m(d, f),
        }
        if kind == "full_attention":
            a = p + "self_attn."
            w |= {
                a + "q_proj.weight": m(c["num_attention_heads"] * hd, d),
                a + "k_proj.weight": m(c["num_key_value_heads"] * hd, d),
                a + "v_proj.weight": m(c["num_key_value_heads"] * hd, d),
                a + "out_proj.weight": m(d, c["num_attention_heads"] * hd),
                a + "q_layernorm.weight": ones(hd),
                a + "k_layernorm.weight": ones(hd),
            }
        else:
            w |= {
                p + "conv.in_proj.weight": m(3 * d, d),
                p + "conv.out_proj.weight": m(d, d),
                p + "conv.conv.weight": m(d, 1, c["conv_L_cache"], scale=0.5),
            }
    return w
