"""LFM3 under d1's set attention in plain numpy (float64), written from Liquid's d1-2B code (modeling_lfm3.py and
attention.py) rather than from ippon's, for checking the MLX implementation on tiny random checkpoints."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from lfm2_numpy import rms, silu


def rope(x: np.ndarray, theta: float, rotated: float) -> np.ndarray:
    """rotate_half RoPE, x [T, H, D] at positions 0..T-1: the first `rotated` share of the angle pairs at the whole
    head's frequencies, the rest at frequency zero (modeling_lfm3.py's zero tail)."""
    t, _, d = x.shape
    angles = int(rotated * d) // 2
    inv = np.concatenate([1.0 / theta ** (np.arange(0, 2 * angles, 2) / d), np.zeros(d // 2 - angles)])
    a = np.arange(t)[:, None] * inv[None]
    cos, sin = np.cos(np.concatenate([a, a], -1))[:, None], np.sin(np.concatenate([a, a], -1))[:, None]
    lo, hi = x[..., : d // 2], x[..., d // 2 :]
    return np.asarray(x * cos + np.concatenate([-hi, lo], -1) * sin)


def forward(
    c: dict[str, Any], w: dict[str, np.ndarray], ids: list[int], groups: Sequence[int] | None = None
) -> np.ndarray:
    """Final-normed hidden states [T, hidden] of one prompt. `groups`: each token's question (-1 on the state); a
    full attention layer's state token sees every token, a question token the state and its own question
    (attention.py's Sees.rows). None: one question, every token sees every token."""
    d, heads, kv, eps = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["norm_eps"]
    hd, t = c["head_dim"], len(ids)
    g = np.full(t, -1) if groups is None else np.asarray(groups)
    full = (g[:, None] < 0) | (g[None, :] < 0) | (g[:, None] == g[None, :])
    gap = np.arange(t)[:, None] - np.arange(t)[None, :]
    sliding = (gap >= 0) & (gap < c["sliding_window"])
    x = w["embed_tokens.weight"][ids].astype(np.float64)
    for i, kind in enumerate(c["layer_types"]):
        p = f"layers.{i}."
        h = rms(x, w[p + "operator_norm.weight"], eps)
        if kind == "conv":
            proj = h @ w[p + "conv.in_proj.weight"].T
            b, cc, u = proj[:, :d], proj[:, d : 2 * d], proj[:, 2 * d :]
            taps = w[p + "conv.conv.weight"][:, 0, :]  # [d, kernel]
            kernel = taps.shape[1]
            padded = np.concatenate([np.zeros((kernel - 1, d)), b * u])
            y = (cc * sum(padded[j : j + t] * taps[:, j] for j in range(kernel))) @ w[p + "conv.out_proj.weight"].T
        else:
            a = p + "self_attn."
            params = c["rope_parameters"][kind]
            q = rms((h @ w[a + "q_proj.weight"].T).reshape(t, heads, hd), w[a + "q_norm.weight"], eps)
            k = rms((h @ w[a + "k_proj.weight"].T).reshape(t, kv, hd), w[a + "k_norm.weight"], eps)
            v = (h @ w[a + "v_proj.weight"].T).reshape(t, kv, hd)
            rotated = params.get("partial_rotary_factor", 1.0)
            q, k = rope(q, params["rope_theta"], rotated), rope(k, params["rope_theta"], rotated)
            k, v = np.repeat(k, heads // kv, axis=1), np.repeat(v, heads // kv, axis=1)
            scores = np.einsum("thd,shd->hts", q, k) * hd**-0.5
            scores = np.where((full if kind == "full_attention" else sliding)[None], scores, -np.inf)
            p_ = np.exp(scores - scores.max(-1, keepdims=True))
            p_ /= p_.sum(-1, keepdims=True)
            o = np.einsum("hts,shd->thd", p_, v)
            gate = 1 / (1 + np.exp(-(h @ w[a + "gate_proj.weight"].T)))  # [t, heads]
            y = (o * gate[..., None]).reshape(t, heads * hd) @ w[a + "o_proj.weight"].T
        x = x + rms(y, w[p + "post_operator_norm.weight"], eps)
        f = p + "feed_forward."
        h = rms(x, w[p + "ffn_norm.weight"], eps)
        mlp = (silu(h @ w[f + "w1.weight"].T) * (h @ w[f + "w3.weight"].T)) @ w[f + "w2.weight"].T
        x = x + rms(mlp, w[p + "post_ffn_norm.weight"], eps)
    return rms(x, w["embedding_norm.weight"], eps)


def tiny_config(layer_types: list[str], window: int = 4) -> dict[str, Any]:
    return {
        "model_type": "lfm3",
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "layer_types": layer_types,
        "conv_L_cache": 7,
        "conv_bias": False,
        "norm_eps": 1e-5,
        "rope_parameters": {
            "full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 1_000_000.0, "rope_type": "default"},
            "sliding_attention": {"rope_theta": 10_000.0, "rope_type": "default"},
        },
        "sliding_window": window,
        "use_sandwich_norm": True,
        "tie_word_embeddings": True,
        "vocab_size": 64,
    }


def tiny_weights(c: dict[str, Any], seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    d, f, hd, heads = c["hidden_size"], c["intermediate_size"], c["head_dim"], c["num_attention_heads"]

    def m(*shape: int, scale: float = 0.2) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def ones(n: int) -> np.ndarray:
        return (1 + rng.standard_normal(n) * 0.1).astype(np.float32)

    w = {"embed_tokens.weight": m(c["vocab_size"], d, scale=1.0), "embedding_norm.weight": ones(d)}
    for i, kind in enumerate(c["layer_types"]):
        p = f"layers.{i}."
        w |= {
            p + "operator_norm.weight": ones(d),
            p + "post_operator_norm.weight": ones(d),
            p + "ffn_norm.weight": ones(d),
            p + "post_ffn_norm.weight": ones(d),
            p + "feed_forward.w1.weight": m(f, d),
            p + "feed_forward.w3.weight": m(f, d),
            p + "feed_forward.w2.weight": m(d, f),
        }
        if kind == "conv":
            w |= {
                p + "conv.in_proj.weight": m(3 * d, d),
                p + "conv.out_proj.weight": m(d, d),
                p + "conv.conv.weight": m(d, 1, c["conv_L_cache"], scale=0.5),
            }
        else:
            a = p + "self_attn."
            w |= {
                a + "q_proj.weight": m(heads * hd, d),
                a + "k_proj.weight": m(c["num_key_value_heads"] * hd, d),
                a + "v_proj.weight": m(c["num_key_value_heads"] * hd, d),
                a + "gate_proj.weight": m(heads, d, scale=1.0),
                a + "o_proj.weight": m(d, heads * hd),
                a + "q_norm.weight": ones(hd),
                a + "k_norm.weight": ones(hd),
            }
    return w
