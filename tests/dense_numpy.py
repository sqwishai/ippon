"""Dense decoders (Llama, Mistral, Qwen2, Qwen3) in plain numpy (float64), written from Hugging Face's
modeling_llama / modeling_qwen2 / modeling_qwen3 and its RoPE initialisation (_compute_llama3_parameters,
_compute_yarn_parameters), not from ippon's code, for checking the MLX implementation on tiny random
checkpoints. A config's "norm_groups" (no Hugging Face field; set by tests) takes each RMSNorm's statistics over that
many equal groups of the channels."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def rms(x: np.ndarray, w: np.ndarray, eps: float, groups: int = 1) -> np.ndarray:
    g = x.reshape(*x.shape[:-1], groups, x.shape[-1] // groups)
    return np.asarray((g / np.sqrt((g * g).mean(-1, keepdims=True) + eps)).reshape(x.shape) * w)


def silu(x: np.ndarray) -> np.ndarray:
    return np.asarray(x / (1 + np.exp(-x)))


def inv_freq(c: dict[str, Any], d: int) -> tuple[np.ndarray, float, float]:
    """(inv_freq, position factor, the factor on cos and sin) as transformers initialises them: default, "linear",
    "llama3" or "yarn"."""
    p = c.get("rope_scaling") or {}
    base = float(c.get("rope_theta", 10_000.0))
    inv = 1.0 / base ** (np.arange(0, d, 2, dtype=np.float64) / d)
    kind = p.get("rope_type", "default")
    if kind == "linear":
        return inv, 1.0 / float(p["factor"]), 1.0
    if kind == "yarn":
        factor, old = float(p["factor"]), float(p["original_max_position_embeddings"])
        attention = p.get("attention_factor") or (0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0)

        def dim(rotations: float) -> float:
            return (d * math.log(old / (rotations * 2 * math.pi))) / (2 * math.log(base))

        low, high = dim(p.get("beta_fast", 32)), dim(p.get("beta_slow", 1))
        if p.get("truncate", True):
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, d - 1)
        if low == high:
            high += 0.001
        ramp = np.clip((np.arange(d // 2, dtype=np.float64) - low) / (high - low), 0, 1)
        extrapolation = 1 - ramp
        return inv / factor * (1 - extrapolation) + inv * extrapolation, 1.0, float(attention)
    if kind == "llama3":
        factor, low, high = p["factor"], p["low_freq_factor"], p["high_freq_factor"]
        old = p["original_max_position_embeddings"]
        low_wavelen, high_wavelen = old / low, old / high
        wavelen = 2 * math.pi / inv
        llama = np.where(wavelen > low_wavelen, inv / factor, inv)
        smooth = (old / wavelen - low) / (high - low)
        smoothed = (1 - smooth) * llama / factor + smooth * llama
        medium = ~(wavelen < high_wavelen) * ~(wavelen > low_wavelen)
        return np.where(medium, smoothed, llama), 1.0, 1.0
    return inv, 1.0, 1.0


def rope(x: np.ndarray, c: dict[str, Any]) -> np.ndarray:
    """rotate_half RoPE; x [T, H, D] at positions 0..T-1."""
    t, _, d = x.shape
    inv, factor, attention = inv_freq(c, d)
    angles = (np.arange(t) * factor)[:, None] * inv[None]
    emb = np.concatenate([angles, angles], -1)
    cos, sin = np.cos(emb)[:, None] * attention, np.sin(emb)[:, None] * attention
    a, b = x[..., : d // 2], x[..., d // 2 :]
    return np.asarray(x * cos + np.concatenate([-b, a], -1) * sin)


def forward(c: dict[str, Any], w: dict[str, np.ndarray], ids: list[int]) -> np.ndarray:
    """Final-normed hidden states [T, hidden] of one prompt (causal)."""
    d, heads, kv, eps = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["rms_norm_eps"]
    hd, t = c.get("head_dim") or d // heads, len(ids)
    family, groups = c["model_type"], c.get("norm_groups", 1)
    x = w["model.embed_tokens.weight"][ids].astype(np.float64)
    for i in range(c["num_hidden_layers"]):
        p, a = f"model.layers.{i}.", f"model.layers.{i}.self_attn."

        def proj(name: str, h: np.ndarray, _a: str = a) -> np.ndarray:
            y = np.asarray(h @ w[_a + name + ".weight"].T)
            bias = w.get(_a + name + ".bias")
            return np.asarray(y + bias) if bias is not None else y

        h = rms(x, w[p + "input_layernorm.weight"], eps, groups)
        q = proj("q_proj", h).reshape(t, heads, hd)
        k = proj("k_proj", h).reshape(t, kv, hd)
        v = proj("v_proj", h).reshape(t, kv, hd)
        if family == "qwen3":
            q, k = rms(q, w[a + "q_norm.weight"], eps), rms(k, w[a + "k_norm.weight"], eps)
        q, k = rope(q, c), rope(k, c)
        k, v = np.repeat(k, heads // kv, axis=1), np.repeat(v, heads // kv, axis=1)
        scores = np.einsum("thd,shd->hts", q, k) * hd**-0.5
        scores = np.where(np.tril(np.ones((t, t), bool))[None], scores, -np.inf)
        p_ = np.exp(scores - scores.max(-1, keepdims=True))
        p_ /= p_.sum(-1, keepdims=True)
        x = x + proj("o_proj", np.einsum("hts,shd->thd", p_, v).reshape(t, heads * hd))
        m = p + "mlp."
        h = rms(x, w[p + "post_attention_layernorm.weight"], eps, groups)
        x = x + (silu(h @ w[m + "gate_proj.weight"].T) * (h @ w[m + "up_proj.weight"].T)) @ w[m + "down_proj.weight"].T
    return rms(x, w["model.norm.weight"], eps, groups)


def tiny_config(family: str, **extra: Any) -> dict[str, Any]:
    c: dict[str, Any] = {
        "model_type": family,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "vocab_size": 64,
        "tie_word_embeddings": False,
    }
    if family == "qwen3":
        c["head_dim"] = 16  # wider than hidden / heads, as Qwen3's are
    return c | extra


def tiny_weights(c: dict[str, Any], seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    d, f, heads, kv = c["hidden_size"], c["intermediate_size"], c["num_attention_heads"], c["num_key_value_heads"]
    hd = c.get("head_dim") or d // heads
    bias = c["model_type"] == "qwen2" or c.get("attention_bias", False)

    def m(*shape: int, scale: float = 0.2) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def ones(n: int) -> np.ndarray:
        return (1 + rng.standard_normal(n) * 0.1).astype(np.float32)

    w = {"model.embed_tokens.weight": m(c["vocab_size"], d, scale=1.0), "model.norm.weight": ones(d)}
    if not c.get("tie_word_embeddings"):
        w["lm_head.weight"] = m(c["vocab_size"], d)
    for i in range(c["num_hidden_layers"]):
        p, a = f"model.layers.{i}.", f"model.layers.{i}.self_attn."
        w |= {
            p + "input_layernorm.weight": ones(d),
            p + "post_attention_layernorm.weight": ones(d),
            a + "q_proj.weight": m(heads * hd, d),
            a + "k_proj.weight": m(kv * hd, d),
            a + "v_proj.weight": m(kv * hd, d),
            a + "o_proj.weight": m(d, heads * hd),
            p + "mlp.gate_proj.weight": m(f, d),
            p + "mlp.up_proj.weight": m(f, d),
            p + "mlp.down_proj.weight": m(d, f),
        }
        if bias:
            w |= {a + f"{x}_proj.bias": m(n, scale=0.5) for x, n in (("q", heads * hd), ("k", kv * hd), ("v", kv * hd))}
        if c.get("attention_bias"):
            w[a + "o_proj.bias"] = m(d, scale=0.5)
        if c["model_type"] == "qwen3":
            w |= {a + "q_norm.weight": ones(hd), a + "k_norm.weight": ones(hd)}
    return w
