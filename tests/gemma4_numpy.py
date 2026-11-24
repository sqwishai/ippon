"""Gemma 4 text in plain numpy (float64), written from Hugging Face's modeling_gemma4 (transformers 5.17) rather than
from ippon's code, for checking the MLX decoder on tiny random checkpoints: Gemma4RMSNorm's plain weight, the
scaled embedding, attention without a 1/sqrt(d) scale, sliding and full layers with their own head sizes, values from
the key projection on full layers (attention_k_eq_v) through a weightless norm, default and proportional RoPE,
masking_utils' sliding window (a key is seen when kv_idx > q_idx - sliding_window), the MLP beside the experts
(Gemma4TextRouter, Gemma4TextExperts) and each layer's layer_scalar. `scripts/check_gemma4_numpy.py` checks this
module against transformers itself."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

ROOT = "model.language_model."


def rms(x: np.ndarray, w: np.ndarray | None, eps: float) -> np.ndarray:
    y = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return np.asarray(y if w is None else y * w)


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    return np.asarray(0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3))))


def inv_freq(c: dict[str, Any], kind: str, d: int) -> np.ndarray:
    p = c["rope_parameters"][kind]
    if p.get("rope_type", "default") == "default":
        return np.asarray(1.0 / p["rope_theta"] ** (np.arange(0, d, 2) / d))
    angles = int(p.get("partial_rotary_factor", 1.0) * d // 2)
    rotated = 1.0 / p["rope_theta"] ** (np.arange(0, 2 * angles, 2) / d)
    return np.asarray(np.concatenate([rotated, np.zeros(d // 2 - angles)]) / p.get("factor", 1.0))


def tables(inv: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
    angles = np.arange(t)[:, None] * inv[None]
    emb = np.concatenate([angles, angles], -1)
    return np.cos(emb), np.sin(emb)


def rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    d = x.shape[-1]
    cos, sin = cos[:, None], sin[:, None]
    a, b = x[..., : d // 2], x[..., d // 2 :]
    return np.asarray(x * cos + np.concatenate([-b, a], -1) * sin)


def experts(c: dict[str, Any], w: dict[str, np.ndarray], p: str, x: np.ndarray) -> np.ndarray:
    """Gemma4TextRouter and Gemma4TextExperts on the residual x [T, hidden]."""
    eps, k = c["rms_norm_eps"], c["top_k_experts"]
    r = rms(x, None, eps) * w[p + "router.scale"] * c["hidden_size"] ** -0.5
    logits = r @ w[p + "router.proj.weight"].T
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    h = rms(x, w[p + "pre_feedforward_layernorm_2.weight"], eps)
    gate_up, down, width = w[p + "experts.gate_up_proj"], w[p + "experts.down_proj"], c["moe_intermediate_size"]
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        chosen = np.argsort(-probs[t], kind="stable")[:k]
        weights = probs[t, chosen] / probs[t, chosen].sum() * w[p + "router.per_expert_scale"][chosen]
        for e, weight in zip(chosen, weights, strict=True):
            gu = gate_up[e] @ h[t]
            out[t] += weight * (down[e] @ (gelu_tanh(gu[:width]) * gu[width:]))
    return out


def forward(
    c: dict[str, Any],
    w: dict[str, np.ndarray],
    ids: list[int],
    rope_tables: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    embed_scale: float | None = None,
    prefix_len: int | None = None,
) -> np.ndarray:
    """Final-normed hidden states [T, hidden] of one prompt. `rope_tables` (cos, sin [T, head_dim] by layer type) and
    `embed_scale` replace the exact ones (transformers computes both in float32 whatever the model's dtype). With
    `prefix_len`, every layer uses Strands Decider v2's prefix-bidirectional mask (no sliding window): state tokens
    `< prefix_len` attend to `< prefix_len`, question tokens `>= prefix_len` attend to every token."""
    heads, eps, t = c["num_attention_heads"], c["rms_norm_eps"], len(ids)
    scale = math.sqrt(c["hidden_size"]) if embed_scale is None else embed_scale
    x = w[ROOT + "embed_tokens.weight"][ids].astype(np.float64) * scale
    ple_dim = int(c.get("hidden_size_per_layer_input") or 0)
    ple: np.ndarray | None = None
    if ple_dim:
        n_layers = len(c["layer_types"])
        ple_vocab = int(c.get("vocab_size_per_layer_input") or c["vocab_size"])
        ple_ids = [i if i < ple_vocab else 0 for i in ids]
        tok_ple = (w[ROOT + "embed_tokens_per_layer.weight"][ple_ids].astype(np.float64) * math.sqrt(ple_dim)).reshape(
            t, n_layers, ple_dim
        )
        mod_ple = (x @ w[ROOT + "per_layer_model_projection.weight"].T) * (c["hidden_size"] ** -0.5)
        mod_ple = rms(mod_ple.reshape(t, n_layers, ple_dim), w[ROOT + "per_layer_projection_norm.weight"], eps)
        ple = (mod_ple + tok_ple) * (2.0**-0.5)
    kv_shared = int(c.get("num_kv_shared_layers") or 0)
    first_shared = len(c["layer_types"]) - kv_shared
    shared_kv: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    q_i, k_i = np.arange(t)[:, None], np.arange(t)[None, :]
    for i, kind in enumerate(c["layer_types"]):
        p, a = f"{ROOT}layers.{i}.", f"{ROOT}layers.{i}.self_attn."
        full = kind == "full_attention"
        hd = c["global_head_dim"] if full else c["head_dim"]
        kv = c["num_global_key_value_heads"] if full and c["attention_k_eq_v"] else c["num_key_value_heads"]
        h = rms(x, w[p + "input_layernorm.weight"], eps)
        q = rms((h @ w[a + "q_proj.weight"].T).reshape(t, heads, hd), w[a + "q_norm.weight"], eps)
        cos, sin = rope_tables[kind] if rope_tables else tables(inv_freq(c, kind, hd), t)
        q = rope(q, cos, sin)
        if i < first_shared:
            k_raw = (h @ w[a + "k_proj.weight"].T).reshape(t, kv, hd)
            v_raw = k_raw if full and c["attention_k_eq_v"] else (h @ w[a + "v_proj.weight"].T).reshape(t, kv, hd)
            k, v = rope(rms(k_raw, w[a + "k_norm.weight"], eps), cos, sin), rms(v_raw, None, eps)
            shared_kv[kind] = (k, v)
        else:
            k, v = shared_kv[kind]
        k_rep, v_rep = np.repeat(k, heads // kv, axis=1), np.repeat(v, heads // kv, axis=1)
        scores = np.einsum("thd,shd->hts", q, k_rep)  # scaling 1.0
        if prefix_len is not None:
            seen = np.where(q_i < prefix_len, k_i < prefix_len, np.ones_like(k_i, dtype=bool))
        else:
            seen = k_i <= q_i
            if full and c.get("bidirectional") == "full":  # Jeff 1.1: full layers read every position
                seen = np.ones_like(seen)
            if not full:
                seen &= k_i > q_i - c["sliding_window"]
        scores = np.where(seen[None], scores, -np.inf)
        probs = np.exp(scores - scores.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        o = np.einsum("hts,shd->thd", probs, v_rep).reshape(t, heads * hd) @ w[a + "o_proj.weight"].T
        x = x + rms(o, w[p + "post_attention_layernorm.weight"], eps)
        h = rms(x, w[p + "pre_feedforward_layernorm.weight"], eps)
        m = p + "mlp."
        f = (gelu_tanh(h @ w[m + "gate_proj.weight"].T) * (h @ w[m + "up_proj.weight"].T)) @ w[m + "down_proj.weight"].T
        if c.get("enable_moe_block"):
            f = rms(f, w[p + "post_feedforward_layernorm_1.weight"], eps)
            f = f + rms(experts(c, w, p, x), w[p + "post_feedforward_layernorm_2.weight"], eps)
        x = x + rms(f, w[p + "post_feedforward_layernorm.weight"], eps)
        if ple is not None:
            g = gelu_tanh(x @ w[p + "per_layer_input_gate.weight"].T) * ple[:, i, :]
            x = x + rms(g @ w[p + "per_layer_projection.weight"].T, w[p + "post_per_layer_input_norm.weight"], eps)
        x = x * w[p + "layer_scalar"][0]
    return rms(x, w[ROOT + "norm.weight"], eps)


def tiny_config(**extra: Any) -> dict[str, Any]:
    return {
        "model_type": "gemma4_text",
        "hidden_size": 32,
        "intermediate_size": 24,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_global_key_value_heads": 1,
        "head_dim": 16,
        "global_head_dim": 32,
        "attention_k_eq_v": True,
        "rms_norm_eps": 1e-6,
        "sliding_window": 5,  # small: a 19-token prompt crosses it
        "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1_000_000.0},
        },
        "hidden_activation": "gelu_pytorch_tanh",
        "enable_moe_block": True,
        "num_experts": 8,
        "top_k_experts": 2,
        "moe_intermediate_size": 12,
        "hidden_size_per_layer_input": 0,
        "num_kv_shared_layers": 0,
        "final_logit_softcapping": 30.0,
        "vocab_size": 64,
        "tie_word_embeddings": True,
    } | extra


def tiny_weights(c: dict[str, Any], seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    d, f, heads = c["hidden_size"], c["intermediate_size"], c["num_attention_heads"]
    e, width = c["num_experts"], c["moe_intermediate_size"]
    ple_dim = int(c.get("hidden_size_per_layer_input") or 0)
    kv_shared = int(c.get("num_kv_shared_layers") or 0)
    first_shared = len(c["layer_types"]) - kv_shared
    double_wide = bool(c.get("use_double_wide_mlp", False))

    def m(*shape: int, scale: float = 0.2) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def norm(n: int) -> np.ndarray:  # Gemma 4 applies w itself: around 1
        return (1.0 + rng.standard_normal(n) * 0.1).astype(np.float32)

    w = {ROOT + "embed_tokens.weight": m(c["vocab_size"], d, scale=0.3), ROOT + "norm.weight": norm(d)}
    if ple_dim:
        n_layers = len(c["layer_types"])
        ple_vocab = int(c.get("vocab_size_per_layer_input") or c["vocab_size"])
        w |= {
            ROOT + "embed_tokens_per_layer.weight": m(ple_vocab, n_layers * ple_dim, scale=0.3),
            ROOT + "per_layer_model_projection.weight": m(n_layers * ple_dim, d),
            ROOT + "per_layer_projection_norm.weight": norm(ple_dim),
        }
    for i, kind in enumerate(c["layer_types"]):
        p, a = f"{ROOT}layers.{i}.", f"{ROOT}layers.{i}.self_attn."
        full = kind == "full_attention"
        hd = c["global_head_dim"] if full else c["head_dim"]
        kv = c["num_global_key_value_heads"] if full and c["attention_k_eq_v"] else c["num_key_value_heads"]
        mlp_w = 2 * f if double_wide and i >= first_shared else f
        w |= {
            p + "input_layernorm.weight": norm(d),
            p + "post_attention_layernorm.weight": norm(d),
            p + "pre_feedforward_layernorm.weight": norm(d),
            p + "post_feedforward_layernorm.weight": norm(d),
            p + "layer_scalar": np.array([0.5 + 0.25 * i], dtype=np.float32),
            a + "q_proj.weight": m(heads * hd, d),
            a + "k_proj.weight": m(kv * hd, d),
            a + "o_proj.weight": m(d, heads * hd),
            a + "q_norm.weight": norm(hd),
            a + "k_norm.weight": norm(hd),
            p + "mlp.gate_proj.weight": m(mlp_w, d),
            p + "mlp.up_proj.weight": m(mlp_w, d),
            p + "mlp.down_proj.weight": m(d, mlp_w),
        }
        if not (full and c["attention_k_eq_v"]):
            w[a + "v_proj.weight"] = m(kv * hd, d)
        if ple_dim:
            w |= {
                p + "per_layer_input_gate.weight": m(ple_dim, d),
                p + "per_layer_projection.weight": m(d, ple_dim),
                p + "post_per_layer_input_norm.weight": norm(d),
            }
        if c.get("enable_moe_block"):
            w |= {
                p + "router.proj.weight": m(e, d, scale=0.1),  # spread: the top-k hold part of the mass
                p + "router.scale": norm(d),
                p + "router.per_expert_scale": norm(e),
                p + "experts.gate_up_proj": m(e, 2 * width, d),
                p + "experts.down_proj": m(e, d, width),
                p + "pre_feedforward_layernorm_2.weight": norm(d),
                p + "post_feedforward_layernorm_1.weight": norm(d),
                p + "post_feedforward_layernorm_2.weight": norm(d),
            }
    return w
