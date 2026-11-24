"""DeBERTa-v2/v3 in plain numpy (float64), written from Hugging Face's modeling_deberta_v2 (DisentangledSelfAttention
with share_att_key and c2p|p2c, build_relative_position, make_log_bucket_position) rather than from ippon's
code: the gathers are torch.gather's, step by step, for checking the MLX encoder on tiny random checkpoints."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def ln(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return np.asarray((x - mean) / np.sqrt(var + eps) * w + b)


def gelu(x: np.ndarray) -> np.ndarray:
    return np.asarray(0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2))))


def log_bucket(rel: np.ndarray, bucket_size: int, max_position: int) -> np.ndarray:
    sign = np.sign(rel)
    mid = bucket_size // 2
    abs_pos = np.where((rel < mid) & (rel > -mid), mid - 1, np.abs(rel)).astype(np.float32)
    log_pos = np.ceil(np.log(abs_pos / mid) / np.log(np.float32((max_position - 1) / mid)) * (mid - 1)) + mid
    return np.where(abs_pos <= mid, rel, log_pos * sign).astype(np.int64)


def forward(c: dict[str, Any], w: dict[str, np.ndarray], ids: list[int]) -> np.ndarray:
    """The last layer's hidden states [T, hidden] of one prompt."""
    d, heads, eps = c["hidden_size"], c["num_attention_heads"], c["layer_norm_eps"]
    hd, t = d // heads, len(ids)
    buckets, max_rel = c["position_buckets"], c["max_relative_positions"]
    span = buckets if buckets > 0 else max_rel
    e = "deberta.embeddings."
    x = w[e + "word_embeddings.weight"][ids].astype(np.float64)
    if c.get("position_biased_input", True):
        x = x + w[e + "position_embeddings.weight"][:t]
    x = ln(x, w[e + "LayerNorm.weight"], w[e + "LayerNorm.bias"], eps)
    rel_emb = w["deberta.encoder.rel_embeddings.weight"].astype(np.float64)[: 2 * span]
    rel_emb = ln(rel_emb, w["deberta.encoder.LayerNorm.weight"], w["deberta.encoder.LayerNorm.bias"], eps)
    rel = np.arange(t)[:, None] - np.arange(t)[None, :]
    if buckets > 0:
        rel = log_bucket(rel, buckets, max_rel)
    for i in range(c["num_hidden_layers"]):
        p = f"deberta.encoder.layer.{i}."

        def lin(name: str, h: np.ndarray, _p: str = p) -> np.ndarray:
            return np.asarray(h @ w[_p + name + ".weight"].T + w[_p + name + ".bias"])

        def split(y: np.ndarray) -> np.ndarray:  # [n, d] -> [heads, n, hd]
            return np.asarray(y.reshape(y.shape[0], heads, hd).transpose(1, 0, 2))

        q, k, v = (split(lin(f"attention.self.{x_}_proj", x)) for x_ in ("query", "key", "value"))
        pos_q, pos_k = split(lin("attention.self.query_proj", rel_emb)), split(lin("attention.self.key_proj", rel_emb))
        scale = math.sqrt(hd * 3)
        scores = q @ k.transpose(0, 2, 1) / scale
        c2p = q @ pos_k.transpose(0, 2, 1)  # [heads, query, 2 span]
        c2p_pos = np.clip(rel + span, 0, 2 * span - 1)
        c2p = np.take_along_axis(c2p, np.broadcast_to(c2p_pos, (heads, t, t)), axis=-1)  # torch.gather(dim=-1)
        p2c = k @ pos_q.transpose(0, 2, 1)  # [heads, key, 2 span]
        p2c_pos = np.clip(-rel + span, 0, 2 * span - 1)
        p2c = np.take_along_axis(p2c, np.broadcast_to(p2c_pos, (heads, t, t)), axis=-1).transpose(0, 2, 1)
        scores = scores + c2p / scale + p2c / scale
        probs = np.exp(scores - scores.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        context = (probs @ v).transpose(1, 0, 2).reshape(t, d)
        a = lin("attention.output.dense", context)
        x = ln(a + x, w[p + "attention.output.LayerNorm.weight"], w[p + "attention.output.LayerNorm.bias"], eps)
        f = lin("output.dense", gelu(lin("intermediate.dense", x)))
        x = ln(f + x, w[p + "output.LayerNorm.weight"], w[p + "output.LayerNorm.bias"], eps)
    return x


def tiny_config(**extra: Any) -> dict[str, Any]:
    return {
        "model_type": "deberta-v2",
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "layer_norm_eps": 1e-7,
        "vocab_size": 64,
        "max_position_embeddings": 64,
        "position_biased_input": False,
        "type_vocab_size": 0,
        "relative_attention": True,
        "share_att_key": True,
        "pos_att_type": ["p2c", "c2p"],
        "position_buckets": 8,  # small, so a 20-token prompt's distances fall into log buckets
        "max_relative_positions": 64,
        "norm_rel_ebd": "layer_norm",
        "hidden_act": "gelu",
        "pooler_hidden_size": 32,
        "pooler_hidden_act": "gelu",
    } | extra


def tiny_weights(c: dict[str, Any], seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    d, f = c["hidden_size"], c["intermediate_size"]
    span = c["position_buckets"] if c["position_buckets"] > 0 else c["max_relative_positions"]

    def m(*shape: int, scale: float = 0.2) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def ones(n: int) -> np.ndarray:
        return (1 + rng.standard_normal(n) * 0.1).astype(np.float32)

    e = "deberta.embeddings."
    w = {
        e + "word_embeddings.weight": m(c["vocab_size"], d, scale=1.0),
        e + "LayerNorm.weight": ones(d),
        e + "LayerNorm.bias": m(d, scale=0.1),
        "deberta.encoder.rel_embeddings.weight": m(2 * span, d, scale=1.0),
        "deberta.encoder.LayerNorm.weight": ones(d),
        "deberta.encoder.LayerNorm.bias": m(d, scale=0.1),
    }
    if c.get("position_biased_input", True):
        w[e + "position_embeddings.weight"] = m(c["max_position_embeddings"], d, scale=0.5)
    for i in range(c["num_hidden_layers"]):
        p = f"deberta.encoder.layer.{i}."
        for name, shape in (
            ("attention.self.query_proj", (d, d)),
            ("attention.self.key_proj", (d, d)),
            ("attention.self.value_proj", (d, d)),
            ("attention.output.dense", (d, d)),
            ("intermediate.dense", (f, d)),
            ("output.dense", (d, f)),
        ):
            w[p + name + ".weight"], w[p + name + ".bias"] = m(*shape), m(shape[0], scale=0.1)
        for name in ("attention.output.LayerNorm", "output.LayerNorm"):
            w[p + name + ".weight"], w[p + name + ".bias"] = ones(d), m(d, scale=0.1)
    return w
