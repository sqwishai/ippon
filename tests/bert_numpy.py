"""BERT, RoBERTa, XLM-RoBERTa and DistilBERT in plain numpy (float64), written from Hugging Face's modeling_bert,
modeling_roberta and modeling_distilbert rather than from ippon's code, for checking the MLX encoder on tiny
random checkpoints (tensor names as their checkpoints store them)."""

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


def names(family: str, i: int) -> dict[str, str]:
    if family == "distilbert":
        p = f"distilbert.transformer.layer.{i}."
        return {
            "q": p + "attention.q_lin",
            "k": p + "attention.k_lin",
            "v": p + "attention.v_lin",
            "o": p + "attention.out_lin",
            "attn_norm": p + "sa_layer_norm",
            "up": p + "ffn.lin1",
            "down": p + "ffn.lin2",
            "mlp_norm": p + "output_layer_norm",
        }
    root = "bert." if family == "bert" else "roberta."
    p = f"{root}encoder.layer.{i}."
    return {
        "q": p + "attention.self.query",
        "k": p + "attention.self.key",
        "v": p + "attention.self.value",
        "o": p + "attention.output.dense",
        "attn_norm": p + "attention.output.LayerNorm",
        "up": p + "intermediate.dense",
        "down": p + "output.dense",
        "mlp_norm": p + "output.LayerNorm",
    }


def forward(c: dict[str, Any], w: dict[str, np.ndarray], ids: list[int], types: list[int] | None = None) -> np.ndarray:
    """The last layer's hidden states [T, hidden] of one prompt; `types` its token types (all 0 by default)."""
    family = c["model_type"]
    distil = family == "distilbert"
    d = c["dim"] if distil else c["hidden_size"]
    heads = c["n_heads"] if distil else c["num_attention_heads"]
    layers = c["n_layers"] if distil else c["num_hidden_layers"]
    eps = 1e-12 if distil else c["layer_norm_eps"]
    root = {"bert": "bert.", "distilbert": "distilbert."}.get(family, "roberta.")
    e = root + "embeddings."
    t, hd = len(ids), d // heads
    offset = c["pad_token_id"] + 1 if family in ("roberta", "xlm-roberta") else 0
    x = (
        w[e + "word_embeddings.weight"][ids].astype(np.float64)
        + w[e + "position_embeddings.weight"][offset : offset + t]
    )
    if not distil:
        x = x + w[e + "token_type_embeddings.weight"][types or [0] * t]
    x = ln(x, w[e + "LayerNorm.weight"], w[e + "LayerNorm.bias"], eps)
    for i in range(layers):
        n = names(family, i)

        def lin(key: str, h: np.ndarray, _n: dict[str, str] = n) -> np.ndarray:
            return np.asarray(h @ w[_n[key] + ".weight"].T + w[_n[key] + ".bias"])

        q, k, v = (lin(key, x).reshape(t, heads, hd) for key in "qkv")
        scores = np.einsum("thd,shd->hts", q, k) / math.sqrt(hd)
        p_ = np.exp(scores - scores.max(-1, keepdims=True))
        p_ /= p_.sum(-1, keepdims=True)
        a = lin("o", np.einsum("hts,shd->thd", p_, v).reshape(t, d))
        x = ln(x + a, w[n["attn_norm"] + ".weight"], w[n["attn_norm"] + ".bias"], eps)
        f = lin("down", gelu(lin("up", x)))
        x = ln(x + f, w[n["mlp_norm"] + ".weight"], w[n["mlp_norm"] + ".bias"], eps)
    return x


def tiny_config(family: str) -> dict[str, Any]:
    if family == "distilbert":
        return {"model_type": "distilbert", "dim": 32, "hidden_dim": 48, "n_heads": 4, "n_layers": 2,
                "vocab_size": 64, "max_position_embeddings": 40, "activation": "gelu"}  # fmt: skip
    c: dict[str, Any] = {
        "model_type": family,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "layer_norm_eps": 1e-12,
        "vocab_size": 64,
        "max_position_embeddings": 40,
        "type_vocab_size": 2 if family == "bert" else 1,
        "hidden_act": "gelu",
        "position_embedding_type": "absolute",
    }
    if family in ("roberta", "xlm-roberta"):
        c |= {"pad_token_id": 1, "max_position_embeddings": 42}
    return c


def tiny_weights(c: dict[str, Any], seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    family = c["model_type"]
    distil = family == "distilbert"
    d, f = (c["dim"], c["hidden_dim"]) if distil else (c["hidden_size"], c["intermediate_size"])
    layers = c["n_layers"] if distil else c["num_hidden_layers"]

    def m(*shape: int, scale: float = 0.2) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def ones(n: int) -> np.ndarray:
        return (1 + rng.standard_normal(n) * 0.1).astype(np.float32)

    root = {"bert": "bert.", "distilbert": "distilbert."}.get(family, "roberta.")
    e = root + "embeddings."
    w = {
        e + "word_embeddings.weight": m(c["vocab_size"], d, scale=1.0),
        e + "position_embeddings.weight": m(c["max_position_embeddings"], d, scale=0.5),
        e + "LayerNorm.weight": ones(d),
        e + "LayerNorm.bias": m(d, scale=0.1),
    }
    if not distil:
        w[e + "token_type_embeddings.weight"] = m(c["type_vocab_size"], d, scale=0.5)
    for i in range(layers):
        n = names(family, i)
        for key, shape in (
            ("q", (d, d)),
            ("k", (d, d)),
            ("v", (d, d)),
            ("o", (d, d)),
            ("up", (f, d)),
            ("down", (d, f)),
        ):
            w[n[key] + ".weight"], w[n[key] + ".bias"] = m(*shape), m(shape[0], scale=0.1)
        for key in ("attn_norm", "mlp_norm"):
            w[n[key] + ".weight"], w[n[key] + ".bias"] = ones(d), m(d, scale=0.1)
    return w
