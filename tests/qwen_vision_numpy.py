"""Qwen3.5's vision tower and patch merger in plain numpy (float64), written from transformers' modeling_qwen3_5.py
(Qwen3_5VisionModel and its parts) and F.interpolate's align_corners=True bilinear rather than from ippon's,
for checking the MLX QwenVision on tiny random checkpoints: each image (or frame) read alone, as transformers splits
its packed rows at cu_seqlens."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return np.asarray((x - mean) / np.sqrt(var + eps) * w + b)


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    return np.asarray(0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3))))


def gelu(x: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf, otypes=[np.float64])
    return np.asarray(0.5 * x * (1 + erf(x / math.sqrt(2))))


def interpolation(side: int, n: int) -> np.ndarray:
    """F.interpolate(bilinear, align_corners=True) along one axis as a dense [n, side] matrix: output i reads the
    table at i * (side - 1) / (n - 1), between its two neighbours."""
    out = np.zeros((n, side))
    for i in range(n):
        src = i * (side - 1) / (n - 1) if n > 1 else 0.0
        lo = min(math.floor(src), side - 1)
        frac = src - lo
        out[i, lo] += 1 - frac
        if frac:
            out[i, lo + 1] += frac
    return out


def block_cells(h: int, w: int, merge: int) -> list[tuple[int, int]]:
    """The (row, column) of each patch in merge-block order: blocks row-major, each block's patches row-major."""
    return [
        (br * merge + ir, bc * merge + ic)
        for br in range(h // merge)
        for bc in range(w // merge)
        for ir in range(merge)
        for ic in range(merge)
    ]


def rotate(x: np.ndarray, rows: np.ndarray, columns: np.ndarray, theta: float) -> np.ndarray:
    """The axial RoPE on [n, heads, head_dim]: of each head's first half, the first quarter's pairs (i, i + half)
    turn by row x freq, the second quarter's by column x freq, freq = theta ** (-2j / half)."""
    half = x.shape[-1] // 2
    inv = theta ** (-np.arange(0, half, 2) / half)
    angles = np.concatenate([rows[:, None] * inv, columns[:, None] * inv], axis=-1)[:, None, :]  # [n, 1, half]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * np.cos(angles) - x2 * np.sin(angles), x2 * np.cos(angles) + x1 * np.sin(angles)], -1)


def frame(c: dict[str, Any], w: dict[str, np.ndarray], pixels: np.ndarray, h: int, wd: int) -> np.ndarray:
    """Qwen3_5VisionModel's pooler_output (merged rows) for one frame's patches [h * wd, patch dim]."""
    v = c["vision_config"]
    d, heads, merge = v["hidden_size"], v["num_heads"], v["spatial_merge_size"]
    hd, n = d // heads, pixels.shape[0]
    side = math.isqrt(v["num_position_embeddings"])
    kernel = w["patch_embed.proj.weight"]  # [d, channels, temporal, patch, patch]
    x = np.einsum("nctpq,dctpq->nd", pixels.reshape(n, *kernel.shape[1:]), kernel) + w["patch_embed.proj.bias"]
    table = w["pos_embed.weight"].reshape(side, side, d)
    raster = np.einsum("hs,swd->hwd", interpolation(side, h), table)
    raster = np.einsum("wt,htd->hwd", interpolation(side, wd), raster)
    cells = block_cells(h, wd, merge)
    x = x + np.stack([raster[r, col] for r, col in cells])
    rows = np.array([r for r, _ in cells], np.float64)
    columns = np.array([col for _, col in cells], np.float64)
    theta = (v.get("rope_parameters") or {}).get("rope_theta", 10_000.0)
    for i in range(v["depth"]):
        b = f"blocks.{i}."
        y = layer_norm(x, w[b + "norm1.weight"], w[b + "norm1.bias"])
        qkv = (y @ w[b + "attn.qkv.weight"].T + w[b + "attn.qkv.bias"]).reshape(n, 3, heads, hd)
        q, k, vv = (qkv[:, j] for j in range(3))
        q, k = rotate(q, rows, columns, theta), rotate(k, rows, columns, theta)
        scores = np.einsum("thd,shd->hts", q, k) * hd**-0.5
        probs = np.exp(scores - scores.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        o = np.einsum("hts,shd->thd", probs, vv).reshape(n, d)
        x = x + o @ w[b + "attn.proj.weight"].T + w[b + "attn.proj.bias"]
        y = layer_norm(x, w[b + "norm2.weight"], w[b + "norm2.bias"])
        y = gelu_tanh(y @ w[b + "mlp.linear_fc1.weight"].T + w[b + "mlp.linear_fc1.bias"])
        x = x + y @ w[b + "mlp.linear_fc2.weight"].T + w[b + "mlp.linear_fc2.bias"]
    x = layer_norm(x, w["merger.norm.weight"], w["merger.norm.bias"]).reshape(-1, d * merge**2)
    x = gelu(x @ w["merger.linear_fc1.weight"].T + w["merger.linear_fc1.bias"])
    return np.asarray(x @ w["merger.linear_fc2.weight"].T + w["merger.linear_fc2.bias"])


def forward(
    c: dict[str, Any], w: dict[str, np.ndarray], images: list[tuple[np.ndarray, tuple[int, int, int]]]
) -> np.ndarray:
    """The merged rows of images (patches, (t, h, w)), every frame of each in order: [rows, out]."""
    out = []
    for pixels, (t, h, wd) in images:
        for f in range(t):
            out.append(frame(c, w, pixels[f * h * wd : (f + 1) * h * wd].astype(np.float64), h, wd))
    return np.concatenate(out)


def tiny_config(hidden: int = 32, heads: int = 4) -> dict[str, Any]:
    return {
        "model_type": "qwen3_5",
        "vision_config": {
            "model_type": "qwen3_5_vision",
            "deepstack_visual_indexes": [],
            "depth": 2,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": hidden,
            "in_channels": 3,
            "intermediate_size": 64,
            "num_heads": heads,
            "num_position_embeddings": 16,
            "out_hidden_size": 40,
            "patch_size": 4,
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "axial"},
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
    }


def tiny_weights(c: dict[str, Any], seed: int = 0) -> dict[str, np.ndarray]:
    """Random weights under the creators' names, relative to the tower's root (`visual.`)."""
    rng = np.random.default_rng(seed)
    v = c["vision_config"]
    d, f, p = v["hidden_size"], v["intermediate_size"], v["patch_size"]
    merged = d * v["spatial_merge_size"] ** 2

    def m(*shape: int, scale: float = 0.2) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def ones(n: int) -> np.ndarray:
        return (1 + rng.standard_normal(n) * 0.1).astype(np.float32)

    w = {
        "patch_embed.proj.weight": m(d, v["in_channels"], v["temporal_patch_size"], p, p, scale=0.1),
        "patch_embed.proj.bias": m(d),
        "pos_embed.weight": m(v["num_position_embeddings"], d, scale=1.0),
        "merger.norm.weight": ones(d),
        "merger.norm.bias": m(d),
        "merger.linear_fc1.weight": m(merged, merged, scale=0.1),
        "merger.linear_fc1.bias": m(merged),
        "merger.linear_fc2.weight": m(v["out_hidden_size"], merged, scale=0.1),
        "merger.linear_fc2.bias": m(v["out_hidden_size"]),
    }
    for i in range(v["depth"]):
        b = f"blocks.{i}."
        w |= {
            b + "attn.qkv.weight": m(3 * d, d),
            b + "attn.qkv.bias": m(3 * d),
            b + "attn.proj.weight": m(d, d),
            b + "attn.proj.bias": m(d),
            b + "mlp.linear_fc1.weight": m(f, d),
            b + "mlp.linear_fc1.bias": m(f),
            b + "mlp.linear_fc2.weight": m(d, f),
            b + "mlp.linear_fc2.bias": m(d),
            b + "norm1.weight": ones(d),
            b + "norm1.bias": m(d),
            b + "norm2.weight": ones(d),
            b + "norm2.bias": m(d),
        }
    return w
