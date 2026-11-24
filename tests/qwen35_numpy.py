"""An independent, slow numpy Qwen3.5, written from Hugging Face's modeling_qwen3_5.py, for checking the MLX one
on tiny random checkpoints: the Gated DeltaNet as its token-by-token recurrence, attention as explicit softmaxes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

F = np.float64


def silu(x: np.ndarray) -> np.ndarray:
    return np.asarray(x / (1.0 + np.exp(-x)))


def rms(x: np.ndarray, eps: float) -> np.ndarray:
    return np.asarray(x / np.sqrt((x * x).mean(-1, keepdims=True) + eps))


def rope(
    x: np.ndarray, positions: np.ndarray, rot: int, theta: float, section: tuple[int, ...] = (11, 11, 10)
) -> np.ndarray:
    """rotate_half RoPE on the first `rot` channels of x [..., n, d], at positions [n], or multimodal positions
    [3, n] (t, h, w) as Qwen3_5TextRotaryEmbedding recomposes them: each axis's angles, then h's and w's frequencies
    put in t's at slices (1, 3 * section[1], 3) and (2, 3 * section[2], 3)."""
    inv = 1.0 / theta ** (np.arange(0, rot, 2, dtype=F) / rot)
    axes = positions if positions.ndim == 2 else np.broadcast_to(positions, (3, len(positions)))
    each = axes[:, :, None].astype(F) * inv[None, None, :]  # [3, n, rot / 2]
    freqs = each[0].copy()
    for axis, start in ((1, 1), (2, 2)):
        at = slice(start, section[axis] * 3, 3)
        freqs[:, at] = each[axis][:, at]
    cos, sin = np.cos(np.concatenate([freqs, freqs], -1)), np.sin(np.concatenate([freqs, freqs], -1))
    xr, xp = x[..., :rot], x[..., rot:]
    half = rot // 2
    rotated = np.concatenate([-xr[..., half:], xr[..., :half]], -1)
    return np.concatenate([xr * cos + rotated * sin, xp], -1)


def forward(
    c: dict[str, Any],
    w: dict[str, np.ndarray],
    ids: list[int],
    causal: bool = True,
    embeds: np.ndarray | None = None,
    positions: np.ndarray | None = None,
    block: int | None = None,
    recurrence: tuple[list[tuple[int, int]], int] | None = None,
) -> np.ndarray:
    """Final-normed hidden states [n, hidden] of one sequence: of `ids`, or of rows `embeds` [n, hidden] (an image's
    rows among text's), at positions 0.. or `positions` ([n], or [3, n] multimodal ones). With `block`, block
    attention as transformers' blockwise overlay makes it: a key is seen when it is earlier or in the query's block
    (the first `block` rows, or the rest). With `recurrence` ([(a, b), ...], P), recurrent depth as Fastino's
    glide/recurrent.py runs it: each block of layers a..b, G, takes its input x_0 through P damped Euler steps
    x_k = (1 - 1/P) x_(k-1) + (1/P) G(x_(k-1))."""
    n = len(ids) if embeds is None else len(embeds)
    eps = c["rms_norm_eps"]
    x = (w["embed_tokens.weight"][ids] if embeds is None else embeds).astype(F)
    pos = np.arange(n) if positions is None else positions
    section = tuple(c["rope_parameters"].get("mrope_section", (11, 11, 10)))

    def layer(i: int, x: np.ndarray) -> np.ndarray:
        kind = c["layer_types"][i]
        p = f"layers.{i}."
        h = rms(x, eps) * (1 + w[p + "input_layernorm.weight"])
        if kind == "full_attention":
            hd, nh, kv = c["head_dim"], c["num_attention_heads"], c["num_key_value_heads"]
            qg = (h @ w[p + "self_attn.q_proj.weight"].T).reshape(n, nh, 2 * hd)
            q, gate = qg[..., :hd], qg[..., hd:].reshape(n, nh * hd)
            k = (h @ w[p + "self_attn.k_proj.weight"].T).reshape(n, kv, hd)
            v = (h @ w[p + "self_attn.v_proj.weight"].T).reshape(n, kv, hd)
            q = rms(q, eps) * (1 + w[p + "self_attn.q_norm.weight"])
            k = rms(k, eps) * (1 + w[p + "self_attn.k_norm.weight"])
            rot = int(hd * c["partial_rotary_factor"])
            q = rope(q.transpose(1, 0, 2), pos, rot, c["rope_parameters"]["rope_theta"], section)  # [nh, n, hd]
            k = rope(k.transpose(1, 0, 2), pos, rot, c["rope_parameters"]["rope_theta"], section)
            v = v.transpose(1, 0, 2)
            rep = nh // kv
            k, v = np.repeat(k, rep, 0), np.repeat(v, rep, 0)
            scores = q @ k.transpose(0, 2, 1) * hd**-0.5
            if causal or block is not None:
                seen = np.tril(np.ones((n, n), bool))
                if block is not None:
                    after = np.arange(n) >= block
                    seen |= after[:, None] == after[None, :]
                scores = np.where(seen, scores, -np.inf)
            probs = np.exp(scores - scores.max(-1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            o = (probs @ v).transpose(1, 0, 2).reshape(n, nh * hd) * (1 / (1 + np.exp(-gate)))
            mixed = o @ w[p + "self_attn.o_proj.weight"].T
        else:
            hk, hv, dk, dv = (
                c["linear_num_key_heads"],
                c["linear_num_value_heads"],
                c["linear_key_head_dim"],
                c["linear_value_head_dim"],
            )
            kern = c["linear_conv_kernel_dim"]
            g_ = p + "linear_attn."
            qkv = h @ w[g_ + "in_proj_qkv.weight"].T
            z = (h @ w[g_ + "in_proj_z.weight"].T).reshape(n, hv, dv)
            b = h @ w[g_ + "in_proj_b.weight"].T
            a = h @ w[g_ + "in_proj_a.weight"].T
            conv_w = w[g_ + "conv1d.weight"][:, 0, :]  # [channels, kernel]
            padded = np.concatenate([np.zeros((kern - 1, qkv.shape[1])), qkv], 0)
            conv = sum(padded[j : j + n] * conv_w[:, j] for j in range(kern))
            conv = silu(conv)
            kd = hk * dk
            q = conv[:, :kd].reshape(n, hk, dk)
            k = conv[:, kd : 2 * kd].reshape(n, hk, dk)
            v = conv[:, 2 * kd :].reshape(n, hv, dv)
            q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) * dk**-0.5
            k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
            rep = hv // hk
            q, k = np.repeat(q, rep, 1), np.repeat(k, rep, 1)
            beta = 1 / (1 + np.exp(-b))
            g = -np.exp(w[g_ + "A_log"]) * np.log1p(np.exp(a + w[g_ + "dt_bias"]))
            state = np.zeros((hv, dk, dv))
            out = np.zeros((n, hv, dv))
            for t in range(n):
                state = state * np.exp(g[t])[:, None, None]
                kv_mem = (state * k[t][:, :, None]).sum(1)
                delta = (v[t] - kv_mem) * beta[t][:, None]
                state = state + k[t][:, :, None] * delta[:, None, :]
                out[t] = (state * q[t][:, :, None]).sum(1)
            y = rms(out, eps) * w[g_ + "norm.weight"] * silu(z)
            mixed = y.reshape(n, hv * dv) @ w[g_ + "out_proj.weight"].T
        x = x + mixed
        h = rms(x, eps) * (1 + w[p + "post_attention_layernorm.weight"])
        return np.asarray(x + (moe(c, w, p + "mlp.", h) if c.get("num_experts") else mlp(w, p + "mlp.", h)))

    blocks, passes = recurrence or ([], 1)
    starts = dict(blocks)
    i = 0
    while i < len(c["layer_types"]):
        if i not in starts or passes == 1:
            x = layer(i, x)
            i += 1
            continue
        for _ in range(passes):
            g = x
            for j in range(i, starts[i] + 1):
                g = layer(j, g)
            x = (1 - 1 / passes) * x + (1 / passes) * g
        i = starts[i] + 1
    return np.asarray(rms(x, eps) * (1 + w["norm.weight"]))


def mlp(w: dict[str, np.ndarray], p: str, h: np.ndarray) -> np.ndarray:
    return np.asarray(
        (silu(h @ w[p + "gate_proj.weight"].T) * (h @ w[p + "up_proj.weight"].T)) @ w[p + "down_proj.weight"].T
    )


def moe(c: dict[str, Any], w: dict[str, np.ndarray], p: str, h: np.ndarray) -> np.ndarray:
    """transformers' Qwen3.5-MoE block: softmax router, top-k renormalized, each expert's SwiGLU weighted and summed,
    plus the shared expert times sigmoid of its gate; one token at a time."""
    k = c["num_experts_per_tok"]
    logits = h @ w[p + "gate.weight"].T
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    out = np.zeros_like(h)
    for t in range(h.shape[0]):
        chosen = np.argsort(-probs[t])[:k]
        weights = probs[t, chosen] / probs[t, chosen].sum()
        for e, weight in zip(chosen, weights, strict=True):
            out[t] += weight * mlp(w, f"{p}experts.{e}.", h[t : t + 1])[0]
    shared = mlp(w, p + "shared_expert.", h)
    gate = 1 / (1 + np.exp(-(h @ w[p + "shared_expert_gate.weight"].T)))
    return np.asarray(out + gate * shared)


def tiny_moe_config(layer_types: list[str]) -> dict[str, Any]:
    c = tiny_config(layer_types) | {
        "model_type": "qwen3_5_moe_text",
        "num_experts": 6,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 24,
    }
    del c["intermediate_size"]
    return c


def tiny_config(layer_types: list[str]) -> dict[str, Any]:
    return {
        "model_type": "qwen3_5_text",
        "hidden_size": 32,
        "layer_types": layer_types,
        "num_hidden_layers": len(layer_types),
        "intermediate_size": 48,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "partial_rotary_factor": 0.5,
        "rope_parameters": {"rope_theta": 10_000.0, "partial_rotary_factor": 0.5, "rope_type": "default"},
        "rms_norm_eps": 1e-6,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "vocab_size": 64,
        "tie_word_embeddings": True,
        "attn_output_gate": True,
        "hidden_act": "silu",
    }


def tiny_weights(c: dict[str, Any], seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    hd, nh, kv, hid, inter = (
        c["head_dim"],
        c["num_attention_heads"],
        c["num_key_value_heads"],
        c["hidden_size"],
        c.get("intermediate_size", 0),
    )
    hk, hv, dk, dv, kern = (
        c["linear_num_key_heads"],
        c["linear_num_value_heads"],
        c["linear_key_head_dim"],
        c["linear_value_head_dim"],
        c["linear_conv_kernel_dim"],
    )

    def m(*shape: int, scale: float = 0.2) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    w = {"embed_tokens.weight": m(c["vocab_size"], hid, scale=1.0), "norm.weight": m(hid, scale=0.1)}
    for i, kind in enumerate(c["layer_types"]):
        p = f"layers.{i}."
        w |= {
            p + "input_layernorm.weight": m(hid, scale=0.1),
            p + "post_attention_layernorm.weight": m(hid, scale=0.1),
        }
        if c.get("num_experts"):
            e_w, s_w = c["moe_intermediate_size"], c["shared_expert_intermediate_size"]
            w[p + "mlp.gate.weight"] = m(c["num_experts"], hid, scale=1.0)
            for e in range(c["num_experts"]):
                for x, shape in (("gate", (e_w, hid)), ("up", (e_w, hid)), ("down", (hid, e_w))):
                    w[f"{p}mlp.experts.{e}.{x}_proj.weight"] = m(*shape)
            for x, shape in (("gate", (s_w, hid)), ("up", (s_w, hid)), ("down", (hid, s_w))):
                w[f"{p}mlp.shared_expert.{x}_proj.weight"] = m(*shape)
            w[p + "mlp.shared_expert_gate.weight"] = m(1, hid)
        else:
            w |= {
                p + "mlp.gate_proj.weight": m(inter, hid),
                p + "mlp.up_proj.weight": m(inter, hid),
                p + "mlp.down_proj.weight": m(hid, inter),
            }
        if kind == "full_attention":
            a = p + "self_attn."
            w |= {
                a + "q_proj.weight": m(nh * hd * 2, hid),
                a + "k_proj.weight": m(kv * hd, hid),
                a + "v_proj.weight": m(kv * hd, hid),
                a + "o_proj.weight": m(hid, nh * hd),
                a + "q_norm.weight": m(hd, scale=0.1),
                a + "k_norm.weight": m(hd, scale=0.1),
            }
        else:
            g = p + "linear_attn."
            conv_dim = 2 * hk * dk + hv * dv
            w |= {
                g + "in_proj_qkv.weight": m(conv_dim, hid),
                g + "in_proj_z.weight": m(hv * dv, hid),
                g + "in_proj_b.weight": m(hv, hid),
                g + "in_proj_a.weight": m(hv, hid),
                g + "conv1d.weight": m(conv_dim, 1, kern, scale=0.5),
                g + "dt_bias": m(hv, scale=0.5),
                g + "A_log": m(hv, scale=0.5),
                g + "norm.weight": (1 + m(dv, scale=0.1)),
                g + "out_proj.weight": m(hid, hv * dv),
            }
    return w


# -- multimodal RoPE positions, as transformers gives them (held to the creator's by test_media_qwen.py) --------


def image_positions(start: int, grid: tuple[int, int, int], merge: int = 2) -> np.ndarray:
    """An image's rows' (t, h, w) positions [3, rows] when its first row is at text position `start`, as
    Qwen3_5Model.get_vision_position_ids gives them: t = start + frame, h = start + block row, w = start + block
    column, row-major over the merged grid."""
    t, h, w = grid
    frames, rows, columns = np.meshgrid(np.arange(t), np.arange(h // merge), np.arange(w // merge), indexing="ij")
    out: np.ndarray = np.stack([frames, rows, columns]).reshape(3, -1).astype(np.int64) + start
    return out


def rope_index(
    length: int, images: Sequence[tuple[int, tuple[int, int, int]]], merge: int = 2
) -> tuple[np.ndarray, int]:
    """Qwen3_5Model.get_rope_index for one sequence of `length` rows: `images` are (row, grid) for each image's
    first row and its (t, h, w) grid in patches, in order. Text rows get t = h = w = a running position; an image at
    position p gets `image_positions(p, grid)`, and the text after it resumes at p + max(h, w) / merge.

    Returns the positions [3, length] int64 and the delta (positions.max() + 1 - length), which rows past the
    sequence add to their index. ValueError when images overlap, run past the end, or touch with no text between:
    the creator reads touching images as one and fails."""
    positions = np.empty((3, length), np.int64)
    row = current = 0
    for n, (start, grid) in enumerate(images):
        count = grid[0] * (grid[1] // merge) * (grid[2] // merge)
        if start < row or start + count > length or count < 1:
            raise ValueError(f"image {n} at row {start} ({count} rows) overlaps another or the end of {length} rows")
        if n and start == row:
            raise ValueError(f"images {n - 1} and {n} touch: Qwen's code reads them as one image")
        text = start - row
        positions[:, row:start] = current + np.arange(text)
        current += text
        positions[:, start : start + count] = image_positions(current, grid, merge)
        current += max(grid[1], grid[2]) // merge
        row = start + count
    positions[:, row:] = current + np.arange(length - row)
    delta = int(np.max(positions)) + 1 - length if length else 0
    return positions, delta
