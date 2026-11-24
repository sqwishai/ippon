"""Rotary position embeddings (RoPE) as a model's config.json sets them, and their application to queries and keys:
rotate_half over a head's first `dims` channels, by MLX's fused RoPE, in fp32 and rounded once to the input's dtype.

A config's rope_parameters (or rope_scaling) names the scaling, read as transformers' rope initialisation reads it:
"default" (theta ** (2i / dims)), "linear" (positions divided by its factor), "llama3" (long wavelengths divided by
its factor, short ones kept, those between interpolated) and "yarn" (interpolated by a ramp between beta_slow and
beta_fast rotations, and an attention factor on the rotated queries and keys). A family with RoPE of its own
(Gemma 4's "proportional") builds a RoPE with its frequencies.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import mlx.core as mx
import numpy as np

from ..errors import UnsupportedModel

F32 = mx.float32


@dataclass(frozen=True)
class RoPE:
    dims: int  # channels rotated, the first of each head's
    theta: float
    freqs: tuple[float, ...] | None = None  # MLX's frequencies (the reciprocals of inv_freq) when scaled
    scale: float = 1.0  # positions times this (linear scaling)
    attention: float = 1.0  # YaRN's factor on the rotated queries and keys: attention applies it to the scores

    @classmethod
    def from_hf(
        cls, dims: int, params: Mapping[str, Any] | None = None, *, theta: float = 10_000.0, max_positions: int = 0
    ) -> RoPE:
        """From a config's rope_parameters (or rope_scaling): `theta` when they name none; `max_positions`
        (max_position_embeddings) for a YaRN factor they don't give."""
        p = dict(params or {})
        kind = p.get("rope_type", p.get("type", "default"))
        theta = float(p.get("rope_theta", theta))
        if kind == "default":
            return cls(dims, theta)
        if kind == "linear":
            return cls(dims, theta, scale=1.0 / float(p["factor"]))
        if kind == "llama3":
            return cls(dims, theta, llama3_freqs(dims, theta, p))
        if kind == "yarn":
            freqs, attention = yarn_freqs(dims, theta, p, max_positions)
            return cls(dims, theta, freqs, attention=attention)
        raise UnsupportedModel(f"rope_type {kind!r} is not supported")

    @cached_property
    def _freqs(self) -> mx.array | None:
        return None if self.freqs is None else mx.array(self.freqs, dtype=F32)

    def __call__(self, x: mx.array, start: int, positions: mx.array | None = None) -> mx.array:
        """x [sequences, heads, n, head_dim] at positions start, start + 1, ...; or each row at its own `positions`
        (a branch pass's rows)."""
        base = None if self.freqs is not None else self.theta
        if positions is None:
            return mx.fast.rope(
                x, self.dims, traditional=False, base=base, scale=self.scale, offset=start, freqs=self._freqs
            )
        rows_first = x.transpose(2, 1, 0, 3)  # each row a batch of one, with its own offset
        y = mx.fast.rope(
            rows_first, self.dims, traditional=False, base=base, scale=self.scale, offset=positions, freqs=self._freqs
        )
        return y.transpose(2, 1, 0, 3)


def llama3_freqs(head_dim: int, theta: float, p: Mapping[str, Any]) -> tuple[float, ...]:
    """Llama 3's scaled RoPE as MLX's frequencies (the reciprocals of Hugging Face's inv_freq): long wavelengths
    divided by `factor`, short ones kept, those between interpolated, as transformers' _compute_llama3_parameters."""
    factor, low, high = float(p["factor"]), float(p["low_freq_factor"]), float(p["high_freq_factor"])
    context = float(p["original_max_position_embeddings"])
    inv = 1.0 / theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    wavelen = 2 * math.pi / inv
    scaled = np.where(wavelen > context / low, inv / factor, inv)
    smooth = (context / wavelen - low) / (high - low)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    medium = ~(wavelen < context / high) & ~(wavelen > context / low)
    return tuple(float(f) for f in 1.0 / np.where(medium, smoothed, scaled))


def yarn_freqs(dims: int, theta: float, p: Mapping[str, Any], max_positions: int) -> tuple[tuple[float, ...], float]:
    """YaRN's RoPE as MLX's frequencies (the reciprocals of inv_freq) and its attention factor, step for step as
    transformers' _compute_yarn_parameters computes them in fp32: the dimensions that turn fewer than beta_slow
    times over the original context interpolated by `factor`, those that turn more than beta_fast times kept, a
    linear ramp between (its ends rounded outward unless "truncate" is false); the factor given, or 0.1 ln(factor)
    + 1 (with mscale and mscale_all_dim, their ratio)."""
    original = float(p["original_max_position_embeddings"])
    factor = float(p.get("factor") or max_positions / original)

    def mscale(scale: float, m: float = 1.0) -> float:
        return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0

    attention = p.get("attention_factor")
    if attention is None:
        m, m_all = p.get("mscale"), p.get("mscale_all_dim")
        attention = mscale(factor, float(m)) / mscale(factor, float(m_all)) if m and m_all else mscale(factor)

    def correction(rotations: float) -> float:
        return dims * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(theta))

    low, high = correction(float(p.get("beta_fast") or 32)), correction(float(p.get("beta_slow") or 1))
    if p.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dims - 1)
    if low == high:
        high += 0.001
    f32 = np.float32
    ramp = np.clip((np.arange(dims // 2, dtype=f32) - f32(low)) / f32(high - low), 0, 1).astype(f32)
    pos = (f32(theta) ** (np.arange(0, dims, 2).astype(f32) / f32(dims))).astype(f32)
    extrapolation = f32(1) - ramp
    inv = (f32(1) / (f32(factor) * pos)) * (f32(1) - extrapolation) + (f32(1) / pos) * extrapolation
    return tuple(float(x) for x in 1.0 / inv.astype(np.float64)), float(attention)


def proportional_freqs(head_dim: int, theta: float, p: Mapping[str, Any]) -> tuple[float, ...]:
    """transformers' proportional RoPE as MLX's periods over the whole head: int(factor * head_dim // 2) angle pairs
    at base ** (2i / head_dim), the other pairs never rotate (an infinite period), all times `factor`."""
    angles = int(float(p.get("partial_rotary_factor", 1.0)) * head_dim // 2)
    scale = float(p.get("factor", 1.0))
    rotated = [scale * theta ** (2 * i / head_dim) for i in range(angles)]
    return tuple(rotated + [math.inf] * (head_dim // 2 - angles))
