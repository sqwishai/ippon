"""Measure supported text RoPE paths against NumPy FP64 rotations through position 16,384.

Run with the GPU lock: python references/rope/measure.py --out RESULT.json. No weights are loaded or changed.
These are rotation errors, not probability errors; model-level verification is a separate experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from ippon._portable import dumps
from ippon.arch.gemma4 import proportional_freqs
from ippon.arch.qwen35 import mrope_axes
from ippon.arch.rope import RoPE
from ippon.machine import describe

LIMIT = 16_384
BOUNDS = (1_024, 4_096, 8_192, LIMIT)


def configurations() -> list[tuple[str, int, RoPE]]:
    """Published local-model parameters and representative dense/Gemma scaling configurations."""
    llama3 = {"rope_type": "llama3", "factor": 8, "low_freq_factor": 1, "high_freq_factor": 4,
              "original_max_position_embeddings": 8192}  # fmt: skip
    yarn = {"rope_type": "yarn", "factor": 4, "original_max_position_embeddings": 4096}
    return [
        ("LFM2", 64, RoPE(64, 1_000_000)),
        ("Qwen3.5 partial", 256, RoPE(64, 10_000_000)),
        ("ModernBERT global", 64, RoPE(64, 160_000)),
        ("ModernBERT local", 64, RoPE(64, 10_000)),
        ("dense default", 128, RoPE(128, 10_000)),
        ("dense high base", 128, RoPE(128, 1_000_000)),
        ("dense linear", 128, RoPE.from_hf(128, {"rope_type": "linear", "factor": 8})),
        ("dense Llama 3", 128, RoPE.from_hf(128, llama3, theta=500_000)),
        ("dense YaRN", 128, RoPE.from_hf(128, yarn)),
        ("Gemma full", 256, RoPE(256, 1_000_000)),
        ("Gemma sliding", 256, RoPE(256, 10_000)),
        (
            "Gemma proportional",
            512,
            RoPE(512, 1_000_000, proportional_freqs(512, 1_000_000, {"partial_rotary_factor": 0.25})),
        ),
    ]


def rotate(x: np.ndarray, rope: RoPE, positions: np.ndarray) -> np.ndarray:
    """FP64 angle construction, trig functions and arithmetic, on the exact supplied input values."""
    d = rope.dims // 2
    periods = np.asarray(rope.freqs) if rope.freqs is not None else rope.theta ** (np.arange(d, dtype=np.float64) / d)
    angles = positions[:, None].astype(np.float64) * rope.scale / periods
    c, s = np.cos(angles), np.sin(angles)
    a, b = x[..., :d].astype(np.float64), x[..., d : 2 * d].astype(np.float64)
    return np.concatenate([a * c - b * s, a * s + b * c, x[..., 2 * d :]], axis=-1)


def errors(got: np.ndarray, want: np.ndarray) -> dict[str, Any]:
    delta = np.abs(got.astype(np.float64) - want)
    return {
        "max_abs": {str(end): float(delta[..., : end + 1, :].max()) for end in BOUNDS},
        "rms": float(np.sqrt(np.mean(delta**2))),
    }


def measure() -> dict[str, Any]:
    mx.set_default_device(mx.gpu)
    positions = np.arange(LIMIT + 1, dtype=np.int32)
    results = []
    for name, width, rope in configurations():
        rng = np.random.default_rng(17)
        sample = rng.standard_normal((1, 1, LIMIT + 1, width)).astype(np.float32)
        rows = []
        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            x = mx.array(sample, dtype=dtype)
            want = rotate(np.asarray(x.astype(mx.float32)), rope, positions)
            got = rope(x, 0)
            explicit = rope(x, 0, mx.array(positions))
            offset = rope(x[..., 8192:, :], 8192)
            mx.eval(got, explicit, offset)
            rows.append(
                {
                    "dtype": str(dtype),
                    **errors(np.asarray(got.astype(mx.float32)), want),
                    "explicit_positions_identical": bool(mx.array_equal(got, explicit).item()),
                    "offset_identical": bool(mx.array_equal(got[..., 8192:, :], offset).item()),
                }
            )
        results.append(
            {
                "name": name,
                "width": width,
                "rotary_dims": rope.dims,
                "theta": rope.theta,
                "scale": rope.scale,
                "custom_frequencies": rope.freqs is not None,
                "results": rows,
            }
        )
        print(name, rows[0]["max_abs"], flush=True)

    # Distinct axes exercise the interleaved M-RoPE path, with the same per-pair assignment as Qwen3.5.
    rope, width = RoPE(64, 10_000_000), 256
    sample = np.random.default_rng(17).standard_normal((1, 1, LIMIT + 1, width)).astype(np.float32)
    axes = [positions, positions // 2, positions // 3]
    assignment = mrope_axes(64, (11, 11, 10), width)
    rows = []
    for dtype in (mx.float32, mx.float16, mx.bfloat16):
        x = mx.array(sample, dtype=dtype)
        turns = [rope(x, 0, mx.array(p)) for p in axes]
        got = mx.where(mx.array(assignment) == 1, turns[1], mx.where(mx.array(assignment) == 2, turns[2], turns[0]))
        ideal = [rotate(np.asarray(x.astype(mx.float32)), rope, p) for p in axes]
        want = np.where(assignment == 1, ideal[1], np.where(assignment == 2, ideal[2], ideal[0]))
        rows.append({"dtype": str(dtype), **errors(np.asarray(got.astype(mx.float32)), want)})
    results.append(
        {
            "name": "Qwen3.5 M-RoPE",
            "width": width,
            "rotary_dims": 64,
            "results": rows,
        }
    )
    return {
        "machine": describe(),
        "seed": 17,
        "max_position": LIMIT,
        "configurations": results,
        "reference": "FP64 rotations on dtype-rounded inputs; YaRN uses its configured FP32-derived frequencies",
        "scope": (
            "All supported text rotary paths; representative scaled dense/Gemma parameters, not every possible config"
        ),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.write_text(dumps(measure(), indent=2) + "\n")
