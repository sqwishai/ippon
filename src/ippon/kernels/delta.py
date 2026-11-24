"""Gated DeltaNet's inputs in one kernel: q and k L2-normalized per head (q also times k_dim**-0.5) and v, all in fp32,
from the rows [q | k | v] of the convolution's output in the compute dtype, and each value head's gates from the
input projection's b and a columns: beta = sigmoid(b) (in fp32, rounded once to the compute dtype: MLX's own sigmoid,
which computes in that dtype, can differ from it by a unit in the last place) and gamma = exp(neg_a * softplus(a +
dt_bias)), in fp32.

    q, k, v, beta, gamma = delta_inputs(mixed, proj, neg_a, dt_bias, k_heads, k_dim, v_heads, v_dim, scale)

The numbers are those of `arch.ops`' MLX composition: each value widened to fp32, x * rsqrt(sum(x * x) + 1e-6)
(times `scale` for q), which takes about five full passes over q and k in fp32 there, and five small kernels for the
gates. One simdgroup per head of a row, its 32 threads striding over the head's channels (a value head's first
thread also computes its gates); the sizes are template constants, the row count the grid's, so a new length
compiles nothing.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import mlx.core as mx

from ._source import metal_source
from .launches import Launches

SOURCE = metal_source("delta.metal")


@cache
def _kernel(scale: float) -> object:
    return mx.fast.metal_kernel(
        name=f"ippon_delta_inputs_{scale!r}".replace(".", "_").replace("-", "m"),
        input_names=["mixed", "proj", "neg_a", "dt_bias"],
        output_names=["q", "k", "v", "beta", "gamma"],
        source=SOURCE.replace("SCALE", f"{scale!r}f"),
    )


def delta_inputs(
    mixed: mx.array,
    proj: mx.array,
    neg_a: mx.array,
    dt_bias: mx.array,
    k_heads: int,
    k_dim: int,
    v_heads: int,
    v_dim: int,
    scale: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """q [.., k_heads, k_dim], k and v [.., v_heads, v_dim], beta and gamma [.., v_heads], all fp32, from mixed
    [.., 2 * k_heads * k_dim + v_heads * v_dim] and the rows of proj whose last 2 * v_heads columns are b and a
    (`neg_a`, `dt_bias`: fp32 [v_heads]); the leading dimensions kept."""
    key = (scale, mixed.dtype, mixed.shape, proj.shape, k_heads, k_dim, v_heads, v_dim)
    found = _LAUNCHES.get(key)
    if found is None:
        lead, width = mixed.shape[:-1], mixed.shape[-1]
        if width != 2 * k_heads * k_dim + v_heads * v_dim:
            raise ValueError(f"{width} columns are not q, k and v of {k_heads}x{k_dim} and {v_heads}x{v_dim}")
        rows, pw = mixed.size // width, proj.shape[-1]
        found = _LAUNCHES[key] = {
            "template": [
                ("T", mixed.dtype),
                ("WIDTH", width),
                ("HK", k_heads),
                ("DK", k_dim),
                ("HV", v_heads),
                ("DV", v_dim),
                ("PW", pw),
                ("BOFF", pw - 2 * v_heads),
            ],
            "grid": (32, rows * (2 * k_heads + v_heads), 1),
            "threadgroup": (32, 1, 1),
            "output_shapes": [  # the rows' layout as written, with the leading dimensions kept
                (*lead, k_heads, k_dim),
                (*lead, k_heads, k_dim),
                (*lead, v_heads, v_dim),
                (*lead, v_heads),
                (*lead, v_heads),
            ],
            "output_dtypes": [mx.float32] * 5,
        }
    q, k, v, beta, gamma = _kernel(float(scale))(inputs=[mixed, proj, neg_a, dt_bias], **found)  # type: ignore[operator]
    return q, k, v, beta, gamma


_LAUNCHES: Launches[dict[str, Any]] = Launches()
