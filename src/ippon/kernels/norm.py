"""The residual add and the norm that follows it, in one read and one write of the rows.

    h, n = add_norm(x, delta, w, eps)        # h = x + delta (rounded to x's dtype), n = norm(h) * w
    n = add_norm(x, None, w, eps)            # the norm of x alone

The numbers are those of `arch.ops.MLX`: h rounded once to x's dtype, its statistics in fp32 over the last axis,
times the fp32 weight, rounded once. `kind` is "rms" (RMSNorm) or "layer" (LayerNorm without a bias). With
`groups`, the statistics are taken over each of that many equal groups of the last axis's channels (a grouped
RMSNorm), the weight still one per channel. One threadgroup per row (per group), its threads striding over its
channels.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import mlx.core as mx

from ._source import metal_source
from .launches import Launches

SOURCE = metal_source("norm.metal")
ADD_SUM = metal_source("norm_add_sum.metal").rstrip("\n")
ADD_STORE = metal_source("norm_add_store.metal").rstrip("\n")


def threads(width: int) -> int:
    """Threads per row: enough to cover a narrow row (a head's 64 to 256 channels) once, at most 256."""
    return min(256, max(32, 1 << (max(width, 1) - 1).bit_length()))


@cache
def _kernel(add: bool, layer: bool, eps: float, count: int, groups: int = 1) -> object:
    source = (
        SOURCE.replace("EPS", f"{eps!r}f")
        .replace("THREADS", str(count))
        .replace("LAYER", "true" if layer else "false")
        .replace("ADD_SUM", ADD_SUM if add else "")
        .replace("ADD_STORE", ADD_STORE if add else "")
        .replace(
            "WEIGHT_AT",
            "0" if groups == 1 else metal_source("norm_weight_at.metal").rstrip("\n").replace("GROUPS", str(groups)),
        )
    )
    tag = f"{int(add)}{int(layer)}_{count}_{eps!r}_g{groups}".replace(".", "_").replace("-", "m")
    return mx.fast.metal_kernel(
        name=f"ippon_add_norm_{tag}",
        input_names=["x", "delta", "w"] if add else ["x", "w"],
        output_names=["h", "n"] if add else ["n"],
        source=source,
    )


def add_norm(
    x: mx.array, delta: mx.array | None, w: mx.array, eps: float, kind: str = "rms", groups: int = 1
) -> tuple[mx.array, mx.array] | mx.array:
    """(x + delta, its norm times w), or the norm of x alone when `delta` is None; `w` is fp32 [channels], the
    statistics over each of `groups` equal groups of the channels."""
    add = delta is not None
    key = (add, kind, eps, groups, x.dtype, x.shape)
    found = _LAUNCHES.get(key)
    if found is None:
        if kind not in ("rms", "layer"):
            raise ValueError(f"unknown norm {kind!r}")
        if x.shape[-1] % groups:
            raise ValueError(f"{x.shape[-1]} channels do not split into {groups} groups")
        width = x.shape[-1] // groups
        count = threads(width)
        found = _LAUNCHES[key] = (
            _kernel(add, kind == "layer", float(eps), count, groups),
            {
                "template": [("T", x.dtype), ("WIDTH", width)],
                "grid": (count, x.size // width, 1),
                "threadgroup": (count, 1, 1),
                "output_shapes": [x.shape] * (2 if add else 1),
                "output_dtypes": [x.dtype] * (2 if add else 1),
            },
        )
    kernel, args = found
    out = kernel(inputs=[x, delta, w] if add else [x, w], **args)  # type: ignore[operator]
    return (out[0], out[1]) if add else out[0]


_LAUNCHES: Launches[Any] = Launches()


# Gated DeltaNet's output norm: y = round(round(w * round(rmsnorm(round(o)))) * silu(z)), per head, as Qwen3.5's
# reference rounds it; with an fp32 w (a checkpoint that stores it so) w * normed is fp32 and is not rounded
# (ROUND_W false), as the reference's product with an fp32 weight is not. o is the recurrence's fp32 output
# [rows, heads, width]; z is read in place from the projection (columns OFFSET.. of ROW-wide rows).
GATED_SOURCE = metal_source("norm_gated.metal")


@cache
def _gated(eps: float, count: int) -> object:
    source = GATED_SOURCE.replace("EPS", f"{eps!r}f").replace("THREADS", str(count))
    return mx.fast.metal_kernel(
        name=f"ippon_gated_norm_{count}_{eps!r}".replace(".", "_").replace("-", "m"),
        input_names=["o", "z", "w"],
        output_names=["y"],
        source=source,
    )


def gated_norm(o: mx.array, proj: mx.array, offset: int, w: mx.array, eps: float, dtype: mx.Dtype) -> mx.array:
    """o [sequences, length, heads, width] fp32; z the columns `offset`.. of proj's rows; w [width] in `dtype` or
    fp32."""
    batch, length, heads, width = o.shape
    count = threads(width)
    out = _gated(float(eps), count)(  # type: ignore[operator]
        inputs=[o, proj, w],
        template=[
            ("T", dtype),
            ("WIDTH", width),
            ("HEADS", heads),
            ("ROW", proj.shape[-1]),
            ("OFFSET", offset),
            ("ROUND_W", w.dtype != mx.float32),
        ],
        grid=(count, batch * length * heads, 1),
        threadgroup=(count, 1, 1),
        output_shapes=[(batch, length, heads * width)],
        output_dtypes=[dtype],
    )
    result: mx.array = out[0]
    return result
