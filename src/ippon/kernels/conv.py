"""Short depthwise convolutions over the sequence, in one read and one write:

    causal_conv(x, window, taps, silu)     # Qwen3.5's Gated DeltaNet input: silu(conv(x)) when `silu`
    gated_conv(proj, window, taps)         # LFM2: C * conv(B * x) from the projection [..., B | C | x]

x and proj are [sequences, length, channels...]; `window` holds the K - 1 inputs before the first position
([1 or sequences, K - 1, channels], broadcast when 1); taps are fp32 [channels, K], tap K - 1 on the current
position. The numbers are `arch.ops.MLX`'s: the sum in fp32 rounded once to x's dtype; LFM2 rounds B * x first and
the convolution before multiplying by C, as its reference does.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import mlx.core as mx

from ._source import metal_source
from .launches import Launches

CAUSAL = metal_source("conv_causal.metal")

# LFM2: bx = round(B * x); y = round(C * round(conv(bx))). proj rows are [B | C | x], each C wide.
GATED_SOURCE = metal_source("conv_gated.metal")
THREADS = 256


@cache
def _causal(silu: bool) -> object:
    return mx.fast.metal_kernel(
        name=f"ippon_causal_conv_{int(silu)}",
        input_names=["x", "window", "taps", "sizes"],
        output_names=["y"],
        source=CAUSAL.replace("SILU", "true" if silu else "false"),
    )


@cache
def _gated() -> object:
    return mx.fast.metal_kernel(
        name="ippon_gated_conv",
        input_names=["proj", "window", "taps", "sizes"],
        output_names=["y"],
        source=GATED_SOURCE,
    )


def _grid(n: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    return ((n + THREADS - 1) // THREADS * THREADS, 1, 1), (THREADS, 1, 1)


# The zero window of a convolution with no inputs before it, by its shape and dtype.
_ZEROS: dict[tuple[Any, ...], mx.array] = {}
_LAUNCHES: Launches[tuple[mx.array, dict[str, Any]]] = Launches()


def causal_conv(x: mx.array, window: mx.array, taps: mx.array, silu: bool) -> mx.array:
    """Over the first `taps.shape[0]` columns of x's rows (x may be a wider projection, read in place)."""
    key = ("causal", silu, x.dtype, x.shape, taps.shape, window.shape[0])
    found = _LAUNCHES.get(key)
    if found is None:
        batch, length, row = x.shape
        channels, k = taps.shape
        n = batch * length * channels
        grid, group = _grid(n)
        found = _LAUNCHES[key] = (
            mx.array([n, length], mx.uint32),
            {
                "template": [("T", x.dtype), ("C", channels), ("ROW", row), ("K", k), ("WB", window.shape[0])],
                "grid": grid,
                "threadgroup": group,
                "output_shapes": [(batch, length, channels)],
                "output_dtypes": [x.dtype],
            },
        )
    sizes, args = found
    result: mx.array = _causal(silu)(inputs=[x, window.astype(x.dtype), taps, sizes], **args)[0]  # type: ignore[operator]
    return result


def gated_conv(proj: mx.array, window: mx.array | None, taps: mx.array) -> mx.array:
    if window is None:
        zeros = ("zeros", taps.shape, proj.dtype)
        if zeros not in _ZEROS:
            _ZEROS[zeros] = mx.zeros((1, taps.shape[1] - 1, taps.shape[0]), proj.dtype)
        window = _ZEROS[zeros]
    key = ("gated", proj.dtype, proj.shape, taps.shape, window.shape[0])
    found = _LAUNCHES.get(key)
    if found is None:
        batch, length, three = proj.shape
        channels, k = three // 3, taps.shape[1]
        n = batch * length * channels
        grid, group = _grid(n)
        found = _LAUNCHES[key] = (
            mx.array([n, length], mx.uint32),
            {
                "template": [("T", proj.dtype), ("C", channels), ("K", k), ("WB", window.shape[0])],
                "grid": grid,
                "threadgroup": group,
                "output_shapes": [(batch, length, channels)],
                "output_dtypes": [proj.dtype],
            },
        )
    sizes, args = found
    inputs = [proj, window.astype(proj.dtype), taps, sizes]
    result: mx.array = _gated()(inputs=inputs, **args)[0]  # type: ignore[operator]
    return result
