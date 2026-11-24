"""A gated linear unit from the two halves of one projection, in one read and one write:

    glu(gate_up, "silu")   # silu(gate) * up      (SwiGLU: Qwen3.5, LFM2)
    glu(gate_up, "gelu")   # gelu(gate) * up      (GeGLU: ModernBERT, exact GELU)

`gate_up` is [..., 2 * width], gate first. The arithmetic is in fp32 and the result is rounded once to the input's
dtype, as `arch.ops.MLX.glu` computes it in six passes.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import mlx.core as mx

from ._source import metal_source
from .launches import Launches

SOURCE = metal_source("glu.metal")
ACTIVATIONS = {name: metal_source(f"activation_{name}.metal").rstrip("\n") for name in ("silu", "gelu", "gelu_tanh")}


def activation_source(name: str | None, value: str) -> str:
    """The Metal statement computing activation `name` (None: identity) of fp32 variable `value` into `a`."""
    source = metal_source("activation_identity.metal").rstrip("\n") if name is None else ACTIVATIONS[name]
    return source.replace("X", value)


THREADS = 256
# erf in fp32 (Abramowitz and Stegun 7.1.26, absolute error below 1.5e-7): Metal has none, MLX's is internal
HEADER = metal_source("erf.metal")


@cache
def _kernel(activation: str) -> object:
    return mx.fast.metal_kernel(
        name=f"ippon_glu_{activation}",
        input_names=["gu", "sizes"],
        output_names=["out"],
        source=SOURCE.replace("ACTIVATION", activation_source(activation, "g")),
        header=HEADER,
    )


def glu(gate_up: mx.array, activation: str = "silu") -> mx.array:
    key = (activation, gate_up.dtype, gate_up.shape)
    found = _LAUNCHES.get(key)
    if found is None:
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}")
        width = gate_up.shape[-1] // 2
        n = gate_up.size // 2
        found = _LAUNCHES[key] = (
            _kernel(activation),
            mx.array([n], mx.uint32),
            {
                "template": [("T", gate_up.dtype), ("WIDTH", width)],
                "grid": ((n + THREADS - 1) // THREADS * THREADS, 1, 1),
                "threadgroup": (THREADS, 1, 1),
                "output_shapes": [(*gate_up.shape[:-1], width)],
                "output_dtypes": [gate_up.dtype],
            },
        )
    kernel, sizes, args = found
    result: mx.array = kernel(inputs=[gate_up, sizes], **args)[0]  # type: ignore[operator]
    return result


_LAUNCHES: Launches[tuple[Any, mx.array, dict[str, Any]]] = Launches()
