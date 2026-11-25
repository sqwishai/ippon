"""The activations that the fused matmul kernels apply in their epilogues (kernels/gemm.py, kernels/qgemm.py), as
Metal statements on an fp32 value:

    silu        SwiGLU (Qwen3.5, LFM2)
    gelu        GeGLU with exact GELU (ModernBERT)
    gelu_tanh   GeGLU with the tanh approximation (Gemma 4)
"""

from __future__ import annotations

from ._source import metal_source

ACTIVATIONS = {name: metal_source(f"activation_{name}.metal").rstrip("\n") for name in ("silu", "gelu", "gelu_tanh")}
# erf in fp32 (Abramowitz and Stegun 7.1.26, absolute error below 1.5e-7): Metal has none, MLX's is internal
HEADER = metal_source("erf.metal")


def activation_source(name: str | None, value: str) -> str:
    """The Metal statement computing activation `name` (None: identity) of fp32 variable `value` into `a`."""
    source = metal_source("activation_identity.metal").rstrip("\n") if name is None else ACTIVATIONS[name]
    return source.replace("X", value)
