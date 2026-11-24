"""The MLX dtype each precision computes in (DTYPES, for the names quantization.PRECISIONS gives), and how "auto"
chooses: fp16 where a model's activations fit in it with room to spare, else bf16.

fp16 keeps 11 significant bits to bf16's 8; the effect on each model's probabilities is measured against its
creator's fp32 answers ([accuracy.md](../../docs/accuracy.md)). Its range ends at 65,504, and some architectures
carry activations near that (an encoder's few outlier dimensions reaching 34,000). So "auto" loads the model in
fp16, answers the warm-up requests while recording the largest magnitude in the residual stream and in every
matmul's output (`PeakOps`), and keeps fp16 when that stays below FP16_CEILING, two thirds of fp16's range;
otherwise the model is loaded again in bf16. Nothing is decided per model by name: the measurement decides.

The warm-up is a sample, not a bound on every possible input: an mmBERT-base encoder's peaks stayed within 1% of its
warm-up's peak (33,600) on every request of its reference, in eleven languages and up to 6,000 tokens. The
reserved range covers those cases. An input that still overflows gives a non-finite answer, which loads the model
again in bf16 for that request and after (`Decider.fallback`). bf16 is the one that fails there: its 8 bits land
0.155 from the creator's fp32, past the creator's own bf16 distance (0.127), where fp16 lands 0.017.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from . import quant
from .arch.ops import Ops
from .quantization import Dtype

FP16_MAX = 65504.0
FP16_CEILING = FP16_MAX / 1.5
DTYPES: dict[Dtype, mx.Dtype] = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}


@dataclass(frozen=True)
class PeakOps(Ops):
    """The same operations as the Ops it copies, keeping the largest magnitude of the residual stream and of every
    matmul's output in `peak` (a lazy fp32 scalar, evaluated with the pass)."""

    peak: list[mx.array] = field(default_factory=lambda: [mx.array(0.0)], hash=False, compare=False)

    def _seen(self, x: mx.array) -> mx.array:
        if x.size:
            self.peak[0] = mx.maximum(self.peak[0], mx.max(mx.abs(x)).astype(mx.float32))
        return x

    def mm(self, x: mx.array, w: quant.Matrix) -> mx.array:
        return self._seen(super().mm(x, w))

    def glu_mm(self, x: mx.array, w: quant.Matrix, activation: str = "silu") -> mx.array:
        return self._seen(super().glu_mm(x, w, activation))

    def add_norm(
        self, x: mx.array, delta: mx.array, w: mx.array, eps: float, kind: str = "rms", groups: int = 1
    ) -> tuple[mx.array, mx.array]:
        h, n = super().add_norm(x, delta, w, eps, kind, groups)
        return self._seen(h), n

    def mm_bias(self, x: mx.array, w: quant.Matrix, b: mx.array) -> mx.array:
        return self._seen(super().mm_bias(x, w, b))

    def add_layer_norm(self, x: mx.array, delta: mx.array, w: mx.array, b: mx.array, eps: float) -> mx.array:
        self._seen(x + delta)  # the residual stream, as add_norm's
        return super().add_layer_norm(x, delta, w, b, eps)

    def value(self) -> float:
        return float(np.array(self.peak[0]))
