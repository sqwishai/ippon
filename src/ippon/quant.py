"""Packed weight matrices in MLX's affine, MXFP8, NVFP4 and MXFP4 encodings, retaining each matrix's mode,
encoded scales and optional NVFP4 checkpoint multiplier.

A matrix is a full-precision array or a `QuantizedMatrix`; `mm` multiplies by either, `rows` reads some of its rows
(an embedding lookup, the label rows of an output matrix) as full precision, `concat` joins matrices that read the
same input (q, k and v; gate and up), keeping them quantized when they share bits and group size. Builds below 8 bits
load only when the caller asks for them (drift.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import mlx.core as mx
import numpy as np

from .errors import UnsupportedModel
from .host_rows import HostRows
from .kernels import gemm, qgemm
from .quantization import QUANTIZE_TO, group_for, unsupported_format

NVFP4_MAX = 448 * 6


def checkpoint_multiplier(scale: mx.array | None, is_amax: bool) -> mx.array | None:
    return scale / NVFP4_MAX if is_amax and scale is not None else scale


@dataclass(frozen=True)
class QuantizedMatrix:
    """A packed matrix and its original per-group encoding, plus an optional checkpoint multiplier."""

    w: mx.array  # [rows, columns * bits / 32] uint32
    scales: mx.array  # [rows, columns / group_size]
    biases: mx.array | None
    bits: int
    group_size: int
    # what choices.py chose for the matrix's shape on this chip (none: MLX's quantized matmul at every size)
    tiles: gemm.Bands[qgemm.QTile] = ()  # the kernels/qgemm.py tile from each row count (None: MLX's)
    glu: tuple[str, gemm.Bands[qgemm.QTile]] | None = None  # a [gate; up]'s fused GLU tiles, with its activation
    unpack_rows: int = 0  # from this many input rows, unpacked on every call and multiplied at full precision

    mode: str = "affine"
    global_scale: mx.array | None = None  # original scale, interpreted by global_scale_is_amax
    compute_dtype: mx.Dtype = mx.bfloat16
    global_scale_is_amax: bool = False  # native MLX amax; otherwise the Ollama/ModelOpt multiplier
    row_multiplier: mx.array | None = None  # [rows] float32: projections joined across different global scales

    @property
    def shape(self) -> tuple[int, int]:
        return self.w.shape[0], self.w.shape[1] * 32 // self.bits

    @property
    def dtype(self) -> mx.Dtype:
        return self.scales.dtype if self.mode == "affine" else self.compute_dtype

    def dense(self, dtype: mx.Dtype | None = None) -> mx.array:
        x = mx.dequantize(
            self.w,
            self.scales,
            self.biases,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            dtype=None if self.mode == "affine" else (dtype or self.compute_dtype),
        )
        multiplier = self.multiplier()
        if self.row_multiplier is not None:
            multiplier = self.row_multiplier[:, None]
        elif multiplier is not None and multiplier.ndim:
            multiplier = multiplier[..., None, None]
        x = scaled(x, multiplier)
        return x if dtype is None else x.astype(dtype)

    def multiplier(self) -> mx.array | None:
        """The checkpoint multiplier its products take: one per row when it joins projections whose global scales
        differ, else its own (None without one)."""
        if self.row_multiplier is not None:
            return self.row_multiplier
        return checkpoint_multiplier(self.global_scale, self.global_scale_is_amax)

    def astype(self, dtype: mx.Dtype) -> QuantizedMatrix:
        """Change the compute dtype, preserving encoded floating-point scale bytes."""
        return replace(
            self,
            compute_dtype=dtype,
            scales=self.scales.astype(dtype) if self.mode == "affine" else self.scales,
            biases=self.biases.astype(dtype) if self.biases is not None else None,
        )

    @property
    def key(self) -> str:
        """Format, group size, compute dtype and shape, as q8g64:fp16:12352x4096; another mode's name in place of
        "q", as "mxfp8:8g32:...". NVFP4 adds its external scale layout: scalar or per-row scales require the fp32
        fallback, and the staged kernel reads a separate multiplier per row when projections with different scales
        have been joined."""
        rows, columns = self.shape
        label = {mx.float16: "fp16", mx.bfloat16: "bf16"}.get(self.dtype, "fp32")
        prefix = "q" if self.mode == "affine" else self.mode + ":"
        scale = ""
        if self.mode == "nvfp4":
            kind = "row" if self.row_multiplier is not None else "scalar" if self.global_scale is not None else "none"
            scale = f":scale-{kind}"
        return f"{prefix}{self.bits}g{self.group_size}:{label}:{rows}x{columns}{scale}"

    def arrays(self) -> list[mx.array]:
        return [a for a in (self.w, self.scales, self.biases, self.global_scale, self.row_multiplier) if a is not None]

    def tile(self, m: int) -> qgemm.QTile | None:
        """The chosen kernels/qgemm.py tile for m input rows; None for MLX's quantized matmul."""
        return gemm.band(self.tiles, m)

    def glu_tile(self, m: int, activation: str) -> qgemm.QTile | None:
        """The chosen fused-GLU tile (`qgemm.glu_qmm`) for m input rows of this [gate; up] with `activation`."""
        return gemm.band(self.glu[1], m) if self.glu is not None and self.glu[0] == activation else None


def scaled(x: mx.array, multiplier: mx.array | None) -> mx.array:
    """Apply a checkpoint multiplier in fp32, restoring x's dtype. NVFP4 products with an external multiplier stay
    in fp32 until this scaling, then their caller restores the input dtype."""
    return x if multiplier is None else (x.astype(mx.float32) * multiplier).astype(x.dtype)


@dataclass(frozen=True)
class JoinedMatrix:
    """Row projections with different formats or scales, retained as their original packed matrices."""

    parts: tuple[mx.array | QuantizedMatrix | JoinedMatrix, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return sum(p.shape[0] for p in self.parts), self.parts[0].shape[1]

    def astype(self, dtype: mx.Dtype) -> JoinedMatrix:
        return JoinedMatrix(tuple(p.astype(dtype) for p in self.parts))

    @property
    def dtype(self) -> mx.Dtype:
        return self.parts[0].dtype

    def dense(self, dtype: mx.Dtype | None = None) -> mx.array:
        return mx.concatenate(
            [
                p.dense(dtype)
                if isinstance(p, QuantizedMatrix | JoinedMatrix)
                else p
                if dtype is None
                else p.astype(dtype)
                for p in self.parts
            ]
        )

    def arrays(self) -> list[mx.array]:
        return arrays(self.parts)


Matrix = mx.array | QuantizedMatrix | JoinedMatrix
Table = Matrix | HostRows  # a vocabulary table: loaded, or read from its file on the CPU (Weights.table)


def _pack(w: mx.array, bits: int | None) -> tuple[mx.array, mx.array, mx.array, int, int] | None:
    """w's rows packed to `bits` (codes, scales, biases, bits, group size), evaluated now so its full-precision copy
    can be freed; None for no `bits`. Groups are as `quantization.group_for` gives them for w's width; None for a
    matrix of a width it groups in none."""
    if bits is None:
        return None
    group = group_for(w.shape[-1])
    if group is None:
        return None
    q, scales, biases = mx.quantize(w, group_size=group, bits=bits)
    mx.eval(q, scales, biases)
    return q, scales, biases, bits, group


def _refuse_bits(bits: int | None) -> None:
    if bits is not None and bits not in QUANTIZE_TO:
        raise unsupported_format(f"{bits}-bit quantization")


def quantize(w: Matrix, bits: int | None) -> Matrix:
    """`w` quantized to `bits` (`_pack`); None, already quantized, or of neither group width: as it is."""
    _refuse_bits(bits)
    if isinstance(w, JoinedMatrix):
        return concat([quantize(p, bits) for p in w.parts])
    if isinstance(w, QuantizedMatrix) or (packed := _pack(w, bits)) is None:
        return w
    return QuantizedMatrix(*packed)


def unpacked(x: mx.array, w: Matrix) -> Matrix:
    """w unpacked to x's dtype where choices.py multiplies x's rows by it at full precision (`w.unpack_rows`), for
    the caller's full-precision kernel choices; else w as it is."""
    if (
        isinstance(w, QuantizedMatrix)
        and w.unpack_rows
        and x.size // x.shape[-1] >= w.unpack_rows
        and x.dtype == w.dtype
        and mx.default_device() == mx.gpu
    ):
        return w.dense(x.dtype)
    return w


def mm(x: mx.array, w: Matrix) -> mx.array:
    """x @ w.T, for a full-precision or a quantized weight. A quantized one is multiplied by MLX's quantized
    matmul, or ippon's (kernels/qgemm.py) at the tile choices.py chose for x's rows (`w.tiles`)."""
    if isinstance(w, JoinedMatrix):
        return mx.concatenate([mm(x, p) for p in w.parts], axis=-1)
    if isinstance(w, QuantizedMatrix):
        m = x.size // x.shape[-1]
        tile = w.tile(m) if w.tiles and x.dtype == w.dtype and mx.default_device() == mx.gpu else None
        if tile is not None:
            k, multiplier = x.shape[-1], w.multiplier()
            if w.mode == "affine":
                y = qgemm.qmm(x.reshape(m, k), w.w, w.scales, w.biases, w.bits, w.group_size, tile)
                return scaled(y.reshape(*x.shape[:-1], w.shape[0]), multiplier)
            # an fp mode's multiplier folds into its group scales inside the kernel: no fp32 pass after, no overflow
            y = qgemm.qmm(x.reshape(m, k), w.w, w.scales, None, w.bits, w.group_size, tile, w.mode, multiplier)
            return y.reshape(*x.shape[:-1], w.shape[0])
        multiplier = w.multiplier()
        # d3-mini NVFP4 on M5 Pro: rounding the unscaled product to fp16 overflowed warm-up, and scaling in fp32
        # first gives a finite peak of 267.5. Packed weights and encoded scales stay unchanged.
        operand = x.astype(mx.float32) if w.mode == "nvfp4" and multiplier is not None else x
        y = mx.quantized_matmul(
            operand, w.w, w.scales, w.biases, transpose=True, group_size=w.group_size, bits=w.bits, mode=w.mode
        )
        return scaled(y, multiplier).astype(x.dtype)
    return x @ w.T


def glu_mm(x: mx.array, w: Matrix, activation: str) -> mx.array | None:
    """act(x @ gate.T) * (x @ up.T) for a quantized [gate; up] at its chosen fused-GLU tile for x's rows, [..., I];
    None where it has none (the caller multiplies, then applies the GLU)."""
    if not isinstance(w, QuantizedMatrix) or (w.mode == "affine" and w.global_scale is not None):
        return None
    m = x.size // x.shape[-1]
    if x.dtype != w.dtype or mx.default_device() != mx.gpu:
        return None
    tile = w.glu_tile(m, activation)
    if tile is None:
        return None
    k = x.shape[-1]
    multiplier = None if w.mode == "affine" else w.multiplier()
    y = qgemm.glu_qmm(
        x.reshape(m, k), w.w, w.scales, w.biases, w.bits, w.group_size, tile, activation, w.mode, multiplier
    )
    return y.reshape(*x.shape[:-1], w.shape[0] // 2)


def rows(w: Table, ids: mx.array) -> mx.array:
    """The rows `ids` of `w` (any shape of indices), in full precision."""
    if isinstance(w, HostRows):
        picked = w.pick(np.asarray(ids))
        if w.packing is None:
            return picked[0].astype(w.dtype)
        bits, group_size, mode = w.packing  # unpacked as the loaded table is (Weights.matrix, then astype)
        biases = picked[2] if len(picked) == 3 else None
        return QuantizedMatrix(picked[0], picked[1], biases, bits, group_size, mode=mode).astype(w.dtype).dense()
    if isinstance(w, JoinedMatrix) or (isinstance(w, QuantizedMatrix) and w.row_multiplier is not None):
        raise TypeError("joined projections are not embedding tables")
    if isinstance(w, QuantizedMatrix):
        return replace(
            w, w=w.w[ids], scales=w.scales[ids], biases=w.biases[ids] if w.biases is not None else None
        ).dense()
    return w[ids]


def concat(parts: Sequence[Matrix]) -> Matrix:
    """Join row projections, retaining separate packed matrices when their formats differ. Projections in one fp
    format join across different global scales, as Ollama's NVFP4 builds give each its own. The joined matrix keeps
    one multiplier per row, which the staged kernel folds into each row's group scales, so they still run as one
    multiply (and a [gate; up] as the fused GLU)."""
    if all(isinstance(p, QuantizedMatrix) for p in parts):
        qs = [p for p in parts if isinstance(p, QuantizedMatrix)]
        multipliers = [q.multiplier() for q in qs]
        one_format = len({(q.bits, q.group_size, q.mode, q.dtype) for q in qs}) == 1
        if one_format and (all(m is None for m in multipliers) or qs[0].mode != "affine"):
            per_row = None
            if any(m is not None for m in multipliers):
                per_row = mx.concatenate(
                    [
                        mx.broadcast_to((qgemm.ONE if m is None else m.astype(mx.float32)).reshape(-1), (q.shape[0],))
                        if q.row_multiplier is None
                        else q.row_multiplier
                        for q, m in zip(qs, multipliers, strict=True)
                    ]
                )
            return QuantizedMatrix(
                mx.concatenate([q.w for q in qs]),
                mx.concatenate([q.scales for q in qs]),
                mx.concatenate([q.biases for q in qs if q.biases is not None]) if qs[0].biases is not None else None,
                qs[0].bits,
                qs[0].group_size,
                mode=qs[0].mode,
                compute_dtype=qs[0].compute_dtype,
                row_multiplier=per_row,
            )
    if all(isinstance(p, mx.array) for p in parts):
        return mx.concatenate(list(parts))  # type: ignore[arg-type]
    return JoinedMatrix(tuple(parts))


@dataclass(frozen=True)
class Stacked:
    """One weight matrix per expert of a mixture-of-experts layer, [experts, rows, columns]: full precision, or
    quantized group-wise as QuantizedMatrix is (w, scales and biases with a leading experts axis). `gather` multiplies
    each row of its input by its own expert's matrix (MLX's gather_mm / gather_qmm)."""

    w: mx.array
    scales: mx.array | None = None
    biases: mx.array | None = None
    bits: int = 0
    group_size: int = 0
    mode: str = "affine"
    global_scale: mx.array | None = None  # one original scale per expert
    global_scale_is_amax: bool = False

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            self.w.shape[0],
            self.w.shape[1],
            self.w.shape[2] * 32 // self.bits if self.scales is not None else self.w.shape[2],
        )

    @classmethod
    def of(cls, parts: Sequence[Matrix]) -> Stacked:
        """Per-expert matrices stacked: all full precision, or all quantized alike."""
        if all(isinstance(p, mx.array) for p in parts):
            return cls(mx.stack(list(parts)))  # type: ignore[arg-type]
        qs = [p for p in parts if isinstance(p, QuantizedMatrix)]
        if len(qs) == len(parts) and len({(q.bits, q.group_size, q.mode) for q in qs}) == 1:
            native_scale = any(q.global_scale_is_amax for q in qs)

            def scale(q: QuantizedMatrix) -> mx.array:
                if q.global_scale is None:
                    return mx.array(float(NVFP4_MAX) if native_scale else 1.0)
                return q.global_scale * NVFP4_MAX if native_scale and not q.global_scale_is_amax else q.global_scale

            return cls(
                mx.stack([q.w for q in qs]),
                mx.stack([q.scales for q in qs]),
                mx.stack([q.biases for q in qs if q.biases is not None]) if qs[0].biases is not None else None,
                qs[0].bits,
                qs[0].group_size,
                qs[0].mode,
                mx.stack([scale(q) for q in qs]) if any(q.global_scale is not None for q in qs) else None,
                native_scale,
            )
        raise UnsupportedModel("an expert bank needs one encoding; different formats among its experts are unsupported")

    @classmethod
    def stored(cls, m: Matrix) -> Stacked:
        """A checkpoint's tensor that already holds every expert ([experts, rows, columns])."""
        if isinstance(m, QuantizedMatrix):
            return cls(m.w, m.scales, m.biases, m.bits, m.group_size, m.mode, m.global_scale, m.global_scale_is_amax)
        if isinstance(m, JoinedMatrix):
            raise UnsupportedModel("expert banks must be dense or packed tensors")
        return cls(m)

    @classmethod
    def join(cls, parts: Sequence[Stacked]) -> Stacked | JoinedStacked:
        """Stacked matrices joined by rows (each expert's [gate; up])."""
        if all(p.scales is None for p in parts):
            return cls(mx.concatenate([p.w for p in parts], axis=1))
        if len({(p.bits, p.group_size, p.mode, p.scales is None) for p in parts}) != 1 or any(
            p.global_scale is not None for p in parts
        ):
            return JoinedStacked(tuple(parts))
        return cls(
            mx.concatenate([p.w for p in parts], axis=1),
            mx.concatenate([p.scales for p in parts if p.scales is not None], axis=1),
            mx.concatenate([p.biases for p in parts if p.biases is not None], axis=1)
            if parts[0].biases is not None
            else None,
            parts[0].bits,
            parts[0].group_size,
            parts[0].mode,
        )

    def quantized(self, bits: int | None) -> Stacked:
        """Quantized to `bits` (`_pack`); None, already quantized, or of neither group width: as it is."""
        _refuse_bits(bits)
        if self.scales is not None or (packed := _pack(self.w, bits)) is None:
            return self
        return Stacked(*packed)

    def astype(self, dtype: mx.Dtype) -> Stacked:
        if self.scales is None:
            return replace(self, w=self.w.astype(dtype))
        return replace(
            self,
            scales=self.scales.astype(dtype) if self.mode == "affine" else self.scales,
            biases=self.biases.astype(dtype) if self.biases is not None else None,
        )

    def arrays(self) -> list[mx.array]:
        return [a for a in (self.w, self.scales, self.biases, self.global_scale) if a is not None]

    def gather(self, x: mx.array, experts: mx.array, sorted_experts: bool = False) -> mx.array:
        """x [pairs, 1, columns] times each pair's expert's matrix, transposed: [pairs, 1, rows]."""
        if self.scales is None:
            return mx.gather_mm(x, self.w.swapaxes(-1, -2), rhs_indices=experts, sorted_indices=sorted_experts)
        # MLX 0.32.4's gather_qmm takes a global scale on the GPU only, and the CPU scales its output instead
        native_scale = self.global_scale is not None and mx.default_device() == mx.gpu
        bank_scale = None if self.global_scale is None else mx.broadcast_to(self.global_scale, (self.w.shape[0],))
        operand = x.astype(mx.float32) if self.mode == "nvfp4" and bank_scale is not None and not native_scale else x
        result = mx.gather_qmm(
            operand,
            self.w,
            self.scales,
            self.biases,
            rhs_indices=experts,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            global_scale=mx.contiguous(bank_scale if self.global_scale_is_amax else bank_scale * NVFP4_MAX)
            if native_scale and bank_scale is not None
            else None,
            sorted_indices=sorted_experts,
        )
        multiplier = (
            None
            if bank_scale is None or native_scale
            else checkpoint_multiplier(bank_scale[experts][..., None, None], self.global_scale_is_amax)
        )
        return scaled(result, multiplier).astype(x.dtype)


@dataclass(frozen=True)
class JoinedStacked:
    """Expert projections whose formats or global scales require separate multiplies."""

    parts: tuple[Stacked, ...]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.parts[0].shape[0], sum(p.shape[1] for p in self.parts), self.parts[0].shape[2]

    def astype(self, dtype: mx.Dtype) -> JoinedStacked:
        return JoinedStacked(tuple(p.astype(dtype) for p in self.parts))

    def quantized(self, bits: int | None) -> JoinedStacked:
        return JoinedStacked(tuple(p.quantized(bits) for p in self.parts))

    def arrays(self) -> list[mx.array]:
        return [a for p in self.parts for a in p.arrays()]

    def gather(self, x: mx.array, experts: mx.array, sorted_experts: bool = False) -> mx.array:
        return mx.concatenate([p.gather(x, experts, sorted_experts) for p in self.parts], axis=-1)


def arrays(values: Sequence[object]) -> list[mx.array]:
    """Every array among `values`, a quantized matrix's three and a tuple's (a norm's weight and bias) included (for
    one `mx.eval`)."""
    out: list[mx.array] = []
    for v in values:
        if isinstance(v, QuantizedMatrix | JoinedMatrix | Stacked | JoinedStacked):
            out += v.arrays()
        elif isinstance(v, mx.array):
            out.append(v)
        elif isinstance(v, tuple | list):
            out += arrays(v)
    return out
