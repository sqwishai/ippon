"""The affine-quantized matmul (x @ w.T for a packed 2- to 8-bit w) on the M5's neural accelerators, with its row
block, its rows per SIMD group and a K split as template parameters, so that each quantized matrix shape and number
of rows can run the variant that fills this chip's GPU best (choices.py picks it).

MLX 0.32.4 runs a prompt's rows through one fixed 64 x 64 x 64 tile of 2 x 2 SIMD groups, and a matrix with few
output tiles through split-K on SIMD-group kernels (not the neural accelerators). This is MLX's next schedule for
the same tile (its main branch, #4682: `QuantizedBlockLoaderStaged` and `qmm_t_nax_staged_impl`, MIT, Copyright
Apple Inc.), built as an `mx.fast.metal_kernel` on the released MLX's NAX fragments (steel/gemm/nax.h):

- the 64 x 64 tile of weights is unpacked into threadgroup memory, double-buffered, one barrier per 64-deep K step;
  each thread fetches the next step's packed words, scale and bias into registers before the current step's
  multiplies, and each SIMD group loads the next step's activations the same way;
- each SIMD group computes SM (16 or 32) rows of a BM-row block (BM / SM SIMD groups share one unpacked tile);
- with S > 1 the K range is split into S parts whose fp32 partial products are summed after, so a matrix with few
  output tiles still fills the GPU.

The unpacking is MLX's own (`dequantize` in quantized_nax.h): the same arithmetic, so the products differ from MLX's
only in the order of the fp32 sums. N must be a multiple of 64; K of 64, of the group size, and of S times both.
Every loop over a thread's registers is unrolled in full, as MLX's are: rolled, the register arrays it indexes spill
to memory and the kernel ran at a quarter of MLX's speed.

Measured on the M5 Pro (group size 64, fp16, in a chain of distinct weight copies), against the faster of
MLX's quantized matmul and its rows given as two halves of a batch, at the best variant: 8-bit 6-90% faster at 16 to
2,048 rows over seven decision models' shapes (Clef-Flash's 4096x12288 down
projection 76% at 16 rows, 35% at 333, 15% at 2,048), 4-bit 2% to 2.2x; 25 to 30 TFLOPS from 1,024 rows, where
MLX's reaches 22 to 27.

`glu_qmm` is a [gate; up] matrix's gated linear unit in the same kernel's epilogue: each 64-row tile of weights it
unpacks is 32 gate rows and the same 32 up rows, so act(gate) * up comes from the fp32 sums of one tile, rounded
once, and the [M, 2I] product is never written.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx

from . import gemm
from ._source import metal_source
from .glu import HEADER as ERF
from .glu import activation_source

QTile = tuple[int, int, int]  # BM (rows per threadgroup), SM (rows per SIMD group), S (K split)

AFFINE_HEADER = metal_source("qgemm_affine.metal")
FP_HEADER = metal_source("qgemm_fp.metal")
STAGED_HEADER = metal_source("qgemm_staged.metal")

SOURCE = metal_source("qgemm.metal")
AFFINE_LOADER = metal_source("qgemm_affine_loader.metal").rstrip("\n")
FP_LOADER = metal_source("qgemm_fp_loader.metal").rstrip("\n")


@cache
def _kernel(activation: str | None = None, fp: bool = False) -> object:
    act = activation_source(activation, "v")
    name = "ippon_qmm_staged" + ("_fp" if fp else "") + ("" if activation is None else f"_{activation}")
    fp_types = (
        gemm.inline("mlx/backend/metal/kernels/fp8.h", set()) + gemm.inline("mlx/backend/metal/kernels/fp4.h", set())
        if fp
        else ""
    )
    header = AFFINE_HEADER + (FP_HEADER if fp else "") + STAGED_HEADER
    return mx.fast.metal_kernel(
        name=name,
        input_names=["x", "w", "scales", "mult" if fp else "biases", "dims"],
        output_names=["out"],
        header=gemm.header() + ERF + fp_types + header,
        source=SOURCE.replace("LOADER", FP_LOADER if fp else AFFINE_LOADER).replace("ACTIVATION", act),
    )


# the variants it computes: 32 to 128 rows per threadgroup, 16 or 32 per SIMD group, K whole or in 2 or 4 parts
CANDIDATES: tuple[QTile, ...] = tuple(
    (bm, sm, s) for bm in (32, 64, 128) for sm in (16, 32) for s in (1, 2, 4) if bm % sm == 0
)


FP_MODES = ("mxfp8", "mxfp4", "nvfp4")  # values fp8 e4m3 or fp4 e2m1, a uint8 scale per group of 32, or 16 (nvfp4)


def fits(tile: QTile, n: int, k: int, bits: int, group_size: int, mode: str = "affine") -> bool:
    """Whether `tile` computes an [n, k] matrix of `bits` and `group_size` in `mode`: whole 64-column tiles and K
    steps, each of its threads' unpacked values whole words in one row, and in one group (affine, as MLX's loader) or
    whole groups (the fp modes)."""
    bm, sm, s = tile
    threads = 32 * bm // sm
    if mode in FP_MODES:
        pack = 8 // bits
        reads = (64 // pack) * 64 // threads  # bytes a thread unpacks a step
        values = reads * pack
        return (
            n % 64 == 0
            and k % (s * max(64, group_size)) == 0
            and (64 // pack) * 64 >= threads
            and reads % 4 == 0
            and values <= 64
            and values % 8 == 0
            and (values % group_size == 0 or group_size % values == 0)
        )
    pack = {3: 8, 5: 8, 6: 4}.get(bits, 8 // bits)
    width = {3: 3, 5: 5, 6: 3}.get(bits, 1)
    reads = (64 // pack) * 64 // threads
    values = reads * pack
    return (
        mode == "affine"
        and bits in (2, 3, 4, 5, 6, 8)
        and n % 64 == 0
        and k % (s * max(64, group_size)) == 0
        and (64 // pack) * 64 >= threads
        and reads * width % 4 == 0
        and values <= min(group_size, 64)
        and values % 8 == 0
    )


def qmm(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array | None,
    bits: int,
    group_size: int,
    tile: QTile,
    mode: str = "affine",
    mult: mx.array | None = None,
) -> mx.array:
    """x [M, K] @ w.T for w [N, K] packed as MLX packs `mode` (affine: w uint32, scales and biases [N, K / group_size]
    in x's dtype; an fp mode: uint8 scales, no biases, `mult` an nvfp4 checkpoint's multiplier, float32: one, or one
    per row of w), at `tile`."""
    m = x.shape[0]
    n = w.shape[0]
    s = tile[2]
    out_dtype = x.dtype if s == 1 else mx.float32
    shape = (m, n) if s == 1 else (s, m, n)
    y = _launch(x, w, scales, biases, mult, bits, group_size, tile, n, n // 64, out_dtype, shape, mode)
    return y if s == 1 else y.sum(axis=0).astype(x.dtype)


def glu_qmm(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array | None,
    bits: int,
    group_size: int,
    tile: QTile,
    activation: str = "silu",
    mode: str = "affine",
    mult: mx.array | None = None,
) -> mx.array:
    """act(x @ gate.T) * (x @ up.T) for a packed w = [gate; up] [2I, K]: [M, I], at `tile` (no K split: the
    activation applies to the whole sums). Each 64-row tile of weights it unpacks is 32 gate rows and the same 32 up
    rows, so the products are those of the plain kernel, and act(gate) * up is taken from their fp32 sums and
    rounded once."""
    m, _ = x.shape
    half = w.shape[0] // 2
    return _launch(
        x, w, scales, biases, mult, bits, group_size, tile, half, half // 32, x.dtype, (m, half), mode, activation
    )


def glu_fits(tile: QTile, n: int, k: int, bits: int, group_size: int, mode: str = "affine") -> bool:
    """Whether `glu_qmm` runs at `tile` for a [gate; up] matrix of n rows: `fits`, gate rows in whole 32s, whole K."""
    return tile[2] == 1 and n % 64 == 0 and fits(tile, n, k, bits, group_size, mode)


ONE = mx.array([1.0], mx.float32)


def _launch(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array | None,
    mult: mx.array | None,
    bits: int,
    group_size: int,
    tile: QTile,
    n: int,
    groups: int,
    out_dtype: mx.Dtype,
    shape: tuple[int, ...],
    mode: str = "affine",
    activation: str | None = None,
) -> mx.array:
    m, k = x.shape
    bm, sm, s = tile
    threads = 32 * bm // sm
    fp = mode in FP_MODES
    fourth = (ONE if mult is None else mult.astype(mx.float32).reshape(-1)) if fp else biases
    per_row = [("PER_ROW", fourth.size > 1)] if fp else []  # type: ignore[union-attr]
    return _kernel(activation, fp)(  # type: ignore[operator, no-any-return]
        inputs=[x, w, scales, fourth, mx.array([m], mx.int32)],
        template=[("T", x.dtype), ("U", out_dtype), ("GS", group_size), ("BITS", bits), ("BM", bm), ("SM", sm),
                  ("K", k), ("N", n), ("S", s), ("GLU", activation is not None), *per_row],
        grid=(groups * threads, (m + bm - 1) // bm, s),
        threadgroup=(threads, 1, 1),
        output_shapes=[shape],
        output_dtypes=[out_dtype],
    )[0]  # fmt: skip
