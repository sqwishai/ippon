"""MLX's own matmul for the M5's neural accelerators, with the tile shape a template parameter, so that each
matrix shape and number of rows can run the tile that fills this chip's GPU best (choices.py picks it).

MLX picks one tile for every bf16 matmul on an M5 (64 x 128 per threadgroup, 32 x 32 per SIMD group, split-K when
K is much larger than M and N). A decision request multiplies a few dozen to a few thousand rows by each matrix,
and at a few hundred rows that tile leaves much of the GPU idle. This builds the same kernel from MLX's own tile
library (steel/gemm/gemm_nax.h, which the wheel ships as headers; the Metal compiler can't read files, so they are
inlined into the source) as an `mx.fast.metal_kernel`, with:

- the threadgroup tile (BM x BN), its grid of SIMD groups (WM x WN), the K depth of a step (BK) and the swizzle
  of threadgroups (SWZ) as parameters;
- optionally a prefetching loop (PIPE): the next 32-deep slices of x and w load into a second pair of register
  tiles while the current pair is multiplied, which hides load latency when few SIMD groups run.

The same machinery and the same order of the K loop as MLX's, so the products equal MLX's bit for bit except
where MLX would have split K. N must be a multiple of BN and K of BK (64 with PIPE); any M. A shape with no chosen
tile runs MLX's matmul.

From 65 to a few hundred rows MLX's regular tile reads the weights at half the bandwidth or less (a thin matmul is
bound by reading w), and MLX's thin kernel (`THIN_SOURCE`, which MLX itself runs only up to 64 rows) does better:
its tiles join the others as (BM, BN, 64, SM, SN, KS, THIN), the SIMD-group grid and the K split in the fields
that hold WM, WN and SWZ for the others.

`glu_gemm` is the gate/up projection with its gated linear unit in the epilogue: for a [gate; up] matrix of 2I rows
each SIMD group accumulates a tile of x @ gate.T and the same tile of x @ up.T in one pass over K (each slice of x
read once for both), and stores act(gate) * up from the fp32 accumulators, rounded once: the [M, 2I] projection is
never written or read back. The matrix keeps its layout. Its loop steps 32 deep whatever a tile's BK.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Callable, Mapping
from functools import cache
from pathlib import Path
from typing import Any, TypeVar

import mlx.core as mx

from . import activations
from ._source import metal_source
from .launches import Launches

HEADER = "mlx/backend/metal/kernels/steel/gemm/gemm_nax.h"
# What mx.fast.metal_kernel's own prelude already includes.
PRELUDE = {
    "mlx/backend/metal/kernels/utils.h",
    "mlx/backend/metal/kernels/bf16.h",
    "mlx/backend/metal/kernels/bf16_math.h",
    "mlx/backend/metal/kernels/complex.h",
    "mlx/backend/metal/kernels/defines.h",
    "mlx/backend/metal/kernels/logging.h",
}
PARAMS = ("BM", "BN", "BK", "WM", "WN", "SWZ", "PIPE")
Tile = tuple[int, int, int, int, int, int, int]  # BM, BN, BK, WM, WN, SWZ, PIPE; a thin tile: BM, BN, 64, SM, SN, KS, 2
THIN = 2  # the PIPE field of a tile of MLX's thin kernel (`THIN_SOURCE`)
SIMD = 3  # the PIPE field of a tile of MLX's SIMD-group library (M1 to M4, `SIMD_SOURCE`)
SIMD_HEADER = "mlx/backend/metal/kernels/steel/gemm/gemm.h"

TILE_LOOP = metal_source("gemm_tile_loop.metal")

SOURCE = metal_source("gemm.metal")


GLU_SOURCE = metal_source("gemm_glu.metal")


# MLX's thin kernel for the neural accelerators (steel/gemm/kernels/steel_gemm_thin_nax.h, which MLX dispatches only
# up to 64 rows), with its tile as template constants and M read at run time: each SIMD group computes a
# (16 UM) x (16 UN) block keeping a 64-deep K chunk of x in registers while the accelerator reads w's; SM x SN SIMD
# groups tile a threadgroup and KS groups of them split K, summed in threadgroup memory. Threadgroups go along M
# first, so the row tiles that run together share w's reads. From 65 to a few hundred rows it reads the weights
# 10-100% faster than MLX's regular tile (M5 Pro, a 2B Qwen3.5 hybrid's shapes). N must be a multiple of
# 16 UN SN and K of 64; any M. With GLU, w is [gate; up] (2N rows) and each SIMD group accumulates its block of both
# from the same registers of x, storing act(gate) * up from the fp32 accumulators (ACTIVATION, as `GLU_SOURCE`).
THIN_SOURCE = metal_source("gemm_thin.metal")


# M1 to M4 have no neural accelerators: MLX multiplies with its SIMD-group tile library (steel/gemm/gemm.h), x's and
# w's tiles staged in threadgroup memory, at one tile per GPU size (64 x 64 x 16 with 2 x 2 SIMD groups on a Pro or
# Max for most shapes). The same GEMMKernel with the tile a template parameter, and a fused GLU on it that stages
# the gate's and the up's tiles beside x's and stores act(gate) * up once.
SIMD_SOURCE = metal_source("gemm_simd.metal")
SIMD_GLU_SOURCE = metal_source("gemm_glu_simd.metal")


def include_dir() -> Path:
    return Path(mx.__file__).parent / "include"


def inline(path: str, seen: set[str]) -> str:
    """A header of the wheel with its #includes of MLX's headers inlined, each once."""
    out = []
    for line in (include_dir() / path).read_text().splitlines():
        m = re.match(r'\s*#include\s+"([^"]+)"', line)
        if m:
            if m.group(1) not in seen:
                seen.add(m.group(1))
                out.append(inline(m.group(1), seen))
            continue
        if line.strip() != "#pragma once":
            out.append(line)
    return "\n".join(out) + "\n"  # a trailing comment must not swallow the kernel's template line


@cache
def header() -> str:
    """The tile library and its loops, as a kernel's header: compacted (`compact`)."""
    return compact(inline(HEADER, set(PRELUDE)) + TILE_LOOP)


def compact(source: str) -> str:
    """The same program in fewer bytes: comments, indentation and blank lines out, every line kept a line (a macro's
    continuations stay). MLX hashes a custom kernel's whole source at every launch: the tile library's 42 KB took
    about 2 us a launch, 32 KB compacted (M5 Pro). Its strings hold no comment markers; a line with a string keeps its
    text."""
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    lines = (line.strip() if '"' in line else re.sub(r"//.*", "", line).strip() for line in source.splitlines())
    return "\n".join(line for line in lines if line) + "\n"


@cache
def simd_header() -> str:
    return compact(inline(SIMD_HEADER, set(PRELUDE)))


@cache
def _simd_kernel(activation: str | None = None) -> object:
    if activation is None:
        return mx.fast.metal_kernel(
            name="ippon_gemm_simd_xwt",
            input_names=["x", "w", "dims"],
            output_names=["out"],
            header=simd_header(),
            source=SIMD_SOURCE,
        )
    return mx.fast.metal_kernel(
        name=f"ippon_glu_gemm_simd_{activation}",
        input_names=["x", "w", "dims"],
        output_names=["out"],
        header=simd_header() + activations.HEADER,
        source=SIMD_GLU_SOURCE.replace("ACTIVATION", activations.activation_source(activation, "v")),
    )


@cache
def _kernel() -> object:
    return mx.fast.metal_kernel(
        name="ippon_gemm_xwt",
        input_names=["x", "w", "dims"],
        output_names=["out"],
        header=header(),
        source=SOURCE,
    )


@cache
def _thin_kernel(activation: str | None = None) -> object:
    act = activations.activation_source(activation, "v")
    return mx.fast.metal_kernel(
        name="ippon_gemm_thin_xwt" if activation is None else f"ippon_glu_gemm_thin_{activation}",
        input_names=["x", "w", "dims"],
        output_names=["out"],
        header=header() + ("" if activation is None else activations.HEADER),
        source=THIN_SOURCE.replace("ACTIVATION", act),
    )


@cache
def _glu_kernel(activation: str) -> object:
    act = activations.activation_source(activation, "v")
    return mx.fast.metal_kernel(
        name=f"ippon_glu_gemm_{activation}",
        input_names=["x", "w", "dims"],
        output_names=["out"],
        header=header() + activations.HEADER,
        source=GLU_SOURCE.replace("ACTIVATION", act),
    )


def is_thin(tile: Tile) -> bool:
    return tile[6] == THIN


def is_simd(tile: Tile) -> bool:
    return tile[6] == SIMD


def fits(tile: Tile, n: int, k: int) -> bool:
    bm, bn, bk, wm, wn, _, pipe = tile
    if pipe == SIMD:
        return n % bn == 0 and k % bk == 0 and bm % (8 * wm) == 0 and bn % (8 * wn) == 0
    if pipe == THIN:
        return n % bn == 0 and k % 64 == 0
    return n % bn == 0 and k % (64 if pipe else bk) == 0 and bm % wm == 0 and bn % wn == 0


def gemm(x: mx.array, w: mx.array, tile: Tile) -> mx.array:
    """x [..., K] @ w [N, K].T (bf16 or fp16) at `tile`: [..., N], x's leading dimensions read as its M
    rows (no reshape either side). Its threadgroups are laid out as MLX lays out its own: (32, WN, WM) threads; a thin
    tile in one line of threadgroups, M first."""
    if is_thin(tile):
        return _launch(_thin_kernel(), "thin", x, w, tile, w.shape[0], _thin_args)
    if is_simd(tile):
        return _launch(_simd_kernel(), "simd", x, w, tile, w.shape[0], _tile_args)
    return _launch(_kernel(), "nax", x, w, tile, w.shape[0], _tile_args)


def glu_gemm(x: mx.array, w: mx.array, tile: Tile, activation: str = "silu") -> mx.array:
    """act(x @ gate.T) * (x @ up.T) for w = [gate; up] [2I, K]: [..., I], at `tile` (which must fit I columns)."""
    if is_thin(tile):
        return _launch(_thin_kernel(activation), f"thin {activation}", x, w, tile, w.shape[0] // 2, _thin_args, True)
    if is_simd(tile):
        return _launch(_simd_kernel(activation), f"simd {activation}", x, w, tile, w.shape[0] // 2, _tile_args, True)
    return _launch(_glu_kernel(activation), f"glu {activation}", x, w, tile, w.shape[0] // 2, _tile_args, True)


_LAUNCHES: Launches[tuple[mx.array, dict[str, Any]]] = Launches()


def _launch(
    kernel: object,
    name: str,
    x: mx.array,
    w: mx.array,
    tile: Tile,
    n: int,
    arguments: Callable[[mx.Dtype, int, int, int, Tile, bool], dict[str, Any]],
    glu: bool = False,
) -> mx.array:
    lead, k = x.shape[:-1], x.shape[-1]
    key = (name, x.dtype, lead, n, k, tile)
    found = _LAUNCHES.get(key)
    if found is None:
        m = x.size // k
        args = arguments(x.dtype, m, n, k, tile, glu)
        found = _LAUNCHES[key] = (
            mx.array([m], mx.int32),
            args | {"output_shapes": [(*lead, n)], "output_dtypes": [x.dtype]},
        )
    dims, args = found
    return kernel(inputs=[x, w, dims], **args)[0]  # type: ignore[operator, no-any-return]


def _thin_args(dtype: mx.Dtype, m: int, n: int, k: int, tile: Tile, glu: bool) -> dict[str, Any]:
    bm, bn, _, sm, sn, ks, _ = tile
    um, un = bm // (16 * sm), bn // (16 * sn)
    groups = ((m + bm - 1) // bm) * (n // bn)
    threads = 32 * sm * sn * ks
    template = [("T", dtype), ("N", n), ("K", k), ("UM", um), ("UN", un), ("SM", sm), ("SN", sn), ("KS", ks)]
    return {"template": [*template, ("GLU", glu)], "grid": (groups * threads, 1, 1), "threadgroup": (threads, 1, 1)}


def _tile_args(dtype: mx.Dtype, m: int, n: int, k: int, tile: Tile, glu: bool) -> dict[str, Any]:
    bm, bn, _, wm, wn, swz, pipe = tile
    grid_x, grid_y = (n // bn) << swz, ((m + bm - 1) // bm + (1 << swz) - 1) >> swz
    return {
        "template": [("T", dtype), ("N", n), ("K", k)]
        + [(p, v) for p, v in zip(PARAMS[:6], tile[:6], strict=True)]
        + [("ALIGN_M", m >= bm), ("SHIFT", m >= bm and m % bm != 0), ("PIPE", bool(pipe))],
        "grid": (grid_x * 32, grid_y * wn, wm),
        "threadgroup": (32, wn, wm),
    }


# the thin tiles, as (SM, SN, KS) of 32 x 32 SIMD-group blocks: MLX's own (1, 1, 4) and wider ones
THIN_SHAPES = (
    (1, 1, 4),
    (1, 1, 2),
    (1, 1, 1),
    (2, 1, 2),
    (3, 1, 1),
    (3, 1, 2),
    (4, 1, 1),
    (4, 1, 2),
    (2, 2, 1),
    (2, 2, 2),
)
THIN_TILES: tuple[Tile, ...] = tuple((32 * sm, 32 * sn, 64, sm, sn, ks, THIN) for sm, sn, ks in THIN_SHAPES)


def candidates(glu: bool = False) -> list[Tile]:
    """Every tile the kernels compute. NAX: SIMD-group tiles of 32 or 64 rows and columns (at most 64 x 32), 1, 2 or 4
    SIMD groups each way, K steps of 128 or 256 (with `glu`, 128 only: the fused GLU's loop steps 32 deep whatever
    BK, so the others would be the same kernels again), swizzle 0 or 2, and the prefetching loop, and MLX's thin
    kernel's tiles (`THIN_TILES`)."""
    out: list[Tile] = []
    for sm in (32, 64):
        for sn in (32, 64):
            if sm * sn > 64 * 32:
                continue
            for wm in (1, 2, 4):
                for wn in (1, 2, 4):
                    for bk in (128,) if glu else (128, 256):
                        for swz in (0, 2):
                            out.append((sm * wm, sn * wn, bk, wm, wn, swz, 0))
                            if bk == 128:
                                out.append((sm * wm, sn * wn, bk, wm, wn, swz, 1))
    return out + list(THIN_TILES)


def simd_candidates(glu: bool = False) -> list[Tile]:
    """Every M1-to-M4 tile: threadgroup tiles of 32, 64 or 128 rows and columns, 1, 2 or 4 SIMD groups each way
    (each at least an 8 x 8 fragment, and no more rows than threads, as MLX's loaders need), K steps of 16 or 32,
    swizzle 0 or 2, within 32 KB of threadgroup memory (x's tile and w's, or with `glu` the gate's and the up's: MLX
    pads each row by 16 bytes)."""
    out: list[Tile] = []
    for bm in (32, 64, 128):
        for bn in (32, 64, 128):
            for bk in (16, 32):
                staged = 2 * ((bm + (2 if glu else 1) * bn) * (bk + 8))
                if staged > 32768:
                    continue
                for wm in (1, 2, 4):
                    for wn in (1, 2, 4):
                        threads = 32 * wm * wn  # MLX's loaders give each thread whole slices of a tile's rows
                        if bm % (8 * wm) or bn % (8 * wn) or wm * wn > 8 or max(bm, bn) > threads:
                            continue
                        for swz in (0, 2):
                            out.append((bm, bn, bk, wm, wn, swz, SIMD))
    return out


T = TypeVar("T")
# A matrix's chosen kernel for each band of input rows: (from rows, the tile from there; None: MLX's own), ascending
Bands = tuple[tuple[int, T | None], ...]


def banded(choose: Callable[[int], T | None], top: int) -> Bands[T]:
    """`choose(rows)` for every row count from 1 to `top` as bands, a new one where the choice changes; the choice
    at `top` holds for every count above."""
    out: list[tuple[int, T | None]] = []
    for rows in range(1, top + 1):
        tile = choose(rows)
        if not out or out[-1][1] != tile:
            out.append((rows, tile))
    return tuple(out)


def band(bands: Bands[T], rows: int) -> T | None:
    """The tile `bands` choose for `rows` input rows; None (MLX's own) below the first band."""
    i = bisect.bisect_right(bands, rows, key=lambda b: b[0]) - 1
    return bands[i][1] if i >= 0 else None


def key(dtype: mx.Dtype, n: int, k: int, glu: str | None = None) -> str:
    """A shape's name: "bf16:NxK" for the matmul, "silu:bf16:NxK" for its fused GLU (N the [gate; up] rows)."""
    name = f"{'bf16' if dtype == mx.bfloat16 else 'fp16'}:{n}x{k}"
    return f"{glu}:{name}" if glu else name


def tile_for(
    table: Mapping[str, Bands[Tile]], dtype: mx.Dtype, n: int, k: int, rows: int, glu: str | None = None
) -> Tile | None:
    """The tile choices.py chose (`table`, by `key`) for `rows` rows of an [n, k] matrix; None for MLX's matmul."""
    found = table.get(key(dtype, n, k, glu))
    return band(found, rows) if found is not None else None


@cache
def failure(simd: bool = False) -> str | None:
    """Why a tile library does not run on this Mac, None when it compiles as these kernels and a small product
    equals MLX's: the neural accelerators' library, or (`simd`) MLX's SIMD-group one."""
    if not mx.metal.is_available():
        return "no Metal GPU"
    dtype, tile = (mx.float16, (64, 64, 16, 2, 2, 0, SIMD)) if simd else (mx.bfloat16, (64, 64, 128, 2, 2, 0, 1))
    try:
        with mx.stream(mx.gpu):
            x = (mx.random.normal((40, 256), key=mx.random.key(0)) * 0.5).astype(dtype)
            w = (mx.random.normal((128, 256), key=mx.random.key(1)) * 0.05).astype(dtype)
            got = gemm(x, w, tile)
            if mx.allclose(got.astype(mx.float32), (x @ w.T).astype(mx.float32), atol=1e-2).item():
                return None
            return "a small product differs from MLX's"
    # no neural accelerators, or a version of the headers this kernel doesn't compile against
    except Exception as error:
        return f"{type(error).__name__}: {error}"


def available() -> bool:
    """True when the neural-accelerator tile library runs on this Mac (`failure`)."""
    return failure() is None
