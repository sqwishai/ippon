"""Which kernel computes each matrix multiply, chosen at load from the chip and the matrix: nothing is measured on
the user's Mac and nothing is stored there.

MLX runs every multiply of a shape with one fixed tile (64 x 128 per threadgroup on an M5), whatever the number of
input rows. A decision request multiplies a few dozen to a few thousand rows by each matrix, and at a few hundred
rows those tiles leave GPU cores idle or reread the weights. On every M5 (its neural accelerators), each distinct
shape gets, for each band of input rows (looked up per call):

- a quantized matrix (affine 4- to 8-bit, MXFP8, MXFP4, NVFP4): kernels/qgemm.py's QTILE from QROWS rows, and for
  MXFP8 WIDE_QTILE from WIDE_ROWS while the input is at most WIDE_INPUT bytes (`qgemm_bands`); below QROWS, MLX's
  quantized matmul; from UNPACK_ROWS, an MXFP8 one is unpacked to full precision on every call and multiplied as a
  full-precision one of its shape;
- a full-precision one: MLX's thin kernel (kernels/gemm.py, reading the weights once for its rows) up to THIN_ROWS,
  its K split in two when its threadgroups fill the GPU's cores fewer than SPLIT_WAVES times, then MID_TILE up to
  MID_ROWS, then MLX's matmul, or DOWN_TILE for a down projection (`gemm_bands`); its fused gate/up the same, with
  GLU_WIDE past MID_ROWS.

Full-precision matrices take these tiles only on an M5 of the M5 Pro's FITTED_CORES GPU cores, where they were
measured faster; on any other M5 they run MLX's matmul, which can't be slower than itself. On the base M5 (10 cores)
the same tiles made the 4B models' whole requests (d3-mini, H2O-Lightning-4B, pplx-decider flash,
Torchcast) take 10 to 24% longer at 1,000 words and more, though the chips have the same memory bandwidth per core:
which tile wins depends on what the spec sheets leave out (cache sizes), so each chip is measured before its tiles
are chosen. Quantized matrices take QTILE on every M5 (d1-2B's Q8_0 up to 28% faster on the base M5). M1 to M4 run
MLX's own matmuls, and on M1, M2 and M4 an MXFP8 matrix is unpacked to full precision from OLDER_UNPACK_ROWS (by
profile: the m4 profile's rows on any Mac running the M4's paths, an M5 before macOS 26.2 among them). The switch
gemm=mlx turns every choice off (MLX's own matmuls throughout: the reference path, and the other arm of an A/B).

Fitted on the M5 Pro (20 GPU cores, MLX 0.32.4, profiles.FITTED_MLX; scripts/measure_kernels.py:
every variant of every shape of the README's builds at 16 to 4,096 rows). Full precision, 255 cells of 17 shapes: a
median 1.7% and a 90th percentile 8.9% longer than each cell's fastest variant, against 1.7% and 10.1% for tiles tuned
on each Mac and 8.9% and 25.5% for MLX's matmul alone. Quantized, 672 cells: QTILE a median 6% (MXFP8) to 46% (NVFP4)
faster than the best of MLX's quantized matmul, its rows as two halves, and the weights unpacked.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import mlx.core as mx

from .arch.base import Layered
from .kernels import gemm, qgemm
from .profiles import FITTED_MLX, Profile
from .quant import JoinedMatrix, QuantizedMatrix

RULE = "m5-7"  # what the choices below are: raised whenever they change, recorded with every measurement
FITTED_CORES = 20  # the M5 Pro's GPU cores: full-precision tiles on M5s of this many only

# Quantized (M5 Pro, 672 cells): QTILE was the fastest variant or within 3% of it in 82-100% of each affine format's
# cells, and in 72% of NVFP4's. MXFP8 (195 cells of the Ollama Clef-Flash and d3-mini builds, 16 to 16,384 rows):
# WIDE_QTILE was the fastest from 256 rows while the input (rows x K, fp16) stayed within WIDE_INPUT; past it, at
# 3,072 to 16,384 rows (62 cells), it fell 5-50% behind MLX's quantized matmul while QTILE was from 0.5% behind to
# 8% ahead. With both bounds the choice took at most 0.5% longer than MLX's and at the 90th percentile 2.9% from each
# cell's fastest variant (47.5% with WIDE_QTILE from 65 rows and no bound, which made 50-question requests 8-9%
# slower than tiles tuned on each Mac).
QTILE: qgemm.QTile = (64, 16, 1)  # 64 rows per threadgroup, 16 per SIMD group, K whole
QROWS = 16
WIDE_QTILE: qgemm.QTile = (128, 16, 1)
WIDE_ROWS = 256
# M5 Pro, d3-mini MXFP8 in fp16, eight fresh-process rounds: the wide tile from 192 rows for 2,560-wide inputs
# or outputs cut 203-token requests from 71.5 to 70.0 ms and 243-token ones from 79.3 to 76.8 ms (3.2%), with no
# regression in eight cells. Starting at 65 rows made 104-token requests take 12% longer (40.0 to 44.8 ms).
WIDE_2560_ROWS = 192
WIDE_INPUT = 24 << 20  # bytes of fp16 input
# MXFP8 unpacked to fp16 on every call and multiplied by MLX's matmul (13 shapes of the same builds, 1,024 to 16,384
# rows, the unpack timed), against the choice above: a median 7.9% longer at 1,024 rows, 0.3% faster at 2,560, 8.4%
# at 4,096 (12 of 13 shapes faster; a down projection 4.4% longer, which MLX's matmul multiplied and DOWN_TILE was
# 16% faster than there) and 12-13% from 8,192 (at worst 0.6% longer); against MLX's quantized matmul a median 12-14%
# faster from 4,096. The kernel trailed a matmul by weights unpacked beforehand by 8-19%.
UNPACK_ROWS = 4096
# M1 and M2 have no neural accelerators, and there MLX's MXFP8 quantized matmul took 34% (M1) and 28% (M2 Pro) longer on
# d1-2B's ~1,100-token prompts than the same model at fp16, and 44% and 16% longer than its Q8_0, which reads as many
# bytes. On an M1 (7 GPU cores, 8 GB, MLX 0.32.4; scripts/measure_kernels.py, d1-2B's six MXFP8 shapes), unpacking on
# every call and MLX's matmul summed 6.3% longer than MLX's quantized matmul at 128 rows, 5.0% less at 192, 10.7% at 256
# and 22-26% from 1,024, within 2-5% of weights unpacked beforehand. Whole requests (fresh processes, 2 rounds) cross
# later, 6% longer at 205 rows and 7% less time at 235, so the threshold sits between them; 500 words took 22-25% less
# time, 1,000 words 17-19%, the 500-word 3-question call 16%, answers identical to every digit. On a 48 GB M4 Pro
# (MLX 0.32.4, the same sweep on d1-2B's and d3-mini's Ollama MXFP8 builds) unpacking took 18-28% longer at 256 rows,
# was 2-6% faster from 1,024 and 10-14% at 4,096, so the M4 unpacks from 1,024. The M3 is unmeasured and keeps MLX's
# quantized matmul. Keyed by profile, the paths a Mac runs: an M5 before macOS 26.2 runs the m4 profile and takes the
# M4's rows (unmeasured there).
OLDER_UNPACK_ROWS = {"m1": 224, "m2": 224, "m4": 1024}
# Full precision (M5 Pro): the thin kernel was the fastest up to about 160 rows and lost to MLX's tile from 256 on
# 2560- to 18432-row matrices; a 64 x 32 tile won from there to 384; MLX's tile was within 2% (median) above.
THIN_ROWS = 160
SPLIT_WAVES = 16
MID_ROWS = 384
MID_TILE: gemm.Tile = (64, 32, 128, 2, 1, 0, 1)
GLU_MID: gemm.Tile = (64, 32, 128, 2, 1, 0, 0)
GLU_WIDE: gemm.Tile = (32, 32, 128, 1, 1, 0, 0)  # a fused gate/up's from MID_ROWS: a median 12% faster than MLX's
# A down projection's (K at least DOWN_RATIO times N) from MID_ROWS: a median 8% faster than MLX's tile there (14
# cells, at worst 1% longer), as tuning on each Mac chose for every such shape of the README builds. Past 4,096 rows
# the product alone was 2-17% faster up to 108 MiB of input and up to 30% longer from 120 MiB (4 shapes, to 16,384
# rows), but whole requests with MLX's tile from 112 MiB took 1.0-4.7% longer with 8 clients at once (Clef-Flash,
# d1-3B, d3-mini) and no different with 50 questions (d1-omni, d3-mini, Perplexity and a fourth model): it has no
# bound. The fused gate/up's GLU_WIDE stayed 7-18% faster to 16,384 rows on 3 of 4 shapes, at worst 4% longer
# (Clef-Flash's).
DOWN_TILE: gemm.Tile = (32, 64, 128, 1, 1, 0, 0)
DOWN_RATIO = 2.5


def thin(split: int) -> gemm.Tile:
    """MLX's thin kernel with one 32 x 32 SIMD-group block per threadgroup and K in `split` parts."""
    return (32, 32, 64, 1, 1, split, gemm.THIN)


# -- the shapes a backbone multiplies by -----------------------------------------------------------------------------


def _packed(value: Any) -> Iterable[QuantizedMatrix]:
    """The quantized matrices `value` is or holds (the parts of a JoinedMatrix of several formats)."""
    if isinstance(value, QuantizedMatrix):
        yield value
    elif isinstance(value, JoinedMatrix):
        for part in value.parts:
            yield from _packed(part)


def staged(w: QuantizedMatrix) -> bool:
    """Whether kernels/qgemm.py computes a matrix: affine-quantized or in an fp mode (MXFP8, MXFP4, NVFP4),
    computing in fp16 or bf16."""
    return (
        isinstance(w, QuantizedMatrix)
        and (w.mode == "affine" or w.mode in qgemm.FP_MODES)
        and w.dtype in (mx.float16, mx.bfloat16)
    )


def gemm_shapes(backbone: Layered) -> list[tuple[mx.Dtype, int, int]]:
    """The distinct (dtype, rows, columns) of a backbone's full-precision bf16 / fp16 weight matrices."""
    found: dict[tuple[str, int, int], tuple[mx.Dtype, int, int]] = {}
    for layer in backbone.layers:
        for key in backbone.linears:
            w = layer.get(key)
            if isinstance(w, mx.array) and w.ndim == 2 and w.dtype in (mx.bfloat16, mx.float16):
                n, k = w.shape
                found.setdefault((str(w.dtype), n, k), (w.dtype, n, k))
    return list(found.values())


def glu_shapes(backbone: Layered) -> list[tuple[mx.Dtype, int, int, str]]:
    """The distinct (dtype, rows, columns, activation) of a backbone's full-precision [gate; up] matrices."""
    gated = backbone.gated_mlp
    if gated is None:
        return []
    key, activation = gated
    found: dict[tuple[str, int, int], tuple[mx.Dtype, int, int, str]] = {}
    for layer in backbone.layers:
        w = layer.get(key)
        if isinstance(w, mx.array) and w.ndim == 2 and w.dtype in (mx.bfloat16, mx.float16):
            n, k = w.shape
            found.setdefault((str(w.dtype), n, k), (w.dtype, n, k, activation))
    return list(found.values())


# -- the rule ----------------------------------------------------------------------------------------------------------


def gemm_bands(n: int, k: int, cores: int, glu: str | None = None) -> gemm.Bands[gemm.Tile]:
    """The tiles of an [n, k] full-precision matrix on an M5 of `cores` GPU cores; with `glu`, its fused GLU's
    ([gate; up], n the both of them). MLX's matmul throughout on an M5 the tiles weren't measured on."""
    if cores != FITTED_CORES:
        return ((1, None),)
    columns = n // 2 if glu else n

    def choose(m: int) -> gemm.Tile | None:
        tile: gemm.Tile | None
        if m <= 64:
            return None
        if m <= THIN_ROWS:
            waves = math.ceil(m / 32) * math.ceil(columns / 32) / cores
            tile = thin(1 if glu or waves >= SPLIT_WAVES else 2)  # the fused GLU's: K whole, as measured
        elif m <= MID_ROWS:
            tile = GLU_MID if glu else MID_TILE
        else:
            tile = GLU_WIDE if glu else DOWN_TILE if k >= DOWN_RATIO * columns else None
        return tile if tile is not None and gemm.fits(tile, columns, k) else None

    return gemm.banded(choose, MID_ROWS + 1)


def qgemm_bands(w: QuantizedMatrix, glu: str | None = None) -> gemm.Bands[qgemm.QTile]:
    """The tiles of a packed matrix with kernels/qgemm.py's variants (with `glu`, its fused GLU's); below the first,
    MLX's quantized matmul."""
    n, k = w.shape
    fits = qgemm.glu_fits if glu else qgemm.fits
    wide_from = WIDE_2560_ROWS if w.dtype == mx.float16 and 2560 in (n, k) else WIDE_ROWS
    widest = WIDE_INPUT // (2 * k)  # the most rows WIDE_QTILE takes

    def choose(m: int) -> qgemm.QTile | None:
        if m < QROWS:
            return None
        tile = WIDE_QTILE if w.mode == "mxfp8" and wide_from <= m <= widest else QTILE
        return tile if fits(tile, n, k, w.bits, w.group_size, w.mode) else None

    return gemm.banded(choose, max(WIDE_ROWS, widest + 1))


# -- at load -------------------------------------------------------------------------------------------------------


def _mlx_reason(profile: Profile) -> str | None:
    """Why MLX's own matmuls run under `profile`, None when the choices apply: the chip's own M5 profile with its
    neural accelerators, the switch gemm=kernel, an MLX of the minor version they were fitted on, and ippon's tile
    library passing its check on this Mac."""
    if profile.device is None or not profile.device.neural_accelerators:
        return "no neural accelerators: M1 to M4"
    if profile.forced or profile.name != "m5":
        return "a forced profile"
    if profile.switches.gemm != "kernel":
        return "the switch gemm=mlx"
    if not profile.mlx_fitted:
        return f"MLX {profile.mlx}, kernel choices fitted on {FITTED_MLX.rpartition('.')[0]}.x"
    if not gemm.available():
        return f"ippon's tile kernels failed their check on this Mac: {gemm.failure()}"
    return None


def active(profile: Profile) -> bool:
    """Whether the choices apply (`_mlx_reason`)."""
    return _mlx_reason(profile) is None


def older(profile: Profile) -> bool:
    """Whether a profile unpacks its MXFP8 matrices (OLDER_UNPACK_ROWS): m1, m2 or m4 with the switch gemm=kernel."""
    return profile.device is not None and profile.name in OLDER_UNPACK_ROWS and profile.switches.gemm == "kernel"


def summary(profile: Profile) -> str:
    """How the matmuls' kernels are chosen under `profile`, in a line (`ippon doctor`, `Decider.details`)."""
    if older(profile):
        return f"MLX's own matmuls, MXFP8 unpacked to full precision from {OLDER_UNPACK_ROWS[profile.name]:,} rows"
    if (reason := _mlx_reason(profile)) is not None:
        return f"MLX's own matmuls ({reason})"
    device = profile.device
    assert device is not None
    full = "" if device.gpu_cores == FITTED_CORES else "; full precision: MLX's, these tiles unmeasured here"
    return f"ippon's for the M5's neural accelerators, {device.gpu_cores} GPU cores (rule {RULE}{full})"


def attach(backbone: Layered, profile: Profile) -> dict[str, gemm.Bands[gemm.Tile]]:
    """Set each quantized matrix's choices on `backbone` and return its full-precision matrices' tiles (by
    `gemm.key`), for `profile`; nothing where MLX's own matmuls run (`active`), apart from the MXFP8 matrices of
    `older` profiles (m1, m2 and m4), which are unpacked from OLDER_UNPACK_ROWS and multiplied by MLX's matmul."""
    if older(profile):
        rows = OLDER_UNPACK_ROWS[profile.name]
        for layer in backbone.layers:
            for key, value in list(layer.items()):  # whole matrices only, as on the M5
                if isinstance(value, QuantizedMatrix) and value.mode == "mxfp8" and staged(value):
                    layer[key] = replace(value, unpack_rows=rows)
        return {}
    if not active(profile):
        return {}
    assert profile.device is not None
    cores = profile.device.gpu_cores or 0  # IOKit not answering: an M5 not measured, MLX's full-precision matmuls
    tables = {gemm.key(d, n, k): gemm_bands(n, k, cores) for d, n, k in gemm_shapes(backbone)}
    tables |= {gemm.key(d, n, k, act): gemm_bands(n, k, cores, act) for d, n, k, act in glu_shapes(backbone)}
    gated = backbone.gated_mlp
    chosen: dict[str, QuantizedMatrix] = {}

    def choose(value: Any, glu: str | None, part: bool = False) -> Any:
        if isinstance(value, JoinedMatrix):  # its parts multiply one by one (`quant.mm`), never unpacked
            return JoinedMatrix(tuple(choose(p, None, part=True) for p in value.parts))
        if not isinstance(value, QuantizedMatrix) or not staged(value):
            return value
        name = f"{'part:' if part else ''}{glu}:{value.key}"
        if name not in chosen:
            tiles = qgemm_bands(value)
            unpack = UNPACK_ROWS if value.mode == "mxfp8" and not part else 0
            if unpack:  # from there a full-precision matrix of its shape, with that shape's choices
                n, k = value.shape
                tables[gemm.key(value.dtype, n, k, glu)] = gemm_bands(n, k, cores, glu)
            glu_tiles = (glu, qgemm_bands(value, glu)) if glu else None
            chosen[name] = replace(value, tiles=tiles, glu=glu_tiles, unpack_rows=unpack)
        found = chosen[name]
        return replace(value, tiles=found.tiles, glu=found.glu, unpack_rows=found.unpack_rows)

    for layer in backbone.layers:
        for key, value in list(layer.items()):
            if isinstance(value, QuantizedMatrix | JoinedMatrix):
                layer[key] = choose(value, gated[1] if gated is not None and key == gated[0] else None)
    return tables
