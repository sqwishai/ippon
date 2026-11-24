"""The neural accelerators' matmul with chosen tiles: band lookup, when Ops takes the tiles, and on the GPU every tile
against MLX's own matmul."""

from __future__ import annotations

import math
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from helpers import MLX_SWITCHES, assert_mlx_close, require_gemm
from ippon.arch.ops import Ops, for_switches
from ippon.kernels import gemm
from ippon.quant import mm

TILE = (64, 64, 128, 2, 2, 0, 1)
WIDE = (64, 128, 256, 2, 4, 2, 0)


def test_bands_pick_the_tile_from_each_row_count() -> None:
    """Chosen bands: each serves from its row count up, by dtype and shape, and a fused GLU's
    bands are its own, apart from the plain matmul's of the same matrix."""
    table = {"bf16:256x512": ((1, None), (12, TILE), (23, WIDE)), "silu:bf16:256x512": ((1, TILE),)}
    picks = [gemm.tile_for(table, mx.bfloat16, 256, 512, m) for m in (1, 11, 12, 22, 23, 9000)]
    assert picks == [None, None, TILE, TILE, WIDE, WIDE]
    assert gemm.tile_for(table, mx.float16, 256, 512, 100) is None  # another dtype: none chosen
    assert gemm.tile_for(table, mx.bfloat16, 512, 512, 100) is None  # another shape
    assert gemm.tile_for(table, mx.bfloat16, 256, 512, 5, "silu") == TILE  # its own band; the plain one's is MLX's
    assert gemm.tile_for(table, mx.bfloat16, 256, 512, 5, "gelu") is None


def test_tiles_fit_only_whole_column_and_k_steps() -> None:
    assert gemm.fits(TILE, 128, 2048)
    assert not gemm.fits(TILE, 96, 2048)  # 96 columns: not whole 64-column tiles
    assert not gemm.fits(WIDE, 128, 2048 + 128)  # K steps of 256
    thin = (64, 32, 64, 2, 1, 2, gemm.THIN)
    assert gemm.fits(thin, 96, 2048 + 64)
    assert not gemm.fits(thin, 80, 2048)  # 80 columns: not whole 32-column tiles
    assert not gemm.fits(thin, 96, 2048 + 32)  # K chunks of 64


def test_ops_on_the_cpu_compute_as_mlx_does_whatever_the_tiles() -> None:
    """Chosen tiles run on the GPU only: on the CPU, with tiles for the very shape, Ops' matmul is MLX's and its GLU
    matmul the GLU of that."""
    ops = Ops(gemm_tiles={"bf16:8x16": ((1, TILE),), "silu:bf16:8x16": ((1, TILE),)})
    x, w = mx.random.normal((2, 3, 16)).astype(mx.bfloat16), mx.random.normal((8, 16)).astype(mx.bfloat16)
    assert mx.array_equal(ops.mm(x, w), mm(x, w))
    assert mx.array_equal(ops.glu_mm(x, w, "silu"), ops.glu(mm(x, w), "silu"))


# -- on the GPU ---------------------------------------------------------------------------------------------------


@pytest.mark.metal
def test_ops_with_a_chosen_tile_match_mlx(gpu: None) -> None:
    require_gemm()
    ops = for_switches(replace(MLX_SWITCHES, gemm="kernel"), {"bf16:256x512": ((1, TILE),)})
    x = mx.random.normal((2, 37, 512)).astype(mx.bfloat16)
    w = (mx.random.normal((256, 512)) * 0.05).astype(mx.bfloat16)
    got, want = ops.mm(x, w), x @ w.T
    assert got.shape == (2, 37, 256)
    assert_mlx_close(got, want, atol=2e-2)


@pytest.mark.metal
@pytest.mark.parametrize("activation", ["silu", "gelu"])
@pytest.mark.parametrize(("m", "i", "k"), [(1, 128, 512), (93, 256, 1024), (300, 384, 2048)])
def test_the_fused_glu_matmul_is_the_glu_of_the_matmul(gpu: None, activation: str, m: int, i: int, k: int) -> None:
    """In fp32 from the accumulators, rounded once: closer to exact than MLX's matmul rounded to bf16, then the GLU."""
    require_gemm()
    x = (mx.random.normal((m, k), key=mx.random.key(m)) * 0.5).astype(mx.bfloat16)
    w = (mx.random.normal((2 * i, k), key=mx.random.key(i)) * 0.05).astype(mx.bfloat16)
    y = np.array(x.astype(mx.float32)).astype(np.float64) @ np.array(w.astype(mx.float32)).astype(np.float64).T
    g, u = y[:, :i], y[:, i:]
    act = g / (1 + np.exp(-g)) if activation == "silu" else 0.5 * g * (1 + np.vectorize(math.erf)(g / math.sqrt(2)))
    exact = act * u
    tiles = gemm.candidates()[m % 5 :: 11] + list(gemm.THIN_TILES)  # every thin tile: its own GLU form
    for tile in tiles + gemm.simd_candidates(glu=True)[m % 7 :: 13]:  # M1 to M4's, which an M5 runs too
        if gemm.fits(tile, i, k):
            got = np.array(gemm.glu_gemm(x, w, tile, activation).astype(mx.float32))
            np.testing.assert_allclose(got, exact, atol=1e-2 * np.abs(exact).max(), rtol=0, err_msg=str(tile))


@pytest.mark.metal
@pytest.mark.parametrize(
    ("kind", "m", "n", "k"),
    [("nax", m, n, k) for m, n, k in ((1, 256, 512), (93, 512, 1024), (300, 384, 2048))]
    + [("thin", m, 256, k) for m in (1, 32, 33, 200) for k in (512, 576)]
    + [("simd", m, n, k) for m, n, k in ((1, 256, 512), (93, 512, 1024), (300, 384, 2048))],
)
def test_matmul_tiles_compute_mlxs_product(gpu: None, kind: str, m: int, n: int, k: int) -> None:
    """Every thin tile and a spread of normal tiles (the neural accelerators', and M1 to M4's SIMD-group ones),
    including partial rows and uneven K splits."""
    require_gemm()
    tiles = {
        "thin": list(gemm.THIN_TILES),
        "nax": gemm.candidates()[m % 5 :: 9],
        "simd": gemm.simd_candidates()[m % 5 :: 7],
    }
    dtypes = {"thin": (mx.bfloat16, mx.float16), "nax": (mx.bfloat16,), "simd": (mx.bfloat16, mx.float16)}
    for dtype in dtypes[kind]:
        x = (mx.random.normal((m, k), key=mx.random.key(m)) * 0.5).astype(dtype)
        w = (mx.random.normal((n, k), key=mx.random.key(k if kind == "thin" else n)) * 0.05).astype(dtype)
        expected = np.array((x @ w.T).astype(mx.float32))
        for tile in tiles[kind]:
            if kind == "thin":
                assert gemm.fits(tile, n, k)
            if gemm.fits(tile, n, k):
                got = np.array(gemm.gemm(x, w, tile).astype(mx.float32))
                np.testing.assert_allclose(got, expected, atol=2e-2, rtol=2e-2, err_msg=f"{tile} {dtype}")
