from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from helpers import assert_mlx_close, require_gemm
from ippon.kernels import qgemm
from ippon.quant import QuantizedMatrix, mm


def packed(n: int, k: int, bits: int, group_size: int, dtype: mx.Dtype = mx.float16) -> QuantizedMatrix:
    w = (mx.random.normal((n, k), key=mx.random.key(n + k + bits)) * 0.05).astype(dtype)
    q, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    return QuantizedMatrix(q, scales, biases, bits, group_size)


def test_on_the_cpu_a_matrix_with_chosen_tiles_computes_as_mlx() -> None:
    """Tiles chosen for the GPU never send the CPU to a Metal kernel: it takes MLX's quantized product."""
    w = packed(128, 256, 8, 64)
    tuned = replace(w, tiles=((1, None), (65, (64, 16, 2)), (257, (128, 32, 1))))
    x = mx.random.normal((300, 256)).astype(mx.float16)  # the tests' default device is the CPU: MLX's product
    want = mx.quantized_matmul(x, w.w, w.scales, w.biases, transpose=True, group_size=64, bits=8)
    assert mx.array_equal(mm(x, tuned), want)


# -- on the GPU ---------------------------------------------------------------------------------------------------


@pytest.mark.metal
def test_chosen_bands_run_the_kernel_inside_mm(gpu: None) -> None:
    require_gemm()
    w = packed(512, 1024, 8, 64)
    x = (mx.random.normal((3, 70, 1024)) * 0.5).astype(mx.float16)  # leading dims flatten to 210 rows
    got = mm(x, replace(w, tiles=((1, None), (129, (64, 16, 2)))))
    ref = mx.quantized_matmul(x, w.w, w.scales, w.biases, transpose=True, group_size=64, bits=8)
    assert_mlx_close(got, ref, atol=2e-2)


# Preserve the affine row/group/dtype grid and floating formats' whole/partial rows.
PRODUCTS: list[tuple[str, int, int, int, int, str | None]] = [
    ("affine", bits, group, m, 256, None)
    for bits, group in ((8, 64), (4, 64), (8, 32), (4, 32))
    for m in (1, 16, 64, 100, 333)
]
GLUS: list[tuple[str, int, int, int, int, str | None]] = [
    ("affine", bits, 64, m, 512, act) for bits in (8, 4) for m in (5, 100, 333) for act in ("silu", "gelu")
]
FLOATS: list[tuple[str, int, int, int, int, str | None]] = [
    (mode, bits, group, m, 256, act)
    for mode, bits, group in (("mxfp8", 8, 32), ("mxfp4", 4, 32), ("nvfp4", 4, 16))
    for m in (3, 333)
    for act in (None, "silu")
]


@pytest.mark.metal
@pytest.mark.parametrize(("mode", "bits", "group", "m", "n", "activation"), PRODUCTS + GLUS + FLOATS)
def test_every_packed_tile_matches_mlxs_product_and_glu(
    gpu: None, mode: str, bits: int, group: int, m: int, n: int, activation: str | None
) -> None:
    """All fitting tiles, with an explicit nonempty guard so deleting candidates cannot turn this into a pass."""
    from ippon.arch.ops import FUSED

    require_gemm()
    k = 1024
    dtypes = (mx.float16,) if mode == "affine" and activation else (mx.float16, mx.bfloat16)
    fits = qgemm.glu_fits if activation else qgemm.fits
    tiles = [t for t in qgemm.CANDIDATES if fits(t, n, k, bits, group, mode)]
    assert tiles, (mode, bits, group, m, activation)
    for dtype in dtypes:
        seed = n + k + bits if mode == "affine" else 1
        w = (mx.random.normal((n, k), key=mx.random.key(seed)) * 0.05).astype(dtype)
        values = mx.quantize(w, group_size=group, bits=bits, mode=mode)
        packed, scales = values[:2]
        biases = values[2] if mode == "affine" else None
        x = (mx.random.normal((m, k), key=mx.random.key(m)) * 0.5).astype(dtype)
        product = mx.quantized_matmul(x, packed, scales, biases, transpose=True, group_size=group, bits=bits, mode=mode)
        expected = FUSED.glu(product, activation) if activation else product
        for tile in tiles:
            got = (
                qgemm.glu_qmm(x, packed, scales, biases, bits, group, tile, activation, mode)
                if activation
                else qgemm.qmm(x, packed, scales, biases, bits, group, tile, mode)
            )
            assert_mlx_close(got, expected, atol=2e-2, rtol=2e-2, err_msg=f"{mode} {tile} {dtype} {activation}")


@pytest.mark.metal
@pytest.mark.parametrize("m", [3, 333])
def test_nvfp4_tiles_apply_scalar_and_joined_projection_scales(gpu: None, m: int) -> None:
    from ippon.arch.ops import FUSED

    require_gemm()
    n, k, bits, group, mode = 256, 1024, 4, 16, "nvfp4"
    w = (mx.random.normal((n, k), key=mx.random.key(1)) * 0.05).astype(mx.bfloat16)
    packed, scales = mx.quantize(w, group_size=group, bits=bits, mode=mode)[:2]
    x = (mx.random.normal((m, k), key=mx.random.key(m)) * 0.5).astype(mx.bfloat16)
    ref = mx.quantized_matmul(x, packed, scales, mode=mode)
    tile = next(t for t in qgemm.CANDIDATES if qgemm.fits(t, n, k, bits, group, mode))
    base = qgemm.qmm(x, packed, scales, None, bits, group, tile, mode)
    twice = qgemm.qmm(x, packed, scales, None, bits, group, tile, mode, mx.array(2.0))
    np.testing.assert_allclose(np.array(twice.astype(mx.float32)), 2 * np.array(base.astype(mx.float32)), rtol=1e-2)
    per_row = mx.linspace(0.25, 4.0, n)
    scaled = (ref.astype(mx.float32) * per_row).astype(ref.dtype)
    glu_tile = next(t for t in qgemm.CANDIDATES if qgemm.glu_fits(t, n, k, bits, group, mode))
    pairs = [
        (qgemm.qmm(x, packed, scales, None, bits, group, tile, mode, per_row), scaled),
        (
            qgemm.glu_qmm(x, packed, scales, None, bits, group, glu_tile, "silu", mode, per_row),
            FUSED.glu(scaled, "silu"),
        ),
    ]
    for got, expected in pairs:
        assert_mlx_close(got, expected, atol=0.1, rtol=2e-2)
