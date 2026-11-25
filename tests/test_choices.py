"""Kernel choices made at load from the chip and the matrix (choices.py): every chosen kernel fits its matrix, the
bands follow the rule fitted on the M5 Pro, and choices reach every matrix of a shape only on the chip's own M5
profile with the switch gemm=kernel."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import mlx.core as mx
import pytest

from helpers import assert_mlx_close, require_gemm
from ippon import choices, quant
from ippon.device import Device
from ippon.kernels import gemm, qgemm
from ippon.profiles import FITTED_MLX, Profile, Switches
from ippon.quant import JoinedMatrix, QuantizedMatrix, quantize

M5 = Device("Apple M5 Pro", "applegpu_g17s", 48 * 2**30, 36 * 2**30, (27, 0), True, gpu_cores=20)
M4 = Device("Apple M4 Pro", "applegpu_g16s", 48 * 2**30, 36 * 2**30, (27, 0), True, gpu_cores=20)
M2 = Device("Apple M2 Pro", "applegpu_g14s", 16 * 2**30, 12 * 2**30, (27, 0), True, gpu_cores=16)
# full-precision [n, k] shapes of the README builds (the d1, d3, Clef-Flash and Perplexity ones)
SHAPES = [(1024, 1024), (1024, 3584), (2048, 8192), (2560, 9216), (4096, 12288), (10240, 2560), (12288, 4096),
          (18432, 2560), (21504, 2048), (24576, 4096), (5248, 1024), (8224, 1024)]  # fmt: skip


def matrix(rows: int, columns: int, bits: int = 8, mode: str = "affine") -> QuantizedMatrix:
    w = mx.random.normal((rows, columns)).astype(mx.float16)
    if mode == "affine":
        q = quantize(w, bits)
        assert isinstance(q, QuantizedMatrix)
        return q
    packed, scales, *_ = mx.quantize(w, group_size=32, bits=8, mode=mode)
    return QuantizedMatrix(packed, scales, None, 8, 32, mode=mode, compute_dtype=mx.float16)


def tiles(bands: gemm.Bands[Any], top: int = 1024) -> dict[int, Any]:
    """The tile at each row count from 1 to `top`."""
    return {m: gemm.band(bands, m) for m in range(1, top + 1)}


@pytest.mark.parametrize(("n", "k"), SHAPES)
def test_every_chosen_kernel_fits_its_matrix(n: int, k: int) -> None:
    plain, fused = tiles(choices.gemm_bands(n, k, 20)), tiles(choices.gemm_bands(2 * n, k, 20, "silu"))
    for m, tile in plain.items():
        assert tile is None or gemm.fits(tile, n, k), (m, tile)
    assert all(t is None or gemm.fits(t, n, k) for t in fused.values())


def test_full_precision_tiles_only_on_the_m5_they_were_measured_on() -> None:
    """The M5 Pro's tiles made the base M5's 4B models 10-24% slower: an M5 of any other core count (an unread one
    too) runs MLX's matmul for every full-precision matrix. On the M5 Pro the thin kernel splits K where its
    threadgroups fill the 20 cores fewer than SPLIT_WAVES times."""
    for cores in (10, 40, 0):
        assert choices.gemm_bands(4096, 4096, cores) == ((1, None),)
        assert choices.gemm_bands(2 * 4096, 4096, cores, "silu") == ((1, None),)
    narrow, wide = tiles(choices.gemm_bands(1024, 4096, 20)), tiles(choices.gemm_bands(4096, 4096, 20))
    assert narrow[160] == choices.thin(2)  # 5 x 32 threadgroups over 20 cores: 8 waves
    assert wide[160] == choices.thin(1)  # 5 x 128: 32 waves


@pytest.mark.parametrize(
    ("n", "k", "mode"),
    [(4096, 4096, "affine"), (4096, 4096, "mxfp8"), (96, 4096, "affine"), (2560, 512, "mxfp8"), (512, 2560, "mxfp8")],
)
def test_every_chosen_quantized_kernel_fits_its_matrix(n: int, k: int, mode: str) -> None:
    """96 rows are not whole 64-row tiles: such a matrix gets no tile at any row count."""
    w = matrix(n, k, mode=mode)
    for m, tile in tiles(choices.qgemm_bands(w), 4096).items():
        assert tile is None or qgemm.fits(tile, n, k, 8, w.group_size, mode), (m, tile)


@pytest.mark.parametrize(("n", "k"), [(2560, 512), (512, 2560)])
def test_short_mxfp8_shapes_keep_the_wide_tile_and_its_memory_cap(n: int, k: int) -> None:
    wide_limit = choices.WIDE_INPUT // (2 * k)
    w = matrix(n, k, mode="mxfp8")
    selected = tiles(choices.qgemm_bands(w), wide_limit + 1)
    assert selected[104] == selected[191] == selected[wide_limit + 1] == choices.QTILE
    assert all(selected[m] == choices.WIDE_QTILE for m in (192, 203, 255, wide_limit))
    assert tiles(choices.qgemm_bands(matrix(n, k)))[203] == choices.QTILE
    assert tiles(choices.qgemm_bands(replace(w, compute_dtype=mx.bfloat16)))[203] == choices.QTILE


@pytest.mark.metal
@pytest.mark.parametrize(("n", "k"), [(2560, 512), (512, 2560)])
def test_short_mxfp8_choice_preserves_mlxs_product(gpu: None, n: int, k: int) -> None:
    require_gemm()
    w = matrix(n, k, mode="mxfp8")
    selected = replace(w, tiles=choices.qgemm_bands(w))
    assert selected.tile(203) == choices.WIDE_QTILE
    x = (mx.random.normal((7, 29, k)) * 0.05).astype(mx.float16)
    expected = mx.quantized_matmul(x, w.w, w.scales, transpose=True, group_size=32, bits=8, mode="mxfp8")
    assert_mlx_close(quant.mm(x, selected), expected, atol=2e-2, rtol=2e-2)


class Backbone:
    """Two layers: a full-precision matrix, a quantized one and its joined pair, and a quantized [gate; up]."""

    linears = ("a", "b", "gate_up")
    gated_mlp = ("gate_up", "silu")
    dtype = mx.float16

    def __init__(self) -> None:
        w, other, gate_up = (matrix(n, 256) for n in (128, 64, 256))
        self.layers: list[dict[str, Any]] = [
            {"a": mx.zeros((128, 256), mx.float16), "b": w, "gate_up": gate_up, "norm": mx.ones((256,))},
            {"a": mx.zeros((128, 256), mx.float16), "b": JoinedMatrix((w, other)), "gate_up": gate_up},
        ]


def profile(device: Device, name: str = "m5", gemm: Literal["mlx", "kernel"] = "kernel") -> Profile:
    return Profile(name, f"m{device.generation}", device, Switches(gemm=gemm))


def test_choices_reach_every_matrix_of_a_shape_on_the_m5_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gemm, "available", lambda: True)
    backbone = Backbone()
    table = choices.attach(backbone, profile(M5))
    assert gemm.tile_for(table, mx.float16, 128, 256, 100) is not None  # the full-precision matrix's shape
    first, second = backbone.layers
    for q in (first["b"], second["b"].parts[0], second["b"].parts[1]):  # alone and joined
        assert q.tile(200) is not None
    assert first["gate_up"].glu_tile(200, "silu") is not None  # its fused GLU's own steps
    for unchosen in (profile(M5, gemm="mlx"), profile(M5, "m4"), profile(M4, "m4")):  # MLX's own matmuls
        plain = Backbone()
        assert choices.attach(plain, unchosen) == {}
        assert plain.layers[0]["b"].tile(200) is None
        assert plain.layers[0]["gate_up"].glu_tile(200, "silu") is None
    assert "20 GPU cores" in choices.summary(profile(M5))
    assert "10 GPU cores" in choices.summary(profile(replace(M5, gpu_cores=10)))
    assert choices.summary(profile(M5, gemm="mlx")) == "MLX's own matmuls (the switch gemm=mlx)"
    assert "M1 to M4" in choices.summary(profile(M4, "m3"))
    # an MLX patch release keeps the choices, another minor version (outside pyproject's range) runs MLX's own
    for patch in (FITTED_MLX, "0.32.9"):
        assert choices.active(replace(profile(M5), mlx=patch)), patch
    newer = replace(profile(M5), mlx="0.33.0")
    assert choices.attach(Backbone(), newer) == {}
    assert choices.summary(newer) == "MLX's own matmuls (MLX 0.33.0, kernel choices fitted on 0.32.x)"


@pytest.mark.metal
def test_m1_m2_and_m4_unpack_whole_mxfp8_matrices_from_their_rows(gpu: None) -> None:
    """On m1, m2 and m4 with gemm=kernel only a whole MXFP8 matrix is unpacked, from the profile's OLDER_UNPACK_ROWS,
    and on m4 a Q8_0 one too, each product MLX's quantized matmul's within fp16 rounding; gemm=mlx and m3 leave every
    matrix to MLX's quantized matmul."""
    w, q8 = matrix(128, 256, mode="mxfp8"), matrix(128, 256)
    backbone = Backbone()
    backbone.layers = [{"b": w, "q": q8, "joined": JoinedMatrix((w, q8))}]
    assert choices.attach(backbone, profile(M2, "m2")) == {}
    layer = backbone.layers[0]
    rows = choices.OLDER_UNPACK_ROWS["m2"]
    assert layer["b"].unpack_rows == rows
    assert [layer["q"].unpack_rows, *(p.unpack_rows for p in layer["joined"].parts)] == [0, 0, 0]
    assert "MXFP8 unpacked" in choices.summary(profile(M2, "m2"))
    x = mx.random.normal((rows, 256)).astype(mx.float16)
    assert quant.unpacked(x[:-1], layer["b"]) is layer["b"]
    expected = mx.quantized_matmul(x, w.w, w.scales, transpose=True, group_size=32, bits=8, mode="mxfp8")
    unpacked = quant.unpacked(x, layer["b"])
    assert isinstance(unpacked, mx.array)
    assert_mlx_close(x @ unpacked.T, expected, atol=2e-2, rtol=2e-2)
    packed, scales, biases = mx.quantize(mx.random.normal((128, 256)).astype(mx.float16), group_size=32, bits=8)
    q8_0 = QuantizedMatrix(packed, scales, biases, 8, 32, compute_dtype=mx.float16)  # GGUF's Q8_0 layout
    for chosen, expected_rows, q8_0_rows in (
        (profile(M2, "m2", gemm="mlx"), 0, 0),
        (profile(M2, "m2"), rows, 0),
        (profile(M4, "m3"), 0, 0),
        (profile(M4, "m4"), 1024, 1024),
    ):
        plain = Backbone()
        plain.layers = [{"b": w, "q8_0": q8_0, "q": q8}]
        choices.attach(plain, chosen)
        assert [plain.layers[0][k].unpack_rows for k in ("b", "q8_0", "q")] == [expected_rows, q8_0_rows, 0]
    assert "MXFP8 and Q8_0 unpacked" in choices.summary(profile(M4, "m4"))
    x = mx.random.normal((1024, 256)).astype(mx.float16)
    expected = mx.quantized_matmul(x, packed, scales, biases, transpose=True, group_size=32, bits=8)
    unpacked = quant.unpacked(x, replace(q8_0, unpack_rows=1024))
    assert isinstance(unpacked, mx.array)
    assert_mlx_close(x @ unpacked.T, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.metal
def test_mxfp8_matrices_multiply_unpacked_from_unpack_rows(gpu: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """From UNPACK_ROWS an MXFP8 matrix, and only a whole one, is a full-precision matrix of its shape: unpacked on
    the GPU and multiplied with that shape's choices, its [gate; up] with the fused GLU's."""
    monkeypatch.setattr(gemm, "available", lambda: True)
    w, gate_up, q8 = matrix(128, 256, mode="mxfp8"), matrix(256, 256, mode="mxfp8"), matrix(128, 256)
    backbone = Backbone()
    backbone.layers = [{"b": w, "gate_up": gate_up}, {"b": JoinedMatrix((w, q8)), "gate_up": q8}]
    table = choices.attach(backbone, profile(M5))
    first, second = backbone.layers
    assert first["b"].unpack_rows == first["gate_up"].unpack_rows == choices.UNPACK_ROWS
    assert [p.unpack_rows for p in second["b"].parts] + [second["gate_up"].unpack_rows] == [0, 0, 0]
    assert set(table) == {gemm.key(mx.float16, 128, 256), gemm.key(mx.float16, 256, 256, "silu")}
    x = mx.random.normal((choices.UNPACK_ROWS, 256)).astype(mx.float16)
    assert quant.unpacked(x[:-1], first["b"]) is first["b"]
    unpacked = quant.unpacked(x, first["b"])
    assert isinstance(unpacked, mx.array)
    assert mx.array_equal(unpacked, w.dense(mx.float16))
    mx.set_default_device(mx.cpu)  # the fixture restores the default after
    assert quant.unpacked(x, first["b"]) is first["b"]
