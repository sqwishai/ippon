"""Published floating-point encodings remain packed through linear, embedding and mixed projection paths."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from helpers import assert_mlx_equal
from ippon.errors import UnsupportedModel
from ippon.quant import JoinedMatrix, JoinedStacked, QuantizedMatrix, Stacked, arrays, concat, mm, rows
from ippon.weights import Weights

FORMATS = [("mxfp8", 8, 32), ("nvfp4", 4, 16), ("mxfp4", 4, 32)]


def matrix(mode: str, bits: int, group: int, scale: float | None = None) -> QuantizedMatrix:
    w = mx.array(np.random.default_rng(14).normal(size=(16, 64)).astype(np.float32))
    packed, scales = mx.quantize(w, group_size=group, bits=bits, mode=mode)[:2]
    tensors = {"x.weight": packed, "x.scales": scales}
    if scale is not None:
        tensors["x.weight.global_scale"] = mx.array(scale)
    loaded = Weights(
        tensors,
        quantization={"mode": mode, "bits": bits, "group_size": group, "global_scale_convention": "multiplier"},
    ).matrix("x.weight", mx.bfloat16)
    assert isinstance(loaded, QuantizedMatrix)
    assert loaded.w is packed
    assert loaded.scales is scales
    return loaded


@pytest.mark.parametrize(("mode", "bits", "group"), FORMATS)
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_format_matches_stock_mlx_and_preserves_encoded_scales(
    mode: str, bits: int, group: int, dtype: mx.Dtype
) -> None:
    q = matrix(mode, bits, group).astype(dtype)
    x = mx.array(np.random.default_rng(3).normal(size=(7, 64)).astype(np.float32)).astype(dtype)
    expected = mx.quantized_matmul(x, q.w, q.scales, mode=mode, bits=bits, group_size=group)
    assert_mlx_equal(mm(x, q), expected)
    assert q.scales.dtype == mx.uint8
    dequantized = mx.dequantize(q.w, q.scales, mode=mode, bits=bits, group_size=group, dtype=dtype)
    assert_mlx_equal(q.dense(), dequantized)
    ids = mx.array([[3, 1], [1, 5]])
    assert_mlx_equal(rows(q, ids), dequantized[ids])


@pytest.mark.parametrize("device", [mx.cpu, pytest.param(mx.gpu, marks=pytest.mark.metal)])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("scale_kind", ["multiplier", "amax", "per-row"])
def test_nvfp4_unscaled_overflow_and_rounding_do_not_change_finite_products(
    device: mx.Device, dtype: mx.Dtype, scale_kind: str
) -> None:
    """Each encoded weight is 6 * 448, scaled down to a practical magnitude. The old fp16 intermediate overflows;
    bf16 rounds too soon. NumPy's float64 sum of the known weight values supplies the independent answer."""
    with mx.stream(device):
        q = QuantizedMatrix(
            mx.full((2, 8), 0x77777777, dtype=mx.uint32),  # E2M1 code 7 is 6
            mx.full((2, 4), 0x7E, dtype=mx.uint8),  # E4M3 code 0x7e is 448
            None,
            4,
            16,
            mode="nvfp4",
            global_scale=mx.array(0.0001),
            compute_dtype=dtype,
        )
        if scale_kind == "amax":
            q = replace(
                q, global_scale=mx.array(float(np.float32(0.0001) * np.float32(2688))), global_scale_is_amax=True
            )
        elif scale_kind == "per-row":
            q = replace(q, global_scale=None, row_multiplier=mx.array([0.0001, 0.0002]))
        x = mx.array(np.random.default_rng(4).uniform(0.5, 1.5, (1, 7, 64)).astype(np.float32)).astype(dtype)
        product = np.array(x.astype(mx.float32)).astype(np.float64).sum(-1, keepdims=True) * 2688
        scale = q.multiplier()
        assert scale is not None
        multiplier = np.array(scale).astype(np.float64)
        expected = mx.array(np.broadcast_to(product * multiplier, (1, 7, 2)).astype(np.float32)).astype(dtype)
        raw = mx.quantized_matmul(x, q.w, q.scales, mode="nvfp4")
        early = (raw.astype(mx.float32) * scale).astype(dtype)
        got = mm(x, q)
        assert got.dtype == dtype
        assert bool(mx.all(mx.isfinite(got)))
        assert_mlx_equal(got, expected)
        if dtype == mx.float16:
            assert not bool(mx.all(mx.isfinite(raw)))
        else:
            assert not bool(mx.array_equal(early, expected))


@pytest.mark.metal
def test_nvfp4_expert_global_scales_match_the_native_metal_operator() -> None:
    with mx.stream(mx.gpu):
        a = matrix("nvfp4", 4, 16, 0.013)
        bank = Stacked.of([a, replace(a, global_scale=mx.array(0.072))])
        x, ids = mx.ones((4, 1, 64), mx.bfloat16), mx.array([1, 0, 1, 0])
        assert bank.global_scale is not None
        assert bank.scales is not None
        expected = mx.gather_qmm(
            x, bank.w, bank.scales, rhs_indices=ids, mode="nvfp4", global_scale=bank.global_scale * 2688
        )
        assert_mlx_equal(bank.gather(x, ids), expected)


def test_nvfp4_expert_global_scales_on_the_cpu_scale_each_experts_output() -> None:
    """MLX's gather_qmm takes no global scale on the CPU: each pair's output is scaled by its own expert's, which may
    be a checkpoint multiplier or MLX's amax, mixed in one bank."""
    a = matrix("nvfp4", 4, 16, 0.013)
    b = replace(a, global_scale=mx.array(0.072 * 2688), global_scale_is_amax=True)
    mixed = Stacked.of([a, b])
    assert mixed.global_scale_is_amax
    x, ids = mx.ones((2, 1, 64), mx.bfloat16), mx.array([0, 1])
    expected = mx.stack([mm(x[0], a), mm(x[1], b)])
    assert_mlx_equal(mixed.gather(x, ids), expected)


def test_nvfp4_cpu_experts_scale_before_an_unscaled_fp16_product_overflows() -> None:
    q = QuantizedMatrix(
        mx.full((2, 8), 0x77777777, dtype=mx.uint32),
        mx.full((2, 4), 0x7E, dtype=mx.uint8),
        None,
        4,
        16,
        mode="nvfp4",
        global_scale=mx.array(0.0001),
        compute_dtype=mx.float16,
    )
    bank = Stacked.of([q, replace(q, global_scale=mx.array(0.0002))])
    x, ids = mx.ones((3, 1, 64), mx.float16), mx.array([1, 0, 1])
    expected = mx.array([64 * 6 * 448 * scale for scale in (0.0002, 0.0001, 0.0002)]).astype(mx.float16)
    expected = mx.broadcast_to(expected[:, None, None], (3, 1, 2))
    got = bank.gather(x, ids)
    assert_mlx_equal(got, expected)


@pytest.mark.metal
def test_native_mlx_expert_scales_are_amax_not_checkpoint_multipliers() -> None:
    from ippon.quantization import inspect
    from ippon.safetensors import Tensor

    with mx.stream(mx.gpu):
        raw = mx.array(np.random.default_rng(5).normal(size=(2, 16, 64)).astype(np.float32))
        amax = mx.max(mx.abs(raw), axis=(-2, -1))
        pairs = [mx.quantize(raw[i], mode="nvfp4", global_scale=amax[i]) for i in range(2)]
        packed, scales = mx.stack([p[0] for p in pairs]), mx.stack([p[1] for p in pairs])
        config = {"mode": "nvfp4", "experts": {"global_scale": True}}
        headers = [
            Tensor("experts.weight", "U32", packed.shape),
            Tensor("experts.scales", "U8", scales.shape),
            Tensor("experts.global_scale", "F32", amax.shape),
        ]
        assert inspect(headers, config) == {"nvfp4": 1}
        q = Weights(
            {"experts.weight": packed, "experts.scales": scales, "experts.global_scale": amax},
            quantization=config,
        ).matrix("experts.weight", mx.bfloat16)
        assert isinstance(q, QuantizedMatrix)
        assert q.global_scale_is_amax
        assert q.w is packed
        assert q.scales is scales
        bank = Stacked.stored(q)
        x, ids = mx.ones((4, 1, 64), mx.bfloat16), mx.array([1, 0, 1, 0])
        expected = mx.gather_qmm(x, packed, scales, rhs_indices=ids, mode="nvfp4", global_scale=amax)
        assert_mlx_equal(bank.gather(x, ids), expected)
        with pytest.raises(UnsupportedModel, match="requires a scale tensor"):
            inspect(headers[:-1], config)
        without_flag = Weights(
            {"experts.weight": packed, "experts.scales": scales, "experts.global_scale": amax},
            quantization={"mode": "nvfp4"},
        ).matrix("experts.weight", mx.bfloat16)
        assert isinstance(without_flag, QuantizedMatrix)
        assert without_flag.global_scale_is_amax


def test_unsupported_encoding_mixture_within_one_bank_is_a_model_error() -> None:
    with pytest.raises(UnsupportedModel, match="one encoding"):
        Stacked.of([matrix("mxfp8", 8, 32), matrix("nvfp4", 4, 16)])


def test_mixed_projections_remain_packed_and_scales_stay_separate() -> None:
    a, b = matrix("mxfp8", 8, 32), matrix("nvfp4", 4, 16, 0.0123)
    dense = mx.ones((3, 64), mx.bfloat16)
    joined = concat([a, b, dense])
    assert isinstance(joined, JoinedMatrix)
    x = mx.ones((2, 64), mx.bfloat16)
    expected = mx.concatenate([mm(x, a), mm(x, b), x @ dense.T], axis=-1)
    assert_mlx_equal(mm(x, joined), expected)
    assert a.w in arrays([joined])


def test_one_format_joins_across_global_scales_with_a_multiplier_per_row() -> None:
    """Ollama's NVFP4 builds give each projection its own global scale: joined, they stay one packed matrix whose
    rows each keep their part's multiplier and values. All rows of a scaled product accumulate in fp32, including
    those whose multiplier is implicitly one."""
    b, plain = matrix("nvfp4", 4, 16, 0.0123), matrix("nvfp4", 4, 16)
    parts = (b, replace(b, global_scale=mx.array(2.0)), plain)
    joined = concat(parts)
    assert isinstance(joined, QuantizedMatrix)
    x = mx.array(np.random.default_rng(3).normal(size=(5, 64)).astype(np.float32)).astype(mx.bfloat16)
    dense = np.concatenate([np.array(p.dense(mx.float32)) for p in parts]).astype(np.float64)
    product = np.array(x.astype(mx.float32)).astype(np.float64) @ dense.T
    expected = mx.array(product.astype(np.float32)).astype(x.dtype)
    assert_mlx_equal(mm(x, joined), expected)
    np.testing.assert_array_equal(
        np.array(joined.dense(mx.float32)), np.concatenate([np.array(p.dense(mx.float32)) for p in parts])
    )
    again = concat([joined, plain])
    assert isinstance(again, QuantizedMatrix)
    np.testing.assert_array_equal(
        np.array(again.dense(mx.float32)),
        np.concatenate([np.array(p.dense(mx.float32)) for p in (*parts, plain)]),
    )
    with pytest.raises(TypeError, match="not embedding tables"):
        rows(joined, mx.array([0]))


def test_affine_projections_fuse_by_their_scales_and_different_bits_stay_packed() -> None:
    """Affine projections of one bits, group size and scale dtype fuse into one matrix, whatever their compute dtype
    field; a mixed-bit MLX build's fused q/k/v or gate/up stays its packed parts (it was dequantized to one
    full-precision matrix before)."""
    q, s, b = mx.quantize(mx.ones((8, 64), mx.float16), bits=8)
    first = QuantizedMatrix(q, s, b, 8, 64)
    assert isinstance(concat([first, first.astype(mx.float16)]), QuantizedMatrix)
    w = mx.array(np.random.default_rng(4).normal(size=(16, 64)).astype(np.float32))
    four, eight = (QuantizedMatrix(*mx.quantize(w, bits=bits), bits, 64) for bits in (4, 8))
    joined = concat([four, eight])
    assert isinstance(joined, JoinedMatrix)
    x = mx.ones((2, 64))
    expected = mx.concatenate([mm(x, four), mm(x, eight)], axis=-1)
    np.testing.assert_array_equal(np.array(mm(x, joined)), np.array(expected))


@pytest.mark.parametrize(("mode", "bits", "group"), FORMATS)
def test_same_format_fuses_and_expert_banks_keep_the_mode(mode: str, bits: int, group: int) -> None:
    a = matrix(mode, bits, group)
    joined = concat([a, a])
    assert isinstance(joined, QuantizedMatrix)
    assert_mlx_equal(joined.dense(), mx.concatenate([a.dense(), a.dense()]))
    x = mx.ones((3, 1, 64), mx.bfloat16)
    bank = Stacked.of([a, a]).astype(mx.bfloat16)
    ids = mx.array([1, 0, 1])
    assert bank.scales is not None
    expected = mx.gather_qmm(x, bank.w, bank.scales, rhs_indices=ids, mode=mode)
    assert_mlx_equal(bank.gather(x, ids), expected)
    other = Stacked.of([matrix("nvfp4", 4, 16, 0.01)] * 2)
    mixed = Stacked.join([bank, other])
    assert isinstance(mixed, JoinedStacked)
    assert_mlx_equal(mixed.gather(x, ids), mx.concatenate([bank.gather(x, ids), other.gather(x, ids)], axis=-1))


@pytest.mark.parametrize("fault", ["mode", "group", "bits", "biases", "scales", "global", "missing"])
def test_bad_format_metadata_is_rejected_before_matrix_execution(fault: str) -> None:
    q = matrix("mxfp8", 8, 32)
    tensors = {"x.weight": q.w, "x.scales": q.scales}
    config: dict[str, Any] = {"mode": "mxfp8", "bits": 8, "group_size": 32}
    if fault == "mode":
        config["mode"] = "unknown"
    elif fault == "group":
        config["group_size"] = 64
    elif fault == "bits":
        config["bits"] = 4
    elif fault == "biases":
        tensors["x.biases"] = mx.ones(q.scales.shape)
    elif fault == "scales":
        tensors["x.scales"] = q.scales.astype(mx.float16)
    elif fault == "global":
        tensors["x.global_scale"] = mx.array(2.0)
    else:
        del tensors["x.scales"]
    with pytest.raises(UnsupportedModel):
        Weights(tensors, quantization=config).matrix("x.weight")


def test_per_layer_metadata_overrides_the_global_format_and_dense_layers_stay_dense() -> None:
    """A layer's own entry overrides the global format, `false` keeps it dense and `true` reads the global one."""
    a, b = matrix("mxfp8", 8, 32), matrix("nvfp4", 4, 16)
    tensors = {
        "a.weight": a.w,
        "a.scales": a.scales,
        "b.weight": b.w,
        "b.scales": b.scales,
        "c.weight": mx.ones((4, 64), mx.bfloat16),
        "d.weight": a.w,
        "d.scales": a.scales,
    }
    w = Weights(
        tensors,
        quantization={
            "mode": "mxfp8",
            "bits": 8,
            "group_size": 32,
            "b": {"mode": "nvfp4", "bits": 4, "group_size": 16},
            "c": False,
            "d": True,
        },
    )
    assert isinstance(w.matrix("a.weight"), QuantizedMatrix)
    second = w.matrix("b.weight")
    assert isinstance(second, QuantizedMatrix)
    assert second.mode == "nvfp4"
    assert w.matrix("c.weight").dtype == mx.bfloat16
    fourth = w.matrix("d.weight")
    assert isinstance(fourth, QuantizedMatrix)
    assert fourth.mode == "mxfp8"


def test_foreign_byte_packing_cannot_be_mistaken_for_dense_weights() -> None:
    from ippon.quantization import inspect
    from ippon.safetensors import Tensor

    with pytest.raises(UnsupportedModel, match="unsupported packed layout"):
        Weights({"x.weight": mx.ones((8, 64), mx.uint8)}).matrix("x.weight", mx.bfloat16)
    with pytest.raises(UnsupportedModel, match="unsupported packed layout"):
        inspect([Tensor("x.weight", "U8", (8, 64))], None)


@pytest.mark.metal
@pytest.mark.parametrize(("mode", "bits", "group"), FORMATS)
@pytest.mark.parametrize("length", [128, 4096])
def test_real_metal_at_decision_input_lengths(mode: str, bits: int, group: int, length: int) -> None:
    with mx.stream(mx.gpu):
        q = matrix(mode, bits, group, 0.013 if mode == "nvfp4" else None)
        x = mx.ones((length, 64), mx.bfloat16)
        expected = mx.quantized_matmul(
            x.astype(mx.float32) if q.global_scale is not None else x, q.w, q.scales, mode=mode
        )
        if q.global_scale is not None:
            expected = (expected.astype(mx.float32) * q.global_scale).astype(mx.bfloat16)
        assert_mlx_equal(mm(x, q), expected)
