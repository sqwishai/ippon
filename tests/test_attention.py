"""Head-size-256 attention's memory paths (kernels/attention.py) on the GPU: which calls they take, MLX's attention in
evaluated query blocks against the whole call, and a both-ways pass whose scores can't be allocated read in blocks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest

from helpers import MLX_SWITCHES
from ippon.arch.ops import for_switches
from ippon.kernels import attention as attention_kernels

pytestmark = [pytest.mark.metal, pytest.mark.usefixtures("gpu")]

D = 256


def inputs(b: int, heads: int, groups: int, lq: int, lk: int, dtype: mx.Dtype) -> tuple[mx.array, mx.array, mx.array]:
    """q, k, v as attention takes them: [b, heads, n, 256] views of [b, n, heads, 256] projections."""
    q = mx.random.normal((b, lq, heads, D), key=mx.random.key(lq))
    k = mx.random.normal((b, lk, heads // groups, D), key=mx.random.key(lk + 1))
    v = mx.random.normal((b, lk, heads // groups, D), key=mx.random.key(lk + 2))
    q, k, v = (x.astype(dtype).transpose(0, 2, 1, 3) for x in (q, k, v))
    return q, k, v


def tree(lq: int, lk: int) -> mx.array:  # a branch: every cached key, then its own rows' lower triangle
    return mx.concatenate([mx.ones((lq, lk - lq), dtype=mx.bool_), mx.tril(mx.ones((lq, lq), dtype=mx.bool_))], 1)


def test_unsupported_shapes_fall_back_without_inspecting_missing_axes() -> None:
    q, k, v = inputs(1, 4, 2, 40, 40, mx.float16)
    for key in (mx.array(1, dtype=mx.float16), k[0], k[:, :0], k[:, :, :0]):
        assert not attention_kernels.applies(q, key, v, None)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("shape", [(1, 8, 4, 257, 513), (2, 4, 2, 333, 517), (1, 4, 1, 4097, 4103)])
def test_bounded_queries_preserve_full_key_attention(dtype: mx.Dtype, shape: tuple[int, int, int, int, int]) -> None:
    q, k, v = inputs(*shape, dtype)
    q, k, v = (mx.stack([x, -x], axis=-1).reshape(*x.shape[:-1], 2 * D)[..., ::2] for x in (q, k, v))
    want = mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5)
    got = attention_kernels.bounded(q, k, v, D**-0.5)
    assert mx.allclose(got, want, atol=0.002 if dtype == mx.float16 else 0.02, rtol=0)


def test_bounded_switch_leaves_small_and_masked_attention_with_mlx() -> None:
    """Below its threshold, or with a mask (which bounded can't take), the bounded switch is MLX's attention exactly."""
    q, k, v = inputs(1, 4, 2, 257, 513, mx.float16)
    ops = for_switches(replace(MLX_SWITCHES, attention="bounded"))
    want = mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5)
    assert mx.array_equal(ops.attention(q, k, v, D**-0.5, None), want)
    threshold_reached = replace(ops, attention_min_scores=1)
    assert mx.allclose(threshold_reached.attention(q, k, v, D**-0.5, None), want, atol=0.002, rtol=0)
    for mask in ("causal", tree(257, 513)):
        got = threshold_reached.attention(q, k, v, D**-0.5, mask)
        want = mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5, mask=mask)
        assert mx.array_equal(got, want)


def test_bounded_mode_runs_eager_layers_even_when_compilation_is_requested(tmp_path: Path) -> None:
    from ippon.arch import qwen35
    from qwen35_numpy import tiny_config, tiny_weights

    config = tiny_config(["full_attention", "full_attention"]) | {"head_dim": D}
    weights = {"model." + k: mx.array(v) for k, v in tiny_weights(config).items()}
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    model = qwen35.load(str(tmp_path), config, causal=False, dtype=mx.float16)
    ids = [i % 64 for i in range(257)]
    model.ops = for_switches(replace(MLX_SWITCHES, compile="layers"))
    want = model.prefill(ids, model.new_cache(len(ids)), list(range(len(ids))))
    mx.eval(want)
    model.ops = replace(
        for_switches(replace(MLX_SWITCHES, attention="bounded", compile="layers")), attention_min_scores=1
    )
    got = model.prefill(ids, model.new_cache(len(ids)), list(range(len(ids))))
    assert mx.allclose(got, want, atol=0.002, rtol=0.002)


def test_default_mode_reads_a_both_ways_pass_in_blocks_when_its_scores_cannot_be_allocated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ippon.arch import qwen35
    from qwen35_numpy import tiny_config, tiny_weights

    config = tiny_config(["full_attention", "full_attention"]) | {"head_dim": D}
    weights = {"model." + k: mx.array(v) for k, v in tiny_weights(config).items()}
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    model = qwen35.load(str(tmp_path), config, causal=False, dtype=mx.float16)
    model.ops = for_switches(replace(MLX_SWITCHES, compile="layers"))
    ids = [i % 64 for i in range(257)]
    want = model.prefill(ids, model.new_cache(len(ids)), list(range(len(ids))))
    mx.eval(want)
    original, calls = attention_kernels.bounded, 0

    def bounded(q: mx.array, k: mx.array, v: mx.array, scale: float) -> mx.array:
        nonlocal calls
        calls += 1
        return original(q, k, v, scale)

    monkeypatch.setattr(attention_kernels, "bounded", bounded)
    monkeypatch.setattr(attention_kernels, "score_limit", lambda: 1)  # as if the GPU had no room for the scores
    got = model.prefill(ids, model.new_cache(len(ids)), list(range(len(ids))))
    assert calls == 2  # both layers, eagerly: a compiled trace cannot evaluate block by block
    assert mx.allclose(got, want, atol=0.002, rtol=0.002)


def test_bounded_queries_release_score_buffers_between_blocks() -> None:
    q, k, v = inputs(1, 16, 4, 4096, 4096, mx.float16)
    mx.eval(q, k, v)
    mx.clear_cache()
    before = mx.get_active_memory()
    mx.reset_peak_memory()
    want = mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5)
    mx.eval(want)
    full_peak = mx.get_peak_memory() - before
    mx.clear_cache()
    before = mx.get_active_memory()
    mx.reset_peak_memory()
    got = attention_kernels.bounded(q, k, v, D**-0.5)
    mx.eval(got)
    bounded_peak = mx.get_peak_memory() - before
    assert mx.array_equal(got, want)
    assert bounded_peak < full_peak // 2, (bounded_peak, full_peak)
