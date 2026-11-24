"""The MLX Qwen3.5 against an independent numpy one on tiny random checkpoints, and what its recurrent layers
add to the runtime (branches on the GPU); the paths every family shares (branches leaving the
prefix's state as it was among them) are test_backbone_contract.py's."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

import backbone_fixtures as fixtures
from backbone_fixtures import QWEN_ATTENTION as ENDS_IN_ATTENTION
from backbone_fixtures import QWEN_DELTANET as ENDS_IN_DELTANET
from backbone_fixtures import qwen35_model as model
from helpers import MLX_SWITCHES
from helpers import tokens as tokens
from ippon.arch import qwen35
from ippon.arch.common import Recurrence
from ippon.arch.ops import for_switches
from ippon.errors import UnsupportedModel
from ippon.plan import Prompt, Readout, plan
from ippon.runtime import run
from qwen35_numpy import forward, tiny_config, tiny_moe_config, tiny_weights


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.metal)])
def test_handing_layers_over_early_changes_nothing(tmp_path: Path, device: str, request: pytest.FixtureRequest) -> None:
    """pipeline=layers starts evaluating the pass every few layers while the rest is built: the same numbers."""
    from dataclasses import replace

    m, _, _ = model(tmp_path, ENDS_IN_ATTENTION + ENDS_IN_ATTENTION)  # 10 layers: handed over twice
    ids = tokens(21, 6)
    if device == "gpu":
        request.getfixturevalue("gpu")
    plain = np.array(m.prefill(ids, m.new_cache(21), list(range(21))))
    m.ops = replace(m.ops, pipeline=4)
    handed = np.array(m.prefill(ids, m.new_cache(21), list(range(21))))
    np.testing.assert_array_equal(handed, plain)


@pytest.mark.metal
@pytest.mark.parametrize("split", [0, 8])
def test_rows_read_on_the_gpu_match_every_row(tmp_path: Path, split: int, request: pytest.FixtureRequest) -> None:
    """On the GPU, a suffix (MLX's causal attention, its queries aligned with the last keys), a run in the middle
    and scattered rows (explicit masks) against the same rows of reading every row."""
    m, _, _ = model(tmp_path, ENDS_IN_ATTENTION)
    ids = tokens(23, 4)
    request.getfixturevalue("gpu")

    def read(rows: list[int]) -> np.ndarray:
        cache = m.new_cache(len(ids))
        if split:
            m.prefill(ids[:split], cache, [])
        return np.array(m.prefill(ids[split:], cache, [r - split for r in rows]))

    every = read(list(range(split, 23)))
    for rows in (list(range(12, 23)), list(range(9, 14)), [22], [17, 9, 21]):
        want = every[[r - split for r in rows]]
        np.testing.assert_allclose(read(rows), want, atol=1e-3 * float(np.abs(want).max()), rtol=0)


def test_bf16_stays_close_to_fp32(tmp_path: Path) -> None:
    m32, config, _ = model(tmp_path, ENDS_IN_ATTENTION)
    m16 = qwen35.load(str(tmp_path), config, dtype=mx.bfloat16)
    ids = tokens(15, 8)
    a = np.array(m32.prefill(ids, m32.new_cache(15), [14]))
    b = np.array(m16.prefill(ids, m16.new_cache(15), [14]).astype(mx.float32))
    assert np.abs(a - b).max() < 0.1 * np.abs(a).max()


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.metal)])
@pytest.mark.parametrize("marks", [(), (5, 11)])
def test_half_precision_recurrence_inputs_stay_close_to_fp32_ones(
    tmp_path: Path, device: str, marks: tuple[int, ...], request: pytest.FixtureRequest
) -> None:
    """gated_delta=half (q, k, v in fp16 into the recurrence, its state in fp32) against fp32 inputs, both in fp16,
    whole and split at marks: as close as fp16's own rounding."""
    _, config, _ = model(tmp_path, ENDS_IN_DELTANET)
    plain = qwen35.load(str(tmp_path), config, dtype=mx.float16)
    half = qwen35.load(str(tmp_path), config, dtype=mx.float16)
    half.ops = for_switches(replace(MLX_SWITCHES, gated_delta="half"))
    ids = tokens(19, 9)
    if device == "gpu":
        request.getfixturevalue("gpu")
    a = np.array(plain.prefill(ids, plain.new_cache(19), list(range(19)), marks).astype(mx.float32))
    b = np.array(half.prefill(ids, half.new_cache(19), list(range(19)), marks).astype(mx.float32))
    assert 0 < np.abs(a - b).max() < 1e-2 * np.abs(a).max()  # another path, as close as fp16's rounding


def test_block_attention_reads_the_prefix_once_and_each_question_both_ways(tmp_path: Path) -> None:
    """elephant-1's block attention against the numpy model's mask (a key earlier, or in the query's block): every
    question branched in one pass from a prefix read once, a question's first token reading its last; a kept prefix
    reused only when it is all of the new one (a state grown at its end is read again)."""
    from ippon.prefixes import Prefixes

    _, config, weights = model(tmp_path, ENDS_IN_ATTENTION)
    m = qwen35.load(str(tmp_path), config, block=True, dtype=mx.float32)
    store = Prefixes(1 << 30)
    questions = [tokens(3, 4), tokens(2, 5), tokens(4, 6)]
    for state, reused in ((tokens(12, 3), 0), (tokens(12, 3), 12), (tokens(12, 3) + tokens(3, 9), 12)):
        prompts = [
            Prompt(
                tuple(state + t),
                (Readout(f"q{i}", len(state)), Readout(f"q{i}", len(state) + len(t) - 1)),
                (len(state),),
            )
            for i, t in enumerate(questions)
        ]
        got = run(m, plan(prompts, block=True), prefixes=store)
        assert store.stats.tokens_reused == reused  # the grown state's first 12 tokens were not reused
        for p, g in zip(prompts, got, strict=True):
            want = forward(config, weights, list(p.tokens), block=len(state))[[r.position for r in p.readouts]]
            np.testing.assert_allclose(np.array(g), want, atol=2e-4, rtol=2e-4)
        causal = forward(config, weights, list(prompts[0].tokens))[len(state)]
        assert not np.allclose(np.array(got[0][0]), causal, atol=1e-3)  # the question's first token reads its last


@pytest.mark.parametrize("bits", [8, 4])
def test_quantized_weights_stay_close_to_full_precision(tmp_path: Path, bits: int) -> None:
    full, config, _ = model(tmp_path, ENDS_IN_ATTENTION)
    small = qwen35.load(str(tmp_path), config, dtype=mx.float32, bits=bits)
    ids = tokens(15, 2)
    a = np.array(full.prefill(ids, full.new_cache(15), [14]))
    b = np.array(small.prefill(ids, small.new_cache(15), [14]))
    assert 0 < np.abs(a - b).max() < (0.05 if bits == 8 else 0.6) * np.abs(a).max()  # quantized, and close


def test_an_output_gate_other_than_silu_is_refused() -> None:
    """The Gated DeltaNet output gate is SiLU; a config naming another activation there would be computed wrongly."""

    with pytest.raises(UnsupportedModel, match="output gate of 'sigmoid'"):
        qwen35.Config.from_hf(tiny_config(ENDS_IN_DELTANET) | {"output_gate_type": "sigmoid"})


@pytest.mark.parametrize("layout", ["per expert", "fused", "switch_mlp"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.metal)])
def test_a_mixture_of_experts_matches_the_numpy_model(
    tmp_path: Path, layout: str, device: str, request: pytest.FixtureRequest
) -> None:
    """Qwen3.5-MoE (router, top-k experts, shared expert), its experts stored one tensor each, as transformers 5
    fuses them, or as mlx-lm stacks them; every row, and a suffix."""
    config = tiny_moe_config(ENDS_IN_ATTENTION)
    weights = tiny_weights(config)
    stored = dict(weights)
    if layout != "per expert":
        for i in range(len(config["layer_types"])):
            p = f"layers.{i}.mlp."
            each = {
                x: np.stack([stored.pop(f"{p}experts.{e}.{x}_proj.weight") for e in range(6)])
                for x in ("gate", "up", "down")
            }
            if layout == "fused":
                stored[p + "experts.gate_up_proj"] = np.concatenate([each["gate"], each["up"]], axis=1)
                stored[p + "experts.down_proj"] = each["down"]
            else:
                stored |= {f"{p}switch_mlp.{x}_proj.weight": a for x, a in each.items()}
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {"model." + k: mx.array(v) for k, v in stored.items()})
    m = qwen35.load(str(tmp_path), config, dtype=mx.float32)
    ids = tokens(17, 7)
    want = forward(config, weights, ids)
    if device == "gpu":
        request.getfixturevalue("gpu")
    got = np.array(m.prefill(ids, m.new_cache(17), list(range(17))))
    suffix = np.array(m.prefill(ids, m.new_cache(17), list(range(9, 17))))
    # On an M5 the fp32 matmuls round about as TF32 does, the router's logits too, which moves the experts' weights by
    # a percent or so (with MLX_METAL_GPU_ARCH=applegpu_g16s, the M4's kernels, this equals the numpy model to 2e-4);
    # a wrong expert or weight would be off by about the output's own size
    tolerance = 2e-4 if device == "cpu" else 0.05 * float(np.abs(want).max())
    np.testing.assert_allclose(got, want, atol=tolerance, rtol=2e-4)
    np.testing.assert_allclose(suffix, want[9:], atol=tolerance, rtol=2e-4)


@pytest.mark.parametrize("family", ["qwen35", "gemma4"])
def test_quantized_experts_stay_close(tmp_path: Path, family: str) -> None:
    build, oracle, _ = fixtures.FAMILIES[family]
    load: Any
    extra = {"hidden_size": 64, "moe_intermediate_size": 64}
    if family == "qwen35":
        config = tiny_moe_config(ENDS_IN_DELTANET) | extra
        weights = tiny_weights(config)
        fixtures.save(tmp_path, weights, "model.")
        load = qwen35.load
        full = load(str(tmp_path), config, dtype=mx.float32)
    else:
        from ippon.arch import gemma4

        full, config, weights = build(tmp_path, intermediate_size=64, **extra)
        load = gemma4.load
    q8 = load(str(tmp_path), config, dtype=mx.float32, bits=8)
    ids = tokens(11, 8)
    a = np.array(full.prefill(ids, full.new_cache(11), [10]))
    b = np.array(q8.prefill(ids, q8.new_cache(11), [10]))
    np.testing.assert_allclose(a, oracle.forward(config, weights, ids)[[10]], atol=2e-4, rtol=2e-4)
    assert 0 < np.abs(a - b).max() < 0.05 * np.abs(a).max()  # quantized, and close


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.metal)])
def test_recurrent_depth_matches_the_numpy_model(tmp_path: Path, device: str, request: pytest.FixtureRequest) -> None:
    """Recurrent depth (GLiDE 9B's "midloop"): two blocks read 3 times each, in a padded both-ways batch, against
    the numpy model's damped Euler steps on the CPU; on the GPU, the compiled layers against the eager ones (an M5's
    fp32 matmuls round about as TF32 does, which three reads of a block carry to a few percent of the output)."""
    from dataclasses import replace

    _, config, weights = model(tmp_path, ENDS_IN_ATTENTION + ENDS_IN_ATTENTION, causal=False)
    depth = Recurrence(((2, 4), (5, 6)), 3)
    m = qwen35.load(str(tmp_path), config, causal=False, recurrence=depth, dtype=mx.float32)
    with pytest.raises(UnsupportedModel, match="both ways only"):
        qwen35.load(str(tmp_path), config, recurrence=depth, dtype=mx.float32)
    prompts = [tokens(13, 1), tokens(9, 2)]
    rows = [list(range(len(p))) for p in prompts]
    if device == "cpu":
        for g, p in zip(m.batch(prompts, rows), prompts, strict=True):
            want = forward(config, weights, p, causal=False, recurrence=([(2, 4), (5, 6)], 3))
            np.testing.assert_allclose(np.array(g), want, atol=2e-4, rtol=2e-4)
        return
    request.getfixturevalue("gpu")
    m.ops = replace(m.ops, compiled=False)
    eager = [np.array(g) for g in m.batch(prompts, rows)]
    m.ops = replace(m.ops, compiled=True)
    for g, e in zip(m.batch(prompts, rows), eager, strict=True):
        np.testing.assert_allclose(np.array(g), e, atol=1e-4 * float(np.abs(e).max()), rtol=0)
