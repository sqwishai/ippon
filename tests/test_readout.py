from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from ippon.readout import LabelSet, label_logits, option_probabilities


def test_options_tokens_combine_by_log_sum_exp_at_a_temperature() -> None:
    """An option's logit is the log-sum-exp of its tokens' (option 0 has two, at logits 1 and 2; option 1 one, at 0),
    divided by a temperature that must be positive; logits that don't match the label tokens, or an option without
    one, are refused."""
    labels = LabelSet(((0, 1), (2,)))
    first = math.log(math.exp(1) + math.exp(2))
    for temperature in (1.0, 0.5):
        probs = option_probabilities(mx.array([[1.0, 2.0, 0.0]]), labels, temperature=temperature)
        assert probs[0, 0].item() == pytest.approx(1 / (1 + math.exp(-first / temperature)), rel=1e-6)
    with pytest.raises(ValueError, match="positive"):
        option_probabilities(mx.array([[1.0, 2.0, 0.0]]), labels, temperature=0)
    with pytest.raises(ValueError, match="label tokens"):
        option_probabilities(mx.array([[1.0, 2.0]]), LabelSet(((0,), (1,), (2,))))
    with pytest.raises(ValueError, match="at least one"):
        LabelSet(((0,), ()))


def test_tiny_and_large_readouts_give_the_matmuls_logits_in_fp32() -> None:
    """Both routes (a multiply-and-sum up to TINY_PRODUCT logits, a matmul past it) against numpy, and fp32 from
    bf16 weights on each."""
    weight = mx.random.normal((300, 64), key=mx.random.key(0))
    for rows, labels in ((1, 2), (4, 10), (16, 40), (40, 30)):
        hidden = mx.random.normal((rows, 64), key=mx.random.key(rows))
        ids = list(range(5, 5 + labels))
        got = label_logits(hidden, weight, ids)
        want = np.array(hidden) @ np.array(weight)[ids].T
        np.testing.assert_allclose(np.array(got), want, rtol=1e-5, atol=1e-5)
        assert label_logits(hidden.astype(mx.bfloat16), weight.astype(mx.bfloat16), ids).dtype == mx.float32


def test_a_code_readout_is_checked_when_it_loads(tmp_path: Path) -> None:
    from ippon.adapter import LoadContext
    from ippon.errors import InvalidRequest, UnsupportedModel
    from ippon.model_dir import ModelDir
    from ippon.readout import CODE_ROWS, CodeReadout

    model = ModelDir.open(tmp_path)

    def saved(rows: int) -> None:
        mx.save_safetensors(str(tmp_path / "readout.safetensors"), {"weight": mx.ones((rows, 4))})

    saved(CODE_ROWS)
    ctx = LoadContext(dtype=mx.float32)
    readout = CodeReadout.load(ctx, model, ["A", "B", "C"], 2.0)
    readout.check("q", 3)
    with pytest.raises(InvalidRequest, match="1 to 3"):  # an option without a code is refused, never left out
        readout.check("q", 4)
    p = readout(mx.array([1.0, 0.0, 0.0, 0.0]), 2)
    np.testing.assert_allclose(np.array(p), [0.5, 0.5], rtol=1e-6)
    refused = [(["A"], 0.0, "temperature"), (["A"], float("nan"), "temperature"), (["A", "A"], 1.0, "distinct")]
    for codes, temperature, match in [*refused, ([], 1.0, "distinct")]:
        with pytest.raises(UnsupportedModel, match=match):
            CodeReadout.load(ctx, model, codes, temperature)
    saved(16)
    with pytest.raises(UnsupportedModel, match="16 codes"):
        CodeReadout.load(ctx, model, ["A"], 1.0)
