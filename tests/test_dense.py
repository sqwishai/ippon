"""The MLX dense decoders (Llama, Mistral, Qwen2, Qwen3) against an independent numpy one on tiny random
checkpoints: every RoPE and norm setting, any rows read, the head; the runtime's paths through it are
test_backbone_contract.py's."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from backbone_fixtures import dense_model as model
from dense_numpy import forward, tiny_config, tiny_weights
from helpers import tokens as tokens
from ippon.arch import dense
from ippon.arch.rope import yarn_freqs
from ippon.errors import ContextTooLong
from ippon.quant import rows

# transformers 5.17's _compute_yarn_parameters on Qwen3's documented YaRN setting (factor 4 over 32,768 positions,
# 128-dim heads, theta 1e6, its default betas and attention factor): its inv_freq
TRANSFORMERS_YARN_INV_FREQ = np.array(
    [
        1.0,
        0.8058422207832336,
        0.6493816375732422,
        0.5232991576194763,
        0.4216965138912201,
        0.33982083201408386,
        0.2738419771194458,
        0.22067339718341827,
        0.17782793939113617,
        0.14330126345157623,
        0.11547820270061493,
        0.09305720031261444,
        0.07498941570520401,
        0.0604296438395977,
        0.04869675263762474,
        0.03924189880490303,
        0.03162277862429619,
        0.025482967495918274,
        0.02053525112569332,
        0.016548171639442444,
        0.01333521492779255,
        0.010746078565716743,
        0.00865964312106371,
        0.006978305988013744,
        0.005375321488827467,
        0.004131738096475601,
        0.0031684227287769318,
        0.002423422411084175,
        0.0018482765881344676,
        0.0014051124453544617,
        0.0010643609566614032,
        0.000802959781140089,
        0.0006029411451891065,
        0.0004503235686570406,
        0.00033424055436626077,
        0.0002462583943270147,
        0.0001798411540221423,
        0.00012993148993700743,
        9.262301318813115e-05,
        6.490394298452884e-05,
        4.4456985051510856e-05,
        3.582531644497067e-05,
        2.8869548259535804e-05,
        2.3264301489689387e-05,
        1.8747356079984456e-05,
        1.5107409126358107e-05,
        1.2174188668723218e-05,
        9.8104746939498e-06,
        7.905693564680405e-06,
        6.370741630234988e-06,
        5.133812464919174e-06,
        4.137042651564116e-06,
        3.3338035336782923e-06,
        2.686519565031631e-06,
        2.1649109385180054e-06,
        1.7445765934098745e-06,
        1.405853367941745e-06,
        1.1328959317324916e-06,
        9.129353202297352e-07,
        7.356817945947114e-07,
        5.928434347879374e-07,
        4.777382400789065e-07,
        3.8498163235090033e-07,
        3.102344408034696e-07,
    ]
)


def test_the_output_matrix_is_the_head_or_the_tied_embedding(tmp_path: Path) -> None:
    """Either read row by row from the checkpoint's file (Weights.table), every row as stored."""
    m, _, weights = model(tmp_path, "llama")
    every = mx.arange(weights["lm_head.weight"].shape[0])
    np.testing.assert_array_equal(np.array(rows(m.output_matrix, every)), weights["lm_head.weight"])
    tied, _, weights = model(tmp_path, "llama-rope3")
    np.testing.assert_array_equal(np.array(rows(tied.output_matrix, every)), weights["model.embed_tokens.weight"])


def test_yarn_matches_transformers_tables() -> None:
    """Qwen3's documented YaRN setting, against the inverse frequencies and attention factor transformers 5.17's
    _compute_yarn_parameters gives for it (recorded): the betas and the attention factor at their defaults."""
    p = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
    freqs, attention = yarn_freqs(128, 1_000_000.0, p, 131072)
    np.testing.assert_allclose(1.0 / np.array(freqs), TRANSFORMERS_YARN_INV_FREQ, rtol=1e-6)
    assert attention == pytest.approx(1.138629436111989)


def test_a_prompt_past_the_sliding_window_is_refused(tmp_path: Path) -> None:
    config = tiny_config("mistral", sliding_window=8)
    weights = tiny_weights(config)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {k: mx.array(v) for k, v in weights.items()})
    m = dense.load(str(tmp_path), config, dtype=mx.float32)
    got = m.prefill(tokens(8, 1), m.new_cache(8), [7])  # within the window: as full attention
    np.testing.assert_allclose(np.array(got), forward(config, weights, tokens(8, 1))[[7]], atol=2e-4, rtol=2e-4)
    with pytest.raises(ContextTooLong, match="sliding window"):
        m.prefill(tokens(9, 1), m.new_cache(9), [8])
