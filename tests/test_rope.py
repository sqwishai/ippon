"""RoPE as configs set it (arch/rope.py): every scaling type read the same way for every family, the ones ippon
doesn't compute refused, and YaRN refused where its attention factor can't be applied to the scores."""

from __future__ import annotations

import numpy as np
import pytest

from ippon.arch import qwen35
from ippon.arch.rope import RoPE
from ippon.errors import UnsupportedModel
from qwen35_numpy import tiny_config


def test_each_scaling_type_is_read() -> None:
    assert RoPE.from_hf(64, None, theta=5.0) == RoPE(64, 5.0)
    assert RoPE.from_hf(64, {"rope_type": "default", "rope_theta": 7.0}) == RoPE(64, 7.0)
    assert RoPE.from_hf(64, {"type": "linear", "factor": 4.0}).scale == 0.25  # transformers' old key, "type"
    llama3 = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
              "original_max_position_embeddings": 8192}  # fmt: skip
    assert RoPE.from_hf(64, llama3).freqs is not None
    yarn = RoPE.from_hf(64, {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096})
    assert yarn.attention == pytest.approx(0.1 * np.log(4.0) + 1.0)
    with pytest.raises(UnsupportedModel, match="dynamic"):
        RoPE.from_hf(64, {"rope_type": "dynamic", "factor": 2.0})


def test_yarn_over_part_of_the_head_is_refused() -> None:
    c = tiny_config(["full_attention"]) | {"partial_rotary_factor": 0.5}
    c["rope_parameters"] = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 64}
    with pytest.raises(UnsupportedModel, match="part of the head"):
        qwen35.Config.from_hf(c)
