"""Checkpoints converted to MLX (quantized matrices as weight/scales/biases, MLX's convolution layout, Qwen3.5's
norm weights stored shifted) load to the same model as the Hugging Face checkpoint quantized the same way."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from ippon.arch import lfm2, qwen35
from ippon.quant import QuantizedMatrix, rows
from ippon.weights import Weights

ZERO_CENTRED = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


GROUPS: dict[str, Any] = {}  # each quantized matrix's entry, as an MLX config's "quantization" lists them


def to_mlx(
    weights: dict[str, np.ndarray], root: str, shift: bool, matrices: tuple[str, ...], embedding: bool = True
) -> dict[str, mx.array]:
    """An MLX conversion of a Hugging Face state dict: 8-bit matrices (those 32 wide or a multiple) and embedding,
    [C, K, 1] convolutions, and Qwen3.5's zero-centred norms as 1 + w when `shift`."""
    out: dict[str, mx.array] = {}
    for name, value in weights.items():
        key = root + name
        if name.endswith(("conv1d.weight", "conv.conv.weight")):
            out[key] = mx.array(value).moveaxis(2, 1)  # [C, 1, K] -> [C, K, 1]
        elif shift and name.endswith(ZERO_CENTRED) and "linear_attn" not in name:
            out[key] = mx.array(value) + 1.0
        elif (
            any(name.endswith(m + ".weight") for m in matrices) or (embedding and name == "embed_tokens.weight")
        ) and value.shape[-1] % 32 == 0:
            group = 64 if value.shape[-1] % 64 == 0 else 32  # as ippon groups, so both sides quantize alike
            q, s, b = mx.quantize(mx.array(value), group_size=group, bits=8)
            base = key.removesuffix(".weight")
            out[key], out[base + ".scales"], out[base + ".biases"] = q, s, b
            GROUPS[base] = {"group_size": group, "bits": 8}
        else:
            out[key] = mx.array(value)
    return out


def qwen_matrices() -> tuple[str, ...]:
    return (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
    )


def test_an_mlx_qwen35_checkpoint_is_the_hf_one_quantized(tmp_path: Path) -> None:
    from qwen35_numpy import tiny_config, tiny_weights

    layers = ["linear_attention", "full_attention"]
    config, weights = tiny_config(layers), tiny_weights(tiny_config(layers))
    converted = Weights(to_mlx(weights, "language_model.model.", True, qwen_matrices()), "mlx", quantization=GROUPS)
    model = qwen35.Qwen35(qwen35.Config.from_hf(config), converted, mx.float32)
    assert isinstance(model.layers[0]["in"], QuantizedMatrix)
    assert isinstance(model.embed, QuantizedMatrix)
    # the same model from Hugging Face's layout, its matrices (and embedding, dequantized) quantized alike
    hf = {"model." + k: mx.array(v) for k, v in weights.items()}
    reference = qwen35.Qwen35(qwen35.Config.from_hf(config), Weights(hf, "hf"), mx.float32, bits=8)
    ids = [3, 17, 5, 60, 2, 9, 41]
    got = np.array(model.prefill(ids, model.new_cache(7), list(range(7))))
    # the reference's embedding is full precision: compare against a model given the dequantized embedding
    reference.embed = rows(model.embed, mx.arange(config["vocab_size"]))
    want = np.array(reference.prefill(ids, reference.new_cache(7), list(range(7))))
    np.testing.assert_allclose(got, want, atol=2e-3, rtol=2e-3)


def test_an_mlx_lfm2_checkpoint_reads_its_convolution_layout(tmp_path: Path) -> None:
    from lfm2_numpy import forward, tiny_config, tiny_weights

    config = tiny_config(["conv", "full_attention", "conv"])
    weights = tiny_weights(config)
    converted = Weights(to_mlx(weights, "language_model.model.", False, (), embedding=False), "mlx")  # the layout
    model = lfm2.LFM2(lfm2.Config.from_hf(config), converted, mx.float32)
    ids = [1, 7, 30, 4, 22]
    got = np.array(model.prefill(ids, model.new_cache(5), list(range(5))))
    np.testing.assert_allclose(got, forward(config, weights, ids), atol=2e-4, rtol=2e-4)
