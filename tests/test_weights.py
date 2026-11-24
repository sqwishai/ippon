"""A PEFT LoRA merged into a checkpoint's weights: the formula, the variants it refuses, and a tiny Qwen3.5 with a
merged LoRA against the numpy model run with the merged matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from ippon.arch import qwen35
from ippon.errors import UnsupportedModel
from ippon.host_rows import HostRows
from ippon.quant import rows
from ippon.weights import Weights
from qwen35_numpy import forward, tiny_config, tiny_weights

CONFIG = {"peft_type": "LORA", "r": 2, "lora_alpha": 8, "bias": "none", "use_dora": False, "use_rslora": False}
UP = "layers.0.mlp.up_proj"


def base() -> Weights:
    rng = np.random.default_rng(0)
    return Weights(
        {
            "model.embed_tokens.weight": mx.array(rng.normal(size=(10, 4)).astype(np.float32)),
            f"model.{UP}.weight": mx.array(rng.normal(size=(6, 4)).astype(np.float32)),
        },
        "base",
    )


def adapter(module: str = UP, extra: dict[str, mx.array] | None = None) -> Weights:
    rng = np.random.default_rng(1)
    tensors = {
        f"base_model.model.{module}.lora_A.weight": mx.array(rng.normal(size=(2, 4)).astype(np.float32)),
        f"base_model.model.{module}.lora_B.weight": mx.array(rng.normal(size=(6, 2)).astype(np.float32)),
    }
    return Weights({**tensors, **(extra or {})}, "adapter")


@pytest.mark.parametrize(
    "setting",
    [{"use_dora": True}, {"bias": "all"}, {"peft_type": "IA3"}],  # a listed variant, trained biases, another method
)
def test_variants_with_another_formula_are_refused(setting: dict[str, Any]) -> None:
    with pytest.raises(UnsupportedModel, match="does not merge"):
        base().with_lora(adapter(), {**CONFIG, **setting})


def test_an_adapter_that_does_not_fit_the_base_is_refused() -> None:
    bias = {f"base_model.model.{UP}.lora_B.bias": mx.zeros((6,))}
    with pytest.raises(UnsupportedModel, match="unexpected adapter tensor"):
        base().with_lora(adapter(extra=bias), CONFIG)
    with pytest.raises(UnsupportedModel, match="matches no matrix"):
        base().with_lora(adapter("layers.0.mlp.down_proj"), CONFIG)
    lora = adapter()
    lora.tensors[f"base_model.model.{UP}.lora_B.weight"] = mx.zeros((5, 2))
    with pytest.raises(UnsupportedModel, match=r"\(5, 4\)"):
        base().with_lora(lora, CONFIG)
    tensors = dict(base().tensors)
    tensors[f"model.{UP}.scales"] = mx.ones((6, 1))
    with pytest.raises(UnsupportedModel, match="quantized"):
        Weights(tensors, "base").with_lora(adapter(), CONFIG)


def test_a_model_with_a_merged_lora_matches_the_numpy_model_with_merged_matrices(tmp_path: Path) -> None:
    layer_types = ["linear_attention", "full_attention"]
    config, weights = tiny_config(layer_types), tiny_weights(tiny_config(layer_types))
    rng = np.random.default_rng(2)
    targets = ["layers.0.linear_attn.in_proj_qkv", "layers.1.self_attn.q_proj", "layers.1.mlp.down_proj"]
    lora, merged = {}, dict(weights)
    for t in targets:
        out, inp = weights[t + ".weight"].shape
        a, b = rng.normal(size=(2, inp)).astype(np.float32), rng.normal(size=(out, 2)).astype(np.float32) * 0.1
        lora[f"base_model.model.{t}.lora_A.weight"], lora[f"base_model.model.{t}.lora_B.weight"] = a, b
        merged[t + ".weight"] = weights[t + ".weight"] + 4.0 * b @ a
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {"model." + k: mx.array(v) for k, v in weights.items()})
    adapter_weights = Weights({k: mx.array(v) for k, v in lora.items()}, "adapter")
    merged_weights = Weights.from_dir(tmp_path).with_lora(adapter_weights, CONFIG)
    m = qwen35.load(str(tmp_path), config, dtype=mx.float32, weights=merged_weights)
    assert isinstance(m.embed, HostRows)  # no LoRA matrix changed it, so it is still read from its file
    ids = [int(t) for t in rng.integers(0, 64, 17)]
    got = m.prefill(ids, m.new_cache(len(ids)), list(range(len(ids))))
    np.testing.assert_allclose(np.array(got), forward(config, merged, ids), atol=2e-4, rtol=2e-4)


def test_a_merge_keeps_the_quantization_of_the_matrices_it_leaves_alone() -> None:
    quantization = {"group_size": 32, "bits": 8, "mode": "affine"}
    w = Weights(dict(base().tensors), "base", quantization=quantization)
    assert w.with_lora(adapter(), CONFIG).quantization == quantization


def test_each_tensor_is_handed_out_once_so_its_memory_goes_with_the_layer_made_from_it(tmp_path: Path) -> None:
    """A loader takes each tensor once: Weights then holds it no more, so loading a model in another dtype than its
    files' (fp16 from bf16, a LoRA merged in fp32) never keeps the source and the result at once."""
    w = Weights({"a.weight": mx.ones((4, 4)), "b.weight": mx.ones((4, 4))}, "t")
    w.get("a.weight", mx.float16)
    assert "a.weight" not in w
    assert "b.weight" in w
    with pytest.raises(ValueError, match="read already"):
        w.matrix("a.weight")
    with pytest.raises(UnsupportedModel, match=r"no tensor 'c\.weight'"):
        w.matrix("c.weight")
    config = tiny_config(["linear_attention", "full_attention"])
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), {"model." + k: mx.array(v) for k, v in tiny_weights(config).items()}
    )
    weights = Weights.from_dir(tmp_path)
    qwen35.load(str(tmp_path), config, dtype=mx.float16, weights=weights)
    assert not [k for k in weights.tensors if ".layers." in k]  # every layer's tensors taken, none still held


@pytest.mark.parametrize("stored", ["bf16", "fp32", "affine", "mxfp8"])
def test_rows_read_from_the_file_are_the_rows_of_the_table_as_loaded(tmp_path: Path, stored: str) -> None:
    """A vocabulary table read row by row from its file (Weights.table) gives the rows the whole table, loaded and
    cast to the load's dtype, gives: at full precision, and packed (an MLX 8-bit or MXFP8 build) unpacked as a load
    unpacks it; and it is taken out of the weights."""
    table = mx.random.normal((300, 64), key=mx.random.key(0))
    quantization: dict[str, Any] | None = None
    if stored in ("bf16", "fp32"):
        tensors = {"embed.weight": table.astype(mx.bfloat16 if stored == "bf16" else mx.float32)}
    else:
        quantization = {"bits": 8, "group_size": 64 if stored == "affine" else 32, "mode": stored}
        packed = mx.quantize(table, **quantization)
        tensors = {"embed.weight": packed[0], "embed.scales": packed[1]} | (
            {"embed.biases": packed[2]} if stored == "affine" else {}
        )
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
    ids = mx.array([[5, 299, 0], [5, 17, 42]])
    weights = Weights.from_dir(tmp_path, quantization=quantization)
    host = weights.table("embed.weight", mx.float16)
    assert isinstance(host, HostRows)
    assert "embed.weight" not in weights
    loaded = Weights.from_dir(tmp_path, quantization=quantization).matrix("embed.weight", mx.float16)
    assert mx.array_equal(rows(host, ids), rows(loaded, ids)).item()
