"""Tiny checkpoints for backbone contracts; independent oracle math stays in *_numpy.py."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import numpy as np

import bert_numpy
import deberta_numpy
import dense_numpy
import gemma3_numpy
import gemma4_numpy
import lfm2_numpy
import lfm3_numpy
import modernbert_numpy
import qwen35_numpy
from helpers import tokens as tokens
from ippon.arch import bert, deberta, dense, gemma3, gemma4, lfm2, lfm3, modernbert, qwen35
from ippon.weights import Weights

LLAMA3 = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 100,  # so the tiny heads' 4 wavelengths (6 to 6,283) fall in all three bands
}

YARN = {  # the tiny heads' 4 dimensions fall below, inside and above the ramp
    "rope_type": "yarn",
    "factor": 16.0,
    "original_max_position_embeddings": 64,
    "beta_fast": 8.0,
    "beta_slow": 1.0,
    "attention_factor": 1.25,
}

VARIANTS: dict[str, dict[str, Any]] = {
    "llama": {},
    "llama-yarn": {"rope_scaling": YARN, "max_position_embeddings": 1024},
    "llama-groups": {"norm_groups": 2},
    "llama-yarn-untruncated": {"rope_scaling": YARN | {"truncate": False, "attention_factor": None}},
    "llama-rope3": {"rope_scaling": LLAMA3, "tie_word_embeddings": True},
    "llama-linear-rope": {"rope_scaling": {"rope_type": "linear", "factor": 4.0}},
    "llama-bias": {"attention_bias": True},
    "qwen2": {},
    "qwen2-window-disabled": {"sliding_window": 8, "use_sliding_window": False},
    "qwen3": {},
}

QWEN_ATTENTION = ["linear_attention", "linear_attention", "full_attention", "linear_attention", "full_attention"]
QWEN_DELTANET = ["linear_attention", "full_attention", "linear_attention"]
LFM_ATTENTION = ["conv", "conv", "full_attention", "conv", "full_attention"]
LFM_CONV = ["conv", "full_attention", "conv"]
LFM3_LAYERS = ["conv", "sliding_attention", "conv", "full_attention", "conv"]  # d1-2B's kinds, its last a conv

MODERN_LAYERS = ["full_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention"]


def save(path: Path, weights: dict[str, np.ndarray], root: str = "") -> None:
    mx.save_safetensors(str(path / "model.safetensors"), {root + k: mx.array(v) for k, v in weights.items()})


def dense_model(tmp_path: Path, variant: str) -> tuple[dense.Dense, dict[str, Any], dict[str, np.ndarray]]:
    family = variant.split("-")[0]
    config = dense_numpy.tiny_config(family, **VARIANTS[variant])
    weights = dense_numpy.tiny_weights(config)
    save(tmp_path, weights)
    c = replace(dense.Config.from_hf(config), norm_groups=config.get("norm_groups", 1))  # no Hugging Face field
    return dense.Dense(c, Weights.from_dir(tmp_path), mx.float32), config, weights


def gemma4_model(
    tmp_path: Path,
    layout: str = "fused",
    bits: int | None = None,
    nested: bool = False,
    per_layer: bool = False,
    **extra: Any,
) -> tuple[gemma4.Gemma4, dict[str, Any], dict[str, np.ndarray]]:
    config = gemma4_numpy.tiny_config(**extra)
    weights = gemma4_numpy.tiny_weights(config)
    stored = dict(weights)
    if layout == "switch_glu":  # mlx-lm's conversion: the experts' gate and up apart
        for name in [k for k in stored if k.endswith("experts.gate_up_proj")]:
            gate, up = np.split(stored.pop(name), 2, axis=1)
            base = name.removesuffix("gate_up_proj") + "switch_glu."
            stored |= {base + "gate_proj.weight": gate, base + "up_proj.weight": up}
            stored[base + "down_proj.weight"] = stored.pop(name.removesuffix("gate_up_proj") + "down_proj")
        stored = {k.replace("model.language_model.", "language_model.model.", 1): v for k, v in stored.items()}
    save(tmp_path, stored)
    loaded_config = {"model_type": "gemma4", "text_config": config} if nested else config
    if per_layer:  # the 12B's config: full layers' widths by layer index, as transformers 5.17 stores them
        full = {"head_dim": config["global_head_dim"], "num_key_value_heads": config["num_global_key_value_heads"]}
        loaded_config = {k: v for k, v in config.items() if k not in ("global_head_dim", "num_global_key_value_heads")}
        loaded_config |= {"model_type": "gemma4_unified_text"}
        loaded_config["per_layer_config"] = {
            f"{i:02d}": full for i, kind in enumerate(config["layer_types"]) if kind == "full_attention"
        }
    bidirectional = config.get("bidirectional") == "full"
    model = gemma4.load(str(tmp_path), loaded_config, dtype=mx.float32, bits=bits, causal=not bidirectional)
    return model, config, weights


def checkpoint(
    oracle: Any,
    load: Any,
    path: Path,
    *args: Any,
    root: str = "",
    explicit_root: bool = False,
    dtype: mx.Dtype = mx.float32,
    causal: bool | None = None,
    sep: int | None = None,
    **extra: Any,
) -> tuple[Any, dict[str, Any], dict[str, np.ndarray]]:
    config = oracle.tiny_config(*args, **extra)
    weights = oracle.tiny_weights(config)
    save(path, weights, root)
    options: dict[str, Any] = {"dtype": dtype}
    if explicit_root:
        options["root"] = root
    if causal is not None:
        options["causal"] = causal
    if sep is not None:
        options["sep"] = sep
    return load(str(path), config, **options), config, weights


bert_model = partial(checkpoint, bert_numpy, bert.load)
deberta_model = partial(checkpoint, deberta_numpy, deberta.load)
gemma3_model = partial(checkpoint, gemma3_numpy, gemma3.load)


def lfm2_model(
    path: Path, layer_types: list[str], root: str = "model.", explicit_root: bool = False
) -> tuple[lfm2.LFM2, dict[str, Any], dict[str, np.ndarray]]:
    return cast(
        tuple[lfm2.LFM2, dict[str, Any], dict[str, np.ndarray]],
        checkpoint(lfm2_numpy, lfm2.load, path, layer_types, root=root, explicit_root=explicit_root),
    )


def lfm3_model(
    path: Path, layer_types: list[str] = LFM3_LAYERS, window: int = 4
) -> tuple[lfm3.LFM3, dict[str, Any], dict[str, np.ndarray]]:
    return cast(
        tuple[lfm3.LFM3, dict[str, Any], dict[str, np.ndarray]],
        checkpoint(lfm3_numpy, lfm3.load, path, layer_types, root="model.language_model.", window=window),
    )


def qwen35_model(
    path: Path, layer_types: list[str], root: str = "model.", causal: bool = True, dtype: mx.Dtype = mx.float32
) -> tuple[qwen35.Qwen35, dict[str, Any], dict[str, np.ndarray]]:
    return cast(
        tuple[qwen35.Qwen35, dict[str, Any], dict[str, np.ndarray]],
        checkpoint(qwen35_numpy, qwen35.load, path, layer_types, root=root, causal=causal, dtype=dtype),
    )


def modernbert_model(path: Path, layers: list[str] = MODERN_LAYERS, local_attention: int = 6) -> Any:
    return checkpoint(
        modernbert_numpy,
        modernbert.load,
        path,
        layers,
        local_attention=local_attention,
        root="encoder.",
        explicit_root=True,
    )


# Builder, independent oracle module, and config parser for the shared contracts.
FAMILIES: dict[str, tuple[Any, Any, Any]] = {
    "bert": (bert_model, bert_numpy, bert.Config.from_hf),
    "deberta": (deberta_model, deberta_numpy, deberta.Config.from_hf),
    "dense": (dense_model, dense_numpy, dense.Config.from_hf),
    "gemma3": (gemma3_model, gemma3_numpy, gemma3.Config.from_hf),
    "gemma4": (gemma4_model, gemma4_numpy, gemma4.Config.from_hf),
    "lfm2": (lfm2_model, lfm2_numpy, lfm2.Config.from_hf),
    "lfm3": (lfm3_model, lfm3_numpy, lfm3.Config.from_hf),
    "modernbert": (modernbert_model, modernbert_numpy, modernbert.Config.from_hf),
    "qwen35": (qwen35_model, qwen35_numpy, qwen35.Config.from_hf),
}
