"""ippon's Qwen3.5 against tests/qwen35_numpy.py in float64 (written from transformers' modeling_qwen3_5.py), on
a checkpoint's own weights over the first tokens of a prompt: the exactness check for a quantized build, whose
creator's fp32 model is another model (docs/accuracy.md). ippon runs in fp32 on the M4's kernels (an M5's round
fp32 matmuls and the chunked recurrence to about TF32); the numpy model reads every tensor dequantized by MLX, one at
a time, with the MLX conversion's conventions undone (zero-centred norms stored as 1 + w, convolutions stored as
[channels, taps, 1]); a GGUF build's tensors as ippon renames them (gguf/names.py). Causal checkpoints (Clef,
Clef-Flash).

    IPPON_PROFILE=m4 uv run python scripts/check_qwen35_float64.py MODEL_DIR_OR_GGUF [--tokens 16]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# isort: off
import ippon  # first: IPPON_PROFILE=m4 takes effect only before MLX loads
import mlx.core as mx
import numpy as np

# isort: on
from ippon import Decider, _emulate, bench
from ippon.quant import NVFP4_MAX
from ippon.quantization import Quantization, global_scale_names
from ippon.schema import parse_request
from ippon.weights import Weights

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from qwen35_numpy import forward

SHIFTED = ("input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight", "k_norm.weight")


class Tensors(dict[str, Any]):
    """The checkpoint's text-model tensors by transformers' names, in float64, each read when asked for."""

    def __init__(self, weights: Weights, group_size: int) -> None:
        super().__init__()
        self.t, self.root, self.group_size = weights.tensors, weights.text_root(), group_size
        self.quantization = weights.quantization or {"group_size": group_size}
        conv = next(k for k in self.t if k.startswith(self.root) and k.endswith("linear_attn.conv1d.weight"))
        self.shifted = self.t[conv].shape[-1] == 1  # converted to MLX: [channels, taps, 1] and norms as 1 + w

    def dequantized(self, base: str, rows: mx.array | None = None) -> np.ndarray:
        w, s = (self.t[base + x] for x in (".weight", ".scales"))
        b = self.t.get(base + ".biases")
        q = Quantization.read(self.quantization, base + ".weight", w.shape, s.shape)
        if rows is not None:
            w, s, b = w[rows], s[rows], b[rows] if b is not None else None
        dense = mx.dequantize(
            w,
            s.astype(mx.float32) if q.mode == "affine" else s,
            b.astype(mx.float32) if b is not None else None,
            q.group_size,
            q.bits,
            mode=q.mode,
            dtype=None if q.mode == "affine" else mx.float32,
        )
        for name in global_scale_names(base + ".weight"):
            if name in self.t:  # an MLX amax (as the loader reads it, weights.py), else the multiplier itself
                convention = q.scale_convention
                amax = convention == "mlx" if convention else q.mlx_global_scale or not name.endswith("_scale_2")
                dense = dense * self.t[name].astype(mx.float32) / (NVFP4_MAX if amax else 1)
        return np.array(dense, dtype=np.float64)

    def __getitem__(self, name: str) -> Any:
        key = self.root + name
        base = key.removesuffix(".weight")
        if name == "embed_tokens.weight":
            return Rows(self)
        if key.endswith(".weight") and base + ".scales" in self.t:
            return self.dequantized(base)
        x = np.array(self.t[key].astype(mx.float32), dtype=np.float64)
        if self.shifted and name.endswith("conv1d.weight"):
            return x.transpose(0, 2, 1)
        if self.shifted and (name.endswith(SHIFTED) or name == "norm.weight"):
            return x - 1.0
        return x


class Rows:
    """The embedding's rows, converted to float64 only for the ids asked for."""

    def __init__(self, tensors: Tensors) -> None:
        self.tensors = tensors

    def __getitem__(self, ids: list[int]) -> np.ndarray:
        base = self.tensors.root + "embed_tokens"
        if base + ".scales" in self.tensors.t:
            return self.tensors.dequantized(base, mx.array(ids))
        return np.array(self.tensors.t[base + ".weight"][mx.array(ids)].astype(mx.float32), np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model")
    p.add_argument("--tokens", type=int, default=16, help="the prompt's first tokens to compare (default 16)")
    p.add_argument("--bound", type=float, default=1e-4, help="the largest relative difference per row (1e-4)")
    args = p.parse_args()
    if _emulate.architecture_for("m4", _emulate.chip_name()) and not _emulate.APPLIED:
        raise SystemExit("on an M5, set IPPON_PROFILE=m4 (its own fp32 kernels round to about TF32)")
    if args.model.startswith("ollama://"):
        from ippon.ollama import open_build

        build = open_build(args.model)
        config = json.loads(build.assets["config.json"].read_text())
        weights, group_size = build.weights(), 64
    elif args.model.endswith(".gguf"):  # a GGUF build: tensors under the creator's names and config
        from ippon import gguf

        directory = gguf.open_build(args.model)
        config, weights, group_size = dict(directory.config), gguf.weights(directory), 32
    else:
        config = json.loads((Path(args.model) / "config.json").read_text())
        weights = Weights.from_dir(
            args.model, quantization=config.get("quantization") or config.get("quantization_config")
        )
        group_size = int((config.get("quantization") or {}).get("group_size", 64))
    text = dict(config.get("text_config", config))
    text["rope_parameters"] = {"rope_theta": text["rope_parameters"]["rope_theta"]}
    text.setdefault(
        "partial_rotary_factor", config.get("text_config", config)["rope_parameters"]["partial_rotary_factor"]
    )
    decider = Decider.load(str(args.model), precision="fp32", warm_up=False, pin_memory=False)
    backbone = decider.backbone
    if not getattr(backbone.config, "causal", True):
        raise SystemExit("a non-causal checkpoint: this check reads a prompt's first tokens alone, as a causal model")
    prompt = decider.format.render(parse_request(next(bench.bodies("fresh", 300, 1, 1, 1)))).prompts[0]
    ids = list(prompt.tokens)[: args.tokens]
    got = backbone.prefill(ids, backbone.new_cache(len(ids)), list(range(len(ids))))
    mine = np.array(got.astype(mx.float32), dtype=np.float64)
    want = forward(text, Tensors(weights, group_size), ids)
    rel = np.abs(mine - want).max(-1) / np.abs(want).max(-1)
    print(f"{args.model} (ippon {ippon.__version__}, fp32 on {_emulate.APPLIED}): {len(ids)} tokens")
    print(f"  largest relative difference per row from float64: {rel.max():.2e} (row {int(rel.argmax())})")
    print("ok" if rel.max() <= args.bound else f"failed: past {args.bound}")
    sys.exit(0 if rel.max() <= args.bound else 1)


if __name__ == "__main__":
    main()
