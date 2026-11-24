"""Run tiny models with MLX's compositions and with the fused kernels on the GPU, against the numpy models: how far
each is, per architecture. Run in a process of its own with MLX_METAL_GPU_ARCH set to a family without neural
accelerators, where MLX's fp32 matmuls are exact (on an M5 they round their inputs to about TF32)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ippon.arch import lfm2, modernbert, qwen35
from ippon.arch.ops import FUSED, MLX

IDS = [5, 9, 1, 33, 7, 2, 60, 11, 4]


def main() -> None:
    import lfm2_numpy
    import modernbert_numpy
    import qwen35_numpy

    mx.set_default_device(mx.gpu)
    cases = {
        "qwen35": (qwen35, qwen35_numpy, ["linear_attention", "full_attention", "linear_attention"], "model.", {}),
        "lfm2": (lfm2, lfm2_numpy, ["conv", "full_attention", "conv"], "model.", {}),
        "modernbert": (
            modernbert,
            modernbert_numpy,
            ["full_attention", "sliding_attention", "full_attention"],
            "encoder.",
            {"root": "encoder."},
        ),
    }
    out: dict[str, dict[str, float]] = {}
    for name, (module, reference, layers, root, kwargs) in cases.items():
        config = reference.tiny_config(layers)
        weights = reference.tiny_weights(config)
        directory = Path(tempfile.mkdtemp())
        mx.save_safetensors(str(directory / "model.safetensors"), {root + k: mx.array(v) for k, v in weights.items()})
        model = module.load(str(directory), config, dtype=mx.float32, **kwargs)
        want = reference.forward(config, weights, IDS)
        out[name] = {}
        for label, ops in (("mlx", MLX), ("fused", FUSED)):
            model.ops = ops
            got = np.array(model.prefill(IDS, model.new_cache(len(IDS)), list(range(len(IDS)))))
            out[name][label] = float(np.abs(got - want).max())
    print(json.dumps(out))


if __name__ == "__main__":
    main()
