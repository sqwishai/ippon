"""FP32 execution of the SAME quantized checkpoint after stock MLX dequantization.

This is not a creator-quality reference and is not an independent architecture implementation. Pair it with
scripts/check_qwen35_float64.py, whose NumPy backbone is independent, and the published-runtime reference.
Only this explicit reference program expands weights. Normal loading retains the packed checkpoint.

    IPPON_PROFILE=m4 python scripts/dequantized_reference.py \
        --reference references/clef/9b-mxfp8.ollama.json.gz --out dequantized.json.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

# isort: off
import ippon  # before MLX: the requested architecture selects exact FP32 kernels on an M5
import mlx.core as mx
# isort: on

from ippon import Decider, _emulate
from ippon.arch.ops import Ops
from ippon.quant import JoinedMatrix, Matrix, QuantizedMatrix
from ippon.schema import parse_request
from ippon.verify import load_reference


class StreamingReference(Ops):
    """Dequantize one matrix at a time so a 9B FP32 reference fits without retaining all dense weights."""

    def mm(self, x: mx.array, w: Matrix) -> mx.array:
        dense = w.dense(mx.float32) if isinstance(w, QuantizedMatrix | JoinedMatrix) else w
        y = super().mm(x, dense)
        mx.eval(y)
        return y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stream-weights", action="store_true")
    args = parser.parse_args()
    if _emulate.architecture_for("m4", _emulate.chip_name()) and not _emulate.APPLIED:
        raise SystemExit("Set IPPON_PROFILE=m4 for exact FP32 on an M5; this does not test physical M4 hardware.")
    ref = load_reference(args.reference)
    with Decider.load(
        ref["model"]["repo"],
        precision="fp32",
        pin_memory=False,
        warm_up=False,
        strategy="reference",
        switches="add_norm=mlx,conv=mlx,gemm=mlx,delta_inputs=mlx,pipeline=off",
    ) as model:
        if args.stream_weights:
            model.backbone.ops = StreamingReference()
            mx.set_cache_limit(0)
        else:
            for layer in model.backbone.layers:
                for key, value in list(layer.items()):
                    if isinstance(value, QuantizedMatrix | JoinedMatrix):
                        layer[key] = value.dense(mx.float32)
                        mx.eval(layer[key])
        mx.clear_cache()
        for case in ref["requests"]:
            request = parse_request(case["request"])
            result = model.answer(request)
            case["probabilities"] = {q.id: list(result[q.id].probabilities) for q in request.questions}
            case["input_tokens"] = result.response["usage"]["input_tokens"]
            print(case["name"], case["probabilities"], flush=True)
    ref.update(
        reference_kind="checkpoint_dequantized",
        bound=0.01,
        runtime={
            "ippon": ippon.__version__,
            "mlx": mx.__version__,
            "precision": "fp32",
            "architecture": _emulate.APPLIED,
        },
        notes=["Same checkpoint, stock MLX dequantization; shared ippon architecture. Not original FP32 quality."],
    )
    args.out.write_bytes(gzip.compress(json.dumps(ref, ensure_ascii=False).encode(), mtime=0))


if __name__ == "__main__":
    main()
