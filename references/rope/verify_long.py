"""Compare shipped RoPE, MLX RoPE and FP64-angle/FP32-trig tables against an existing long creator reference.

Experimental tables stay in this script. Run each mode in a fresh process under the GPU lock with a scratch
IPPON_CACHE. Normal model memory limits stay enabled. This uses the same verify_model check as the CLI.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from ippon import Decider
from ippon._portable import dumps
from ippon.arch.rope import RoPE
from ippon.invariance import run as invariance
from ippon.machine import describe
from ippon.verify import load_reference, verify_model


@lru_cache(maxsize=32)
def tables(rope: RoPE) -> tuple[mx.array, mx.array]:
    pairs = rope.dims // 2
    periods = (
        np.asarray(rope.freqs) if rope.freqs is not None else rope.theta ** (np.arange(pairs, dtype=np.float64) / pairs)
    )
    angles = np.arange(32_768, dtype=np.float64)[:, None] * rope.scale / periods
    return mx.array(np.cos(angles).astype(np.float32)), mx.array(np.sin(angles).astype(np.float32))


def exact_angles(self: RoPE, x: mx.array, start: int, positions: mx.array | None = None) -> mx.array:
    cos, sin = tables(self)
    at = mx.arange(start, start + x.shape[-2]) if positions is None else positions
    c, s = cos[at], sin[at]
    y, d = x.astype(mx.float32), self.dims // 2
    a, b = y[..., :d], y[..., d : 2 * d]
    return mx.concatenate([a * c - b * s, a * s + b * c, y[..., 2 * d :]], axis=-1).astype(x.dtype)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--case", default="long-12k-mixed")
    p.add_argument("--mode", choices=("shipped", "mlx", "tables"), required=True)
    p.add_argument("--invariance", action="store_true")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    source = load_reference(a.reference)
    cases = [c for c in source["requests"] if c["name"] == a.case]
    if len(cases) != 1 or max(map(len, cases[0]["prompts"])) >= 32_768:
        p.error("select exactly one reference case whose positions fit the experiment's tables")
    reference = {**source, "requests": cases}
    # Derived references can carry completion counts for the original set.
    for key in ("planned_requests", "completed_requests"):
        if key in reference:
            reference[key] = 1
    options: dict[str, Any] = {"pin_memory": True}
    if a.mode != "shipped":
        options["switches"] = "norm_rope=mlx,compile=off"
    captured: list[dict[str, list[float]]] = []
    original = Decider._answer

    def answered(d: Decider, request: Any) -> Any:
        result = original(d, request)
        if request.state == cases[0]["request"]["state"]:
            captured.append({q: list(v.probabilities) for q, v in result.answers.items()})
        return result

    replacement = exact_angles if a.mode == "tables" else RoPE.__call__
    with tempfile.TemporaryDirectory(prefix="ippon-rope-") as directory:
        path = Path(directory) / "reference.json.gz"
        path.write_bytes(gzip.compress(json.dumps(reference).encode()))
        with patch.object(RoPE, "__call__", replacement), patch.object(Decider, "_answer", answered):
            report = verify_model(a.model, path, **options)
            print(report.text(), flush=True)
            checks = None
            if a.invariance:
                with Decider.load(a.model, **options) as d:
                    checks = invariance(d, cases, float(source["bound"]), a.model)
                    print(checks.text(), flush=True)
    result = {
        "machine": describe(),
        "mode": a.mode,
        "options": options,
        "reference": str(a.reference),
        "reference_sha256": hashlib.sha256(a.reference.read_bytes()).hexdigest(),
        "case": a.case,
        "verify": report.to_json(),
        "probabilities": captured,
        "invariance": checks.to_json() if checks else None,
        "peak_active_bytes": mx.get_peak_memory(),
    }
    a.out.write_text(dumps(result, indent=2) + "\n")
    if not report.passed or (checks is not None and not checks.passed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
