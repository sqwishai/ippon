"""Run the usual creator-reference check and record blocked attention graphs and peak MLX allocations.

Use a fresh process per switch. This is a correctness check,
not a timing benchmark. Graph counts include warm-up and the availability probe. Compiled layers can replay a
graph without calling Python again, so these are not counts of GPU launches. Long query/key shapes establish that the
long references reached the selected path rather than passing solely through MLX's existing fused path.
Use normal Decider.load memory residency/cache limits: verify_model otherwise disables them. Record retained
buffers as well as active allocations so a small active high-water mark cannot conceal a large free-buffer pool.

    PYTHONPATH=src python scripts/verify_attention.py MODEL --reference FILE --switch attention=bounded --out FILE
"""

from __future__ import annotations

import argparse
import hashlib
import os
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

from ippon._portable import dumps


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--switch", default="attention=bounded")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--chunk", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    # isort: off
    import ippon  # before MLX, so requested chip emulation is applied at import
    import mlx.core as mx
    # isort: on

    from ippon import Decider, Request, Result, choices
    from ippon.device import detect
    from ippon.kernels import attention
    from ippon.verify import load_reference, verify_model

    reference = load_reference(args.reference)
    if any("probabilities" not in c for c in reference["requests"]):
        parser.error("every request needs numerical reference probabilities")
    if args.out.exists():
        parser.error("use a new output path; evidence is not overwritten")
    root = Path(ippon.__file__).resolve().parent
    sources = [
        root / "arch/ops.py",
        root / "arch/common.py",
        root / "profiles.py",
        root / "verify.py",
        root / "quant.py",
        root / "choices.py",
        *sorted((root / "kernels").glob("*.py")),
        *sorted((root / "kernels/metal").glob("*.metal")),
    ]
    hashes = {str(p.relative_to(root)): digest(p) for p in sources if p.is_file()}
    graphs: Counter[tuple[object, ...]] = Counter()
    original_bounded = attention.bounded
    original_answer = Decider.answer
    builds = []
    request_memory = []

    def recorded_answer(decider: Decider, request: Request) -> Result:
        if "precision" in decider.details:  # skip the preliminary warm-up before auto has chosen a dtype
            build = {key: decider.details.get(key) for key in ("precision", "peak_activation", "weights")}
            build["kernels"] = decider.details.get("kernels")
            if build not in builds:
                builds.append(build)
        result = original_answer(decider, request)
        if "precision" in decider.details:
            memory = {
                "input_tokens": result.response["usage"]["input_tokens"],
                "active_bytes": mx.get_active_memory(),
                "cached_bytes": mx.get_cache_memory(),
                "peak_active_bytes": mx.get_peak_memory(),
            }
            request_memory.append(memory)
            print(
                f"request {len(request_memory)}: {memory['input_tokens']} tokens, "
                f"active {memory['active_bytes'] / 2**30:.2f} GiB, "
                f"cached {memory['cached_bytes'] / 2**30:.2f} GiB, "
                f"peak active {memory['peak_active_bytes'] / 2**30:.2f} GiB",
                flush=True,
            )
        return result

    def recorded_bounded(q: mx.array, k: mx.array, v: mx.array, scale: float) -> mx.array:
        graphs[("mlx-bounded", str(q.dtype), q.shape, k.shape, "none")] += 1
        return original_bounded(q, k, v, scale)

    options = {"pin_memory": True} | ({"chunk": args.chunk} if args.chunk is not None else {})
    mx.reset_peak_memory()
    with (
        patch.object(attention, "bounded", recorded_bounded),
        patch.object(Decider, "answer", recorded_answer),
    ):
        report = verify_model(args.model, args.reference, switches=args.switch, precision=args.precision, **options)
    evidence = {
        "schema": "ippon-attention-verification/1",
        "created": datetime.now(UTC).isoformat(),
        "device": asdict(detect()),
        "mlx": version("mlx"),
        "ippon": ippon.__version__,
        "switches": args.switch,
        "precision": args.precision,
        "actual_builds": builds,
        "load_options": options,
        "chunk": args.chunk,
        "reference": {"file": args.reference.name, "sha256": digest(args.reference)},
        "source_sha256": hashes,
        "runner_sha256": digest(Path(__file__)),
        "peak_active_mlx_bytes": mx.get_peak_memory(),
        "request_memory": request_memory,
        "memory_note": "Fresh-process peak MLX active allocations, including model loading and warm-up; not RSS.",
        "kernel_rule": choices.RULE,
        "environment": {k: os.environ.get(k) for k in ("IPPON_PROFILE", "IPPON_SWITCHES")},
        "prompt_lengths": {c["name"]: [len(p) for p in c["prompts"]] for c in reference["requests"]},
        "attention_graphs": [
            {"implementation": f, "dtype": dt, "q": q, "k": k, "mask": m, "count": n}
            for (f, dt, q, k, m), n in sorted(graphs.items())
        ],
        "graph_count_note": "Python graph builds, including probes/warm-up; compiled replay is not counted.",
        "report": report.to_json(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(report.text(), flush=True)
    print(f"{len(graphs)} distinct blocked graph shapes; evidence: {args.out}", flush=True)
    raise SystemExit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
