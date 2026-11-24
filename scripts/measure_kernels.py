"""Time every matmul kernel variant for every matrix shape of the given models on this Mac: the measurements
choices.py's rule is fitted to and checked against (docs/kernels.md). A maintainer's tool: users never run it.

    uv run python scripts/measure_kernels.py --out kernels-m5pro MODEL [MODEL ...]

Each model loads in a process of its own (MLX's own matmuls, `gemm=mlx`). For each distinct shape it has
(full-precision matrices and fused GLUs on the neural accelerators, packed ones with kernels/qgemm.py), at each
row count: every fitting variant timed in a chain of distinct weight copies (4 repetitions after 2 warm-ups,
`coarse`), then the 6 fastest and every baseline alternately for 6 rounds (`precise`, medians). Baselines: MLX's
matmul (with a GLU, then the fused GLU kernel); packed ones also unpacked to full precision on every call (`dense`,
the unpack timed) and, affine ones, as two halves; with `--bound`, also multiplied by copies unpacked beforehand
(`fp16`: the dense product alone, the most an unpacking path could reach). `--rule` also times the installed
choices.py's packed-matrix production path, including fused GLUs and unpacking, in every precise round. A
variant whose product differs from the baseline's is listed under `failed`. One JSON line per shape, in
OUT/<model>.jsonl; shapes already there are skipped, so a run resumes. `--rows` replaces the row counts and
`--kinds` limits the kinds measured (gemm, glu, qgemm, qglu), e.g. to extend the packed ones to longer requests. On
M1 to M4 the tiles are MLX's SIMD-group library's (kernels/gemm.py `simd_candidates`), and packed matrices time only
their baselines and `--rule` (kernels/qgemm.py needs the neural accelerators); `--simd` measures those tiles on an M5
too (to check the script, not to fit M1 to M4's choices).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

GEMM_ROWS = (64, 96, 128, 160, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)
QGEMM_ROWS = (16, 32, 64, 96, 128, 192, 256, 384, 512, 1024, 2048, 4096)
COPIES, TOP, ROUNDS = 8, 6, 6


def measured(out: Path) -> set[tuple[str, str]]:
    return {
        (row["kind"], row["key"])
        for path in out.glob("*.jsonl")
        for line in path.read_text().splitlines()
        if line.strip() and (row := json.loads(line))
    }


def worker(
    model: str,
    out: Path,
    rows: Sequence[int] | None,
    kinds: set[str],
    bound: bool = False,
    rule: bool = False,
    simd: bool = False,
) -> None:
    import mlx.core as mx

    from ippon import Decider, choices, profiles
    from ippon.arch.ops import FUSED
    from ippon.kernels import gemm, qgemm
    from ippon.quant import NVFP4_MAX, QuantizedMatrix, mm, scaled

    def halves(x: mx.array, w: QuantizedMatrix) -> mx.array:
        """x @ w.T with x's rows given to MLX's quantized matmul as a batch of two halves (an odd count padded with
        a zero row): a batch never takes MLX's split-K route on SIMD-group kernels for a narrow matrix."""
        k = x.shape[-1]
        m = x.size // k
        flat = x.reshape(m, k)
        multiplier = w.multiplier()
        if w.mode == "nvfp4" and multiplier is not None:
            flat = flat.astype(mx.float32)
        if m % 2:
            flat = mx.concatenate([flat, mx.zeros((1, k), flat.dtype)])
        biases = w.biases[None] if w.biases is not None else None
        y = mx.quantized_matmul(flat.reshape(2, -1, k), w.w[None], w.scales[None], biases, transpose=True,
                                group_size=w.group_size, bits=w.bits, mode=w.mode)  # fmt: skip
        n = y.shape[-1]
        return scaled(y.reshape(-1, n)[:m].reshape(*x.shape[:-1], n), multiplier).astype(x.dtype)

    nax = gemm.available() and not simd  # an M5: the neural accelerators' tiles; M1 to M4: MLX's SIMD-group library's
    if not nax and gemm.failure(simd=True) is not None:
        sys.exit("neither tile library compiles on this Mac")
    tiles = gemm.candidates if nax else lambda glu=False: gemm.simd_candidates(glu)
    if mx.__version__ != profiles.FITTED_MLX:  # every record says "mlx": the version a rule fitted to it records
        print(f"measuring on MLX {mx.__version__}: a rule fitted to these records sets profiles.FITTED_MLX to it "
              f"(now {profiles.FITTED_MLX})", flush=True)  # fmt: skip
    decider = Decider.load(model, switches="gemm=mlx")
    backbone = decider.backbone
    gemm_shapes, glu_shapes = choices.gemm_shapes(backbone), choices.glu_shapes(backbone)
    if not kinds & {"qgemm", "qglu"}:
        # full-precision shapes are all the sweep needs: the weights go, so the copies it times and every tile it
        # compiles have the GPU to themselves (on an 8 GB M1, d1-2B's 3.6 GB held beside them stalled the first shape)
        decider.close()
        del decider, backbone
        gc.collect()
        mx.clear_cache()
    seen = measured(out)
    rows_given = rows
    sink = (out / f"{Path(model).name.replace(':', '_')}.jsonl").open("a")

    def chain(f: Callable[..., mx.array], x: mx.array, ws: Sequence[Any], reps: int = 1, warm: int = 0) -> float:
        """ms per call of f(x, w) over the copies, each call's input depending on the one before (as the norms
        between a model's matmuls make it): no cache flatters a small product."""

        def once() -> list[mx.array]:
            a, made = x, []
            for w in ws:
                y = f(a, w)
                a = a * (1 + 0 * y[0, 0])
                made.append(y)
            return made

        for _ in range(warm):
            mx.eval(once())
        start = time.perf_counter()
        for _ in range(reps):
            mx.eval(once())
        return (time.perf_counter() - start) / reps / len(ws) * 1000

    def measure(
        kind: str,
        key: str,
        meta: dict[str, Any],
        rows: Sequence[int],
        ws: Sequence[Any],
        baselines: dict[str, Callable[..., mx.array]],
        candidates: Sequence[tuple[int, ...]],
        kernel: Callable[..., mx.array],
    ) -> None:
        if kind not in kinds:
            return
        rows = rows_given or rows
        if (kind, key) in seen:
            return
        print(f"{kind} {key}", flush=True)
        cells = []
        k = meta["k"]
        for m in rows:
            x = mx.random.normal((m, k)).astype(meta["compute"])
            mx.eval(x)
            ref = baselines["mlx"](x, ws[0]).astype(mx.float32)
            bound = 1e-2 * mx.max(mx.abs(ref))
            coarse: dict[str, float] = {}
            failed = []
            for tile in candidates:  # compile every variant before any is timed
                with contextlib.suppress(Exception):  # a variant the compiler or the GPU won't take
                    mx.eval(kernel(x, ws[0], tile))
            for name, base in baselines.items():
                if name == "rule" and (mx.max(mx.abs(base(x, ws[0]).astype(mx.float32) - ref)) > bound).item():
                    raise ValueError(f"production rule differs from MLX beyond the bound: {kind} {key}, {m} rows")
                coarse[name] = chain(base, x, ws, reps=4, warm=2)
            for tile in candidates:
                try:
                    if (mx.max(mx.abs(kernel(x, ws[0], tile).astype(mx.float32) - ref)) > bound).item():
                        failed.append(list(tile))
                        continue
                    coarse[str(tile)] = chain(lambda a, w, t=tile: kernel(a, w, t), x, ws, reps=4, warm=2)
                except Exception:
                    failed.append(list(tile))
            fastest = sorted((ms, name) for name, ms in coarse.items() if name not in baselines)[:TOP]
            arms = dict(baselines)
            for _, name in fastest:
                tile = tuple(int(v) for v in name.strip("()").split(", "))
                arms[name] = lambda a, w, t=tile: kernel(a, w, t)
            times: dict[str, list[float]] = {name: [] for name in arms}
            for r in range(ROUNDS):
                for name in list(arms) if r % 2 == 0 else reversed(list(arms)):
                    times[name].append(chain(arms[name], x, ws))
            precise = {name: statistics.median(t) for name, t in times.items()}
            cells.append({"rows": m, "coarse": coarse, "precise": precise, "failed": failed})
        meta = {k_: v for k_, v in meta.items() if k_ != "compute"}
        record = {"kind": kind, "key": key, **meta, "build": model, "time": dt.datetime.now().astimezone().isoformat(),
                  "mlx": mx.__version__, "rule": choices.RULE if rule else None}  # fmt: skip
        sink.write(json.dumps({**record, "cells": cells}) + "\n")
        sink.flush()
        seen.add((kind, key))

    for dtype, n, k in gemm_shapes:
        ws = [(mx.random.normal((n, k)) * 0.02).astype(dtype) for _ in range(COPIES)]
        mx.eval(ws)
        measure(
            "gemm",
            gemm.key(dtype, n, k),
            {"n": n, "k": k, "dtype": str(dtype), "compute": dtype},
            GEMM_ROWS,
            ws,
            {"mlx": lambda a, w: a @ w.T},
            [t for t in tiles() if gemm.fits(t, n, k)],
            lambda a, w, t: gemm.gemm(a, w, t),
        )
    for dtype, n, k, act in glu_shapes:
        ws = [(mx.random.normal((n, k)) * 0.02).astype(dtype) for _ in range(COPIES)]
        mx.eval(ws)
        measure(
            "glu",
            gemm.key(dtype, n, k, act),
            {"n": n, "k": k, "dtype": str(dtype), "act": act, "compute": dtype},
            GEMM_ROWS,
            ws,
            {"mlx": lambda a, w, act=act: FUSED.glu(a @ w.T, act)},
            [t for t in tiles(glu=True) if gemm.fits(t, n // 2, k)],
            lambda a, w, t, act=act: gemm.glu_gemm(a, w, t, act),
        )

    def copies(w: QuantizedMatrix) -> list[QuantizedMatrix]:
        """Synthetic copies in `w`'s own format: its scale layout and convention, a non-unity multiplier."""
        scale = (
            None
            if w.global_scale is None
            else mx.full(w.global_scale.shape, 0.75 * (NVFP4_MAX if w.global_scale_is_amax else 1), mx.float32)
        )
        per_row = (
            None
            if w.row_multiplier is None
            else mx.linspace(0.5, 1.5, w.row_multiplier.size, dtype=mx.float32).reshape(w.row_multiplier.shape)
        )
        made = []
        n, k = w.shape
        for _ in range(COPIES):
            weights = (mx.random.normal((n, k)) * 0.02).astype(w.dtype)
            if w.mode == "affine":
                packed, scales, biases = mx.quantize(weights, group_size=w.group_size, bits=w.bits)
            else:
                packed, scales, *_ = mx.quantize(weights, group_size=w.group_size, bits=w.bits, mode=w.mode)
                biases = None
            made.append(replace(w, w=packed, scales=scales, biases=biases, global_scale=scale, row_multiplier=per_row))
        mx.eval([c.arrays() for c in made])
        return made

    if kinds & {"qgemm", "qglu"}:  # packed matrices: the loaded model's own
        gated = backbone.gated_mlp
        packed: dict[str, tuple[QuantizedMatrix, str | None]] = {}
        for layer in backbone.layers:
            for name, value in layer.items():
                for w in choices._packed(value):
                    if isinstance(w, QuantizedMatrix) and choices.staged(w):
                        glu = gated[1] if gated is not None and name == gated[0] and w is value else None
                        packed.setdefault(w.key, (w, glu))
        selected: dict[str, QuantizedMatrix] = {}
        rule_ops = FUSED
        if rule:
            profile = replace(decider.profile, switches=replace(decider.profile.switches, gemm="kernel"))
            rule_ops = replace(FUSED, gemm_tiles=choices.attach(backbone, profile))
            for layer in backbone.layers:
                for value in layer.values():
                    for w in choices._packed(value):
                        selected.setdefault(w.key, w)
        for key, (w, glu) in packed.items():
            n, k = w.shape
            qs = copies(w)
            meta = {"n": n, "k": k, "dtype": str(w.dtype), "mode": w.mode, "bits": w.bits, "group": w.group_size}
            meta["compute"] = w.dtype
            bases: dict[str, Callable[..., mx.array]] = {"mlx": mm, "dense": lambda a, q: a @ q.dense(a.dtype).T}
            if w.mode == "affine":
                bases["halves"] = halves
            if bound:
                unpacked = {id(q): q.dense(w.dtype) for q in qs}
                mx.eval(list(unpacked.values()))
                bases["fp16"] = lambda a, q, u=unpacked: a @ u[id(q)].T
            if rule:
                chosen = selected[key]
                rule_copies = {
                    id(q): replace(q, tiles=chosen.tiles, glu=chosen.glu, unpack_rows=chosen.unpack_rows) for q in qs
                }
                bases["rule"] = lambda a, q, copies=rule_copies: rule_ops.mm(a, copies[id(q)])
                meta["rule_tiles"] = chosen.tiles
                meta["rule_glu"] = chosen.glu
                meta["rule_unpack_rows"] = chosen.unpack_rows

            def qmm(a: mx.array, q: QuantizedMatrix, t: tuple[int, int, int]) -> mx.array:
                return qgemm.qmm(a, q.w, q.scales, q.biases, q.bits, q.group_size, t, q.mode, q.multiplier())

            fitting = [t for t in qgemm.CANDIDATES if nax and qgemm.fits(t, n, k, w.bits, w.group_size, w.mode)]
            measure("qgemm", key, meta, QGEMM_ROWS, qs, bases, fitting, qmm)
            if glu is not None:

                def glu_qmm(a: mx.array, q: QuantizedMatrix, t: tuple[int, int, int], act: str = glu) -> mx.array:
                    return qgemm.glu_qmm(
                        a, q.w, q.scales, q.biases, q.bits, q.group_size, t, act, q.mode, q.multiplier()
                    )

                glued = {name: (lambda a, q, b=b, act=glu: FUSED.glu(b(a, q), act)) for name, b in bases.items()}
                if rule:
                    glued["rule"] = lambda a, q, copies=rule_copies, act=glu: rule_ops.glu_mm(a, copies[id(q)], act)
                fits = [t for t in qgemm.CANDIDATES if nax and qgemm.glu_fits(t, n, k, w.bits, w.group_size, w.mode)]
                measure("qglu", f"{glu}:{key}", {**meta, "act": glu}, QGEMM_ROWS, qs, glued, fits, glu_qmm)
    sink.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("models", nargs="*", help="model paths or names, each measured in a process of its own")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--worker", help=argparse.SUPPRESS)
    p.add_argument("--rows", type=lambda v: [int(r) for r in v.split(",")], help="row counts, comma-separated")
    p.add_argument("--kinds", default="gemm,glu,qgemm,qglu", help="kinds to measure, comma-separated")
    p.add_argument("--bound", action="store_true", help="also time packed matrices unpacked beforehand (fp16)")
    p.add_argument(
        "--rule", action="store_true", help="also time choices.py's packed-matrix path in every precise round"
    )
    p.add_argument("--simd", action="store_true", help="the SIMD-group tiles of M1 to M4, even on an M5")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.worker:
        kinds = set(args.kinds.split(","))
        worker(args.worker, args.out.resolve(), args.rows, kinds, args.bound, args.rule, args.simd)
        return
    for model in args.models:
        print(f"{dt.datetime.now():%H:%M} {model}", flush=True)
        command = [sys.executable, __file__, "--worker", model, "--out", str(args.out), "--kinds", args.kinds]
        if args.rows:
            command += ["--rows", ",".join(map(str, args.rows))]
        if args.bound:
            command.append("--bound")
        if args.rule:
            command.append("--rule")
        if args.simd:
            command.append("--simd")
        if code := subprocess.run(command, env=os.environ, check=False).returncode:
            print(f"{model}: exit {code}", flush=True)


if __name__ == "__main__":
    main()
