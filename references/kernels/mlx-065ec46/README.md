# MLX 065ec46 against the M5 quantized rule

ippon keeps rule `m5-5` with MLX 065ec46, because no completed whole-request comparison promotes the all-MLX arm.
The measurements ran once on 30 October 2026 on an Apple M5 Pro with 20 GPU cores and 48 GiB, macOS 27.0.1 and
Python 3.12.15. The unmodified upstream checkout built successfully in an isolated environment and reports
`0.32.5.dev20261030+065ec4683`. Its metallib contains the quantized NAX and gated-delta NAX symbols listed in
[protocol.json](protocol.json). No existing environment or dependency requirement changed.

The new environment has ippon `c4c2bd4`, and the existing MLX 0.32.4 environment has ippon `d3406e1`. Their choices,
quantization, matmul launchers and Ops sources are byte-identical ([hashes](kernel-code-hashes.json)), and both use
rule `m5-5`.

The source identifies itself as `v0.32.4-31-g065ec4683`. No fetched release tag contains it, so its source version
is the unreleased 0.32.5 development line. This evidence does not show that a later release keeps these timings.

## Upstream changes

The [complete 31-commit list](mlx-commits.txt) includes these relevant changes.

- [`bff10220c`](https://github.com/ml-explore/mlx/commit/bff10220c) adds a staged affine NAX matmul with split-K.
- [`c6abf0dbb`](https://github.com/ml-explore/mlx/commit/c6abf0dbb) adds quantized MMA/NAX routes for 6 to 16 rows with large N/K.
- [`007478f1a`](https://github.com/ml-explore/mlx/commit/007478f1a) widens the affine wide-vector row tiles to eight.
- [`52a6c234a`](https://github.com/ml-explore/mlx/commit/52a6c234a) makes FP4 unpacking faster, which is outside the supported 8-bit scope.
- [`065ec4683`](https://github.com/ml-explore/mlx/commit/065ec4683) and
  [`360078a8d`](https://github.com/ml-explore/mlx/commit/360078a8d) change NAX attention, separately from quantized matmul.

The NAX quantized kernels and `gated_delta_update_nax` already exist in the v0.32.4 tree. This range has no
gated-delta kernel change. `53c187057` fixes CUDA gather-QMM offsets, which is not the Metal path measured here.

## Kernel measurements

The five builds are `ollama://clef-flash:9b-mxfp8`, `ollama://d3:4b-mxfp8`, `ollama://d3:9b-mxfp8`,
`ollama://d1:2b-mxfp8`, and `mlx-community/clef-flash-8bit` (the local `clef-flash-mlx-8bit` checkout). All compute
in fp16. Affine weights are 8-bit in groups of 64, and MXFP8 uses groups of 32. There are 25 distinct
operation/shape records (21 plain multiplies and four fused GLUs), which gives 300 cells per runtime. D3-flash shares
all its measured shapes with Clef MXFP8, so the script does not time them again.

The row counts are 16, 32, 64, 96, 128, 192, 256, 384, 512, 1,024, 2,048 and 4,096. Each cell uses eight synthetic
copies in the encoding of the build. It runs four coarse repetitions after two warm-ups, then six alternating precise
rounds of the six fastest candidates plus every baseline. The `--rule` baseline calls the installed production Ops
path, including the chosen tile, fused GLU and 4,096-row MXFP8 unpacking. The other baselines are MLX's quantized
multiply, affine batch halves, unpacking on each call, and an already-unpacked full-precision upper bound
(`--bound`). Every production-rule output and candidate passed the bound of the script, which allows a maximum
absolute error of at most 1% of the largest absolute MLX reference output. These local operation checks are not
model verification.

The signed speed differences below are medians of `100 * (MLX_ms / rule_ms - 1)` over cells. A positive value means
ippon is faster, and a negative value means it is slower. Any negative difference counts as a slower cell, with no
significance threshold. Each comparison is within one runtime, and the tables do not compare absolute times between
runtime sweeps. The [script as measured](measurement-script.py.gz) matches the hash in `protocol.json`. Since that
measurement, the checked-in script changed only the help text of the new option.

| Format | Rows | Cells | vs MLX 065ec46 | Slower cells | vs MLX 0.32.4 | Slower cells |
|---|---:|---:|---:|---:|---:|---:|
| Affine 8-bit | 16 | 6 | -1.60% | 4 | +14.47% | 0 |
| Affine 8-bit | 32 to 128 | 24 | -0.10% | 12 | +34.46% | 0 |
| Affine 8-bit | 192 to 384 | 18 | -0.65% | 14 | +21.88% | 0 |
| Affine 8-bit | 512 to 2,048 | 18 | -0.51% | 15 | +14.16% | 0 |
| Affine 8-bit | 4,096 | 6 | -0.46% | 4 | +11.31% | 0 |
| MXFP8 | 16 | 19 | -19.72% | 18 | +31.61% | 1 |
| MXFP8 | 32 to 128 | 76 | +17.89% | 4 | +18.07% | 5 |
| MXFP8 | 192 to 384 | 57 | +6.17% | 11 | +6.14% | 8 |
| MXFP8 | 512 to 2,048 | 57 | +8.74% | 2 | +8.54% | 2 |
| MXFP8 | 4,096 | 19 | +11.98% | 1 | +12.09% | 1 |

The plain and fused operations differ enough that a rule change must consider them separately.

| Format / operation | Rows | Cells | vs MLX 065ec46 | Slower cells | vs MLX 0.32.4 | Slower cells |
|---|---:|---:|---:|---:|---:|---:|
| Affine 8-bit plain | 16 | 5 | -2.33% | 4 | +13.88% | 0 |
| Affine 8-bit plain | 32 to 128 | 20 | -0.46% | 12 | +38.20% | 0 |
| Affine 8-bit plain | 192 to 384 | 15 | -0.77% | 14 | +21.82% | 0 |
| Affine 8-bit plain | 512 to 2,048 | 15 | -0.55% | 15 | +13.01% | 0 |
| Affine 8-bit plain | 4,096 | 5 | -0.50% | 4 | +10.27% | 0 |
| Affine 8-bit fused GLU | 16 | 1 | +1.43% | 0 | +15.06% | 0 |
| Affine 8-bit fused GLU | 32 to 128 | 4 | +1.63% | 0 | +32.00% | 0 |
| Affine 8-bit fused GLU | 192 to 384 | 3 | +2.20% | 0 | +22.89% | 0 |
| Affine 8-bit fused GLU | 512 to 2,048 | 3 | +2.54% | 0 | +14.73% | 0 |
| Affine 8-bit fused GLU | 4,096 | 1 | +2.70% | 0 | +13.50% | 0 |
| MXFP8 plain | 16 | 16 | -22.29% | 15 | +43.36% | 1 |
| MXFP8 plain | 32 to 128 | 64 | +18.25% | 4 | +18.14% | 5 |
| MXFP8 plain | 192 to 384 | 48 | +5.90% | 10 | +5.75% | 7 |
| MXFP8 plain | 512 to 2,048 | 48 | +8.23% | 2 | +8.19% | 2 |
| MXFP8 plain | 4,096 | 16 | +11.50% | 1 | +11.50% | 1 |
| MXFP8 fused GLU | 16 | 3 | -17.13% | 3 | +13.50% | 0 |
| MXFP8 fused GLU | 32 to 128 | 12 | +13.43% | 0 | +14.95% | 0 |
| MXFP8 fused GLU | 192 to 384 | 9 | +8.65% | 1 | +8.98% | 1 |
| MXFP8 fused GLU | 512 to 2,048 | 9 | +11.19% | 0 | +11.16% | 0 |
| MXFP8 fused GLU | 4,096 | 3 | +20.56% | 0 | +21.64% | 0 |

The d1 matrix with N=3,088, K=2,048 has no fitting custom tile, so both routes call MLX below 4,096 rows, with
different Python dispatch overhead. The table includes it, but its negative cells are not custom kernel losses. At
4,096 rows, the rule unpacks MXFP8 on each call and uses its full-precision route, so those cells do not measure the
packed quantized kernel alone. This one sweep does not make the small affine differences statistically established
wins. [kernels-065ec46.jsonl.gz](kernels-065ec46.jsonl.gz) and [kernels-0.32.4.jsonl.gz](kernels-0.32.4.jsonl.gz)
hold all raw cell medians, variants and these dispatch annotations.

## Whole requests

A is the installed default rule, and B is `--switch gemm=mlx`. The A/B ran four alternating fresh-process rounds,
with 20 requests per cell in total and two warm-ups per process. The CLI `--sizes 300,500,1000` gives words, and the
tables below give the actual token counts. `gemm=mlx` also disables the full-precision choices and the fused
matmul/GLU path. These results therefore do not isolate a hypothetical change only at 16 rows or only for affine
plain matmuls.

All jobs held both shared locks and started in High Power and on AC, with a locked screen and the GPU at 0 to 1% for
20 s. The first Clef MXFP8 job was stopped when the screen unlocked during round 1, arm B. It has no complete report
or four-round verdict, and on the user's instruction it was not restarted.

| Build | Verdict for all-MLX | Median time increase, six cells | Caveat |
|---|---|---:|---|
| [d3-mini MXFP8](ab-d3-mini-mxfp8.json.gz) | reject | +6.7% to +16.4% | Background VM flag in round 4 A, 500 words / 3 questions (61% CPU). |
| [d3-flash MXFP8](ab-d3-flash-mxfp8.json.gz) | reject | +2.5% to +14.1% | No busy-process flag. |
| [d1-2B MXFP8](ab-d1-mxfp8.json.gz) | reject | +8.3% to +16.6% | No busy-process flag. |
| [Clef-Flash affine 8-bit](ab-clef-affine8.json.gz) | reject | +1.3% to +4.4% | Median intervals cross zero, so the verdict comes from the p95 tails. |
| Clef-Flash MXFP8 | Incomplete | none | Stopped in round 1 and not repeated. |

| Build | Words | Questions | Tokens | Default median ms | All-MLX median ms | Time change | 95% interval |
|---|---:|---:|---:|---:|---:|---:|---:|
| d3-mini MXFP8 | 300 | 1 | 403 | 133.18 | 152.14 | +14.2% | +13.9% to +14.4% |
| d3-mini MXFP8 | 300 | 3 | 1,205 | 363.12 | 396.24 | +9.1% | +8.9% to +9.4% |
| d3-mini MXFP8 | 500 | 1 | 603 | 182.44 | 212.34 | +16.4% | +16.0% to +16.7% |
| d3-mini MXFP8 | 500 | 3 | 1,805 | 547.91 | 598.74 | +9.3% | +8.1% to +10.4% |
| d3-mini MXFP8 | 1,000 | 1 | 1,103 | 336.01 | 376.46 | +12.0% | +11.1% to +12.9% |
| d3-mini MXFP8 | 1,000 | 3 | 3,305 | 1020.38 | 1089.13 | +6.7% | +5.3% to +8.6% |
| d3-flash MXFP8 | 300 | 1 | 403 | 236.84 | 268.00 | +13.1% | +12.9% to +13.4% |
| d3-flash MXFP8 | 300 | 3 | 1,205 | 649.44 | 692.87 | +6.7% | +6.6% to +7.2% |
| d3-flash MXFP8 | 500 | 1 | 603 | 327.51 | 373.66 | +14.1% | +13.9% to +14.6% |
| d3-flash MXFP8 | 500 | 3 | 1,805 | 979.90 | 1064.22 | +8.6% | +7.3% to +10.2% |
| d3-flash MXFP8 | 1,000 | 1 | 1,103 | 602.29 | 668.46 | +11.0% | +9.5% to +12.7% |
| d3-flash MXFP8 | 1,000 | 3 | 3,305 | 1882.91 | 1930.16 | +2.5% | +1.2% to +4.3% |
| d1-2B MXFP8 | 300 | 1 | 335 | 43.30 | 50.47 | +16.6% | +16.3% to +16.8% |
| d1-2B MXFP8 | 300 | 3 | 406 | 54.90 | 59.47 | +8.3% | +7.7% to +8.6% |
| d1-2B MXFP8 | 500 | 1 | 535 | 67.07 | 74.75 | +11.4% | +10.8% to +11.7% |
| d1-2B MXFP8 | 500 | 3 | 606 | 71.86 | 82.76 | +15.2% | +15.0% to +15.7% |
| d1-2B MXFP8 | 1,000 | 1 | 1,035 | 124.73 | 137.76 | +10.5% | +10.3% to +10.6% |
| d1-2B MXFP8 | 1,000 | 3 | 1,106 | 132.63 | 147.11 | +10.9% | +10.4% to +11.1% |
| Clef-Flash affine 8-bit | 300 | 1 | 456 | 241.73 | 244.86 | +1.3% | -0.2% to +3.5% |
| Clef-Flash affine 8-bit | 300 | 3 | 610 | 316.46 | 323.97 | +2.4% | -0.8% to +5.9% |
| Clef-Flash affine 8-bit | 500 | 1 | 656 | 341.02 | 353.81 | +3.8% | -0.6% to +8.1% |
| Clef-Flash affine 8-bit | 500 | 3 | 810 | 419.89 | 436.12 | +3.9% | -1.0% to +8.8% |
| Clef-Flash affine 8-bit | 1,000 | 1 | 1,156 | 608.34 | 635.09 | +4.4% | -0.6% to +10.0% |
| Clef-Flash affine 8-bit | 1,000 | 3 | 1,310 | 688.15 | 709.08 | +3.0% | -0.6% to +8.7% |

## Decision and limits

Keep `m5-5`. The new MLX wins the 16-row MXFP8 kernel cells and mostly closes the affine plain-matmul gap, but no
completed whole-request comparison promotes the all-MLX arm. A narrower version-gated rule has no end-to-end test,
and the requested request sizes do not directly test the 16-row boundary. The evidence therefore does not meet the
repository's promotion bar for such a source change, and ippon does not win every cell. Keep the kernels, including
the fused GLUs, and the current dependency range. The version-gated adjustment stays unvalidated.

No additional A/B rounds or repeated timings ran. No `ippon verify` ran, because dispatch and computation did not
change. Formatting, lint, mypy and the full suite passed, with 932 tests and no skips or failures
([validation](validation.json)). The suite ran after timing, under both shared locks, without syncing environments.

To reproduce the kernel commands in separately prepared runtimes, set `MODELS` to the five identifiers above. Set
`OUT` to a fresh directory for each runtime. Then run this command with the Python of the applicable environment.

```sh
python scripts/measure_kernels.py --out "$OUT" --kinds qgemm,qglu --bound --rule $MODELS
```

The whole-request command ran once per build in the isolated new runtime.

```sh
ippon bench "$MODEL" --ab "" "--switch gemm=mlx" --sizes 300,500,1000 --questions 1,3 --rounds 4 --out "$OUT"
```

Run these commands only with exclusive GPU access, High Power, AC, a locked screen and an idle machine. They
describe the measurements. Do not silently resume the interrupted run or count it as a completed verdict. The
published evidence abbreviates the home directory to `~`. All numeric measurements are unchanged, and
`protocol.json` records the hashes of the original files before that abbreviation.
