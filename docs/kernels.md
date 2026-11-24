# Kernels chosen for your Mac's chip

There is no tuning step. When a model loads, ippon picks the kernel for each of its matrix multiplies
(`src/ippon/choices.py`). The choice depends on the chip (its GPU family and core count) and the matrix's shape and
format. It also depends on how many input rows a request multiplies by the matrix. Nothing is measured on your Mac, and nothing is
stored there. `ippon doctor` says which choices apply. `--switch gemm=mlx` turns them all off and runs MLX's own
matmuls, which are the reference path.

## Why the choice matters

The MLX 0.32.4 baseline uses a fixed tile for each shape, which is 64 × 128 output values per threadgroup for an M5
dense matmul. A decision request multiplies a few dozen to a few thousand rows by each matrix. At a few hundred rows,
those tiles leave GPU cores idle. On the M5 Pro's 20 cores, ippon timed every tile. Where MLX's tile filled the GPU
fewer than 4 times over, a better tile existed in 85 to 100% of the row bands. Where it filled the GPU 16 times or
more, a better tile existed in 23% of the row bands.

ippon's matmul kernels (`kernels/gemm.py`, `kernels/qgemm.py`) are MLX's own kernels, with the tile, the rows per
SIMD group and a split of K as parameters. So they fill the GPU where MLX's fixed tile does not.

## What runs where

| Mac | Quantized weights (4- to 8-bit affine, MXFP8, MXFP4 or NVFP4) | Full-precision weights (fp16, bf16) |
|---|---|---|
| any M5 (base, Pro, Max, Ultra) | ippon's quantized kernel from 16 rows, with MXFP8 unpacked to full precision from 4,096 | on an M5 with 20 GPU cores (the M5 Pro), the thin kernel to 160 rows, a 64 × 32 tile to 384, then MLX's matmul or, for a down projection, a 32 × 64 tile. Other M5s use MLX's matmul |
| M1 and M2 | MLX's quantized matmul, with MXFP8 unpacked to full precision from 224 | MLX's matmul |
| M3 | MLX's quantized matmul | MLX's matmul |
| M4, and an M5 before macOS 26.2 | MLX's quantized matmul, with MXFP8 unpacked to full precision from 1,024 | MLX's matmul |

Every Mac runs ippon's other kernels and its planning either way. These are the fused norm, GLU and convolution
kernels and one reading of a state for all of a request's questions. They also include the last layer for the answer
rows only, and kept prefixes ([architecture.md](architecture.md)).

The choices also need an MLX of the minor version they were fitted on (`profiles.FITTED_MLX`, 0.32.4). ippon's
kernels are MLX's own with other tiles, and which tile wins moves with MLX. Any MLX 0.32 keeps the choices. Only an
install outside ippon's range brings another minor version. With that version, ippon runs MLX's own matmuls, and
`ippon doctor` says so.

The same check covers the other facts measured on MLX's kernels. These facts are which attention calls MLX fuses
itself and which Gated DeltaNet head counts its kernels take. Under another MLX, ippon assumes neither.

The rule (`choices.py`) was fitted on the M5 Pro (20 GPU cores, MLX 0.32.4, 27 and 28 October 2026). The data came
from `scripts/measure_kernels.py`, which timed every variant of every matrix shape of the README's builds at 16 to
4,096 rows. It timed the quantized shapes again at 1,024 to 16,384 rows.

- For quantized weights, the variant `(64, 16, 1)` runs from 16 rows. This variant has 64 rows per threadgroup, 16
  per SIMD group and K whole. For MXFP8, `(128, 16, 1)` runs from 256 rows while the input is at most 24 MiB (3,072
  rows of a 4,096-wide matrix). Past that size, `(128, 16, 1)` fell 5 to 50% behind MLX's quantized matmul. Fewer
  than 16 rows run MLX's quantized matmul.

  From 4,096 rows, MXFP8 weights are unpacked to full precision on every call. MLX's matmul then multiplies them as a
  full-precision matrix of their shape. Across 13 shapes, this was a median 8.4% faster than the kernel at 4,096 rows
  and 12 to 13% faster from 8,192 rows. It took 7.9% longer at 1,024 rows and was even at about 2,560. It was also 12
  to 14% faster than MLX's quantized matmul.

  FP16 MXFP8 matrices with a 2,560-wide input or output take the wide tile from 192 rows. On the M5 Pro, d3-mini
  MXFP8's 203-token median fell from 71.5 to 70.0 ms, and its 243-token median fell from 79.3 to 76.8 ms (3.2%).
  These results come from 8 fresh-process rounds with no regressing cell. When the wide tile started at 65 rows
  instead, 104-token requests took 12% longer (40.0 to 44.8 ms). So the smaller tile stays below 192 rows
  ([Batching evidence](../references/batching/README.md)).
- For full-precision weights, MLX's thin kernel runs from 65 to 160 rows and reads the weights once for all its
  rows. It splits K in 2 when its threadgroups fill the GPU's cores fewer than 16 times. A 64 × 32 tile runs to 384
  rows. Above that, MLX's matmul runs. For a down projection (K at least 2.5 times N), a 32 × 64 tile runs instead,
  and it is a median 8% faster there.

  Past 4,096 rows, that tile alone fell up to 30% behind MLX's matmul from 120 MiB of input. But whole requests with
  MLX's matmul there took 1 to 5% longer with 8 clients at once, and the same time with 50 questions. So the rule
  keeps the tile. A fused gate/up runs the thin kernel, the 64 × 32 tile, and from 385 rows a 32 × 32 tile. Up to
  16,384 rows, the 32 × 32 tile was 7 to 18% faster than MLX's matmul and the GLU kernel on 3 of 4 shapes. At worst,
  it took 4% longer.

Where one kernel overtakes another depends on a core's memory bandwidth against its arithmetic. This balance is about
the same on every M5 chip. How well a kernel fills the GPU depends on the chip's core count, and the rule reads that
count from the chip. The tables below show how the rule compared on the M5 Pro, per matrix.

| | Full precision (255 cells, 17 shapes) | |
|---|---:|---:|
| | median, against each cell's fastest variant | 90th percentile |
| the rule | 1.7% longer | 8.9% longer |
| the former per-Mac tuning's choices | 1.7% longer | 10.1% longer |
| MLX's matmul | 8.9% longer | 25.5% longer |

| Quantized format | Cells | `(64, 16, 1)` against the best of MLX's paths, median | Longer in |
|---|---:|---:|---:|
| 8-bit, groups of 32 | 216 | 16 to 20% faster | 5 cells, at most 6% |
| 8-bit, groups of 64 | 144 | 19 to 21% faster | 1 cell |
| MXFP8 | 156 | 6 to 10% faster (the rule, with `(128, 16, 1)` from 256 rows, was a median 10% faster and took longer in 2 cells by at most 0.4%) | 6 cells, at most 3% |

MLX's paths are its quantized matmul and, for the affine formats, its rows given as 2 halves of a batch and its
weights unpacked to full precision. Each path was timed from 16 to 4,096 rows, with the plain multiply and the fused
gate/up together.

At 3,072 to 16,384 rows, the m5-2 rule was a median 3% faster than MLX's quantized matmul for MXFP8 and 10% faster for
affine 8-bit. Unpacking pays only for MXFP8. For affine 8-bit, unpacking was a median 1 to 2% faster from 12,288 rows
and slower below 8,192 rows. The MXFP8 kernel trailed a matmul with weights unpacked beforehand by 8 to 19%.

This table gives the time that the choices save on whole requests on the M5 Pro, against MLX's own matmuls
(`--switch gemm=mlx`). The runs were fresh-process A/Bs of 4 rounds, with 100 and 1,000 words and 1 and 8 questions.

| Model | Time saved |
|---|---:|
| d1-3B, Q8_0 GGUF | 17 to 28% |
| Clef-Flash 9B, MLX 8-bit | 10 to 18% |
| d1-omni-600M | 7 to 23% |
| d1-3B | 6 to 25% |
| pplx-decider flash v1.2 | 4 to 6% |

Liquid's d1-3B and d1-omni-600M (LFM2) were measured before ippon dropped them for d1-2B and d1-Omni-2B. The former
per-Mac tuning stored its own choices. Against those choices, the rule was no different on Clef-Flash 8-bit, and
within 3% on d1-3B and pplx-decider flash. On each of these 2 models, the rule was slower in one cell of 161 to 384
rows.

Full-precision matrices take ippon's tiles only on an M5 with the M5 Pro's 20 GPU cores, where the tiles were
measured (rule `m5-6`). The base M5 (10 GPU cores) was measured on 3 November 2026. There, the same tiles made whole
requests take 10 to 24% longer on d3-mini, H2O-Lightning-4B, pplx-decider flash and Torchcast, at 1,000 words and
more. On d1-2B
and d1-Omni-2B, the tiles were neutral to 20% faster.

Every M5 has about 15 GB/s of memory bandwidth per GPU core. So the difference comes from what Apple does not
publish, such as cache sizes. ippon chooses a chip's tiles only after that chip is measured. Any other M5 runs MLX's
matmul for full-precision matrices, which cannot be slower than itself. Quantized matrices take ippon's kernels on
every M5, and d1-2B's Q8_0 was up to 28% faster on the base M5.

M1 to M4 run MLX's matmuls. ippon's quantized kernel and its fastest tiles use the M5's neural accelerators, which
earlier chips do not have. No measurement on those chips shows another of its tiles helping there yet. The one
exception is MXFP8 on M1, M2 and M4.

On M1 and M2, MLX's MXFP8 matmul took 34% (M1) and 28% (M2 Pro) longer on d1-2B's 1,100-token prompts than the same
model in fp16. It took 44% and 16% longer than the model's Q8_0, which reads as many bytes. So from 224 rows, an
MXFP8 matrix is unpacked to full precision on every call and multiplied by MLX's matmul, as an M5 does from 4,096.

On an M1 (7 GPU cores, `scripts/measure_kernels.py --kinds qgemm,qglu --bound --rule`), d1-2B's 6 MXFP8 shapes took
6.3% longer that way at 128 rows. They took 5.0% less at 192 rows and 22 to 26% less from 1,024 rows, within 2 to 5%
of weights unpacked beforehand. Whole requests on d1-2B's MXFP8 build cross later. They took 6% longer at 205 rows
and 7% less time at 235 rows, hence the threshold of 224. They took 17 to 19% less time at 1,000 words and 22 to 25%
less at 500, with answers identical to every digit.

On a 48 GB M4 Pro, the same sweep on d1-2B's and d3-mini's MXFP8 builds took 18 to 28% longer at 256 rows. It took 2
to 6% less from 1,024 rows and 10 to 14% less at 4,096, so the M4 unpacks from 1,024. The M3 keeps MLX's quantized
matmul until an M3 is measured. `ippon suite` measures the choices on any Mac it runs on (its `ab-kernels` step), and
`--switch gemm=mlx` turns them off.

On 30 October 2026, the M5 Pro (20 GPU cores, 48 GiB) ran the same 300 quantized operation/row cells on 2 MLX
versions. These were MLX 0.32.4 and unmodified commit `065ec4683` (`0.32.5.dev20261030+065ec4683`, no containing
release tag). The cells covered Clef-Flash MXFP8 and affine 8-bit, d3-mini, d3-flash and d1-2B MXFP8 shapes at 16 to
4,096 rows. On the newer MLX, ippon's 16-row MXFP8 rule took a median 19.7% longer, against 31.6% faster on 0.32.4.
At 32 to 4,096 rows, it stayed 6.2 to 17.9% faster by row band. Affine plain multiplies took a median 0.6% longer,
while the fused GLU stayed 2.2% faster.

The 4 completed whole-request A/Bs all rejected `gemm=mlx`. Median times increased 6.7 to 16.4% on d3-mini, 2.5 to
14.1% on d3-flash, 8.3 to 16.6% on d1, and 1.3 to 4.4% on affine Clef. For affine Clef, the median intervals overlap,
so the verdict comes from the p95 tails. Clef MXFP8 stopped in round 1 on screen unlock and was not repeated. d3-mini
flagged one cell for background VM activity.

The rule then stayed at `m5-5`, because a narrower version-gated rule had no promotion verdict. The current rule is
`m5-7`. The NAX quantized and gated-delta
kernels already existed in 0.32.4. This range adds staged affine split-K and the quantized route for 6 to 16 rows.
[Per-band tables, raw measurements, upstream commits and limits](../references/kernels/mlx-065ec46/README.md).

## Numbers

The choices keep the published weights, and packed formats keep their encoding. Products accumulate in fp32 before
the output cast to the compute dtype. So answers move only by the order of those sums. In the fused gate/up kernel,
answers also move by one rounding. That kernel applies the activation to the fp32 sums and rounds once, where the
unfused path rounds each matmul first. `ippon verify` checks a model with its choices applied.

## Refitting the rule (maintainers)

`uv run python scripts/measure_kernels.py --out DIR MODEL ...` times every variant of every shape of the given models
on the Mac it runs on. It runs one model per process, and it resumes.

1. If a kernel or MLX changes, or you have measurements from another chip, refit the thresholds of `choices.py` from
   the script's output.
2. Raise `choices.RULE`.

Add `--rule` to time the installed packed-matrix dispatch in every precise round, alongside the candidate variants.
This dispatch includes fused GLUs and unpacking. Each record includes the MLX version and the rule's dispatch bands.

## Every Metal source

The kernels are `.metal` files in `src/ippon/kernels/metal/`, and the Python launchers beside them assemble them.
Each source below has a production caller, except `gemm_simd.metal` and `gemm_glu_simd.metal`. No kernel choice
gives a SIMD-group tile, so only `scripts/measure_kernels.py` and `tests/test_gemm.py` run those 2. The
[audit of 29 October 2026](../references/kernel-audit/README.md) traced the sources of that day to their callers.

| Sources | When it runs | Launcher and caller |
|---|---|---|
| `gemm.metal`, `gemm_glu.metal`, `gemm_thin.metal`, `gemm_tile_loop.metal` | an M5, at rows the rule gives a tile | `gemm.gemm`, `gemm.glu_gemm` → `Ops.mm`, `Ops.glu_mm` |
| `gemm_simd.metal`, `gemm_glu_simd.metal` | when `scripts/measure_kernels.py` (on M1 to M4, or with `--simd`) or `tests/test_gemm.py` picks a SIMD-group tile | `gemm.gemm`, `gemm.glu_gemm` with a tile of `gemm.simd_candidates` |
| `qgemm.metal`, `qgemm_staged.metal`, `qgemm_affine.metal`, `qgemm_affine_loader.metal`, `qgemm_fp.metal`, `qgemm_fp_loader.metal` | an M5, quantized weights (4- to 8-bit affine, MXFP8, MXFP4 or NVFP4) from 16 rows | `qgemm.qmm`, `qgemm.glu_qmm` → `quant.mm`, `quant.glu_mm` |
| `glu.metal` | every Mac | `glu.glu` → `Ops.glu`, for dense and expert MLPs, and `Ops.glu_mm` without a tile |
| `activation_silu.metal`, `activation_gelu.metal`, `activation_gelu_tanh.metal`, `activation_identity.metal`, `erf.metal` | inside the GLU and matmul kernels | `glu.activation_source`, with SiLU (dense, Qwen3.5, LFM2 and LFM3), exact GELU (ModernBERT), tanh GELU (Gemma 3 and 4), none for a plain multiply |
| `norm.metal`, `norm_add_store.metal`, `norm_add_sum.metal`, `norm_weight_at.metal` | every Mac | `norm.add_norm` → `Ops.norm`, `Ops.add_norm`, for decoders, encoders, vision towers and grouped norms |
| `norm_gated.metal`, `conv_causal.metal`, `delta.metal` | every Mac | `norm.gated_norm`, `conv.causal_conv`, `delta.delta_inputs` → Qwen3.5's DeltaNet layers |
| `conv_gated.metal` | every Mac | `conv.gated_conv` → LFM2 and LFM3 mixers |
| `norm_rope.metal` | `--switch norm_rope=kernel` | `norm.norm_rope` → `Ops.norm_rope` |

ippon's fused attention kernels were removed on 2 November 2026. End to end, they were never faster than MLX's
attention. On the M5 Pro, pplx-decider flash at 4,096 tokens measured 0.89%, and an M1 at 1,000 and 4,000 words
showed no difference. Also, `attention=bounded` saves the same memory on long prompts
([accuracy.md](accuracy.md#attention-at-head-size-256-switch-attention)).
