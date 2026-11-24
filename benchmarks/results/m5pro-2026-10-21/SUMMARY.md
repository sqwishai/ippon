# Recorded benchmark summary

## Recorded runtime comparisons

Measured 2026-10-21 on Apple M5 Pro, 48 GiB, macOS 27.0.1.

In-process ippon ran commit `7344347` with MLX 0.32.4.
The HTTP arms did not record the library commit or the MLX version, so that commit applies only to the in-process arms. These timings do not measure the current commit.

The typical cell is the measured nominal ~1,000-state-token request with one question. Input tokens include the model's prompt. Times are pooled medians in milliseconds.

| Model / build | Other runtime | Input tokens | Other ms | ippon ms | Speed change |
|---|---|---:|---:|---:|---:|
| Clef-Flash 9B · Q8_0 | llama.cpp b11757 | 1341 to 1345 | 1,033.2 | 827.3 | +25% |
| Clef-Flash 9B · MLX 8-bit | mlx-vlm 0.7.8 | 1341 to 1345 | 784.6 | 783.4 | 0% |

Speed change is `(other median / ippon median − 1) × 100%`, so a negative value means ippon is slower. It is not a reduction in latency. Short and Long are geometric means of the cell ratios at the short and long state sizes, with the 1-question and 3-question shapes weighted equally. The Short / long sizes column lists the nominal state sizes that were measured.

| Model / build | Other runtime | Short | Long | Short / long sizes | Max probability gap | Top answers changed |
|---|---|---:|---:|---|---:|---:|
| Clef-Flash 9B · Q8_0 | llama.cpp b11757 | +26% | +31% | 100, 300 / 1000, 2000, 4000 | 0.002609 | 0/400 |
| Clef-Flash 9B · MLX 8-bit | mlx-vlm 0.7.8 | 0% | +1% | 100, 300 / 1000, 2000, 4000 | 0.017600 | 0/400 |

Agreement uses every matched timed question. The probability gap is the largest absolute difference, not a percentage. A matching top answer does not imply matching probabilities.

## Older creators' code comparison

Measured 2026-10-21 on Apple M5 Pro, 48 GiB, macOS 27.0.1.

In-process ippon ran commit `7344347` with MLX 0.32.4.

The typical cell is the measured nominal ~1,000-state-token request with one question. Input tokens include the model's prompt. Times are pooled medians in milliseconds.

| Model / build | Other runtime | Input tokens | Other ms | ippon ms | Speed change |
|---|---|---:|---:|---:|---:|
| Clef-Flash 9B · full checkpoint | Cloudflare's code · Transformers 5.17.0 · PyTorch 2.14.1 | 1341 to 1345 | 2,899.8 | 758.4 | +3.8x |
| pplx-decider flash v1.2 · BF16 | Perplexity's package · Transformers 5.17.0 · PyTorch 2.14.0 · BF16 | 1281 to 1285 | 2,843.2 | 392.4 | +7.2x |
| d3-mini · full checkpoint | vllm-sr's server · Transformers 5.17.0 · PyTorch 2.14.1 | 1289 to 1293 | 2,550.5 | 401.1 | +6.4x |

Speed change is `(other median / ippon median − 1) × 100%`, so a negative value means ippon is slower. It is not a reduction in latency. Short and Long are geometric means of the cell ratios at the short and long state sizes, with the 1-question and 3-question shapes weighted equally. The Short / long sizes column lists the nominal state sizes that were measured.

| Model / build | Other runtime | Short | Long | Short / long sizes | Max probability gap | Top answers changed |
|---|---|---:|---:|---|---:|---:|
| Clef-Flash 9B · full checkpoint | Cloudflare's code · Transformers 5.17.0 · PyTorch 2.14.1 | +4x | +3.9x | 100, 300 / 1000, 2000, 4000 | 0.015249 | 0/200 |
| pplx-decider flash v1.2 · BF16 | Perplexity's package · Transformers 5.17.0 · PyTorch 2.14.0 · BF16 | +7.6x | +7.4x | 100, 300 / 1000, 2000, 4000 | 0.040343 | 1/200 |
| d3-mini · full checkpoint | vllm-sr's server · Transformers 5.17.0 · PyTorch 2.14.1 | +6.6x | +6.5x | 100, 300 / 1000, 2000, 4000 | 0.013899 | 0/200 |

Agreement uses every matched timed question. The probability gap is the largest absolute difference, not a percentage. A matching top answer does not imply matching probabilities.

## Per-shape evidence

### Clef-Flash 9B · Q8_0 against llama.cpp b11757

Measured 2026-10-21 on Apple M5 Pro, 48 GiB, macOS 27.0.1.

These arms did not record the library commit or the MLX version, so their timings do not measure the current commit.

ippon compute: fp16, 2 paired rounds.

| State size | Questions | Other ms | ippon ms | Saved ms | Speed change | Per-round speed change |
|---:|---:|---:|---:|---:|---:|---|
| 100 | 1 | 270.9 | 199.4 | 71.5 | +36% | +38%, +34% |
| 100 | 3 | 373.1 | 313.8 | 59.4 | +19% | +21%, +17% |
| 300 | 1 | 374.0 | 301.3 | 72.7 | +24% | +26%, +23% |
| 300 | 3 | 549.3 | 430.8 | 118.5 | +28% | +29%, +26% |
| 1000 | 1 | 1,033.2 | 827.3 | 205.9 | +25% | +26%, +24% |
| 1000 | 3 | 1,239.1 | 963.9 | 275.2 | +29% | +29%, +28% |
| 2000 | 1 | 1,841.9 | 1,423.8 | 418.1 | +29% | +30%, +29% |
| 2000 | 3 | 1,954.8 | 1,523.5 | 431.4 | +28% | +28%, +25% |
| 4000 | 1 | 3,985.1 | 2,888.8 | 1,096.3 | +38% | +38%, +39% |
| 4000 | 3 | 4,065.5 | 2,997.7 | 1,067.8 | +36% | +36%, +37% |

Recorded requests:

- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-21/r1/clef-flash-llama-Q8_0.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-21/r2/clef-flash-llama-Q8_0.json)
- [Recorded run 3](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-21/r1/clef-flash-gguf-q8-ippon-serve.json)
- [Recorded run 4](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-21/r2/clef-flash-gguf-q8-ippon-serve.json)

### Clef-Flash 9B · MLX 8-bit against mlx-vlm 0.7.8

Measured 2026-10-21 on Apple M5 Pro, 48 GiB, macOS 27.0.1.

In-process ippon ran commit `7344347` with MLX 0.32.4.

ippon compute: fp16, 2 paired rounds.

| State size | Questions | Other ms | ippon ms | Saved ms | Speed change | Per-round speed change |
|---:|---:|---:|---:|---:|---:|---|
| 100 | 1 | 192.7 | 193.6 | -0.9 | 0% | −1%, 0% |
| 100 | 3 | 303.5 | 302.7 | 0.8 | 0% | 0%, 0% |
| 300 | 1 | 291.7 | 290.9 | 0.9 | 0% | 0%, +1% |
| 300 | 3 | 410.8 | 411.8 | -0.9 | 0% | −1%, +1% |
| 1000 | 1 | 784.6 | 783.4 | 1.2 | 0% | −2%, +2% |
| 1000 | 3 | 923.8 | 926.7 | -2.9 | 0% | −1%, +1% |
| 2000 | 1 | 1,378.0 | 1,369.4 | 8.5 | +1% | 0%, +1% |
| 2000 | 3 | 1,485.7 | 1,468.0 | 17.7 | +1% | +2%, +1% |
| 4000 | 1 | 2,854.1 | 2,795.4 | 58.7 | +2% | +2%, +1% |
| 4000 | 3 | 2,955.0 | 2,929.7 | 25.3 | +1% | +2%, 0% |

Recorded requests:

- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-21/r1/clef-flash-mlx-8bit.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-21/r2/clef-flash-mlx-8bit.json)
- [Recorded run 3](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-21/r1/clef-flash-mlx-8bit-ippon.json)
- [Recorded run 4](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-21/r2/clef-flash-mlx-8bit-ippon.json)

### Clef-Flash 9B · full checkpoint against Cloudflare's code · Transformers 5.17.0 · PyTorch 2.14.1

Measured 2026-10-21 on Apple M5 Pro, 48 GiB, macOS 27.0.1.

In-process ippon ran commit `7344347` with MLX 0.32.4.

ippon compute: fp16, 1 paired rounds.

| State size | Questions | Other ms | ippon ms | Saved ms | Speed change | Per-round speed change |
|---:|---:|---:|---:|---:|---:|---|
| 100 | 1 | 663.5 | 165.8 | 497.7 | +4x | +4x |
| 100 | 3 | 1,132.8 | 268.1 | 864.7 | +4.2x | +4.2x |
| 300 | 1 | 1,030.0 | 265.3 | 764.6 | +3.9x | +3.9x |
| 300 | 3 | 1,521.4 | 379.2 | 1,142.1 | +4x | +4x |
| 1000 | 1 | 2,899.8 | 758.4 | 2,141.5 | +3.8x | +3.8x |
| 1000 | 3 | 3,425.9 | 889.2 | 2,536.7 | +3.9x | +3.9x |
| 2000 | 1 | 5,414.1 | 1,310.9 | 4,103.2 | +4.1x | +4.1x |
| 2000 | 3 | 5,591.5 | 1,385.3 | 4,206.2 | +4x | +4x |
| 4000 | 1 | 10,225.9 | 2,660.4 | 7,565.5 | +3.8x | +3.8x |
| 4000 | 3 | 10,620.8 | 2,900.1 | 7,720.7 | +3.7x | +3.7x |

Recorded requests:

- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-21/makers/clef-flash-tf.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-21/makers/clef-flash-ippon.json)

### pplx-decider flash v1.2 · BF16 against Perplexity's package · Transformers 5.17.0 · PyTorch 2.14.0 · BF16

Measured 2026-10-21 on Apple M5 Pro, 48 GiB, macOS 27.0.1.

In-process ippon ran commit `7344347` with MLX 0.32.4.

ippon compute: fp16, 1 paired rounds.

| State size | Questions | Other ms | ippon ms | Saved ms | Speed change | Per-round speed change |
|---:|---:|---:|---:|---:|---:|---|
| 100 | 1 | 512.6 | 67.6 | 445.0 | +7.6x | +7.6x |
| 100 | 3 | 1,648.0 | 193.6 | 1,454.4 | +8.5x | +8.5x |
| 300 | 1 | 852.6 | 120.3 | 732.3 | +7.1x | +7.1x |
| 300 | 3 | 2,576.8 | 359.4 | 2,217.4 | +7.2x | +7.2x |
| 1000 | 1 | 2,843.2 | 392.4 | 2,450.8 | +7.2x | +7.2x |
| 1000 | 3 | 9,201.2 | 1,168.8 | 8,032.4 | +7.9x | +7.9x |
| 2000 | 1 | 5,320.9 | 701.3 | 4,619.6 | +7.6x | +7.6x |
| 2000 | 3 | 14,903.8 | 2,072.5 | 12,831.4 | +7.2x | +7.2x |
| 4000 | 1 | 10,986.8 | 1,431.9 | 9,554.9 | +7.7x | +7.7x |
| 4000 | 3 | 31,570.1 | 4,491.3 | 27,078.8 | +7x | +7x |

Recorded requests:

- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-21/makers/pplx-flash-maker-mps-bf16.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-21/makers/pplx-flash-ippon.json)

### d3-mini · full checkpoint against vllm-sr's server · Transformers 5.17.0 · PyTorch 2.14.1

Measured 2026-10-21 on Apple M5 Pro, 48 GiB, macOS 27.0.1.

In-process ippon ran commit `7344347` with MLX 0.32.4.

ippon compute: fp16, 1 paired rounds.

| State size | Questions | Other ms | ippon ms | Saved ms | Speed change | Per-round speed change |
|---:|---:|---:|---:|---:|---:|---|
| 100 | 1 | 467.4 | 70.4 | 397.0 | +6.6x | +6.6x |
| 100 | 3 | 1,426.1 | 211.7 | 1,214.4 | +6.7x | +6.7x |
| 300 | 1 | 829.0 | 125.6 | 703.5 | +6.6x | +6.6x |
| 300 | 3 | 2,501.8 | 381.6 | 2,120.2 | +6.6x | +6.6x |
| 1000 | 1 | 2,550.5 | 401.1 | 2,149.4 | +6.4x | +6.4x |
| 1000 | 3 | 7,345.8 | 1,195.2 | 6,150.6 | +6.1x | +6.1x |
| 2000 | 1 | 4,446.7 | 687.6 | 3,759.0 | +6.5x | +6.5x |
| 2000 | 3 | 13,908.1 | 2,084.3 | 11,823.8 | +6.7x | +6.7x |
| 4000 | 1 | 9,796.2 | 1,481.2 | 8,315.0 | +6.6x | +6.6x |
| 4000 | 3 | 29,657.5 | 4,421.7 | 25,235.7 | +6.7x | +6.7x |

Recorded requests:

- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-21/makers/d3-mini-tf.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-21/makers/d3-mini-ippon.json)
