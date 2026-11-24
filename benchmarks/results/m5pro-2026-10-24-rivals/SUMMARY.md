# Recorded benchmark summary

## Recorded runtime comparisons

Measured 2026-10-24 on Apple M5 Pro, 48 GiB, macOS 27.0.1.

These arms did not record the library commit or the MLX version, so their timings do not measure the current commit.
oMLX's version is the `kit/setup.sh` pin, and its result metadata does not record a version.

The typical cell is the measured nominal ~1,000-state-token request with one question. Input tokens include the model's prompt. Times are pooled medians in milliseconds.

| Model / build | Other runtime | Input tokens | Other ms | ippon ms | Speed change |
|---|---|---:|---:|---:|---:|
| Clef-Flash 9B · Q8_0 | llama.cpp b11669 | 1341 to 1345 | 1,067.0 | 718.0 | +49% |
| Clef-Flash 9B · Ollama MXFP8 | Ollama 0.40.9 | 1341 to 1345 | 833.8 | 780.6 | +7% |
| d3-mini · Ollama MXFP8 | Ollama 0.40.9 | 1289 to 1293 | 563.2 | 404.8 | +39% |
| Clef-Flash 9B · MLX 8-bit | oMLX 0.7.2 (setup pin) | 1341 to 1345 | 804.2 | 699.7 | +15% |
| d3-mini · checkpoint | oMLX 0.7.2 (setup pin) | 1289 to 1293 | 404.0 | 359.0 | +13% |

Speed change is `(other median / ippon median − 1) × 100%`, so a negative value means ippon is slower. It is not a reduction in latency. Short and Long are geometric means of the cell ratios at the short and long state sizes, with the 1-question and 3-question shapes weighted equally. The Short / long sizes column lists the nominal state sizes that were measured.

| Model / build | Other runtime | Short | Long | Short / long sizes | Max probability gap | Top answers changed |
|---|---|---:|---:|---|---:|---:|
| Clef-Flash 9B · Q8_0 | llama.cpp b11669 | +63% | +48% | 100 / 1000, 4000 | 0.002393 | 0/240 |
| Clef-Flash 9B · Ollama MXFP8 | Ollama 0.40.9 | −1% | +2% | 100 / 1000, 4000 | 0.051407 | 0/240 |
| d3-mini · Ollama MXFP8 | Ollama 0.40.9 | +55% | +37% | 100 / 1000 | 0.017602 | 2/160 |
| Clef-Flash 9B · MLX 8-bit | oMLX 0.7.2 (setup pin) | +28% | +12% | 100 / 1000, 4000 | 0.020200 | 0/240 |
| d3-mini · checkpoint | oMLX 0.7.2 (setup pin) | +25% | +11% | 100 / 1000, 4000 | 0.018493 | 0/240 |

Agreement uses every matched timed question. The probability gap is the largest absolute difference, not a percentage. A matching top answer does not imply matching probabilities.

Clef-Flash MXFP8 is about level overall, at +1.4% across its measured cells, against +6.8% for the typical cell. Its maximum probability gap is 0.051407, with 0/240 top answers changed.

## Per-shape evidence

### Clef-Flash 9B · Q8_0 against llama.cpp b11669

Measured 2026-10-24 on Apple M5 Pro, 48 GiB, macOS 27.0.1.

These arms did not record the library commit or the MLX version, so their timings do not measure the current commit.

ippon compute: fp16, 2 paired rounds.

| State size | Questions | Other ms | ippon ms | Saved ms | Speed change | Per-round speed change |
|---:|---:|---:|---:|---:|---:|---|
| 100 | 1 | 272.7 | 152.9 | 119.7 | +78% | +78%, +79% |
| 100 | 3 | 378.1 | 253.3 | 124.8 | +49% | +47%, +52% |
| 1000 | 1 | 1,067.0 | 718.0 | 349.0 | +49% | +43%, +51% |
| 1000 | 3 | 1,296.9 | 885.9 | 411.0 | +46% | +42%, +48% |
| 4000 | 1 | 4,238.6 | 2,826.1 | 1,412.5 | +50% | +44%, +54% |
| 4000 | 3 | 4,259.1 | 2,883.8 | 1,375.3 | +48% | +43%, +53% |

Recorded requests:

- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-24-rivals/r1/clef-flash-llama-Q8_0.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-24-rivals/r2/clef-flash-llama-Q8_0.json)
- [Recorded run 3](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-24-rivals/r1/clef-flash-gguf-q8-ippon-serve.json)
- [Recorded run 4](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-24-rivals/r2/clef-flash-gguf-q8-ippon-serve.json)

### Clef-Flash 9B · Ollama MXFP8 against Ollama 0.40.9

Measured 2026-10-24 on Apple M5 Pro, 48 GiB, macOS 27.0.1.

These arms did not record the library commit or the MLX version, so their timings do not measure the current commit.

ippon compute: fp16, 2 paired rounds.

| State size | Questions | Other ms | ippon ms | Saved ms | Speed change | Per-round speed change |
|---:|---:|---:|---:|---:|---:|---|
| 100 | 1 | 178.2 | 180.5 | -2.4 | −1% | −3%, 0% |
| 100 | 3 | 283.8 | 283.2 | 0.6 | 0% | −1%, +1% |
| 1000 | 1 | 833.8 | 780.6 | 53.2 | +7% | +5%, +8% |
| 1000 | 3 | 978.2 | 940.5 | 37.6 | +4% | +2%, +5% |
| 4000 | 1 | 3,087.0 | 3,110.5 | -23.5 | −1% | −3%, +1% |
| 4000 | 3 | 3,183.3 | 3,190.5 | -7.2 | 0% | −1%, 0% |

Recorded requests:

- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-24-rivals/r1/clef-flash-mxfp8-ollama.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-24-rivals/r2/clef-flash-mxfp8-ollama.json)
- [Recorded run 3](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-24-rivals/r1/clef-flash-mxfp8-ollama-ippon-serve.json)
- [Recorded run 4](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-24-rivals/r2/clef-flash-mxfp8-ollama-ippon-serve.json)

### d3-mini · Ollama MXFP8 against Ollama 0.40.9

Measured 2026-10-24 on Apple M5 Pro, 48 GiB, macOS 27.0.1.

These arms did not record the library commit or the MLX version, so their timings do not measure the current commit.

ippon compute: fp16, 2 paired rounds.

| State size | Questions | Other ms | ippon ms | Saved ms | Speed change | Per-round speed change |
|---:|---:|---:|---:|---:|---:|---|
| 100 | 1 | 115.5 | 74.2 | 41.4 | +56% | +60%, +52% |
| 100 | 3 | 336.9 | 219.7 | 117.2 | +53% | +59%, +48% |
| 1000 | 1 | 563.2 | 404.8 | 158.4 | +39% | +43%, +35% |
| 1000 | 3 | 1,695.7 | 1,253.2 | 442.5 | +35% | +39%, +31% |

Recorded requests:

- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-24-rivals/r1/d3-mini-mxfp8-ollama.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-24-rivals/r2/d3-mini-mxfp8-ollama.json)
- [Recorded run 3](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-24-rivals/r1/d3-mini-mxfp8-ollama-ippon-serve.json)
- [Recorded run 4](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-24-rivals/r2/d3-mini-mxfp8-ollama-ippon-serve.json)

### Clef-Flash 9B · MLX 8-bit against oMLX 0.7.2 (setup pin)

Measured 2026-10-24 on Apple M5 Pro, 48 GiB, macOS 27.0.1.

These arms did not record the library commit or the MLX version, so their timings do not measure the current commit.
oMLX's version is the `kit/setup.sh` pin, and its result metadata does not record a version.

ippon compute: fp16, 2 paired rounds.

| State size | Questions | Other ms | ippon ms | Saved ms | Speed change | Per-round speed change |
|---:|---:|---:|---:|---:|---:|---|
| 100 | 1 | 198.7 | 153.0 | 45.7 | +30% | +30%, +29% |
| 100 | 3 | 309.0 | 245.6 | 63.4 | +26% | +26%, +25% |
| 1000 | 1 | 804.2 | 699.7 | 104.5 | +15% | +17%, +14% |
| 1000 | 3 | 954.2 | 840.3 | 113.9 | +14% | +15%, +12% |
| 4000 | 1 | 2,990.0 | 2,720.0 | 270.0 | +10% | +11%, +9% |
| 4000 | 3 | 3,054.9 | 2,775.0 | 279.9 | +10% | +10%, +9% |

Recorded requests:

- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-24-rivals/r1/clef-flash-mlx-8bit-omlx.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-24-rivals/r2/clef-flash-mlx-8bit-omlx.json)
- [Recorded run 3](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-24-rivals/r1/clef-flash-mlx-8bit-ippon-serve.json)
- [Recorded run 4](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-24-rivals/r2/clef-flash-mlx-8bit-ippon-serve.json)

### d3-mini · checkpoint against oMLX 0.7.2 (setup pin)

Measured 2026-10-24 on Apple M5 Pro, 48 GiB, macOS 27.0.1.

These arms did not record the library commit or the MLX version, so their timings do not measure the current commit.
oMLX's version is the `kit/setup.sh` pin, and its result metadata does not record a version.

ippon compute: fp16, 2 paired rounds.

| State size | Questions | Other ms | ippon ms | Saved ms | Speed change | Per-round speed change |
|---:|---:|---:|---:|---:|---:|---|
| 100 | 1 | 79.7 | 64.8 | 15.0 | +23% | +23%, +23% |
| 100 | 3 | 240.1 | 190.4 | 49.7 | +26% | +27%, +26% |
| 1000 | 1 | 404.0 | 359.0 | 45.0 | +13% | +12%, +13% |
| 1000 | 3 | 1,219.0 | 1,094.0 | 125.0 | +11% | +11%, +12% |
| 4000 | 1 | 1,625.8 | 1,452.9 | 172.9 | +12% | +12%, +12% |
| 4000 | 3 | 4,890.4 | 4,490.4 | 400.0 | +9% | +10%, +6% |

Recorded requests:

- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-24-rivals/r1/d3-mini-omlx.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-24-rivals/r2/d3-mini-omlx.json)
- [Recorded run 3](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-24-rivals/r1/d3-mini-ippon-serve.json)
- [Recorded run 4](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-24-rivals/r2/d3-mini-ippon-serve.json)
