# README benchmark round

Measured 2026-11-05 on Apple M1, 8 GiB, macOS 26.5.2, MLX 0.32.4, ippon commit `7a2f105885b93805f5f1282d1d69214b421748eb`.

Both workloads send the same nominal ~1,000-token text. A has one yes/no question, and B has 50 mixed yes/no, choice and score questions. Latency includes client encoding, HTTP, model execution and response decoding. Each runtime is already loaded, and warm-up requests are untimed.

2026-11-05 (ippon `7a2f105`): a shape stops once its median's 95% interval is within ±3%, after at least 6 and at most 20 timed requests, or after 90 s. Pairs within 10% in the first round or with overlapping intervals are eligible for another round in the opposite order.

Each pair uses its newest supplied campaign with complete results, and only that campaign's paired rounds are pooled. A timed request that overlapped another busy process is set aside in its result and the next generated request of that shape is timed instead, at most n times per shape. Past that, the busy requests stay in the run and are counted. No observation is dropped for its time. Speed change is `(other median / ippon median − 1) × 100%`, not percentage latency reduction. Both chart panels show milliseconds, and a 50-question request takes `50,000 / questions per second` ms. Each rival bar has ippon's time on the same build directly above it. Percentages use unrounded times, and negative values are shown as slower. The eight-client chart alone uses requests per second, where higher is better.

## One-question latency

| Model / build | Other runtime | Other ms | ippon ms | Speed change | Top answers agree | Max probability gap |
|---|---|---:|---:|---:|---:|---:|
| d1-2B · Q8_0 | llama.cpp b11960 | 2,539.5 | 1,955.3 | +30% | 6/6 | 0.000382 |
| d1-2B · Ollama BF16 | Ollama 0.40.13 | 2,749.2 | 2,067.2 | +33% | 4/6 | 0.004903 |
| d1-2B · Ollama MXFP8 | Ollama 0.40.13 | 4,009.0 | 2,696.3 | +49% | 6/6 | 0.007263 |

## 50-question request time

| Model / build | Other runtime | Other ms | ippon ms | Other questions/s | ippon questions/s | Speed change | Top answers agree | Max probability gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| d1-2B · Q8_0 | llama.cpp b11960 | 131,018.5 | 10,929.0 | 0.4 | 4.6 | +12x | 270/300 | 0.407409 |
| d1-2B · Ollama BF16 | Ollama 0.40.13 | 14,990.7 | 11,702.8 | 3.3 | 4.3 | +28% | 300/300 | 0.019349 |
| d1-2B · Ollama MXFP8 | Ollama 0.40.13 | 21,899.6 | 15,395.7 | 2.3 | 3.2 | +42% | 295/300 | 0.017230 |

Matching top answers do not establish numerical parity, and the maximum probability gap is an absolute difference. These comparisons measure runtime agreement, not task accuracy.

## Request paths and sources

### d1-2B · Q8_0 against llama.cpp b11960

Other runtime: native HTTP, one request. ippon: native HTTP. 1 paired round, 6 to 7 timed requests per arm per workload.

Measured 2026-11-05 on Apple M1, 8 GiB, macOS 26.5.2, MLX 0.32.4, ippon commit `7a2f105885b93805f5f1282d1d69214b421748eb`.

- 1 question, per-round speed change: +29.9%.
- 50 questions, per-round speed change: +12x.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m1-2026-11-05-readme/r1/d1-2B-llama-Q8_0.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m1-2026-11-05-readme/r1/d1-2B-gguf-q8-ippon-serve.json)

### d1-2B · Ollama BF16 against Ollama 0.40.13

Other runtime: native HTTP, one request. ippon: native HTTP. 1 paired round, 6 to 9 timed requests per arm per workload.

Measured 2026-11-05 on Apple M1, 8 GiB, macOS 26.5.2, MLX 0.32.4, ippon commit `7a2f105885b93805f5f1282d1d69214b421748eb`.

- 1 question, per-round speed change: +33.0%.
- 50 questions, per-round speed change: +28.1%.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m1-2026-11-05-readme/r1/d1-2B-bf16-ollama.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m1-2026-11-05-readme/r1/d1-2B-bf16-ollama-ippon-serve.json)

### d1-2B · Ollama MXFP8 against Ollama 0.40.13

Other runtime: native HTTP, one request. ippon: native HTTP. 1 paired round, 6 timed requests per arm per workload.

Measured 2026-11-05 on Apple M1, 8 GiB, macOS 26.5.2, MLX 0.32.4, ippon commit `7a2f105885b93805f5f1282d1d69214b421748eb`.

- 1 question, per-round speed change: +48.7%.
- 50 questions, per-round speed change: +42.2%.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m1-2026-11-05-readme/r1/d1-2B-mxfp8-ollama.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m1-2026-11-05-readme/r1/d1-2B-mxfp8-ollama-ippon-serve.json)

# Practical workload results

Speed change is other milliseconds / ippon milliseconds minus one. With several clients at once it is ippon requests/s / other requests/s minus one. Waits run from sending a request to its answer, and the 90th percentile is the nearest rank.

## typical

milliseconds for a typical call: about 500 tokens, 3 questions.

| Model / build | Rival | ippon | Rival | Speed change |
|---|---|---:|---:|---:|
| d1-2B · Q8_0 | llama.cpp b11960 | 1,106.6 | 3,650.1 | +3.3x |
| d1-2B · Ollama BF16 | Ollama 0.40.13 | 1,220.7 | 1,538.4 | +26.0% |
| d1-2B · Ollama MXFP8 | Ollama 0.40.13 | 1,628.5 | 2,255.4 | +38.5% |

- 2026-11-05, Apple M1, ippon `7a2f105`, MLX 0.32.4, source campaign: `benchmarks/results/m1-2026-11-05-readme`.

## concurrent

requests per second, 8 clients at once.

| Model / build | Rival | ippon req/s | Rival req/s | ippon median ms | Rival median ms | ippon 90th percentile ms | Rival 90th percentile ms | Speed change |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| d1-2B · Q8_0 | llama.cpp b11960 | 0.4 | 0.4 | 17,879.8 | 5,390.9 | 18,034.2 | 106,852.1 | +21.3% |
| d1-2B · Ollama BF16 | Ollama 0.40.13 | 0.5 | 0.4 | 17,470.8 | 22,037.1 | 17,601.3 | 22,127.7 | +26.2% |
| d1-2B · Ollama MXFP8 | Ollama 0.40.13 | 0.3 | 0.3 | 23,994.4 | 31,653.8 | 24,142.0 | 32,012.6 | +32.5% |

- 2026-11-05, Apple M1, ippon `7a2f105`, MLX 0.32.4, source campaign: `benchmarks/results/m1-2026-11-05-readme`.

## image

milliseconds for one question with an image.

| Model / build | Rival | ippon | Rival | Speed change |
|---|---|---:|---:|---:|
| d1-2B · Q8_0 | llama.cpp b11960 | 2,761.2 | 3,294.3 | +19.3% |
| d1-2B · Ollama BF16 | Ollama 0.40.13 | 2,775.3 | 3,766.9 | +35.7% |
| d1-2B · Ollama MXFP8 | Ollama 0.40.13 | 3,456.1 | 4,432.6 | +28.3% |

- 2026-11-05, Apple M1, ippon `7a2f105`, MLX 0.32.4, source campaign: `benchmarks/results/m1-2026-11-05-readme`.

## repeat

milliseconds for a new question about a state already sent.

| Model / build | Rival | ippon | Rival | Speed change |
|---|---|---:|---:|---:|
| d1-2B · Q8_0 | llama.cpp b11960 | 2,046.1 | 2,691.9 | +31.6% |
| d1-2B · Ollama BF16 | Ollama 0.40.13 | 2,197.0 | 2,761.9 | +25.7% |
| d1-2B · Ollama MXFP8 | Ollama 0.40.13 | 2,951.7 | 4,050.1 | +37.2% |

- 2026-11-05, Apple M1, ippon `7a2f105`, MLX 0.32.4, source campaign: `benchmarks/results/m1-2026-11-05-readme`.

## grow

milliseconds for a state sent again with one more message.

| Model / build | Rival | ippon | Rival | Speed change |
|---|---|---:|---:|---:|
| d1-2B · Q8_0 | llama.cpp b11960 | 4,098.3 | 4,639.4 | +13.2% |
| d1-2B · Ollama BF16 | Ollama 0.40.13 | 3,719.8 | 4,509.1 | +21.2% |
| d1-2B · Ollama MXFP8 | Ollama 0.40.13 | 4,675.8 | 5,817.3 | +24.4% |

- 2026-11-05, Apple M1, ippon `7a2f105`, MLX 0.32.4, source campaign: `benchmarks/results/m1-2026-11-05-readme`.

## burst

requests per second, 16 clients sending short requests at once.

| Model / build | Rival | ippon req/s | Rival req/s | ippon median ms | Rival median ms | ippon 90th percentile ms | Rival 90th percentile ms | Speed change |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| d1-2B · Q8_0 | llama.cpp b11960 | 1.6 | 1.5 | 10,011.3 | 1,385.2 | 10,134.5 | 59,956.4 | +9.9% |
| d1-2B · Ollama BF16 | Ollama 0.40.13 | 1.7 | 1.3 | 9,520.6 | 12,436.4 | 9,667.6 | 12,479.4 | +30.4% |
| d1-2B · Ollama MXFP8 | Ollama 0.40.13 | 1.2 | 0.9 | 13,263.6 | 18,307.3 | 13,375.4 | 18,343.3 | +37.7% |

- 2026-11-05, Apple M1, ippon `7a2f105`, MLX 0.32.4, source campaign: `benchmarks/results/m1-2026-11-05-readme`.
