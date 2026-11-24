# Practical workload results

Speed change is other milliseconds / ippon milliseconds minus one. With several clients at once it is ippon requests/s / other requests/s minus one. Waits run from sending a request to its answer, and the 90th percentile is the nearest rank.

## typical

milliseconds for a typical call: about 500 tokens, 3 questions.

| Model / build | Rival | ippon | Rival | Speed change |
|---|---|---:|---:|---:|
| d3-mini · Ollama MXFP8 | Ollama 0.40.13 | 3,877.7 | 5,552.1 | +43.2% |
| d1-2B · Q8_0 | llama.cpp b11960 | 437.3 | 1,192.0 | +2.7x |
| H2O-Lightning-4B · Q8_0 | llama.cpp b11960 | 1,281.3 | 1,430.9 | +11.7% |
| d1-2B · Ollama BF16 | Ollama 0.40.13 | 399.3 | 476.8 | +19.4% |
| d1-2B · Ollama MXFP8 | Ollama 0.40.13 | 508.3 | 703.7 | +38.4% |
| Kettei Core | ippon alone | 1,136.1 | — | — |
| Kettei Dot | ippon alone | 284.4 | — | — |

- 2026-11-07, Apple M2 Pro, ippon `3591e44`, MLX 0.32.4, source campaign: `benchmarks/results/m2pro-2026-11-07-readme`.

## concurrent

requests per second, 8 clients at once.

| Model / build | Rival | ippon req/s | Rival req/s | ippon median ms | Rival median ms | ippon 90th percentile ms | Rival 90th percentile ms | Speed change |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| d3-mini · Ollama MXFP8 | Ollama 0.40.13 | 0.4 | 0.2 | 20,745.0 | 39,901.4 | 20,747.7 | 41,607.8 | +93.3% |
| d1-2B · Q8_0 | llama.cpp b11960 | 1.3 | 1.2 | 6,052.1 | 1,612.3 | 6,070.9 | 32,231.9 | +6.6% |
| H2O-Lightning-4B · Q8_0 | llama.cpp b11960 | 0.5 | 0.4 | 17,545.8 | 9,821.5 | 17,584.1 | 31,400.1 | +11.1% |
| d1-2B · Ollama BF16 | Ollama 0.40.13 | 1.5 | 1.2 | 5,454.5 | 6,657.2 | 5,500.4 | 6,849.5 | +22.8% |
| d1-2B · Ollama MXFP8 | Ollama 0.40.13 | 1.1 | 0.8 | 7,053.5 | 9,721.0 | 7,080.0 | 9,723.6 | +38.0% |
| Kettei Core | ippon alone | 0.6 | — | 13,625.4 | — | 13,683.1 | — | — |
| Kettei Dot | ippon alone | 2.2 | — | 3,552.6 | — | 3,580.2 | — | — |

- 2026-11-07, Apple M2 Pro, ippon `3591e44`, MLX 0.32.4, source campaign: `benchmarks/results/m2pro-2026-11-07-readme`.

## repeat

milliseconds for a new question about a state already sent.

| Model / build | Rival | ippon | Rival | Speed change |
|---|---|---:|---:|---:|
| d3-mini · Ollama MXFP8 | Ollama 0.40.13 | 2,633.7 | 4,487.9 | +70.4% |
| d1-2B · Q8_0 | llama.cpp b11960 | 790.9 | 829.1 | +4.8% |
| H2O-Lightning-4B · Q8_0 | llama.cpp b11960 | 132.6 | 2,385.9 | +18x |
| d1-2B · Ollama BF16 | Ollama 0.40.13 | 716.5 | 838.5 | +17.0% |
| d1-2B · Ollama MXFP8 | Ollama 0.40.13 | 919.3 | 1,251.3 | +36.1% |
| Kettei Core | ippon alone | 134.0 | — | — |
| Kettei Dot | ippon alone | 42.0 | — | — |

- 2026-11-07, Apple M2 Pro, ippon `3591e44`, MLX 0.32.4, source campaign: `benchmarks/results/m2pro-2026-11-07-readme`.

## grow

milliseconds for a state sent again with one more message.

| Model / build | Rival | ippon | Rival | Speed change |
|---|---|---:|---:|---:|
| d3-mini · Ollama MXFP8 | Ollama 0.40.13 | 3,513.3 | 12,958.0 | +3.7x |
| d1-2B · Q8_0 | llama.cpp b11960 | 1,237.1 | 1,292.8 | +4.5% |
| H2O-Lightning-4B · Q8_0 | llama.cpp b11960 | 198.9 | 3,696.9 | +19x |
| d1-2B · Ollama BF16 | Ollama 0.40.13 | 1,136.5 | 1,309.7 | +15.2% |
| d1-2B · Ollama MXFP8 | Ollama 0.40.13 | 1,455.7 | 1,956.1 | +34.4% |
| Kettei Core | ippon alone | 230.6 | — | — |
| Kettei Dot | ippon alone | 68.2 | — | — |

- 2026-11-07, Apple M2 Pro, ippon `3591e44`, MLX 0.32.4, source campaign: `benchmarks/results/m2pro-2026-11-07-readme`.

## burst

requests per second, 16 clients sending short requests at once.

| Model / build | Rival | ippon req/s | Rival req/s | ippon median ms | Rival median ms | ippon 90th percentile ms | Rival 90th percentile ms | Speed change |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| d3-mini · Ollama MXFP8 | Ollama 0.40.13 | 1.3 | 0.9 | 12,654.7 | 18,628.9 | 12,658.1 | 18,637.5 | +47.2% |
| d1-2B · Q8_0 | llama.cpp b11960 | 4.8 | 4.3 | 3,325.5 | 460.0 | 3,361.0 | 20,008.7 | +10.3% |
| H2O-Lightning-4B · Q8_0 | llama.cpp b11960 | 1.5 | 1.2 | 10,471.0 | 9,209.6 | 10,512.7 | 16,128.6 | +27.6% |
| d1-2B · Ollama BF16 | Ollama 0.40.13 | 5.3 | 4.1 | 2,987.5 | 3,898.8 | 3,025.4 | 3,902.4 | +30.3% |
| d1-2B · Ollama MXFP8 | Ollama 0.40.13 | 4.1 | 2.8 | 3,894.9 | 5,658.4 | 3,908.2 | 5,664.3 | +45.4% |
| Kettei Core | ippon alone | 2.0 | — | 7,857.6 | — | 7,875.1 | — | — |
| Kettei Dot | ippon alone | 7.8 | — | 2,050.5 | — | 2,065.7 | — | — |

- 2026-11-07, Apple M2 Pro, ippon `3591e44`, MLX 0.32.4, source campaign: `benchmarks/results/m2pro-2026-11-07-readme`.
