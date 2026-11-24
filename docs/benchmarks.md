# Benchmarks

Each speed claim on this page names the model, the runtime, the request, the Mac and how it was measured. The
benchmark kit (`benchmarks/kit/`, [its README](../benchmarks/README.md)) records the requests, answers, timings and the
environment metadata that is available. Where a historical result has no provenance record, the page says so beside it.

## Published comparisons

The library's current execution path does not include packed prefill. The
[archived packed-prefill experiment](../references/prefill/README.md) records a separate 4-round test on M5 Pro
48 GB, macOS 27.0.1 and MLX 0.32.4. Packed prefill made growing-agent sessions faster, but on all 3 tested builds it
slowed fresh requests by more than the 1% regression limit. Its dispatch policy and timings describe an experimental
candidate.

## ippon against Ollama

<!-- speed headline: benchmarks/kit/summarize.py --readme -->
ippon against Ollama 0.40.13 on the same builds, in an agent's loop (the same ~1,000-token context sent again with one more message) and in a typical call (about 500 tokens, 3 questions).

| Model, build | Agent's loop | Typical call |
|---|---|---|
| d3-mini, MXFP8 | 518 ms against 696, Ollama takes 34% longer | 580 ms against 835, Ollama takes 44% longer |
| Clef-Flash 9B, MXFP8 | 123 ms against 179, Ollama takes 46% longer | 494 ms against 510, Ollama takes 3% longer |
| d3-flash, MXFP8 | 934 ms against 1,153, Ollama takes 23% longer | 1,045 ms against 1,349, Ollama takes 29% longer |
| d1-2B, BF16 | 160 ms against 174, Ollama takes 8% longer | 67 ms against 78, Ollama takes 17% longer |

Clef-Flash 9B answers one question about ~1,000 tokens in 654 ms on ippon, where llama.cpp b11960 takes 57% longer (1,028 ms) on the same Q8_0 file. Every result here was measured on an Apple M5 Pro with 48 GB, macOS 27.0.1 and MLX 0.32.4, on 28 October to 24 November 2026.
<!-- /speed headline -->

## On an 8 GB M1

On a base M1 Air with 8 GB, d1-2B answers sooner on ippon than on llama.cpp b11960 and Ollama 0.40.13 in every
workload measured. Over the typical, repeat, grow, burst and image workloads, those 2 runtimes take 35% longer on
average. Each cell below gives the other runtime's time against ippon's. For several clients at once, each cell gives
how many more answers a second ippon gives.

ippon picks its own matmul kernels on the M5. On an M1 to M4 it runs MLX's own kernels, the same ones that every MLX
runtime uses. The exception is MXFP8 on an M1, an M2 and an M4, where ippon unpacks an MXFP8 matrix to full precision
before MLX multiplies it, from 224 rows on M1 and M2 and from 1,024 rows on M4 ([kernels](kernels.md)).

| Workload | llama.cpp, Q8_0 | Ollama, BF16 | Ollama, MXFP8 |
|---|---:|---:|---:|
| Typical call, about 500 tokens and 3 questions | 3.3x as long | 26% longer | 39% longer |
| The same context with a new question | 32% longer | 26% longer | 37% longer |
| The context sent again with one more message | 13% longer | 21% longer | 24% longer |
| One question with an image | 19% longer | 36% longer | 28% longer |
| 16 clients sending short requests at once | 10% more answers a second | 30% more answers a second | 38% more answers a second |
| 8 clients at once | 21% more answers a second | 26% more answers a second | 33% more answers a second |

Each rival ran the same model file as ippon. The round was measured on 5 November 2026 with macOS 26.5.2, MLX 0.32.4
and ippon `7a2f105`. The [round's summary](../benchmarks/results/m1-2026-11-05-readme/SUMMARY.md) has the
milliseconds, the waits of every client and the recorded runs.

The MXFP8 column predates the unpacking (`99600a3`). On the same M1, in fresh processes, MXFP8 d1-2B with the
unpacking answered a ~500-word question in 22 to 25% less time than without it. It answered a ~1,000-word question in
17 to 19% less time and the typical call in 16% less time, and every answer was identical to the last digit. With the
unpacking, MXFP8 d1-2B runs at about the speed of its Q8_0 file, which reads the same number of bytes for each weight.
The [A/B and the matrix sweep](../benchmarks/results/m1-2026-11-07-mxfp8) were recorded on 7 November 2026.

Kettei Core and Dot also run on the same 8 GB M1 Air. Core runs at 8-bit (`--precision q8`, about 3.7 GiB) to fit its
5 GiB GPU working set. `ippon verify` gives Dot 0.0041 (bound 0.032) and Core 8-bit 0.030 (bound 0.037), so both stay
within their bounds against the reference. The M1 GPU reads each new context, so those calls take seconds. A context
that ippon has already read comes back fast.

| Workload | Kettei Dot | Kettei Core, 8-bit |
|---|---:|---:|
| The same context with a new question | 161 ms | 729 ms |
| The context sent again with one more message | 282 ms | 1,142 ms |
| Typical call, about 500 tokens and 3 questions | 1,344 ms | 7,219 ms |
| A new context of about 1,000 tokens, one question | 2,385 ms | 12,229 ms |
| The same, 50 questions | 8,168 ms | 43,204 ms |
| 16 clients sending short requests at once, the last answer | 11,185 ms | 55,158 ms |

The table gives medians, measured on 5 November 2026 with macOS 26.5.2, MLX 0.32.4 and ippon `90ccacd`
([results](../benchmarks/results/m1-2026-11-05-kettei)).

## On a 16 GB M2 Pro

On an M2 Pro with 16 GB (16 GPU cores), every model answers sooner on ippon than on llama.cpp b11960 and Ollama
0.40.13 in every workload measured. For d1-2B, the other runtimes take 29% longer on average over the typical, repeat,
grow, 8-client and 16-client workloads. Over the 5 pairs together, they take 71% longer. Each cell gives the other
runtime's time against ippon's. For several clients at once, each cell gives how many more answers a second ippon
gives.

| Workload | d1-2B, llama.cpp Q8_0 | d1-2B, Ollama BF16 | d1-2B, Ollama MXFP8 | d3-mini, Ollama MXFP8 | H2O-Lightning-4B, llama.cpp Q8_0 |
|---|---:|---:|---:|---:|---:|
| Typical call, about 500 tokens and 3 questions | 2.7x as long | 19% longer | 38% longer | 43% longer | 12% longer |
| The same context with a new question | 5% longer | 17% longer | 36% longer | 70% longer | 18x as long |
| The context sent again with one more message | 5% longer | 15% longer | 34% longer | 3.7x as long | 19x as long |
| 8 clients at once | 7% more answers a second | 23% more answers a second | 38% more answers a second | 93% more answers a second | 11% more answers a second |
| 16 clients sending short requests at once | 10% more answers a second | 30% more answers a second | 45% more answers a second | 47% more answers a second | 28% more answers a second |

d1-2B reads its context and its questions in both directions, so no runtime can keep what it read for a new
question. Its single-question cells against llama.cpp therefore come down to reading speed, and both runtimes run near
the compute limit of this GPU. The typical call takes 2.7x as long on llama.cpp because ippon reads the context once
for all 3 questions. With several clients at once, the figure is requests answered a second. llama.cpp's median wait
is lower there, because it finishes the requests of some clients while other clients wait. Its 90th percentile is 32 s
with 8 clients, against 6 s for ippon.

Kettei Core and Dot have no rival build and ran on ippon alone.

| Workload | Kettei Dot | Kettei Core |
|---|---:|---:|
| The same context with a new question | 42 ms | 134 ms |
| The context sent again with one more message | 68 ms | 231 ms |
| Typical call, about 500 tokens and 3 questions | 284 ms | 1,136 ms |

Each rival ran the same model file as ippon. The round was measured on 7 November 2026 with macOS 27.2, MLX 0.32.4
and ippon `3591e44`, before ippon unpacked MXFP8 on M1 and M2. oMLX did not run, because the M2 Pro has no Xcode Metal
toolchain for its build. Clef-Flash 9B did not run, because it needs more than 16 GB. The
[round's summary](../benchmarks/results/m2pro-2026-11-07-readme/SUMMARY.md) has the milliseconds, the waits of every
client and the recorded runs.

## At the pace an agent calls

At the pace an agent calls, ippon kept its lead over Ollama, oMLX and llama.cpp in 2 rounds. An agent calls its
decision model seconds apart, while its LLM works. After a pause, every runtime's call is a little slower (15 to 25 ms
on these models), because the GPU's clock falls while it is idle. The kit timed a typical call of about 500 tokens and
3 questions after 0, 2, 5 and 15 seconds idle, with each runtime in its own server on the same build as ippon. The
second round ran the runtimes in the other order.

Over Ollama, ippon's lead was 18% to 21% at every pause (the geometric mean over its 4 builds). Over oMLX it was 19%
back to back and 21% to 27% after a pause (Clef-Flash 9B's MLX 8-bit build, and the d3-mini and d3-flash checkpoints in
the second round). Over llama.cpp it was 2.4x to 2.5x on the Q8_0 files of d1-2B and Clef-Flash 9B. The rounds ran on
an Apple M5 Pro, 48 GB, on 5 November 2026 with ippon `01aeb68`, Ollama 0.40.13, oMLX 0.7.2 and llama.cpp b11960. Each
pause had 8 requests after it, and the runtimes took turns. The results are in
[m5pro-2026-11-05-agent-pace](../benchmarks/results/m5pro-2026-11-05-agent-pace) and
[m5pro-2026-11-05-agent-pace-r2](../benchmarks/results/m5pro-2026-11-05-agent-pace-r2).

## Memory beside other runtimes

On the same builds, ippon's highest memory footprint was lower than Ollama's and oMLX's on every build. On average it
was 45% below Ollama's and 9% below oMLX's (the geometric mean over each runtime's builds). A decision model runs beside
the agent that calls it, so the highest memory it uses while it answers matters as much as its speed.

| Build | Highest, ippon | Highest, other runtime |
|---|---:|---:|
| d3-mini, Ollama's MXFP8 | 6.8 GB | Ollama 21.0 GB |
| d1-2B, Ollama's BF16 | 5.8 GB | Ollama 7.6 GB |
| d1-2B, Ollama's MXFP8 | 4.4 GB | Ollama 6.4 GB |
| d3-mini, checkpoint | 10.6 GB | oMLX 11.7 GB |
| d3-flash, checkpoint | 17.6 GB | oMLX 19.4 GB |

"Highest" is the largest physical footprint (`lifetime_max_phys_footprint`) that each server reached over the
typical, repeat, grow, 16-client and image-again workloads. Each workload ran in a fresh server. ippon's footprint
includes the contexts and media that it keeps to answer the next call sooner. Workload by workload, ippon was higher
only in d1-2B's MXFP8 text sessions (repeat, grow and 16 clients), by 4 to 7%. The round ran on an Apple M5 Pro, 48 GB,
on 19 November 2026 with macOS 27.0.1, MLX 0.32.4, ippon `136fefa9`, Ollama 0.40.13 and oMLX 0.7.2.
[The round](../benchmarks/results/m5pro-2026-11-19-memory) has the recorded runs.

<!-- README round: benchmarks/kit/summarize.py -->
## Current README round

Measured 2026-10-28 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `d8725098aefe28d01e7aa0c022b635f3c7842f96`.
Measured 2026-10-29 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `d27c0ad9523d0c5f025ec05c2d65a7a84c78aa9e`.
Measured 2026-10-29 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `d3406e18b95fbd4a84410884782c47a481648582`.
Measured 2026-11-03 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `1610112a907dfcc06fed2e35720db8541d623d69`.
Measured 2026-11-04 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `08a7b8a465789bd611defa32663ab123915e0863`.
Measured 2026-11-07 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `24701b604b40a1c4ab754508865783f3454cab40`.
Measured 2026-11-10 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `1c9234ecac4f9c9d2da39be520bfaa45bed4682a`.
Measured 2026-11-18 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `7b972d6621230c25ab741d09831647cd0e39da32`.
Measured 2026-11-24 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `5ba345cb75f3071879d83bc632f262ea3f104be2`.

Both workloads send the same nominal ~1,000-token text. A has one yes/no question, and B has 50 mixed yes/no, choice and score questions. Latency includes client encoding, HTTP, model execution and response decoding. Each runtime is already loaded, and warm-up requests are untimed.

In the rounds with ippon `d27c0ad`, `d3406e1`, `08a7b8a` and `24701b6`, a shape stops once its median's 95% interval is within ±3%, after at least 6 and at most 10 timed requests, or after 90 s. Pairs within 10% in the first round, or with overlapping intervals, can have another round in the opposite order.

In the rounds with ippon `1610112`, `1c9234e` and `5ba345c`, a shape stops once its median's 95% interval is within ±3%, after at least 6 and at most 20 timed requests, or after 90 s. Pairs within 10% in the first round, or with overlapping intervals, can have another round in the opposite order.

In the rounds with ippon `7b972d6`, a shape stops once its median's 95% interval is within ±3%, after at least 6 and at most 32 timed requests, or after 90 s. Pairs within 10% in the first round, or with overlapping intervals, can have another round in the opposite order.

Each pair uses its newest supplied campaign with complete results, and only that campaign's paired rounds are pooled. A timed request that overlapped another busy process is set aside in its result and the next generated request of that shape is timed instead, at most n times per shape. Past that, the busy requests stay in the run and are counted. No observation is dropped for its time. Speed change is `(other median / ippon median − 1) × 100%`, and it is not a reduction in latency. Both chart panels show milliseconds, and a 50-question request takes `50,000 / questions per second` ms. Each rival bar has ippon's time on the same build directly above it. Percentages use unrounded times, and negative values are shown as slower. The charts of 8 and 16 clients at once use requests per second, where higher is better.

### One-question latency

| Model / build | Other runtime | Other ms | ippon ms | Speed change | Top answers agree | Max probability gap |
|---|---|---:|---:|---:|---:|---:|
| Clef-Flash 9B · Q8_0 | llama.cpp b11960 | 1,028.1 | 654.1 | +57% | 6/6 | 0.000359 |
| Clef-Flash 9B · Ollama MXFP8 | Ollama 0.40.13 | 834.4 | 795.4 | +5% | 7/7 | 0.005449 |
| d3-mini · Ollama MXFP8 | Ollama 0.40.13 | 570.9 | 417.5 | +37% | 6/6 | 0.014654 |
| Clef-Flash 9B · MLX 8-bit | oMLX 0.7.2 | 795.7 | 659.9 | +21% | 10/10 | 0.003600 |
| d3-mini · checkpoint | oMLX 0.7.2 | 404.1 | 378.1 | +7% | 10/10 | 0.008309 |
| Clef-Flash 9B · MLX 8-bit | mlx-vlm 0.7.10 | 778.3 | 659.9 | +18% | 10/10 | 0.003600 |
| pplx-decider flash v1.3 · BF16 | Perplexity's package · Transformers 5.17.0 · PyTorch 2.14.0 · BF16 | 2,867.8 | 388.6 | +7.4x | 6/6 | 0.002273 |
| d3-mini · full checkpoint | vllm-sr's server · Transformers 5.17.0 · PyTorch 2.14.1 | 2,574.0 | 374.9 | +6.9x | 6/6 | 0.002964 |
| d1-2B · Q8_0 | llama.cpp b11960 | 187.7 | 119.4 | +57% | 6/6 | 0.000779 |
| d1-Omni-2B · Q8_0 | llama.cpp b11960 | 187.7 | 119.0 | +58% | 6/6 | 0.000653 |
| H2O-Lightning-4B · Q8_0 | llama.cpp b11960 | 551.7 | 342.8 | +61% | 6/6 | 0.000421 |
| d1-2B · Ollama MXFP8 | Ollama 0.40.13 | 152.4 | 131.6 | +16% | 6/6 | 0.005715 |
| d1-Omni-2B · Ollama MXFP8 | Ollama 0.40.13 | 153.2 | 136.9 | +12% | 6/6 | 0.001706 |
| d3-flash · Ollama MXFP8 | Ollama 0.40.13 | 903.3 | 750.3 | +20% | 8/8 | 0.009787 |
| d3-flash · checkpoint | oMLX 0.7.2 | 729.6 | 662.8 | +10% | 6/6 | 0.008660 |
| d1-2B · BF16 | Liquid's code · Transformers 5.19.0 · PyTorch 2.14.1 | 204.4 | 119.0 | +72% | 4/6 | 0.010738 |
| d1-Omni-2B · BF16 | Liquid's code · Transformers 5.19.0 · PyTorch 2.14.1 | 204.3 | 118.6 | +72% | 5/6 | 0.006542 |
| d3-flash · full checkpoint | vllm-sr's server · Transformers 5.17.0 · PyTorch 2.14.1 | 2,855.1 | 662.8 | +4.3x | 6/6 | 0.005297 |
| Torchcast Decision 4B · BF16 | Torchcast's server · Transformers 5.8.1 · PyTorch 2.9.1 | 963.4 | 377.5 | +2.6x | 6/6 | 0.012044 |
| elephant-1 · BF16 | Base Labs' runtime · Transformers 5.19.0 · PyTorch 2.8.0 | 3,159.8 | 700.5 | +4.5x | 4/6 | 0.005016 |
| d1-2B · Ollama BF16 | Ollama 0.40.13 | 132.6 | 115.1 | +15% | 3/6 | 0.012624 |
| d1-Omni-2B · Ollama BF16 | Ollama 0.40.13 | 132.0 | 115.0 | +15% | 4/6 | 0.006279 |
| GLiDE 9B · BF16 | Fastino's server · Transformers 5.17.0 · PyTorch 2.14.0 | 4,176.9 | 871.2 | +4.8x | 12/12 | 0.004809 |
| GLiDE 2B · BF16 | Fastino's server · Transformers 5.17.0 · PyTorch 2.14.0 | 959.8 | 143.1 | +6.7x | 6/6 | 0.008151 |
| Jeff-1.1-Qwen-4B · BF16 | Jonathan Geuter's server · Transformers 5.17.0 · PyTorch 2.13.0 | 2,621.8 | 373.8 | +7x | 6/6 | 0.000000 |
| Kev-4B · BF16 | Jared Palmer's server · kev with mlx-lm 0.31.3 | 453.3 | 341.6 | +33% | 6/6 | 0.007114 |
| Kev-9B · BF16 | Jared Palmer's server · kev with mlx-lm 0.31.3 | 785.1 | 601.9 | +30% | 6/6 | 0.005714 |
| Strands-Decider-E2B · BF16 | AWS Strands' server · Transformers 5.24.0 · PyTorch 2.15.0 | 387.7 | 91.4 | +4.2x | 6/6 | 0.003200 |
| Strands-Decider-E4B · BF16 | AWS Strands' server · Transformers 5.24.0 · PyTorch 2.15.0 | 673.9 | 286.0 | +2.4x | 6/6 | 0.006200 |
| Strands-Decider-12B · BF16 | AWS Strands' server · Transformers 5.24.0 · PyTorch 2.15.0 | 1,613.9 | 1,112.5 | +45% | 6/6 | 0.005600 |

### 50-question request time

| Model / build | Other runtime | Other ms | ippon ms | Other questions/s | ippon questions/s | Speed change | Top answers agree | Max probability gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Clef-Flash 9B · Q8_0 | llama.cpp b11960 | 4,905.5 | 3,238.5 | 10.2 | 15.4 | +51% | 299/300 | 0.002407 |
| Clef-Flash 9B · Ollama MXFP8 | Ollama 0.40.13 | 3,682.1 | 3,462.7 | 13.6 | 14.4 | +6% | 500/500 | 0.103344 |
| d3-mini · Ollama MXFP8 | Ollama 0.40.13 | 25,796.4 | 19,582.6 | 1.9 | 2.6 | +32% | 798/800 | 0.018509 |
| Clef-Flash 9B · MLX 8-bit | oMLX 0.7.2 | 3,669.9 | 3,352.6 | 13.6 | 14.9 | +9% | 500/500 | 0.063900 |
| d3-mini · checkpoint | oMLX 0.7.2 | 19,716.0 | 17,761.6 | 2.5 | 2.8 | +11% | 546/550 | 0.014024 |
| Clef-Flash 9B · MLX 8-bit | mlx-vlm 0.7.10 | 3,631.1 | 3,352.6 | 13.8 | 14.9 | +8% | 499/500 | 0.034800 |
| pplx-decider flash v1.3 · BF16 | Perplexity's package · Transformers 5.17.0 · PyTorch 2.14.0 · BF16 | 186,547.0 | 18,080.8 | 0.3 | 2.8 | +10x | 1680/1700 | 0.040081 |
| d3-mini · full checkpoint | vllm-sr's server · Transformers 5.17.0 · PyTorch 2.14.1 | 125,384.7 | 19,017.3 | 0.4 | 2.6 | +6.6x | 796/800 | 0.014526 |
| d1-2B · Q8_0 | llama.cpp b11960 | 9,462.3 | 738.0 | 5.3 | 67.7 | +13x | 270/300 | 0.407987 |
| d1-Omni-2B · Q8_0 | llama.cpp b11960 | 9,427.1 | 734.5 | 5.3 | 68.1 | +13x | 244/300 | 0.331452 |
| H2O-Lightning-4B · Q8_0 | llama.cpp b11960 | 4,246.9 | 1,170.9 | 11.8 | 42.7 | +3.6x | 300/300 | 0.299474 |
| d1-2B · Ollama MXFP8 | Ollama 0.40.13 | 808.5 | 730.4 | 61.8 | 68.5 | +11% | 298/300 | 0.022122 |
| d1-Omni-2B · Ollama MXFP8 | Ollama 0.40.13 | 815.8 | 747.6 | 61.3 | 66.9 | +9% | 399/400 | 0.017642 |
| d3-flash · Ollama MXFP8 | Ollama 0.40.13 | 42,831.6 | 34,076.2 | 1.2 | 1.5 | +26% | 636/650 | 0.019082 |
| d3-flash · checkpoint | oMLX 0.7.2 | 36,577.6 | 32,795.0 | 1.4 | 1.5 | +12% | 498/500 | 0.019302 |
| d1-2B · BF16 | Liquid's code · Transformers 5.19.0 · PyTorch 2.14.1 | 1,328.5 | 737.6 | 37.6 | 67.8 | +80% | 297/300 | 0.018023 |
| d1-Omni-2B · BF16 | Liquid's code · Transformers 5.19.0 · PyTorch 2.14.1 | 1,328.5 | 734.7 | 37.6 | 68.1 | +81% | 297/300 | 0.015263 |
| d3-flash · full checkpoint | vllm-sr's server · Transformers 5.17.0 · PyTorch 2.14.1 | 146,625.9 | 32,795.0 | 0.3 | 1.5 | +4.5x | 500/500 | 0.018445 |
| Torchcast Decision 4B · BF16 | Torchcast's server · Transformers 5.8.1 · PyTorch 2.9.1 | 65,361.7 | 18,082.5 | 0.8 | 2.8 | +3.6x | 499/500 | 0.602000 |
| elephant-1 · BF16 | Base Labs' runtime · Transformers 5.19.0 · PyTorch 2.8.0 | 15,548.3 | 2,453.8 | 3.2 | 20.4 | +6.3x | 298/300 | 0.016234 |
| d1-2B · Ollama BF16 | Ollama 0.40.13 | 700.7 | 707.0 | 71.4 | 70.7 | −1% | 300/300 | 0.023690 |
| d1-Omni-2B · Ollama BF16 | Ollama 0.40.13 | 700.9 | 708.9 | 71.3 | 70.5 | −1% | 299/300 | 0.014886 |
| GLiDE 9B · BF16 | Fastino's server · Transformers 5.17.0 · PyTorch 2.14.0 | 202,899.2 | 46,923.6 | 0.2 | 1.1 | +4.3x | 800/800 | 0.015192 |
| GLiDE 2B · BF16 | Fastino's server · Transformers 5.17.0 · PyTorch 2.14.0 | 46,764.2 | 7,117.4 | 1.1 | 7.0 | +6.6x | 596/600 | 0.018342 |
| Jeff-1.1-Qwen-4B · BF16 | Jonathan Geuter's server · Transformers 5.17.0 · PyTorch 2.13.0 | 131,926.4 | 18,678.4 | 0.4 | 2.7 | +7.1x | 350/350 | 0.011793 |
| Kev-4B · BF16 | Jared Palmer's server · kev with mlx-lm 0.31.3 | 1,234.3 | 746.3 | 40.5 | 67.0 | +65% | 298/300 | 0.016963 |
| Kev-9B · BF16 | Jared Palmer's server · kev with mlx-lm 0.31.3 | 2,084.3 | 1,251.0 | 24.0 | 40.0 | +67% | 300/300 | 0.017860 |
| Strands-Decider-E2B · BF16 | AWS Strands' server · Transformers 5.24.0 · PyTorch 2.15.0 | 1,958.9 | 960.6 | 25.5 | 52.0 | +2x | 398/400 | 0.016800 |
| Strands-Decider-E4B · BF16 | AWS Strands' server · Transformers 5.24.0 · PyTorch 2.15.0 | 3,410.7 | 1,838.2 | 14.7 | 27.2 | +86% | 450/450 | 0.014700 |
| Strands-Decider-12B · BF16 | AWS Strands' server · Transformers 5.24.0 · PyTorch 2.15.0 | 8,133.0 | 5,207.9 | 6.1 | 9.6 | +56% | 391/400 | 0.075200 |

Matching top answers do not establish numerical parity, and the maximum probability gap is an absolute difference. These comparisons measure runtime agreement, not task accuracy.

### Request paths and sources

#### Clef-Flash 9B · Q8_0 against llama.cpp b11960

The other runtime over native HTTP, one request, and ippon over native HTTP. 1 paired round, 6 to 10 timed requests per arm per workload.

Measured 2026-10-29 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `d27c0ad9523d0c5f025ec05c2d65a7a84c78aa9e`.

- 1 question, per-round speed change +57.2%.
- 50 questions, per-round speed change +51.5%.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/clef-flash-llama-Q8_0.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/clef-flash-gguf-q8-ippon-serve.json)

#### Clef-Flash 9B · Ollama MXFP8 against Ollama 0.40.13

The other runtime over native HTTP, one request, and ippon over native HTTP. 1 paired round, 6 to 7 timed requests per arm per workload.

Measured 2026-10-29 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `d27c0ad9523d0c5f025ec05c2d65a7a84c78aa9e`.

- 1 question, per-round speed change +4.9%.
- 50 questions, per-round speed change +6.3%.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/clef-flash-mxfp8-ollama.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/clef-flash-mxfp8-ollama-ippon-serve.json)

#### d3-mini · Ollama MXFP8 against Ollama 0.40.13

The other runtime over native HTTP, one request, and ippon over native HTTP. 1 paired round, 6 timed requests per arm per workload.

Measured 2026-10-29 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `d27c0ad9523d0c5f025ec05c2d65a7a84c78aa9e`.

- 1 question, per-round speed change +36.8%.
- 50 questions, per-round speed change +31.7%.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/d3-mini-mxfp8-ollama.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/d3-mini-mxfp8-ollama-ippon-serve.json)

#### Clef-Flash 9B · MLX 8-bit against oMLX 0.7.2

The other runtime over native HTTP, one request, and ippon over native HTTP. 1 paired round, 10 timed requests per arm per workload.

Measured 2026-10-28 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `d8725098aefe28d01e7aa0c022b635f3c7842f96`.

- 1 question, per-round speed change +20.6%.
- 50 questions, per-round speed change +9.5%.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-28-readme/r1/clef-flash-mlx-8bit-omlx.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-28-readme/r1/clef-flash-mlx-8bit-ippon-serve.json)

#### d3-mini · checkpoint against oMLX 0.7.2

The other runtime over native HTTP, one request, and ippon over native HTTP. 1 paired round, 10 timed requests per arm per workload.

Measured 2026-10-28 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `d8725098aefe28d01e7aa0c022b635f3c7842f96`.

- 1 question, per-round speed change +6.9%.
- 50 questions, per-round speed change +11.0%.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-28-readme/r1/d3-mini-omlx.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-28-readme/r1/d3-mini-ippon-serve.json)

#### Clef-Flash 9B · MLX 8-bit against mlx-vlm 0.7.10

The other runtime over native HTTP, one native /v1/systemone request containing all questions, and ippon over native HTTP. 1 paired round, 10 timed requests per arm per workload.

Measured 2026-10-28 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `d8725098aefe28d01e7aa0c022b635f3c7842f96`.

- 1 question, per-round speed change +17.9%.
- 50 questions, per-round speed change +8.3%.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-28-readme/r1/clef-flash-mlx-8bit.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-28-readme/r1/clef-flash-mlx-8bit-ippon-serve.json)

#### pplx-decider flash v1.3 · BF16 against Perplexity's package · Transformers 5.17.0 · PyTorch 2.14.0 · BF16

The other runtime over benchmark HTTP wrapper, one decide call, the package's default mode: a separate sequence per question, and ippon over native HTTP. 1 paired round, 6 to 13 timed requests per arm per workload.

Measured 2026-11-18 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `7b972d6621230c25ab741d09831647cd0e39da32`.

- 1 question, per-round speed change +7.4x.
- 50 questions, per-round speed change +10x.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-11-18-pplx-v1.3-text/r1/pplx-flash-maker-mps-bf16.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-11-18-pplx-v1.3-text/r1/pplx-flash-ippon-serve.json)

#### d3-mini · full checkpoint against vllm-sr's server · Transformers 5.17.0 · PyTorch 2.14.1

The other runtime over native HTTP, one native /v1/systemone request containing all questions, run in batches of up to eight questions, and ippon over native HTTP. 1 paired round, 6 timed requests per arm per workload.

Measured 2026-10-29 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `d27c0ad9523d0c5f025ec05c2d65a7a84c78aa9e`.

- 1 question, per-round speed change +6.9x.
- 50 questions, per-round speed change +6.6x.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/d3-mini-tf.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/d3-mini-ippon-serve.json)

#### d1-2B · Q8_0 against llama.cpp b11960

The other runtime over native HTTP, one request, and ippon over native HTTP. 1 paired round, 6 timed requests per arm per workload.

Measured 2026-10-29 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `d27c0ad9523d0c5f025ec05c2d65a7a84c78aa9e`.

- 1 question, per-round speed change +57.2%.
- 50 questions, per-round speed change +13x.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/d1-2B-llama-Q8_0.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/d1-2B-gguf-q8-ippon-serve.json)

#### d1-Omni-2B · Q8_0 against llama.cpp b11960

The other runtime over native HTTP, one request, and ippon over native HTTP. 1 paired round, 6 timed requests per arm per workload.

Measured 2026-10-29 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `d27c0ad9523d0c5f025ec05c2d65a7a84c78aa9e`.

- 1 question, per-round speed change +57.7%.
- 50 questions, per-round speed change +13x.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/d1-omni-2B-llama-Q8_0.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/d1-omni-2B-gguf-q8-ippon-serve.json)

#### H2O-Lightning-4B · Q8_0 against llama.cpp b11960

The other runtime over native HTTP, one request, and ippon over native HTTP. 1 paired round, 6 timed requests per arm per workload.

Measured 2026-11-03 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `1610112a907dfcc06fed2e35720db8541d623d69`.

- 1 question, per-round speed change +60.9%.
- 50 questions, per-round speed change +3.6x.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-11-03-h2o-llama/r1/h2o-4b-llama-Q8_0.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-11-03-h2o-llama/r1/h2o-4b-gguf-q8-ippon-serve.json)

#### d1-2B · Ollama MXFP8 against Ollama 0.40.13

The other runtime over native HTTP, one request, and ippon over native HTTP. 1 paired round, 6 timed requests per arm per workload.

Measured 2026-10-29 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `d27c0ad9523d0c5f025ec05c2d65a7a84c78aa9e`.

- 1 question, per-round speed change +15.8%.
- 50 questions, per-round speed change +10.7%.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/d1-2B-mxfp8-ollama.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/d1-2B-mxfp8-ollama-ippon-serve.json)

#### d1-Omni-2B · Ollama MXFP8 against Ollama 0.40.13

The other runtime over native HTTP, one request, and ippon over native HTTP. 1 paired round, 6 timed requests per arm per workload.

Measured 2026-10-29 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `d27c0ad9523d0c5f025ec05c2d65a7a84c78aa9e`.

- 1 question, per-round speed change +11.8%.
- 50 questions, per-round speed change +9.1%.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/d1-omni-2B-mxfp8-ollama.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/d1-omni-2B-mxfp8-ollama-ippon-serve.json)

#### d3-flash · Ollama MXFP8 against Ollama 0.40.13

The other runtime over native HTTP, one request, and ippon over native HTTP. 1 paired round, 6 timed requests per arm per workload.

Measured 2026-10-29 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `d27c0ad9523d0c5f025ec05c2d65a7a84c78aa9e`.

- 1 question, per-round speed change +20.4%.
- 50 questions, per-round speed change +25.7%.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/d3-flash-mxfp8-ollama.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/d3-flash-mxfp8-ollama-ippon-serve.json)

#### d3-flash · checkpoint against oMLX 0.7.2

The other runtime over native HTTP, one request, and ippon over native HTTP. 1 paired round, 6 timed requests per arm per workload.

Measured 2026-10-29 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `d27c0ad9523d0c5f025ec05c2d65a7a84c78aa9e`.

- 1 question, per-round speed change +10.1%.
- 50 questions, per-round speed change +11.5%.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/d3-flash-omlx.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/d3-flash-ippon-serve.json)

#### d1-2B · BF16 against Liquid's code · Transformers 5.19.0 · PyTorch 2.14.1

The other runtime over benchmark HTTP wrapper, one system_one call containing all questions, and ippon over native HTTP. 1 paired round, 6 to 10 timed requests per arm per workload.

Measured 2026-10-29 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `d27c0ad9523d0c5f025ec05c2d65a7a84c78aa9e`.

- 1 question, per-round speed change +71.8%.
- 50 questions, per-round speed change +80.1%.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/d1-2B-tf.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/d1-2B-ippon-serve.json)

#### d1-Omni-2B · BF16 against Liquid's code · Transformers 5.19.0 · PyTorch 2.14.1

The other runtime over benchmark HTTP wrapper, one system_one call containing all questions, and ippon over native HTTP. 1 paired round, 6 to 10 timed requests per arm per workload.

Measured 2026-10-29 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `d27c0ad9523d0c5f025ec05c2d65a7a84c78aa9e`.

- 1 question, per-round speed change +72.2%.
- 50 questions, per-round speed change +80.8%.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/d1-omni-2B-tf.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/d1-omni-2B-ippon-serve.json)

#### d3-flash · full checkpoint against vllm-sr's server · Transformers 5.17.0 · PyTorch 2.14.1

The other runtime over native HTTP, one native /v1/systemone request containing all questions, run in batches of up to eight questions, and ippon over native HTTP. 1 paired round, 6 timed requests per arm per workload.

Measured 2026-10-29 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `d27c0ad9523d0c5f025ec05c2d65a7a84c78aa9e`.

- 1 question, per-round speed change +4.3x.
- 50 questions, per-round speed change +4.5x.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/d3-flash-tf.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/d3-flash-ippon-serve.json)

#### Torchcast Decision 4B · BF16 against Torchcast's server · Transformers 5.8.1 · PyTorch 2.9.1

The other runtime over native HTTP, one native /v1/systemone request containing all questions, run every question in one padded batch, and ippon over native HTTP. 1 paired round, 6 to 9 timed requests per arm per workload.

Measured 2026-10-29 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `d27c0ad9523d0c5f025ec05c2d65a7a84c78aa9e`.

- 1 question, per-round speed change +2.6x.
- 50 questions, per-round speed change +3.6x.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/torchcast-4b-maker-mps.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/torchcast-4b-ippon-serve.json)

#### elephant-1 · BF16 against Base Labs' runtime · Transformers 5.19.0 · PyTorch 2.8.0

The other runtime over native HTTP, one native /v1/systemone request containing all questions, run as forks of the block, and ippon over native HTTP. 1 paired round, 6 to 9 timed requests per arm per workload.

Measured 2026-10-29 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `d27c0ad9523d0c5f025ec05c2d65a7a84c78aa9e`.

- 1 question, per-round speed change +4.5x.
- 50 questions, per-round speed change +6.3x.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/elephant-1-maker-mps.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/elephant-1-ippon-serve.json)

#### d1-2B · Ollama BF16 against Ollama 0.40.13

The other runtime over native HTTP, one request, and ippon over native HTTP. 1 paired round, 6 timed requests per arm per workload.

Measured 2026-10-29 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `d3406e18b95fbd4a84410884782c47a481648582`.

- 1 question, per-round speed change +15.2%.
- 50 questions, per-round speed change −0.9%.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-30-new/r1/d1-2B-bf16-ollama.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-30-new/r1/d1-2B-bf16-ollama-ippon-serve.json)

#### d1-Omni-2B · Ollama BF16 against Ollama 0.40.13

The other runtime over native HTTP, one request, and ippon over native HTTP. 1 paired round, 6 timed requests per arm per workload.

Measured 2026-10-29 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `d3406e18b95fbd4a84410884782c47a481648582`.

- 1 question, per-round speed change +14.8%.
- 50 questions, per-round speed change −1.1%.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-30-new/r1/d1-omni-2B-bf16-ollama.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-30-new/r1/d1-omni-2B-bf16-ollama-ippon-serve.json)

#### GLiDE 9B · BF16 against Fastino's server · Transformers 5.17.0 · PyTorch 2.14.0

The other runtime over native HTTP, one native /v1/systemone request containing all questions, run batched with other requests' questions (its MicroBatcher, under a 16,384-token budget), and ippon over native HTTP. 1 paired round, 6 to 8 timed requests per arm per workload.

Measured 2026-11-07 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `24701b604b40a1c4ab754508865783f3454cab40`.

- 1 question, per-round speed change +4.8x.
- 50 questions, per-round speed change +4.3x.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-11-07-glide/r1/glide-9b-maker-mps.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-11-07-glide/r1/glide-9b-ippon-serve.json)

#### GLiDE 2B · BF16 against Fastino's server · Transformers 5.17.0 · PyTorch 2.14.0

The other runtime over native HTTP, one native /v1/systemone request containing all questions, run batched with other requests' questions (its MicroBatcher, under a 16,384-token budget), and ippon over native HTTP. 1 paired round, 6 to 9 timed requests per arm per workload.

Measured 2026-11-07 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `24701b604b40a1c4ab754508865783f3454cab40`.

- 1 question, per-round speed change +6.7x.
- 50 questions, per-round speed change +6.6x.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-11-07-glide/r1/glide-2b-maker-mps.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-11-07-glide/r1/glide-2b-ippon-serve.json)

#### Jeff-1.1-Qwen-4B · BF16 against Jonathan Geuter's server · Transformers 5.17.0 · PyTorch 2.13.0

The other runtime over native HTTP, one native /v1/systemone request containing all questions, run batched with other requests' questions (jeff.batching), and ippon over native HTTP. 1 paired round, 6 timed requests per arm per workload.

Measured 2026-11-10 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `1c9234ecac4f9c9d2da39be520bfaa45bed4682a`.

- 1 question, per-round speed change +7x.
- 50 questions, per-round speed change +7.1x.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-11-10-creators/r1/jeff-qwen-4b-maker-mps.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-11-10-creators/r1/jeff-qwen-4b-ippon-serve.json)

#### Kev-4B · BF16 against Jared Palmer's server · kev with mlx-lm 0.31.3

The other runtime over native HTTP, one native /v1/systemone request containing all questions, run the state once and every question a row after it, batched with other requests (its MLX backend), and ippon over native HTTP. 1 paired round, 6 timed requests per arm per workload.

Measured 2026-11-10 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `1c9234ecac4f9c9d2da39be520bfaa45bed4682a`.

- 1 question, per-round speed change +32.7%.
- 50 questions, per-round speed change +65.4%.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-11-10-creators/r1/kev-4b-maker-mps.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-11-10-creators/r1/kev-4b-ippon-serve.json)

#### Kev-9B · BF16 against Jared Palmer's server · kev with mlx-lm 0.31.3

The other runtime over native HTTP, one native /v1/systemone request containing all questions, run the state once and every question a row after it, batched with other requests (its MLX backend), and ippon over native HTTP. 1 paired round, 6 timed requests per arm per workload.

Measured 2026-11-10 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `1c9234ecac4f9c9d2da39be520bfaa45bed4682a`.

- 1 question, per-round speed change +30.4%.
- 50 questions, per-round speed change +66.6%.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-11-10-creators/r1/kev-9b-maker-mps.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-11-10-creators/r1/kev-9b-ippon-serve.json)

#### Strands-Decider-E2B · BF16 against AWS Strands' server · Transformers 5.24.0 · PyTorch 2.15.0

The other runtime over native HTTP, one native /v1/systemone request containing all questions, run the state once and its questions in chunks of up to 32 (strands-decider on MPS), and ippon over native HTTP. 1 paired round, 6 to 9 timed requests per arm per workload.

Measured 2026-11-24 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `5ba345cb75f3071879d83bc632f262ea3f104be2`.

- 1 question, per-round speed change +4.2x.
- 50 questions, per-round speed change +2x.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-11-24-strands/r1/strands-e2b-maker-mps.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-11-24-strands/r1/strands-e2b-ippon-serve.json)

#### Strands-Decider-E4B · BF16 against AWS Strands' server · Transformers 5.24.0 · PyTorch 2.15.0

The other runtime over native HTTP, one native /v1/systemone request containing all questions, run the state once and its questions in chunks of up to 32 (strands-decider on MPS), and ippon over native HTTP. 1 paired round, 6 to 9 timed requests per arm per workload.

Measured 2026-11-24 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `5ba345cb75f3071879d83bc632f262ea3f104be2`.

- 1 question, per-round speed change +2.4x.
- 50 questions, per-round speed change +85.5%.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-11-24-strands-e4b/r1/strands-e4b-maker-mps.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-11-24-strands-e4b/r1/strands-e4b-ippon-serve.json)

#### Strands-Decider-12B · BF16 against AWS Strands' server · Transformers 5.24.0 · PyTorch 2.15.0

The other runtime over native HTTP, one native /v1/systemone request containing all questions, run the state once and its questions in chunks of up to 32 (strands-decider on MPS), and ippon over native HTTP. 1 paired round, 6 to 9 timed requests per arm per workload.

Measured 2026-11-24 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `5ba345cb75f3071879d83bc632f262ea3f104be2`.

- 1 question, per-round speed change +45.1%.
- 50 questions, per-round speed change +56.2%.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-11-24-strands-12b/r1/strands-12b-maker-mps.json)
- [Recorded run 2](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-11-24-strands-12b/r1/strands-12b-ippon-serve.json)

### Kettei Core and Kettei Dot on ippon

| Model | 1 question, ms | 50 questions, ms | Requests per median |
|---|---:|---:|---:|
| Kettei Core | 324.8 | 1,211.5 | 6 |
| Kettei Dot | 82.0 | 349.6 | 6 to 10 |

Kettei Core - Measured 2026-10-29 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `d27c0ad9523d0c5f025ec05c2d65a7a84c78aa9e`.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/kettei-core-ippon-serve.json)

Kettei Dot - Measured 2026-10-29 on Apple M5 Pro, 48 GiB, macOS 27.0.1, MLX 0.32.4, ippon commit `d27c0ad9523d0c5f025ec05c2d65a7a84c78aa9e`.
- [Recorded run 1](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-29-readme-targeted/r1/kettei-dot-ippon-serve.json)

### Runtimes that failed

- Clef-Flash 9B · full checkpoint with Cloudflare's code · Transformers 5.10.2 · PyTorch 2.11.0 - out of memory. [Recorded failure](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-10-28-readme/r1/clef-flash-tf.json).
- Jeff-1.1-Medium · BF16 with Jonathan Geuter's server · Transformers 5.17.0 · PyTorch 2.13.0 - out of memory. [Recorded failure](https://github.com/sqwishai/ippon/blob/main/benchmarks/results/m5pro-2026-11-04-jeff/r1/jeff-medium-maker-mps.json).
<!-- /README round -->

<!-- runtime comparisons: benchmarks/kit/summarize.py -->
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
<!-- /runtime comparisons -->

<!-- creators comparisons: benchmarks/kit/summarize.py -->
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
<!-- /creators comparisons -->

## Sources and method

The [24 October summary](../benchmarks/results/m5pro-2026-10-24-rivals/SUMMARY.md) and its
[evidence ledger](../benchmarks/results/m5pro-2026-10-24-rivals/EVIDENCE.json) identify the selected raw files,
SHA-256 hashes and unrounded calculations. The [21 October summary](../benchmarks/results/m5pro-2026-10-21/SUMMARY.md)
and [older evidence ledger](../benchmarks/results/m5pro-2026-10-21/EVIDENCE.json) keep the separate creator and MLX
comparisons. Their source commits and dates do not describe the current checkout.

The newer campaign compares llama.cpp b11669, Ollama 0.40.9 and the oMLX setup pinned to 0.7.2. The older campaign also includes llama.cpp b11757, the community Clef port
and the creators' PyTorch/MPS code. The table names each interface and build. GGUF pairs use
identical files, and Ollama packed pairs use its imported checkpoint.

On a full-precision checkpoint ippon reads the same files as the creator's code and computes in its default
precision, which is fp16 where the model's activations fit ([accuracy](accuracy.md#the-default-is-fp16-where-it-fits-precisionauto)).
A BF16 build therefore runs in fp16 on ippon and in bf16 in its creator's code. Kev-4B timed on ippon in bf16 came
within 3% of its fp16 times on every workload, so the speed comparison holds in either format.

The kit generates seeded support tickets with unique state identifiers and sends the same requests to both arms.
Caller-side timing includes rendering, tokenization and, for servers, the local HTTP round trip. Each runtime loads and
warms up in its own process and environment, then answers timed requests.

The recorded campaigns usually use 2 warm-ups and 10 measured requests per shape. Close comparisons have 2 rounds,
with the runtime order swapped in the second. The older creator comparisons have 1 round. The selected raw files keep
the exact sample counts and any different settings.

The 24 October campaign selected nominal state sizes 100, 1,000 and 4,000, with 1 and 3 questions. These sizes are
targets for building requests. Questions add tokens, so the final prompt length can be longer. The D3 Ollama pair
does not have the 4,000-token state cell. The older campaign adds 300 and 2,000, and separate extended records cover
longer contexts. The tables do not interpolate a missing cell, and a truncated input does not count as a long-context
result.

Run one runtime at a time. Keep unrelated work idle, and record competing activity. The kit's default cooldown is 30
seconds between runtimes. llama.cpp uses all layers on the GPU, a shared KV cache, 16 slots and a prompt batch large
enough for the complete prompt. Since 31 October it also gets an 8 GB prompt cache (`--cache-ram 8192`), which b11960
leaves off for decision models.

Timed again with the cache on (`results/m5pro-2026-10-31-llama-cache`), llama.cpp reused most of the slot of each
re-sent prompt, but it still processed every token. Clef-Flash's re-sent ~1,000-token context took 1,039 ms against
ippon's 76 ms (14x), and H2O-Lightning-4B's took 584 ms against 31 ms (19x), the same as without the cache. d1-2B's
took 188 ms against 120 ms. Each timing comes with the probability differences and the changed top answers. The same
top answer does not mean the same probabilities.

## `ippon bench`

    ippon bench MODEL --sizes 300,1000,2000,4000 --questions 1,3,5 --scenario fresh --n 20 --out r.json

Each scenario (`--scenario`) is a different kind of traffic.

| scenario | requests | what it shows |
|---|---|---|
| `fresh` | a new state each time, starting with a unique identifier | a service answering many different tickets, where no cache can help |
| `repeat` | one state, a new set of questions each time | an agent asking about the same state again (kept prefixes) |
| `grow` | the state grows by about 40 words before each request | an agent's conversation, one turn at a time |
| `idle` | fresh requests after `--gaps` seconds idle (1,5,15,30) | a service answering now and then (GPU residency and clocks) |
| `concurrent` | `--clients` clients (1,4,8) each sending fresh requests one after another, through `serve`'s batcher in the same process | a service with several callers at once, measured as each request's latency, requests per second, and how many each pass of the model answered. `--batch-window-ms` sets the batcher's window, for an A/B of it. On 21 October 2026 on an M5 Pro, a 2 ms window made d1-3B 8-bit 5 to 11% faster with 4 or 8 clients on 100 words and every single client 3 to 7% slower, and d1-omni's single clients 25 to 50% slower, so `serve` keeps 0 |

The requests depend only on the scenario, the shape and `--seed`, so 2 runs with the same seed time the same
requests and their answers can be compared.

## Against other runtimes of the same model (`--compare`)

    ippon bench ~/models/clef-flash-mlx-8bit --compare community-clef=~/venvs/mlxvlm/bin/python

`--compare` times ippon against other runtimes of the same model. An arm is a runtime and the Python that has it
installed. Every arm, ippon included, runs in its own process from its own environment, and ippon never imports
another runtime. In a round, every arm answers the same file of requests and times each call in its own process,
around the call that the model's card documents. Each round starts from a different arm.

Each cell gives each arm's median and ippon's percentage speed change (`(arm median / ippon median − 1) × 100%`,
positive means ippon is faster), with a bootstrap 95% interval and the same change at p95. It also gives the largest
probability difference between their answers and how many most-likely answers differ. The arms are in
`src/ippon/compare_arms/`, one short script each, under one contract.

| arm | runs | for |
|---|---|---|
| `community-clef` | the build's published loader and server (executes model-repository code, so read it first) | community Clef MLX builds |
| `glide-package` | the checkpoint's own `glide` package on MPS in bf16 (executes model-repository code, so read it first) | Fastino's GLiDE models |
| `mlx-vlm-model` | mlx-vlm backbone with ippon's prompts and head | supported multimodal decoder comparisons |
| `mlx-lm-model` | mlx-lm backbone with ippon's prompts and head | bring-your-own decoders |
| `mlx-embeddings-model` | mlx-embeddings encoder with ippon's prompts and head | supported encoders |
| `transformers` | transformers model and head on MPS | supported bring-your-own models |

A model's adapter lists the arms that fit it (`compare_arms`).

## Comparing two configurations (`--ab`)

    ippon bench MODEL --ab "--switch glu=mlx" "--switch glu=kernel" --rounds 4 --n 20
    ippon bench MODEL --ab "" "MLX_METAL_FAST_SYNCH=1"

`--ab` times 2 configurations of ippon against each other. Each arm is a set of extra options, and `NAME=value` words
set environment variables for its process. Each arm runs in `--rounds` fresh processes (loading and warm-up untimed),
in the order A B, then B A, so drift affects both arms. In a round both arms time the same requests. The table gives
both medians and the percentage time change (`(B / A − 1) × 100%`, negative means B is faster), with a bootstrap 95%
interval and the same change at p95. It also gives the largest probability difference between the arms' answers to the
same requests, and the JSON keeps the underlying ratios unchanged.

The promotion rule decides each change to ippon's defaults. B is faster on a cell when its median is at least 3% below
A's and the interval of B/A lies below 1. B is slower when its median (or its p95) is more than 1% above A's, with the
interval above 1. A change ships when it is faster on the shapes it targets, slower on none, and `ippon verify` still
passes. A significant regression takes precedence, so a faster median cannot cancel out a slower p95. If only p95
flags cause a mixed verdict, run more rounds before you decide.

## What is published

Each published benchmark has its raw results (JSON, one file per runtime and request set), a summary table generated
from them, and the machine it ran on. The measured tables never include figures measured elsewhere, such as a
creator's model card or another machine. Where such figures are shown, they are in a separate table that names their
hardware and source.
