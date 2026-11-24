# Models

Each model family has an adapter that reimplements the creator's own prompt and readout code, and each adapter is
checked against that code. "Prompts verified" means every prompt token, readout position and token count equals what
the creator's code gives on the [standard requests](../references/requests/standard.json) that it accepts.
"Answers verified" also puts every probability within the model's bound of the creator's code run in fp32
([accuracy.md](accuracy.md)). ippon claims no speed for a model until `ippon bench` measures it
([benchmarks.md](benchmarks.md)).

| Adapter | Models | Architecture | How it runs | Status |
|---|---|---|---|---|
| `vllm-sr/d3` | vllm-sr/d3-mini and d3-flash (text, images, videos) | Qwen3.5, non-causal attention, with Qwen's vision tower for images and video frames | every question's prompt whole, in shared batches, and an image's or video's rows encoded once | prompts verified (20 of 21, both). Answers 0.0085 and 0.0017 from vllm-sr's fp32 (bounds 0.029, 0.016), images 0.0090 and 0.012 (bounds 0.019, 0.022), videos 0.020 and 0.0022 (bounds 0.048, 0.025) |
| `liquid/d1` | LiquidAI/d1-2B (text, images, videos) and d1-Omni-2B (text, images, audio clips, spoken questions, videos) | LFM3 under set attention. A gated conv and attention tower for images and timestamped video frame pairs. A log-mel front end, strided convolutions and a causal LFM2 for clips | up to 16 questions after their state in one prompt, prompts whole in shared batches, and each medium's rows encoded once | prompts verified (30 of 30, both, plus 17 of 17 with images, 18 of 18 with clips, 12 of 12 with videos). Answers 0.0012 and 0.0018 from the creator's fp32 (bound 0.016, 0.017), images 0.0016 and 0.0048 (0.01, 0.019), clips 0.0010 (0.014), videos 0.0015 and 0.0021 (0.01, 0.0145). Explicit Omni video bf16 0.0156, past bound |
| `torchcast/decision` | torchcast-ai/torchcast-decision-4b (Qwen3.5-4B, v1.3.1) | Qwen3.5, full attention both ways, letter labels at the last token | every question's prompt whole, in shared batches, and a state past 16,384 tokens cut around its middle as the package cuts it | prompts verified (26 of 26 it reads in one pass, against its serving package). Answers 0.0024 from its fp32 (bound 0.043, its own bf16 server's distance). With images at `<image>` marks (else ahead of the state), prompts verified (21) and 0.0080 (bound 0.020) |
| `h2oai/h2o-lightning` | h2oai/h2o-lightning-4b (Qwen3.5-4B), and its Q8_0 GGUF build | Qwen3.5, full attention both ways (its GGUF one way), 255 option labels at the last token | every question's prompt whole, in shared batches, and the GGUF build's questions share the record's reading | prompts verified (26 of 27, as the server takes a choice's criteria as a map only). Answers 0.0040 from its fp32 (bound 0.019), and the Q8_0 file 0.0043 from llama.cpp b11960's. With images ahead of the record, prompts verified (18) and 0.0097 from the server's fp32 (bound 0.079) |
| `baselabs/elephant` | baselabs/elephant-1 (Qwen3.5-9B) | Qwen3.5, block attention, slot labels at the last token | every question its own prompt after one block (system turn and state), the block read once and both ways, each question both ways within itself | prompts verified (27 of 27, against OneJev's qev with the checkpoint's serve.patch). Answers 0.0024 from its fp32 (bound 0.025, its own bf16 distance). With images at the state's `<image:N>` (else its start), prompts verified (19) and 0.012 from its training path's fp32 (bound 0.020) |
| `cloudflare/clef` | Cloudflare/clef-flash (text, images) | Qwen3.5, joint schema head | one prompt for every question, and the head reads every position | prompts verified (20 of 21), head equal to the creator's on a random instance. With images after "STATE:", prompts verified |
| `perplexity/pplx-decider` | perplexity-ai/pplx-decider-flash-v1.3 (text, images) | Qwen3.5, non-causal full attention, 255-code readout | every question its own sequence after the request's context (system turn, images, state), a readout marker after it, in shared batches | prompts verified (15 of 15 of its own requests, against the creator's `pplx_decider` encoder). Answers 0.0023 from the creator's fp32 (bound 0.0169). With images ahead of the state, prompts verified (17) and 0.0030 (bound 0.033) |
| `autotrust/decider` | autotrust/AutoTrust-Decider-Mini-9B (text) | Qwen3.5, causal, its LoRA merged at load, a 24-slot fp32 head at the last token | every question its own prompt, the state after the question's kind | prompts verified (32 of 32, 14 refusals included, against the model card's System 1 code). Answers 0.0019 from its fp32 (bound 0.014). Scores take 6 levels and choices 2 to 16 options, as the card's code does |
| `fastino/glide` | fastino/GLiDE-2B and GLiDE-9B (text) | Qwen3.5 with non-causal full attention (the 9B reading 2 blocks of middle layers twice), 255-code readout | every question's prompt whole, in shared batches | prompts verified (20 of 21, as the creator's code takes a choice's criteria as a map only, against Fastino's `glide` package). Answers 0.0022 and 0.0063 from its fp32 (bounds 0.016, 0.011) |
| `jgeuter/jeff` | jgeuter/Jeff-1.1-Medium (Gemma 4 12B) and Jeff-1.1-Qwen-4B (Qwen3.5 4B), text | the base with a merged LoRA and a decision head, its full-attention layers reading the prompt both ways | every question its own prompt | Medium: answers 0.0474 from the creator's bf16 (bound 0.057), no fp32 reference. Qwen-4B: answers 0.0033 from its fp32 (bound 0.01) |
| `strands/decider` | StrandsAgents/strands-decider-{E2B,E4B,12B}-gemma4-v2-2611 (text) | Gemma 4 (E2B/E4B per-layer embeddings and KV-shared layers; 12B dense), every layer in prefix-bidirectional mode with no sliding window, 255-code readout (`readout.safetensors`) | the state read once both ways, then each question over the state and both ways within itself; E2B/E4B keep their per-layer embedding tables on host | prompts verified (430 of 430 text golden requests on E2B, E4B and 12B). Answers against the creator's bf16 golden vectors: E2B 0.0031 mean TV (max 0.025, 180/180 top-1), E4B 0.0037 mean TV (max 0.024, 90/90 top-1), 12B 0.0039 mean TV (max 0.034, 89/90 top-1 up to 32.7K tokens) |
| `jaredpalmer/kev` | jaredpalmer/kev-0.8b, kev-4b and kev-9b (Qwen3.5 bases), text | Qwen3.5 with a LoRA adapter and a pointer head | the state read once, every question its own causal row after it | answers 0.0026, 0.0023 and 0.0027 from fp32 (bounds 0.020, 0.010, 0.011) |

Sqwish Kettei Core and Kettei Dot run on ippon. Both have verified prompts on 20 accepted requests. Their answers
differ from the SDK's fp32 by at most 0.0067 and 0.0026 (bounds 0.037 and 0.032).

Quantized MLX builds load as they are, such as the community build mlx-community/clef-flash-8bit (its packed weights
and Qwen3.5's norm weights stored as 1 + w). ippon checks it against the runtime it was published for, in fp32 on its
own weights (`verify --against`, [accuracy.md](accuracy.md)). It passes, and ippon's fp16 lands closer to the exact
computation than that runtime as published.

GGUF builds (llama.cpp's format) load as their creators publish them, in Q8_0, which ippon reads as MLX's affine
groups of 32. Load one by name, as in `Decider.load("LiquidAI/d1-2B-GGUF:Q8_0")`, or from the downloaded file. Each
build is pinned (repository, revision, file, sha256, header digest) to its creator's repository at a revision. ippon
reads the config and tokenizer at that revision. ippon renames the GGUF's tensors to the names in the creator's
checkpoint and undoes llama.cpp's conversion. The adapter then reads them as it reads the checkpoint.

Before anything loads, ippon checks the GGUF's metadata against that config. The metadata includes dimensions, heads,
RoPE, causality, and the decision settings such as calibration temperatures. ippon refuses other quantized types (Q3_K,
IQ quants), and the refusal names the Q8_0 file or the MLX builds instead. ippon refuses them because MLX would unpack
them to fp16, the model at full size. Against each creator's fp32 reference, the GGUF builds measure as follows.

| Build | Worst difference | Bound | |
|---|---|---|---|
| LiquidAI/d1-2B-GGUF, Q8_0 | 0.0102 | 0.0162 | passes, 0.0023 from llama.cpp b11960 on the same file |
| LiquidAI/d1-Omni-2B-GGUF, Q8_0 | 0.045 | 0.0172 | the build's own, as llama.cpp b11960 on the same file is 0.0033 from ippon and 0.043 from fp32 |

ippon reads a GGUF's tensors itself. Where MLX reads a file, ippon reads the same values bit for bit. ippon keeps BF16
as BF16, which MLX rounds to fp16.

ippon checks ggml-org/Clef-Flash-GGUF (Q8_0) in parts, because Clef-Flash's creator publishes no fp32 reference that
fits here (the model is 38 GB in fp32). All 427 backbone and 122 head tensors renamed back equal the creator's
checkpoint within Q8_0's rounding (0.40% of a tensor's largest value). This holds with llama.cpp's Qwen3.5 transforms
undone. The GGUF stores norms as 1 + w and A_log as -exp(A_log). It stores the Gated DeltaNet's 32 value heads tiled
over its 16 key heads. ippon moves them back as whole rows and packed column groups. ippon's fp32 hidden states on it
are 2.3e-6 from the float64 numpy Qwen3.5 [reference](../scripts/check_qwen35_float64.py). Its answers are 0.031 from
the full-precision checkpoint's, and no top answer changed (Cloudflare's own, through ippon, on the 20 reference requests).

Mixtures of experts (Qwen3.5-MoE) are not yet verified on a real checkpoint, because no fp32 reference of a 35B model
fits a 48 GB Mac. The Qwen3.5 architecture runs the mixture-of-experts MLP
of Qwen3.5-35B-A3B and the decision models built on it (none supported yet). It uses transformers' router (softmax in
fp32, top-k renormalized). It sends each token's experts through MLX's gather matmul with the rows sorted by expert,
and runs the shared expert behind its sigmoid gate.

Expert weights are stored one tensor per expert, as transformers 5 fuses them, or as mlx-lm stacks them. With `q8`,
ippon quantizes them at load. The MoE MLP equals an independent numpy model on tiny random checkpoints (exactly in fp32
on the CPU and on the M4's GPU kernels).

Gemma 4 (E2B/E4B per-layer embeddings and KV-shared layers, dense 12B/31B, and the 26B-A4B mixture of experts) equals an
independent numpy model on tiny random checkpoints. Strands Decider v2 E2B and Jeff-1.1-Medium run on it, checked
against their creators' bf16 references, with no fp32 reference. That numpy model equals
transformers' own Gemma4TextModel. The [reference check](../scripts/check_gemma4_numpy.py) agrees within 1e-5 in
float64, the size of transformers' float32 norms and RoPE tables. A wrong formula is off by about 1.

## Running each model

Every build below loads by name (`Decider.load(...)`, `ippon run`, `ippon serve`) or from a downloaded folder or file.
Each Hub build loads the commit its answers were checked at, shown after `@`, so a pull never takes a later change to
the model's main branch.
ippon downloads only weights, configs and tokenizers. Use `ippon pull --list` for `FAMILY:TAG` aliases and the
recommended build for each bare family. A tag describes the published weights, while `--precision` selects computation
or explicit weight conversion. Before anything downloads, `ippon pull NAME --dry-run` tells you what each precision
needs on your Mac and what the model's table below records of it. Memory is what the weights take, and a request adds its
activations.

Each model has its own section, which you can link to. In its Checked column, "verified" means within the bound of the creator's fp32 code. For a quantized build, it means
within the bound of the code it was published for. "prompts" means token for token the creator's, with answers not
checked against fp32. "past bound" means checked, and moved past it. "experimental" means that the build loads and
runs, but a check that its note names is still open or past its bound.

<!-- builds: scripts/models_table.py -->
### Clef-Flash

`ippon pull clef-flash` downloads the build that `ippon pull --list` stars. Its creator publishes it at
[Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash), and its licence is Apache-2.0.

| Load | Weights | Checked |
|---|---:|---|
| `Cloudflare/clef-flash@8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3` | 19.1 GB (bf16) | prompts (and its head on a random instance) |
| `mlx-community/clef-flash-8bit@5a195dc147926b63b1d139691ed4480ca9f92c0e` | 10.7 GB | verified (within the bound of the build's own code) |
| `ggml-org/Clef-Flash-GGUF:Q8_0` | 9.7 GB | verified (every tensor, and 0.031 from the full checkpoint) |
| `ollama://clef-flash:9b-mxfp8` | 12.4 GB | experimental (0.00113 from the same weights in fp32. Ollama 0.40.13's own answers 0.378 from them, past 0.01. Original fp32 quality unmeasured) |

### d1-2B

`ippon pull d1-2b` downloads the build that `ippon pull --list` stars. Its creator publishes it at
[LiquidAI/d1-2B](https://huggingface.co/LiquidAI/d1-2B), and its licence is LFM Open License v1.0.

| Load | Weights | Checked |
|---|---:|---|
| `LiquidAI/d1-2B@10e7671d44e4b37700020b1eed3f594c5f2af0b4` | 3.8 GB (bf16) | verified (0.0012, bound 0.016. Images 0.0016, bound 0.01. Videos 0.0015, bound 0.01) |
| `LiquidAI/d1-2B-GGUF:Q8_0` | 1.8 GB | verified (0.010, bound 0.016. 0.0023 from llama.cpp b11960's on the same file) |
| `ollama://d1:2b` | 3.9 GB | verified (Ollama's default d1 build, BF16, at 0.0012, bound 0.016. Ollama 0.40.13's own answers 0.0125) |
| `ollama://d1:2b-mxfp8` | 2.4 GB | past bound (0.10, bound 0.016, one top answer changed. Ollama 0.40.13's own answers 0.081 on the requests it takes) |

### d1-Omni-2B

`ippon pull d1-omni-2b` downloads the build that `ippon pull --list` stars. Its creator publishes it at
[LiquidAI/d1-Omni-2B](https://huggingface.co/LiquidAI/d1-Omni-2B), and its licence is LFM Open License v1.0.

| Load | Weights | Checked |
|---|---:|---|
| `LiquidAI/d1-Omni-2B@c8d48c08dada0191c6b362820f055cfb49c99426` | 4.3 GB (bf16) | verified (0.0018, bound 0.017. Images 0.0048, bound 0.019. Clips 0.0010, bound 0.014. Videos 0.0021, bound 0.0145. Explicit video bf16 0.0156, past bound) |
| `LiquidAI/d1-Omni-2B-GGUF:Q8_0` | 1.8 GB | past bound (the build's own Q8_0 at 0.045 from the full checkpoint's fp32, bound 0.017. 0.0033 from llama.cpp b11960's on the same file) |
| `ollama://d1-omni:2b` | 4.4 GB | verified (Ollama's default d1-omni build, BF16, at 0.0018, bound 0.017. Ollama 0.40.13's own answers 0.029, past it) |
| `ollama://d1-omni:2b-mxfp8` | 2.9 GB | past bound (0.15, bound 0.017. Ollama 0.40.13's own answers 0.15 on the requests it takes) |

### pplx-decider flash v1.3

`ippon pull pplx-flash` downloads the build that `ippon pull --list` stars. Its creator publishes it at
[perplexity-ai/pplx-decider-flash-v1.3](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.3), and its licence is Apache-2.0.

| Load | Weights | Checked |
|---|---:|---|
| `perplexity-ai/pplx-decider-flash-v1.3@d06fb3f2d97e0a52325a0d463c450cf0338f44fe` | 9.1 GB (bf16) | verified (0.0023, bound 0.017. Images 0.0030, bound 0.033) |

### elephant-1

`ippon pull elephant` downloads the build that `ippon pull --list` stars. Its creator publishes it at
[baselabs/elephant-1](https://huggingface.co/baselabs/elephant-1), and its licence is MIT.

| Load | Weights | Checked |
|---|---:|---|
| `baselabs/elephant-1@b3f8a93462fb1051a9de031e94a7ff380e254072` | 18.8 GB (bf16) | verified (0.0024, bound 0.025. Images 0.012, bound 0.020 (bf16 0.048, past it)) |

### d3-mini

`ippon pull d3-mini` downloads the build that `ippon pull --list` stars. Its creator publishes it at
[vllm-sr/d3-mini](https://huggingface.co/vllm-sr/d3-mini), and its licence is Apache-2.0.

| Load | Weights | Checked |
|---|---:|---|
| `vllm-sr/d3-mini@97dd054df4b3e0b3beb6af23ab8c82b7c0052e0e` | 9.1 GB | verified (0.0085, bound 0.029. Images 0.0090, bound 0.019. Videos 0.020, bound 0.048) |
| `ollama://d3:4b-mxfp8` | 5.6 GB | past bound (0.062 from the creator's fp32, bound 0.029, and 0.0059 from the build's own weights in fp32. Ollama 0.40.13's own answers 0.050 on the requests it takes) |
| `ollama://d3:4b-nvfp4` | 4.0 GB | experimental (0.00895 from the same weights in fp32, bound 0.01. Up to 0.018 from Ollama's own answers. Original fp32 quality unmeasured. Loads only by name, with a drift warning) |

### d3-flash

`ippon pull d3-flash` downloads the build that `ippon pull --list` stars. Its creator publishes it at
[vllm-sr/d3-flash](https://huggingface.co/vllm-sr/d3-flash), and its licence is Apache-2.0.

| Load | Weights | Checked |
|---|---:|---|
| `vllm-sr/d3-flash@13dc1cc8ef23caefa765907e1d764751e5629f7d` | 16.8 GB | verified (0.0017, bound 0.016. Images 0.012, bound 0.022. Videos 0.0022, bound 0.025) |
| `ollama://d3:9b-mxfp8` | 10.1 GB | past bound (0.044, bound 0.016, one top answer changed. Ollama 0.40.13's own answers 0.039 on the requests it takes) |

### Torchcast Decision 4B

`ippon pull torchcast` downloads the build that `ippon pull --list` stars. Its creator publishes it at
[torchcast-ai/torchcast-decision-4b](https://huggingface.co/torchcast-ai/torchcast-decision-4b), and its licence is Apache-2.0.

| Load | Weights | Checked |
|---|---:|---|
| `torchcast-ai/torchcast-decision-4b@492bfb19662f3c6e601cc9c4bd2911ac95609ed4` | 10.4 GB (bf16) | verified (0.0024, bound 0.043. Images 0.0080, bound 0.020 (bf16 0.031, past it)) |

### H2O-Lightning-4B

`ippon pull h2o` downloads the build that `ippon pull --list` stars. Its creator publishes it at
[h2oai/h2o-lightning-4b](https://huggingface.co/h2oai/h2o-lightning-4b), and its licence is Apache-2.0.

| Load | Weights | Checked |
|---|---:|---|
| `h2oai/h2o-lightning-4b@4ab7b8d60408495cb2a8c14a9454a1ac30d8c12f` | 9.3 GB (bf16) | verified (0.0040, bound 0.019. Images 0.0097, bound 0.079 (one way 0.0035, bound 0.030)) |
| `h2oai/h2o-lightning-4b-gguf:Q8_0` | 4.5 GB | past bound (the build's own Q8_0, read one way, at 0.024, bound 0.0104. 0.0043 from llama.cpp's on the same file) |

### Jeff-1.1-Medium

`ippon pull jeff` downloads the build that `ippon pull --list` stars. Its creator publishes it at
[jgeuter/Jeff-1.1-Medium](https://huggingface.co/jgeuter/Jeff-1.1-Medium), and its licence is CC BY-NC 4.0.

| Load | Weights | Checked |
|---|---:|---|
| `jgeuter/Jeff-1.1-Medium@540c40924dd67c715d3367309ba6744bac754b00` | 25.4 GB (bf16) | verified (0.0474 from the creator's bf16 on the Mac's GPU, bound 0.057: its fp32 fits no 48 GB Mac) |

### Jeff-1.1-Qwen-4B

`ippon pull jeff-qwen-4b` downloads the build that `ippon pull --list` stars. Its creator publishes it at
[jgeuter/Jeff-1.1-Qwen-4B](https://huggingface.co/jgeuter/Jeff-1.1-Qwen-4B), and its licence is CC BY-NC 4.0.

| Load | Weights | Checked |
|---|---:|---|
| `jgeuter/Jeff-1.1-Qwen-4B@5401940b5a8c959cba82187ebfa2dce5a21abc0a` | 9.1 GB (bf16) | verified (0.0033, bound 0.01) |

### Kev 0.8B

`ippon pull kev-0.8b` downloads the build that `ippon pull --list` stars. Its creator publishes it at
[jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b), and its licence is Apache-2.0.

| Load | Weights | Checked |
|---|---:|---|
| `jaredpalmer/kev-0.8b@bf75a6a8848ea6960ff2ed108d9ed44c2941174f` | 1.8 GB (bf16, its base included) | verified (0.0026, bound 0.020) |

### Kev 4B

`ippon pull kev-4b` downloads the build that `ippon pull --list` stars. Its creator publishes it at
[jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b), and its licence is Apache-2.0.

| Load | Weights | Checked |
|---|---:|---|
| `jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52` | 9.5 GB (bf16, its base included) | verified (0.0023, bound 0.010) |

### Kev 9B

`ippon pull kev-9b` downloads the build that `ippon pull --list` stars. Its creator publishes it at
[jaredpalmer/kev-9b](https://huggingface.co/jaredpalmer/kev-9b), and its licence is Apache-2.0.

| Load | Weights | Checked |
|---|---:|---|
| `jaredpalmer/kev-9b@db029f08b290afd9fee4aa4bbcd9ae48602d1eb0` | 19.5 GB (bf16, its base included) | verified (0.0027, bound 0.011) |

### GLiDE 2B

`ippon pull glide-2b` downloads the build that `ippon pull --list` stars. Its creator publishes it at
[fastino/GLiDE-2B](https://huggingface.co/fastino/GLiDE-2B), and its licence is Fastino Community License.

| Load | Weights | Checked |
|---|---:|---|
| `fastino/GLiDE-2B@00f122c120cdeed82284106cb77d9bab5a50ad16` | 3.8 GB (bf16) | verified (0.0022, bound 0.016) |

### GLiDE 9B

`ippon pull glide-9b` downloads the build that `ippon pull --list` stars. Its creator publishes it at
[fastino/GLiDE-9B](https://huggingface.co/fastino/GLiDE-9B), and its licence is Fastino Community License.

| Load | Weights | Checked |
|---|---:|---|
| `fastino/GLiDE-9B@ab54df5acba0eebf7df45560ffdaacd64e32df3b` | 15.9 GB (bf16) | verified (0.0063, bound 0.011, its recurrent depth included) |

### AutoTrust-Decider-Mini-9B

`ippon pull autotrust-mini-9b` downloads the build that `ippon pull --list` stars. Its creator publishes it at
[autotrust/AutoTrust-Decider-Mini-9B](https://huggingface.co/autotrust/AutoTrust-Decider-Mini-9B), and its licence is Apache-2.0.

| Load | Weights | Checked |
|---|---:|---|
| `autotrust/AutoTrust-Decider-Mini-9B@7ac669b8fb476ba926491e3d1c39f9330d68dbed` | 18.1 GB (bf16) | verified (0.0019, bound 0.014) |

### Strands-Decider-E2B

`ippon pull strands-e2b` downloads the build that `ippon pull --list` stars. Its creator publishes it at
[StrandsAgents/strands-decider-E2B-gemma4-v2-2611](https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v2-2611), and its licence is Apache-2.0.

| Load | Weights | Checked |
|---|---:|---|
| `StrandsAgents/strands-decider-E2B-gemma4-v2-2611@eca68159a790d1e7ff93559d166dc3fbe24d5975` | 10.3 GB (bf16) | verified (0.0031 mean TV (max 0.025, 180/180 top-1) against the creator's bf16 golden vectors; prompts verified (430 of 430)) |

### Strands-Decider-E4B

`ippon pull strands-e4b` downloads the build that `ippon pull --list` stars. Its creator publishes it at
[StrandsAgents/strands-decider-E4B-gemma4-v2-2611](https://huggingface.co/StrandsAgents/strands-decider-E4B-gemma4-v2-2611), and its licence is Apache-2.0.

| Load | Weights | Checked |
|---|---:|---|
| `StrandsAgents/strands-decider-E4B-gemma4-v2-2611@1717bca732ba4ef8c8c531fbc9875fafaf858811` | 16.0 GB (bf16) | verified (0.0037 mean TV (max 0.024, 90/90 top-1) against the creator's bf16 golden vectors; prompts verified (430 of 430)) |

### Strands-Decider-12B

`ippon pull strands-12b` downloads the build that `ippon pull --list` stars. Its creator publishes it at
[StrandsAgents/strands-decider-12B-gemma4-v2-2611](https://huggingface.co/StrandsAgents/strands-decider-12B-gemma4-v2-2611), and its licence is Apache-2.0.

| Load | Weights | Checked |
|---|---:|---|
| `StrandsAgents/strands-decider-12B-gemma4-v2-2611@41d5dc0d20a4deb2835690eb7529ee92463f6c9f` | 23.9 GB (bf16) | verified (0.0039 mean TV (max 0.034, 89/90 top-1, up to 32.7K tokens) against the creator's bf16 golden vectors; prompts verified (430 of 430)) |

### Kettei Core

`ippon pull kettei-core` downloads the build that `ippon pull --list` stars. Its creator publishes it at
[Sqwish/Kettei-Core](https://huggingface.co/Sqwish/Kettei-Core), and its licence is Kettei Community Model License 1.0.

| Load | Weights | Checked |
|---|---:|---|
| `Sqwish/Kettei-Core@5a9821633c8bb15b92b677d368083ec04720d164` | 10.1 GB (bf16) | verified (0.0067, bound 0.037 (bf16 0.045, past it)) |

### Kettei Dot

`ippon pull kettei-dot` downloads the build that `ippon pull --list` stars. Its creator publishes it at
[Sqwish/Kettei-Dot](https://huggingface.co/Sqwish/Kettei-Dot), and its licence is Kettei Community Model License 1.0.

| Load | Weights | Checked |
|---|---:|---|
| `Sqwish/Kettei-Dot@a8ef982f8736cbac710899810f87bb53668be6d1` | 2.2 GB (bf16) | verified (0.0026, bound 0.032) |
<!-- /builds -->

```python
from ippon import Decider, YesNo

with Decider.load("ggml-org/Clef-Flash-GGUF:Q8_0") as model:  # the file llama.cpp runs, read as it is
    result = model.decide("...", {"escalate": YesNo("Should a human take over?")})
```

```sh
ippon serve LiquidAI/d1-2B-GGUF:Q8_0 --port 8000    # POST /v1/systemone
```

Perplexity's models enforce the creator's 32,768-token per-sequence limit. Each question is its own sequence, with
the request's complete state and images. If one question exceeds the limit, the whole request fails with
`context_too_long` before inference, and nothing is truncated. Usage counts every sequence's input tokens. To set
another limit, as the creator's `max_length` does, put `{"max_length": 8192}` in a `ippon.json` file (or
`--settings FILE`). `max_length: null` disables the limit, as in the creator's API.

## Limits you may meet

These come from the models, not from ippon, and they were measured on real photos, clips and calls.

- d3-mini and d3-flash read 32 frames of a video at most, spread evenly over it, as vllm-sr's own code does. Its
  server refuses videos over 300 seconds, and ippon reads any length. In a 3- or 10-minute video, a 4.5-second event
  fell between the frames read and was missed every time. d1-2B and d1-Omni-2B read up to 768 frames and found it.
  For long videos, send the part you ask about, or use d1.
- A screenshot of a 404 or a blocked page, and a reCAPTCHA page (d3-mini), can be judged as a page that loaded. A phone
  number in a document and a cut-off corner of a document can be missed.
- d1-Omni-2B is weaker on screen recordings (73%) and on questions about the end of a video.
- d1-Omni-2B's "is this spam?" flags genuine voicemails at 0.5, and sounds that resemble each other fool it (a cat as a
  crying baby at 0.92, geese as a car horn at 0.89). Its "alarm?" fires on doorbells and phone rings.
- Both models call fireworks fire, and d1-Omni-2B calls fireplaces and bonfires dangerous.

A threshold of 0.5 is rarely the right one. `ippon eval examples.jsonl --model MODEL` picks one for each question from
your own labelled examples and checks it on examples held out from that choice.

## What each adapter accepts beyond its creator's code

ippon answers a few requests that a creator's code refuses, with the closest prompt that the creator's format allows.
It does this because it takes the System One schema as a whole. Each case is documented in its adapter's module
docstring.

- a choice given as a list of labels (vllm-sr d3, Liquid d1, Clef-Flash, Fastino GLiDE), shown as the labels
  without descriptions
- a request without `model` (Clef-Flash's code requires one).

ippon refuses everything else that a creator's code refuses, with `InvalidRequest`. This includes questions without
instructions where its prompt needs them, and states it cannot read.

## Licences

The adapters are Apache-2.0, as ippon is. Weights keep their creators' licences, and ippon downloads them from the
creators' repositories. ippon redistributes none. Liquid's d1 models are under the LFM Open License v1.0, and Jonathan
Geuter's Jeff models are under CC BY-NC 4.0. Base Labs' elephant-1 is under MIT, and Fastino's GLiDE models are under the
Fastino Community License. Sqwish's Kettei Core and Dot are under the Kettei Community Model License 1.0. The others
listed here are under Apache-2.0.
