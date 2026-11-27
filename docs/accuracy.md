# Accuracy

Verification measures how closely ippon computes a published model. The contract below keeps 3 things apart - prompt
parity, numerical execution, and the distance of a quantized build from its creator's original checkpoint.
[models.md](models.md) records the status of each build and identifies experimental builds and builds past their bound.

## The reference

The reference is the model creator's own code, run in fp32 on a fixed set of requests. fp32 is the reference because
it gives the model's answer with the least rounding. bf16, fp16 and quantized runs, the creator's own included, are
all approximations of it.

Jeff-1.1-Medium is the exception. Its fp32 weights need the whole of a 48 GB Mac, so its reference is the creator's
code in bf16 on the Mac's GPU, and its bound is how far that code moves when it runs on the CPU
(`references/jeff/make_reference.py`). That check shows ippon is as close to the creator's GPU run as the creator's own
CPU run is. It does not measure a distance from fp32.

## The checks

Verification checks these 4 things on every request of the set.

1. The prompt tokens are the same. The format renders exactly the token ids that the creator's code renders. A
   different token anywhere makes a different prompt, so this check is exact and has no bound.
2. Every probability is within the bound of the reference's probability.
3. The most likely option is the same, except near ties. If the reference's top 2 probabilities differ by no more
   than the bound, verification reports a changed top answer and does not fail.
4. The input token count is the same, when the reference supplies one.

`ippon verify` performs these checks. A tokens-only reference shows that the prompt and the token count match, and it
says nothing about answer accuracy.

## The bound

The reference file stores the bound. The creator-reference generators usually use the larger of 0.01 and the
creator's BF16 distance from FP32 plus 0.0005, rounded to 4 decimal places. That small allowance is part of the
recorded calculation, and it does not permit a wider bound for a reference that fails. Other fixtures can keep an
established bound explicitly. For a same-checkpoint check, a build's own published runtime can supply the comparison
precision in place of BF16. The generic BYO generator records the larger of 0.01 and the BF16 distance, rounded to 4
decimal places, without that allowance.

## The request set

For each model, the set covers every question type the model supports, the edge cases of its format and realistic
requests. The edge cases include the order of yes/no criteria, missing descriptions, the most options the model takes,
a blank or missing state and non-ASCII text. The longer requests are a support ticket and states of about 300, 1,000,
3,000 and 6,000 tokens, with 1 to 40 questions (`references/requests/make_standard.py`). The set and the reference
probabilities are committed with the model's support, so anyone can run the check again.

## Reporting

Each model's documentation lists, for each precision, the worst difference from fp32, how many requests exceed the
bound and how many top answers change. A precision that fails is listed with its numbers and marked as failed in the
build catalog. Explicit precision options do not run an automatic reference check, so run `ippon verify` for the model
and precision that you intend to use.

## Prompts past pplx-decider's limit

ippon's handling of Perplexity's sequence limit is checked against its creator's `Encoder.plan`. On the M5 Pro 48 GB,
macOS 27.0.1, MLX 0.32.4, a numerical fixture with an explicit 256-token limit reads 7 questions as 7 sequences, each
with the complete state. It stays within 0.00076 of the creator's FP32 probabilities (bound 0.0169), and its
invariance differs by at most 0.0000021. If a question cannot fit, ippon refuses the whole request before inference.

A token-only fixture tests the actual 32,768-token default. It reads 2 questions of 32,664 tokens each as 2 sequences
and refuses a 32,835-token question. The default-limit test runs no model forward pass.
[Fixtures, reproduction and reports](../references/perplexity/README.md) keep those checks apart.

## The default is fp16 where it fits (`precision="auto"`)

FP16 keeps 11 significant bits to BF16's 8, and the recorded FP16 runs below are closer to creator FP32 than the
corresponding BF16 runs. The improvement varies by model. Recorded M5 Pro d1-3B timings at 300 and 2,000 words were the
same within noise in both dtypes. The table gives the largest probability differences on the recorded standard
requests.

| Model | fp16 | bf16 | Recorded bound |
|---|---|---|---|
| Perplexity pplx-decider flash v1.3 (its own 15 requests, one question per sequence) | 0.0023 | 0.0111 | 0.0169 |
| Bespoke Nimble-9B v3, which ippon no longer supports (27 requests, the standard ones it accepts and 7 of its own) | 0.0101 | 0.0593 | 0.6396 |
| Torchcast Decision 4B (26 requests, the standard ones it reads in one pass and 6 of its own) | 0.0024 | 0.0157 | 0.0429 |
| H2O-Lightning-4B, two-way (27 requests, the standard ones and 6 of its own) | 0.0040 | 0.0204 | 0.0193 |
| Base Labs elephant-1, block attention (27 requests, the standard ones and 6 of its own) | 0.0024 | 0.0136 | 0.0249 |
| Liquid d1-2B (28 requests, the standard ones it accepts and 8 of its own, up to 33 questions in 3 rows) | 0.0012 | 0.0160 | 0.0162 |
| Liquid d1-Omni-2B (the same 28) | 0.0018 | 0.0072 | 0.0172 |
| Liquid d1-2B, images (the 17 image requests, with EXIF, odd aspects, 12 MP under d1's cap, several images) | 0.0016 | 0.0090 | 0.01 |
| Liquid d1-Omni-2B, images (the same 17) | 0.0048 | 0.0232 | 0.0185 |
| Liquid d1-Omni-2B, audio clips (18 audio requests, with 0.3 to 40 s, PCM16 and float32, 22,050 Hz stereo, silence, a tone, spoken questions) | 0.0010 | 0.0074 | 0.0144 |
| Liquid d1-2B, videos (12 requests, 7 clips, images and videos together, 17 questions in 2 rows) | 0.001456 | 0.005923 | 0.0100 |
| Liquid d1-Omni-2B, videos (the same 12, with no video soundtrack) | 0.002105 | 0.015616 (past bound on 2 requests) | 0.0145 |
| vllm-sr d3-mini (the 20 standard requests it accepts) | 0.0085 | 0.0219 | 0.0291 |
| vllm-sr d3-flash (the same 20) | 0.0017 | 0.0185 | 0.0161 |
| vllm-sr d3-mini, images (the 17 image requests of media-qwen.json) | 0.0090 | 0.0247 | 0.0185 |
| vllm-sr d3-mini, videos (14 video requests in MP4, QuickTime, WebM and Matroska, 4 to 32 frames, images and videos together) | 0.0197 | 0.0585 | 0.0477 |
| vllm-sr d3-flash, images (the same 17) | 0.0119 | 0.0195 | 0.0219 |
| vllm-sr d3-flash, videos (the same 14) | 0.0022 | 0.0346 | 0.0249 |
| Sqwish Labs' Kettei Core (22 requests: the 20 standard ones it accepts, and 2 with 120 and 255 options) | 0.0109 | 0.0362 | 0.0369 |
| Sqwish Labs' Kettei Dot (the same 22) | 0.0048 | 0.0428 (past bound on 40-questions) | 0.0324 |
| Base Labs elephant-1, images (19 image requests at its `<image:N>` placeholders or the state's start) | 0.0122 | 0.0479 | 0.0195 |
| H2O-Lightning-4B, two-way, images (18 image requests) | 0.0097 | 0.0750 | 0.0787 |
| H2O-Lightning-4B, one-way, images (the same 18) | 0.0035 | 0.0277 | 0.0302 |
| Torchcast Decision 4B, images (21 image requests) | 0.0080 | 0.0312 | 0.0204 |
| Perplexity pplx-decider flash v1.3, images (the 17 image requests of media-qwen.json) | 0.0030 | 0.0226 | 0.0330 |

In bf16, ippon is about as far from fp32 as the creators' own bf16 runs, and closer for some. In fp16 it is several
times closer than any of them. Nimble v3's engine (a model ippon no longer supports) in bf16 (its CUDA precision, here
on the Mac's GPU through MPS) lands 0.64 from its fp32 and changes answers, where ippon's bf16 lands 0.059 and its fp16
0.010. Torchcast's own bf16 server on the Mac's GPU lands 0.042 from its fp32, where ippon's bf16 lands 0.016 and its
fp16 0.0024.

Liquid's own d1-2B and d1-Omni-2B in bf16 on the Mac's GPU (the way its card runs them on a Mac) land 0.0157 and
0.0167 from their fp32. Base Labs' qev runtime in bf16 on the Mac's GPU lands 0.024 from its fp32 for elephant-1.
vllm-sr's own d3-mini and d3-flash in bf16 on the Mac's GPU land 0.0180 and 0.0214 from their fp32 on the images, and
0.0472 and 0.0244 on the videos.

The video references use short clips (references/media/videos). The largest is 1280 x 720 in d3's references and
426 x 240 in d1's, so a 1080p or 4K video has no reference of its own. Its frames take the same path to the tower,
resized to the model's frame budget, and d3 reads 32 of them at most (see [models](models.md)).

ippon also ran the quantized builds that the creators publish, computing in fp16, against the same fp32 references.

| Build | Worst difference | Bound | Top answers |
|---|---|---|---|
| h2oai/h2o-lightning-4b-gguf Q8_0, one-way, against the server's one-way fp32 | 0.0245 (19 November 2026) | 0.0104 | top answers unchanged on all 27 requests, and 3 past the bound (noul 0.0227, several-questions 0.0245, state-1000 0.0105). 0.0043 from llama.cpp b11960 on the same file |
| LiquidAI/d1-2B-GGUF Q8_0 | 0.0102 | 0.0162 | passes, 0.0023 from llama.cpp b11960 on the same file (its one-question requests) |
| LiquidAI/d1-Omni-2B-GGUF Q8_0 | 0.045 | 0.0172 | json-state changes its top answer, as llama.cpp's does. ippon is 0.0033 from llama.cpp b11960 on the same file (its one-question requests), which is itself 0.043 away |
| LiquidAI/d1-2B-GGUF Q8_0 with its Q8_0 mmproj, images (17) | 0.0138 | 0.01 | past the bound on one 3-question request (wright-three-questions), against 0.0016 for the checkpoint. llama.cpp b11960 on the same files is within 0.0003 of ippon where it reads the image as the creator does, but it ignores EXIF rotation (0.998 away) and scales 16-bit grayscale where Liquid's code clips it to 8 bits, nearly all white (0.94 away) |
| LiquidAI/d1-Omni-2B-GGUF Q8_0 with its Q8_0 mmproj, images (17) | 0.0085 | 0.0185 | passes |
| LiquidAI/d1-Omni-2B-GGUF Q8_0 with its Q8_0 mmproj, audio (18) | 0.0075 | 0.0144 | passes. On the requests llama.cpp b11960 takes, 0.0071 from fp32 against llama.cpp's 0.0072 (it refuses requests without a state, and spoken questions) |
| h2oai/h2o-lightning-4b-gguf Q8_0 with its F16 mmproj, one-way, images (18) | 0.0071 | 0.0302 | passes |

These differences come from the quantization, because an 8-bit build is a different model from its creator's fp32
model. ippon is responsible for computing the build itself faithfully, and that check needs a reference on the build's
own weights.

## Builds added later

These builds came later and were checked at ippon's default precision (fp16 where it fits) with `ippon verify`,
against the creator's fp32 reference (Jeff 1.1 Medium against the creator's bf16). The table above was measured in
both dtypes. `verified.py` records each result.

| Build | Largest difference | Bound | Result |
|---|---:|---:|---|
| Jeff 1.1 Qwen 4B (`jeff-qwen-4b`) | 0.0033 | 0.01 | verified |
| Jeff 1.1 Medium (`jeff`) | 0.0474 | 0.057 | verified against the creator's bf16 on the Mac's GPU, where the creator's own bf16 on the CPU is 0.0565 away. 16 of the 18 requests are within 0.011, and the states of about 3,000 and 6,000 tokens are 0.0474 and 0.0304 away |
| Kev 0.8B (`kev-0.8b`) | 0.0026 | 0.020 | verified |
| Kev 4B (`kev-4b`) | 0.0023 | 0.010 | verified |
| Kev 9B (`kev-9b`) | 0.0027 | 0.011 | verified |
| GLiDE 2B (`glide-2b`) | 0.0022 | 0.016 | verified |
| GLiDE 9B (`glide-9b`) | 0.0063 | 0.011 | verified, its recurrent depth included |
| AutoTrust Decider Mini 9B (`autotrust-mini-9b`) | 0.0019 | 0.014 | verified |
| Clef-Flash 8-bit MLX build (`clef-flash:8bit`) | see the next section | | verified, within the bound of the build's own code |
| Clef-Flash Q8_0 (`clef-flash:q8_0`) | | | verified on every tensor, and 0.031 from the full checkpoint |
| Clef-Flash checkpoint (`clef-flash:bf16`) | | | prompts checked, and its head on a random instance |
| d3-mini NVFP4 (`d3-mini:nvfp4`, Ollama's build) | 0.00895 from the same weights in fp32 | 0.01 | experimental. Up to 0.018 from Ollama's own answers. It loads only by name, with a drift warning |
| Clef-Flash MXFP8 (`clef-flash:mxfp8`, Ollama's build) | 0.00113 from the same weights in fp32 | | experimental. Ollama 0.40.13's own answers are 0.378 from them |

## A build against the runtime it was published for (`ippon verify MODEL --against ARM=PYTHON`)

`--against` holds a quantized build to its publisher's runtime, as a reference file holds a model to its creator's
code. The arm (a `bench --compare` arm, here the community port for a community build) answers the requests twice in
its own environment. The first answers are in fp32 on the build's own weights (recast with MLX's `set_dtype`, on the
M4's exact kernels), and they are the reference. The second answers are the build as published. The default bound is
the larger of 0.01 and the published run's distance from FP32 plus 0.0005, rounded to 4 decimal places, and `--bound`
can set an explicit bound. ippon answers at its default, on the M5's own paths.

This was measured on an M5 Pro on 21 October 2026, on each model's reference requests (for Clef-Flash, the requests of
its tokens reference).

| Build | Arm | ippon (fp16) from the arm's fp32 | The arm as published, from its fp32 | |
|---|---|---|---|---|
| mlx-community/clef-flash-8bit | the repository's clef_mlx.py | 0.0040 | 0.0229 | passes |

ippon's default lands closer to the build computed exactly than the runtime it was published for does. In fp32 on the
M4's kernels, ippon and the community Clef port are 0.0069 apart. The [float64 reference check](../scripts/check_qwen35_float64.py)
shows which of the 2 is closer. It runs tests/qwen35_numpy.py (a float64 Qwen3.5 written from transformers'
modeling_qwen3_5.py) on the build's own weights over the first 32 tokens of a prompt. ippon's fp32 hidden states are
3e-6 (8-bit) from it, and the port's are 6e-4 from it from the first token (its own kernels, not looked into further).
The prompts are already token-exact against Cloudflare's code and the joint head is equal to Cloudflare's, so ippon
computes those builds as Cloudflare's model computes them.

fp16's range ends at 65,504, and some architectures carry activations near it. ModernBERT-large and mmBERT-base
encoders have reached about 31,600 and 33,600 in a few outlier dimensions of the residual stream and the MLP's output.
The Qwen3.5 decision models stay under 200 and the LFM2 ones under 1,600. Those peaks come from the model, not from the
input. The mmBERT-base encoder's peak stayed within 1% of its warm-up peak on all 51 requests of its reference (11
languages, states up to 6,000 tokens). So the default, "auto", measures the peak and does not assume it.

"auto" loads the model in fp16 and answers the warm-up requests. During the warm-up it records the largest magnitude in
the residual stream and in every matmul's output. It keeps fp16 if that magnitude stays below 43,669 (2/3 of fp16's
range). Otherwise, or if an answer overflowed, it loads the model again in bf16. `decider.precision` and
`decider.details` give the precision it chose and the peak it measured.

Until 21 October 2026, the limit was 1/16 of the range. That limit put the mmBERT-base encoder in bf16, where it failed
its bound with 0.155 on a 300-word state with 3 questions (a choice between near-equal options), where fp16 lands
0.0165. If a later request still overflows fp16 (an input far outside the warm-up's), its answer comes out non-finite.
The decider then loads the model again in bf16, warns once, answers that request again and stays in bf16. "auto" chose
fp16 for d1-omni-600M (peak 1,561) and for vllm-sr's d3-lite (peak 102).

Several checks below were measured on models that ippon no longer supports. d3-lite is a Qwen3.5-0.8B decision model,
dropped on 28 October 2026 because it ranked on neither decision index. Liquid's d1-3B and d1-omni-600M (LFM2) were
dropped on 29 October 2026 for their LFM3 successors d1-2B and d1-Omni-2B. Their measurements stay, because they set
the switches and paths that those checks describe.

## The same answers however they are run (`ippon verify MODEL --invariance`)

ippon reads a request in chunks, shares a state's reading between questions, resumes kept and grown states and batches
requests together, and none of that may change an answer beyond rounding. `--invariance` first answers each request of
a reference's set the plain way (alone, default chunks, nothing kept). It then answers each request in chunks of 512
and 2,048, with every prompt read whole, resumed from a kept prefix, and resumed from the prefix of a shorter version
of its state. It also answers each request with its questions reversed (formats that read each question from its own
prompt) and together with other requests. It reports the largest probability difference from the plain answers (M5
Pro, precision auto).

| Model | chunks | whole | kept | grown | order | together | Bound |
|---|---|---|---|---|---|---|---|
| Liquid d1-3B | 0.0007 | 0.0019 | 0.0004 | 0.0016 | 0.0011 | 0.0014 | 0.0389 |
| Liquid d1-omni-600M | 0.0000 | 0.0045 | | | 0.0000 | 0.0018 | 0.0474 |
| Bespoke Nimble-9B (its 7 requests) | 0.0010 | 0.0023 | 0.0004 | | | 0.0004 | 0.1459 |

No top answer changed. The differences are rounding, because another chunking or batch adds in another order, and the
M5's fused recurrence rounds about as TF32 does. d1-omni-600M read every prompt whole by design, so chunks changed
nothing for it and it kept no prefixes.

By default (`delta_inputs=kernel`), one kernel prepares Gated DeltaNet's inputs, which are the normalized q, k and v,
and the gates. On the M5 Pro, d3-lite's recorded worst distance from creator FP32 with this path is 0.0017. The
rounding of the fused recurrence can move when its input computation changes.

## Recurrences in the compute dtype (switch `gated_delta`)

ippon computes Gated DeltaNet's inputs in fp32 at every precision. They were also measured in the compute dtype against
the same fp32 reference (largest probability difference at precision="auto").

| Model | as before | switch | Bound | |
|---|---|---|---|---|
| vllm-sr d3-lite | 0.0031 | `gated_delta=half`, 0.038 | 0.0231 | fails |

With the inputs (q, k, v) in fp16 and the recurrence's state still in fp32, the answers move out of bound on long
states. The differences grow with the state's length, as the rounding compounds through the recurrence. The inputs
therefore stay in fp32, and `gated_delta=half` stays as a switch for measurements.

## Attention at head size 256 (switch `attention`)

ippon's default is MLX's own attention (`attention=mlx`). The opt-in `attention=bounded` keeps MLX's softmax but works
through at most 128 query rows at a time, so it never allocates the whole score matrix. It applies to unmasked fp16
and bf16 calls with at least 512 MiB of scores, and it turns off compiled layers. Its rounding can differ, and it is
not faster. Both stay within the models' bounds on the M5 Pro, at precision auto with MLX's own matmuls
([reports](../references/attention/validation/)), on prompts up to 4,096 tokens and 5 layouts per model (a mixed
batch, 3 single requests and an equal-length pair).

| Model | Longest prompt | `mlx` worst | `bounded` worst | Bound |
|---|---:|---:|---:|---:|
| d3-lite | 4,096 | 0.004707 | 0.004707 | 0.0231 |
| pplx-decider flash v1.2 | 4,088 | 0.000337 | 0.000337 | 0.0119 |

Both also stay within the bounds on 15 longer cases per model, at about 4,000, 8,000 and 12,000 tokens, built from the
creator's own answers ([reference generators](../references/attention/README.md)).

| Model, 15 cases | `mlx` worst | `bounded` worst | Bound |
|---|---:|---:|---:|
| d3-lite | 0.003024 | 0.003024 | 0.0231 |
| pplx-decider flash v1.2 | 0.000499 | 0.000499 | 0.0119 |

On those cases `bounded` also lowered the peak memory that MLX allocated, so its use is the memory it saves on long
prompts. In a fresh process through loading, warm-up and all 15 cases, the peak went from 5.18 to 3.84 GiB on d3-lite
and from 14.91 to 11.50 GiB on pplx-decider flash. These figures are MLX's active allocations, not the resident memory
of the process. If MLX cannot allocate the scores of a pass at all, the default also reads that pass in such blocks.

ippon's own fused attention kernels, checked here as `kernel` and `large` until 2 November 2026, were removed. They
saved the same memory as `bounded`, and where measured they were never faster than MLX's attention (pplx-decider flash
at 4,096 tokens, 0.89% in a 4-round A/B on 25 October, [report](../benchmarks/results/m5pro-2026-10-25-attention-4k/SUMMARY.md),
and an M1 at 1,000 and 4,000 words, no difference). d3-lite and the first Nimble release have been dropped since, and
pplx-decider flash v1.3 replaced v1.2, but their rows stay as measured. Clef-Flash's reference is tokens only, so it
has no rows here.

## Quantized weights (`precision="q8"`)

`precision="q8"` quantizes eligible weight matrices at load with MLX's affine groups of 64 columns. It runs only when
you select it explicitly, and ippon never quantizes a model silently when it does not fit. Embeddings and readout rows
keep floating-point weights. Quantized checkpoints that creators or the community publish load in their stored format,
as [formats.md](formats.md) describes.

Quantization can change decisions. The public d1-Omni-2B Q8_0 GGUF build above differs from its creator's FP32
reference by 0.045, against a 0.0172 bound, and changes a top answer. ippon is nevertheless 0.0033 from llama.cpp on
the same file, and the 2 checks answer different questions. To establish the numerical result of an application build,
use `ippon verify MODEL --reference FILE --precision q8` (or the precision you selected). Do not assume that a bit
width is accurate for every model. A build quantized at load is another model than the one the reference ran, so
`ippon verify` grades it the way `ippon setup` tags it ([formats](formats.md)). It passes when its answers move under
10 points with every top answer the same, and it lists the requests past full precision's bound.

Kettei at q8 (`kettei-core:q8`, `kettei-dot:q8`), on the 22 requests above, with every top answer unchanged:

| Model | Largest difference at q8 | Recorded bound | Past the bound |
|---|---|---|---|
| Kettei Core | 0.0414 | 0.0369 | the 255-option request only. The 120-option one is 0.026 |
| Kettei Dot | 0.0498 | 0.0324 | 40-questions (0.0498) and state-3000 (0.0342) |

## Ollama's builds, on ippon and on Ollama

The README times ippon against Ollama 0.40.13 on the builds that Ollama's default tags pull on a Mac, so both runtimes
read the same weights. Pulled with Ollama 0.40.13 on the M5 Pro, `d1:2b` and `d1-omni:2b` are the BF16 builds.
`d3:4b`, `d3:9b` and `clef-flash:9b` are the MXFP8 builds, with the same configuration and weights as the `-mxfp8`
tags. Fetched from the registry with curl, the manifests of those 3 tags name Q8_0 GGUF files instead.

Each build's answers were measured twice against its creator's fp32 reference on 30 October 2026 (M5 Pro 48 GB, macOS
27.0.1). One run was on ippon with `ippon verify ollama://TAG --reference FILE`. The other sent the same reference
requests to Ollama's `/v1/systemone` ([references/ollama/answers.py](../references/ollama/answers.py)). The worst
difference is over the requests that both runtimes answered.

| Model | Ollama build | Requests both answered | ippon | Ollama | Bound | Past the bound (ippon / Ollama) |
|---|---|---:|---:|---:|---:|---:|
| d1-2B | `d1:2b`, the default (BF16) | 23 of 30 | 0.0012 | 0.0125 | 0.0162 | 0 / 0 |
| d1-Omni-2B | `d1-omni:2b`, the default (BF16) | 23 of 30 | 0.0018 | 0.0286 | 0.0172 | 0 / 1 |
| d1-2B | `d1:2b-mxfp8` | 23 of 30 | 0.1045 | 0.0811 | 0.0162 | 12 / 13 |
| d1-Omni-2B | `d1-omni:2b-mxfp8` | 23 of 30 | 0.1469 | 0.1470 | 0.0172 | 11 / 11 |
| d3-mini | `d3:4b-mxfp8`, the build `d3:4b` pulls | 19 of 20 | 0.0624 | 0.0502 | 0.0291 | 3 / 4 |
| d3-flash | `d3:9b-mxfp8`, the build `d3:9b` pulls | 19 of 20 | 0.0440 | 0.0387 | 0.0161 | 8 / 10 |

On the BF16 builds, which hold the weights Liquid published, ippon lands 10 to 16 times closer to Liquid's fp32 answers
than Ollama does. On both runtimes, the MXFP8 builds move past their bounds by about the same amount. Their 8-bit
weights change the answers, and ippon computes them closely (d3-mini's MXFP8 build is 0.0059 from the same weights run
in fp32). Each runtime changed one top answer on `d1:2b-mxfp8` and on `d3:9b-mxfp8`.

Clef-Flash has no creator fp32 reference that fits a 48 GB Mac. Its default build, `clef-flash:9b` (MXFP8), was
therefore measured against the same MXFP8 weights computed in fp32 (references/clef/9b-mxfp8.dequantized.json.gz, 18
requests from 128 to 4,096 tokens). That reference runs ippon's architecture code, and an independent NumPy fp64
backbone matches it to a relative error of 0.0000027 ([formats.md](formats.md)). So the reference isolates how each
runtime computes one set of weights.

| Model | Ollama build | Requests both answered | ippon | Ollama | Bound | Past the bound (ippon / Ollama) |
|---|---|---:|---:|---:|---:|---:|
| Clef-Flash 9B | `clef-flash:9b`, the default (MXFP8) | 18 of 18 | 0.0011 | 0.3784 | 0.01 | 0 / 9 |

Ollama's difference grows with the context, from 0.020 at 512 tokens to 0.109 at 2,048 and 0.378 at 4,096. Ollama
changed the top answer of 3 urgency questions (at 512, 2,048 and 4,096 tokens), and ippon changed none.

Ollama refused 7 of d1's 30 reference requests that Liquid's code and ippon answer. These were questions with missing
or blank instructions, options given as a list, 30 and 40 options, a request with no state, and yes/no criteria written
in a shape other than an object. Through the `-mxfp8` tags, Ollama refused d3's 6,000-token request, which is past the
4,096-token context that those packages set. The default `d3:4b` and `d3:9b` tags take it and answer all 20 requests
with the same worst differences, 0.0502 and 0.0387. [references/ollama](../references/ollama/) has the raw results per
request, for Ollama and for ippon.

## Perplexity's Q8_0 file and llama.cpp b11960

llama.cpp b11960 answered none of the 15 reference requests on Perplexity's flash v1.2 Q8_0 file, and ippon answered
all 15. ippon no longer loads flash v1.2 or this file, because flash v1.3 replaced them and Perplexity publishes no
GGUF of v1.3. The check stays as recorded. It ran once on 30 October 2026 on an Apple M5 Pro, 48 GB, macOS 27.0.1,
with MLX 0.32.4 and ippon `0c11bc8`. Both runtimes received the 15 requests in Perplexity's fp32 reference, and both
loaded the creator's unchanged `decider-flash-v1.2-q8_0.gguf`, SHA-256 `ffb27ade8976765b0e6943a0b7f5f3fd0f7042d45ad90256a76d608bcd3942c2`.

| Runtime | Answered | Largest probability difference from fp32 | Bound | Result |
|---|---:|---:|---:|---|
| ippon, default precision | 15/15 | 0.019995 | 0.0119 | 2 requests past bound, all top answers and prompt tokens match |
| llama.cpp b11960 (`8642541b2`) | 0/15 | unavailable | 0.0119 | HTTP 501, `This model is not a decision model` |

The `pplx-decider` type in this llama.cpp release does not implement protocol v3 of the file. Its decision loader
looks for `<architecture>.decision.type` and a `systemone` chat template, but this file carries the creator's
`pplx_decider.*` protocol metadata. The
[creator's pinned GGUF card](https://huggingface.co/perplexity-ai/pplx-decider-flash-gguf-v1.2/blob/2514d5f0b7fbbe1ee1bfec9ffb5801a60289b4cc/README.md#gguf-files)
also documents that incompatibility. `/v1/decisions` returns 404. Loading the matching F16 mmproj succeeds, but text
and image decisions both still return 501, so this pair has no shared image workload either (ippon has read the mmproj
since 3 November). No speed campaign was run, and no numerical claim can be made about llama.cpp's answers.

[The results and reproduction commands](../references/llama/README.md) keep the verify output, every refused reference
request and the endpoint preflight. The README comparison uses the full BF16 checkpoint and Perplexity's Python code,
and it is not a timing of this GGUF.

## llama.cpp and oMLX against the creators' fp32

On the same weights, llama.cpp b11960 and oMLX 0.7.2 land several times further from the creators' fp32 answers than
ippon. Each runtime answered the full reference request set, and only the requests that both runtimes answered count.

| Model and build | Runtime | Requests answered | Worst difference | ippon's worst | Top answers changed |
|---|---|---|---|---|---|
| d1-Omni-2B, Liquid's Q8_0 file and mmproj | llama.cpp b11960 | 28 of 30 | 0.441 | 0.045 | 5 on llama.cpp, 1 on ippon |
| d1-2B, Liquid's Q8_0 file and mmproj, its 12 video requests | llama.cpp b11960 | 12 of 12 | 0.153 | 0.019 | 1 on llama.cpp, none on ippon |
| d3-flash, the creators' checkpoint | oMLX 0.7.2 | 20 of 20 | 0.0155 | 0.0017 | none |
| d3-mini, the creators' checkpoint | oMLX 0.7.2 | 20 of 20 | 0.0231 | 0.0085 | none |

llama.cpp refused 2 requests, one noul without instructions and one choice with a list of criteria. On its
one-question requests alone it is 0.043 from fp32 (the d1-Omni-2B Q8_0 row above).
The records and the oMLX script are in [references/llama](../references/llama/) and
[references/omlx](../references/omlx/). They were measured on an Apple M5 Pro with 48 GB on 4 November 2026, and the video row on 19 November 2026.

## fp32 on an M5

On an M5, `precision="fp32"` is TF32-like, so the check against a creator's fp32 code runs with the M4 paths. MLX
0.32's fp32 matrix multiplies and attention run on the neural accelerators with their inputs rounded to about TF32. On
an M5 Pro, a random fp32 matmul is about 8e-4 (relative) from numpy's result, and it is exact with an M4's kernels. The
check that an implementation matches its creator's code in fp32 therefore runs with the M4 paths (`IPPON_PROFILE=m4`,
which has MLX take the M4 family's kernels) on the same Mac. The models' own precisions (fp16, bf16) are checked on the
M5's own paths.

Keep 2 things in mind when you run that check (measured 21 October 2026, M5 Pro, MLX 0.32.4, against float64 numpy).

- `IPPON_PROFILE=m4` acts only if ippon is imported before MLX. A script that imports `mlx.core` first keeps the M5's
  kernels. `--profile m4` on the command line switches only ippon's own paths and leaves MLX's kernels as they are.
- `MLX_METAL_GPU_ARCH` (the M4's family) or `MLX_ENABLE_TF32=0` alone makes fp32 matmuls exact (2e-6 relative), and
  the 2 together bring the TF32 rounding back (8e-4). MLX's chunked Gated DeltaNet kernel rounds to about TF32 on the
  M5 from 64 tokens (2e-3) and is exact on the M4's kernels (4e-7). Below 64 tokens it is exact on both.

## Rotary positions through 16,384

On M5 Pro 48 GB, macOS 27.0.1, MLX 0.32.4, `mx.fast.rope` was compared with NumPy FP64 rotations at every integer
position through 16,384. The largest FP32 component errors were 0.00185 for LFM2, 0.00170 for Qwen3.5 partial rotary,
0.00087 for interleaved M-RoPE, 0.00253 and 0.00438 for ModernBERT global and local, and 0.00047 to 0.00540 across
representative dense and Gemma configurations, including linear, Llama 3, YaRN and proportional scaling. Explicit
positions and chunk offsets were bit-identical. These are activation errors, not probability errors.

For the existing 3-question long-state references (up to 12,263 tokens per prompt), d3-lite's shipped probabilities
differ from creator FP32 by at most 0.001941 (bound 0.0231), and Perplexity flash v1.2's by 0.000191 (bound 0.0119).
Both pass invariance. An experimental implementation with FP64 angles and FP32 trigonometry reduces those errors to
0.000892 and 0.000077 (flash v1.3 has since replaced v1.2). Neither original case violates its bound, so the runtime
stays unchanged. This establishes no end-to-end long-state result for the other model families or for arbitrary
scaling parameters. The [measurements, scope and reproduction](../references/rope/README.md) are in its README.
