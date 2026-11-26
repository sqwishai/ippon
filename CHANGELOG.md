# Changelog

## Unreleased

- **Fewer of ippon's own kernels** - The switches `glu` and `norm_rope` are gone. Remove them from `IPPON_SWITCHES`,
  `--switch` and `switches=`, where ippon now refuses them as unknown. MLX's own operations do that work, with the
  same fp32 arithmetic rounded once. On a 48 GB M4 Pro the GLU kernel took 1 to 4% longer than MLX's, and
  `norm_rope` was off by default and no faster.
- **Q8_0 on M4** - The M4 unpacks Q8_0 matrices from 1,024 rows, as it does MXFP8 ones. On a 48 GB M4 Pro, d1-2B's and
  H2O-Lightning-4B's Q8_0 builds answer in 4 to 8% less time from 2,000 words, with the same answers to the last
  digit.
- **Short prompts on an M4 Pro** - From 65 rows, MLX's matmul on an M4 uses tiles of 64 rows, so a 65-token prompt
  costs almost as much as a 128-token one. On an M4 with 20 GPU cores, ippon uses MLX's 32-row tiles at 65 to 96 and
  129 to 160 rows. There, the matmuls of d1-2B, Kettei Core and Kettei Dot took 4 to 33% less time, and their fused
  gate and up projections 20 to 28% less.

## 0.1.0 (25 November 2026)

This is the first release of ippon. ippon runs decision models on Macs with Apple silicon, as their creators published
them. It records how each build was checked against the creator's own code (its prompts,
and its answers in fp32 where a reference exists). It also keeps what an agent's requests have already read, so a
context sent again or grown by a message is answered from where it was left.

**Before you upgrade** - nothing, since this is the first release. ippon needs macOS on Apple silicon, Python 3.11 or
newer and MLX 0.32, which installs with it.

### Measured

Averaged over every model and workload, ippon is 34% faster than Ollama 0.40.13, 55% faster than oMLX 0.7.2, 3.4x
faster than llama.cpp b11960 and 4x faster than the creators' own code. These numbers come from an Apple M5 Pro with 48 GB,
macOS 27.0.1 and MLX 0.32.4, from 28 October to 24 November 2026. Each runtime ran in its own server over local HTTP,
on the same requests and the same model builds. Each number is a median of up to 32 timed requests after 2 warm-ups, as each round stops once its median settles
([benchmarks](docs/benchmarks.md)).

| Model | Typical call (~500 tokens, 3 questions) | Same context again | Context grown | 16 clients at once |
|---|---:|---:|---:|---:|
| Kettei Core | 183 ms | 44 ms | 48 ms | 12.0 req/s |
| Kettei Dot | 51 ms | 14 ms | 16 ms | 46.4 req/s |

With 16 agents asking at once, ippon answers 44% more requests a second than Ollama. At its peak, ippon needs 45% less
memory than Ollama and 9% less than oMLX. A support agent that asks a 9B decision model about a refund after each of
41 customer messages gets all 41 answers in 8.1 s on ippon, 14.2 s on Ollama, 148 s on oMLX and 207 s on llama.cpp.

On each model's reference requests, ippon's answers differ from its creator's code in fp32 by at most these amounts.
ippon ran at its default precision, which is fp16 where it fits and bf16 otherwise (see [accuracy](docs/accuracy.md)).

| Model, build | Largest difference | Bound |
|---|---:|---:|
| d1-2B, BF16 | 0.0012 | 0.016 |
| d1-2B, Q8_0 file | 0.010 | 0.016 |
| d1-Omni-2B, BF16 | 0.0018 | 0.017 |
| d3-mini | 0.0085 | 0.029 |
| d3-flash | 0.0017 | 0.016 |
| pplx-decider flash v1.3 | 0.0023 | 0.017 |
| Torchcast Decision 4B | 0.0024 | 0.043 |
| H2O-Lightning-4B | 0.0040 | 0.019 |
| elephant-1 | 0.0024 | 0.025 |
| AutoTrust-Decider-Mini-9B | 0.0019 | 0.014 |
| Kettei Core | 0.0067 | 0.037 |
| Kettei Dot | 0.0026 | 0.032 |

On Ollama's own default builds, Ollama 0.40.13 lands 0.0125 (d1-2B) and 0.029 (d1-Omni-2B) from Liquid's fp32 on the
same reference requests, where ippon lands 0.0012 and 0.0018. On Clef-Flash's MXFP8 build, compared with the same
weights computed in fp32, Ollama moves up to 0.378 at 4,096 tokens and ippon moves 0.0011.

### What's in it

- **Python & Decide API** - `ippon.Decider.load("kettei-core:q8", accept_license=True)` runs any supported model in
  Python with typed `YesNo`, `Choice`, and `Score` questions (`model.decide(...)`), System One requests
  (`model.systemone(...)`), or Sqwish's decide API shape (`model.sqwish(...)`). `ippon serve` answers the same decide
  request over HTTP at `POST /v1/decide` for another process or language. Images, videos, and audio clips can be passed
  directly by file path in Python, via `--image`/`--video`/`--audio` on the CLI, or in base64 over HTTP.
- **Models** - These models load by short name (`ippon pull kettei-core`). They are Sqwish's Kettei Core and Dot,
  Cloudflare's Clef-Flash 9B, vllm-sr's d3-mini and d3-flash, Liquid AI's d1-2B and d1-Omni-2B, Perplexity's
  pplx-decider flash v1.3, Torchcast Decision 4B, H2O-Lightning-4B, Base Labs' elephant-1, Fastino's GLiDE 2B and
  GLiDE 9B, Jonathan Geuter's Jeff-1.1-Medium and Jeff-1.1-Qwen-4B, Jared Palmer's Kev 0.8B, Kev 4B and Kev 9B,
  AutoTrust's AutoTrust-Decider-Mini-9B, and StrandsAgents' Strands-Decider-E2B, Strands-Decider-E4B and
  Strands-Decider-12B. Clef-Flash, d1, d3, pplx-decider, Torchcast, H2O and elephant-1 read images. Their image answers
  are checked for d1, d3, pplx-decider, Torchcast, H2O and elephant-1, and their image prompts for Clef-Flash.
  d1-Omni-2B reads audio, and d1 and d3 read video. d1's questions also take instructions and option descriptions that
  are JSON other than text, written as Liquid's prompt code of 29 October writes them.
- **Strands Decider v2** - Native Apple Silicon MLX support for Gemma 4 `Strands-Decider-E2B`, `Strands-Decider-E4B`
  and `Strands-Decider-12B` (`ippon pull strands-e2b`), including per-layer embeddings kept on host so E2B uses 3.8 GB
  and E4B uses 8.0 GB GPU memory, shared KV layers, prefix-bidirectional attention and the 255-code readout head. All
  three sizes are verified against the creator's CUDA BF16 golden vectors (0.0031 to 0.0039 mean TV) and run 2.9x to
  5.0x faster than the creator's PyTorch MPS server on typical calls (13x to 17x faster on repeated context).
- **Names** - A short name loads a model's recommended build (`kettei-core`), and a tag picks one build
  (`d1-2b:q8_0`). A precision in place of the tag runs the recommended full-precision build at that precision
  (`kettei-core:q8`, as `--precision q8` does). `FAMILY:q8` first takes the family's own verified 8-bit build where
  it has one (`d1-2b:q8` loads the creator's Q8_0 file).
- **GLiDE** - Fastino's GLiDE 2B and 9B run by short name (`ippon pull glide-2b`) and read text only. The 9B reads 2
  blocks of its middle layers twice in one pass, as Fastino's own code does. Both sizes land within their bounds of
  that code in fp32.
- **pplx-decider flash v1.3** - `ippon pull pplx-flash` loads Perplexity's flash v1.3, with its chunked prompt
  (protocol 4). Each question is its own sequence after the request's context, as Perplexity's package
  reads it by default, so an answer never depends on the other questions. The sequence limit is the package's 32,768
  tokens. On an Apple M5 Pro, its answers land 0.0023 from Perplexity's fp32 (bound 0.017), and 0.0030 on images
  (bound 0.033). It has no GGUF build, because Perplexity publishes no GGUF of v1.3.
- **Less GPU memory** - ippon reads the rows of a model's vocabulary tables from the checkpoint's file on the CPU,
  and does not hold the tables on the GPU. The answers are the same either way. A decision model reads a few hundred
  rows of these tables for each prompt. At bf16 on an M5 Pro, this saves 4.1 GB on Clef-Flash 9B and elephant-1,
  2.6 GB on Kettei Core, 2.5 GB on Torchcast Decision 4B and 2.0 GB on d3-flash and Jeff-1.1-Medium. It saves 1.3 GB
  on d3-mini, H2O-Lightning-4B and pplx-decider flash, and 0.5 and 0.4 GB on d1-2B and Kettei Dot. MLX 8-bit, Ollama
  and GGUF Q8_0 builds read their tables packed. Clef-Flash 9B's Ollama MXFP8 build takes 7.6 GB, against 11.7 with
  the tables on the GPU, its MLX 8-bit build 7.8 against 10.0 and its Q8_0 file 8.3 against 10.6. d3-mini's MXFP8
  build takes 3.7 against 5.0, and d1-2B's 1.5 against 2.0. The memory estimate that admits a model leaves the tables
  out too, so Kettei Core at 8-bit is estimated at 3.74 GiB (3.7 GiB measured on an M5 Pro). That is within an 8 GB
  Mac's 4.3 GiB, where an estimate with the tables in it is 6.1.
- **A request sent again, answered at once** - By default, the decider keeps the answers to its last 1,024 requests
  (`--answer-cache`, `Decider.load(..., answer_cache=1024)`). It answers an identical request without reading it, with the
  same answers. On Kettei Dot through `ippon serve`, the README's Quick start request takes 0.5 ms, against 15 ms to read it
  again (M5 Pro). `--answer-cache 0` turns it off, so every request is read.
- **Ready between an agent's calls** - A decider loaded in your own program (`Decider.load`) keeps its model's
  memory wired for 5 minutes after each answer, then lets the Mac rest, as `ippon serve` does. The option is
  `keep_awake`, which is on by default, and None turns it off. An agent calls its decision model seconds apart.
  Without this option, a short call on Kettei Dot after a 5 to 15 s pause took 109 to 115 ms instead of 30 ms (M5
  Pro). MLX's buffer cache is capped at 1 GiB (about 0.5 GiB on an 8 GB Mac). With the cap, the speed is the same and
  d3-mini's server footprint is 5.3 GB, against 8.2 GB without it.
- **Set up in one step** - The installer ends in `ippon setup`. Setup shows the Mac's chip and memory and lists every
  model with what it reads and whether it fits, with the one for this Mac first. The ← and → keys turn a model through
  full precision, 8, 6 and 4 bits. A row turned below full precision is tagged with how far its answers move, as
  measured (✓ answers hold, ⚠ check thresholds, ⚠ answers may change). Enter takes the model. Setup then downloads
  it, answers a first question, and asks what to use it with and how the server runs. The server either starts when
  something needs it or runs from login. `curl ... | sh -s -- kettei-dot` names the model instead. Without a
  terminal, setup asks nothing and writes nothing to an agent's settings.
- **An agent in one command** - `ippon claude`, `ippon codex` and `ippon opencode` start the shared server, or reuse
  the one that runs. Then they start the agent for that session with the model that setup chose beside it, through
  ippon's MCP tools (and, in Claude Code, ippon's skill). Nothing is written to the agent's own configuration.
  Arguments after `--` go to the agent, and the server keeps running for other clients.
- **Serving your own code** - `ippon start` serves the chosen model in the background on port 8000. If another program
  holds 8000, it uses a free port and names it. `ippon status` shows the server (`ippon ps` does the same),
  `ippon logs` follows it and `ippon stop` stops it. In a terminal, `ippon serve` prints a line for each
  answer, and more lines when its answers outgrow the terminal, with each answer whole. A line gives the client's name
  (the `X-Ippon-Client` header), the model's time per decision, the tokens read and how many of them were kept from an
  earlier reading. `ippon status` watches a server live. It shows the model's time per decision and the questions it
  answers a second (a request of 10 questions counts 10). It also shows how many clients are connected and asking, and
  a bar of each client's decisions. A result's `diagnostics` gives the kept tokens as `kept_tokens`. `ippon remove
  AGENT` takes ippon out of an agent's settings, and `ippon remove login` stops serving from login. `ippon remove all
  --models` takes out everything ippon added (agents' settings, the login job, its servers, settings and caches, and
  the models it downloaded). It keeps a revision that also holds files another app downloaded, such as another
  quantization of the same model. Run it before `uv tool uninstall ippon`.
- **Safe to leave running** - A server that listens beyond this Mac (`--host 0.0.0.0`) needs an API key, and every
  request must carry it. If Python can't read a request's JSON (nested too deep, an integer too long), the request gets
  an error. A request that expands past 1,000 times its compressed size also gets an error. In both cases, the server
  and an MCP session go on. ippon refuses a crafted GGUF file and gives the reason. It clears a server record whose
  process is gone, or whose process now belongs to another user.
- **Checked on your Mac** - `ippon verify MODEL` checks a named build against the reference that its creator's code
  made. The package carries that reference, so the check needs no clone and no `--reference`.
- **Builds** - ippon loads creators' safetensors checkpoints, MLX 8-bit builds, GGUF Q8_0 files and the MLX builds
  that Ollama has already pulled (`ollama://d3:4b`). By default, it loads them as published, at full precision or
  8-bit. `ippon pull --list` marks the 8-bit files that move past their bound. A GGUF build reads images and audio from
  its mmproj file, as llama.cpp's `--mmproj` does. ippon finds that file beside a local file, or downloads it with a
  `repo:Q8_0` build (d1-2B, d1-Omni-2B, H2O-Lightning-4B), and reads it on the first image or clip. On the creators'
  reference requests on an Apple M5 Pro, d1-Omni-2B's Q8_0 files land 0.0085 from its fp32 on images and 0.0075 on
  audio. H2O-Lightning-4B's land 0.0071 on images (one-way). d1-2B's land 0.0138 on images, against 0.0016 for the
  checkpoint, with one three-question request past the full-precision bound of 0.01. Where llama.cpp b11960 reads the
  same request, ippon and llama.cpp land equally close to fp32 (0.0071 and 0.0072 on audio).
- **Fewer bits, only when you ask** - 4-, 5- and 6-bit MLX builds, NVFP4 and MXFP4 (Ollama's NVFP4 tags among them),
  GGUF Q4_K_M and Q5_K_M files, and `--precision q4` or `q6` (`kettei-core:q4`) load only by their own name, tag,
  file or precision. A bare name, `ippon pull`, `ippon recommend` and `ippon setup` never choose one.
  Every such load warns how far the model drifts from its creator's answers. The warning comes as a `LowBitWarning`,
  once on the command line, in `details["drift"]` and in each response's `warnings`. ippon's accuracy claims and speed
  charts stay at full precision and 8-bit. ippon refuses 2- and 3-bit builds and W8A8
  ([formats](docs/formats.md)).
- **Interfaces** - The command line has `setup`, `claude`, `codex`, `opencode`, `pi`, `start`, `decide`, `status`,
  `stop`, `logs`, `pull`, `recommend`, `serve`, `lint`, `eval`, `verify`, `bench`, `doctor`, `remove`, `mcp` and
  `skills`. It also has the maintenance commands `check`, `prompt`, `info`,
  `profile`, `suite` and `new-adapter`, which `ippon --help` leaves out. Python has `Decider.load`, `decide` and
  `decide_many`. HTTP has `POST /v1/systemone` in TypeSafe's System One format and `POST /v1/decisions` in
  Perplexity's and SGLang's format. `ippon mcp` is an MCP server for Claude Code, Cursor and other hosts. `ippon
  skills` gives agents skills to read on when a decision model fits, how to ask it and how to act on its
  probabilities.
- **Serving** - One shared server serves many clients and several models. It loads a model on first use and unloads
  the least recently used model under memory pressure. Requests that arrive together are answered in shared passes.
  Readings are kept in memory, images included. A request that sends the same images again, or one more after them,
  reads only what is new, and an image inside a kept reading is not encoded again. On H2O-Lightning-4B's GGUF build on
  an Apple M5 Pro, adding a painting to a wall of 39 took 344 ms instead of 695 ms. A model that reads both ways (d1,
  d3) reads its attention scores in blocks of queries, so it takes prompts longer than those scores fit in one GPU
  buffer. Read in one buffer, d3-mini stops at about 30,800 tokens a read on an Apple M5 Pro (48 GB), where those
  scores pass its 30.15 GB buffer limit.
- **Limits are the model's** - A question takes as many options and levels as the model has labels for, and ippon
  sets no limits of its own. d1-2B and d1-Omni-2B take 702 options (A to ZZ) and 10 score levels, since their prompt
  asks for one digit. elephant-1 takes 256 options, and d3, pplx-decider, H2O-Lightning-4B and Kettei take 255.
  Torchcast takes any number, asked in its creator's rounds (groups of 25, then their finalists). Jeff-1.1-Medium
  takes 256, asked as its server asks a choice past 16 options (groups of up to 16, then their winners). Clef-Flash
  and label-token models have no fixed count. `ippon prompt` reports the model's `max_options`. A request holds as
  many images and audio clips as the model's window does. ippon reads a video of any length and frame size from its
  sampled frames. A local file passed from Python or the command line has no size cap. Over HTTP and MCP, a request
  may hold up to `ippon serve --max-body-mb`, and a medium up to 1 GiB. By default, that limit is a 32nd of the Mac's
  memory, from 64 MiB to 1.5 GiB (256 MiB on an 8 GB Mac, 1 GiB on 32 GB). A served request has no deadline and waits
  for the model however long it reads, unless `ippon serve --timeout-s` sets one. For example, d3 took more than 120 s
  on a 1,000-token record with 50 questions on an M2 Pro with 16 GB.
- **Longer prompts than a creator's window, by choice** - `--window TOKENS` (`Decider.load(window=...)`) replaces the
  window that a creator's code sets below what the model reads. Examples are H2O-Lightning's 40,944 tokens (its
  server's 40,960-token context less 16 tokens of slack) and the 16,384 tokens of Clef-Flash and Torchcast. Past the creator's window the answers are
  unverified, and the load warns and records it in `details["window"]`. Without the option, every model keeps its
  creator's window.
- **Long records for H2O-Lightning-4B** - ippon fits a record past the server's 40,960-token context as that server
  fits it, and never refuses it. It cuts the record to the passages that the creator's own BM25 selection finds most
  relevant to the question (its selector_v2.py, reimplemented). Otherwise it keeps the record's head and tail.
  `truncated` says how much was left out. The server's cap of 4 images a request is not the model's, and ippon reads
  more.
- **Audio files as they come** - ippon decodes these formats in its native library. They are WAV at any rate, depth
  and channel count, MP3, M4A or AAC, FLAC, Ogg Vorbis, AIFF and CAF. It trims the encoder's priming and padding as
  FFmpeg trims them, from an edit list, Apple's iTunSMPB tag or an MP3's header. ippon reads a 16-bit or float WAV as
  Liquid's code reads it. The other formats go through Liquid's resampler, where Liquid's own code hands them to
  FFmpeg or librosa. Opus (a voice note, or a browser's WebM recording) is decoded by FFmpeg through PyAV.
- **New questions about media already sent** - A video's frames and a recording's encoded audio are kept by size
  (256 MB), so a new question about a 66-second video does not read the video again. On an M5 Pro, d1-Omni-2B answers
  that question about a low-resolution clip in 75 ms, against 242 ms when it reads the clip again, and one about a
  1080p 20-second clip takes about 1.1 s. A new question about a 30-second AAC recording takes 48 ms, against 69 ms
  when it reads the recording again. Both give the same answers. ippon reads a chained Ogg file whole, and reads a WAV
  that FFmpeg streamed without its size to its end. Resampling from an odd rate such as 47,999 Hz peaks at 0.48 GB,
  where its whole filter at once would take 3 GB. A clip longer than about 43 minutes is refused with a request to send
  it in parts. `decide --states` keeps `unsure_below` and the request's other options on every line.
- **Questions in several prompts, batched by cost** - ippon puts prompts read whole into the batches that cost least.
  It weighs a pass's padding against reading the weights again, so a short prompt does not pad to a long one's
  length. On an M5 Pro, d1-2B's 50 questions (4 prompts of up to 16) take 4.7% less time at 1,000 words than with that
  padding (590.7 ms against 619.7). With the padding, ippon was 10% behind Ollama's same build on an M4 Pro (2.95 s
  against 2.65 s).
- **A shorter command list** - `ippon --help` lists the 21 commands a user reaches for. The starter requests are in
  docs/evaluating.md, where a test lints and answers them. `ippon pull MODEL --dry-run` says what a build needs,
  whether it fits and its licence, without downloading (as `ippon check` does). `ippon lint --model MODEL --prompt`
  shows the text the model reads (as `ippon prompt` does). `ippon doctor` starts with this Mac and how ippon runs on
  it (as `ippon info` does). `check`, `prompt`, `info` and the maintainers' tools (`profile`, `suite`, `new-adapter`)
  run unlisted, and so do the tuning flags (`--switch`, `--chunk`, `--profile`, `--keep-awake`, `--settings`).
- **From a first-time user's run** (a clean home, the installer, the docs as written) - `ippon eval` shows a
  chosen threshold that flags exactly the rows that the exact threshold flags. Printed as 0.799, a threshold of 0.7985
  would leave its row out. `ippon eval` also says how many rows a held-out check covered when folds had no threshold,
  and exits 1 when a compared model didn't load. `ippon decide` checks a request before it downloads or loads anything.
  Ctrl-C while waiting stops with exit 130 and no traceback. `ippon ps` lists each model as loaded or not (`--json` has
  the rest), and `pull` points to the docs' starter requests. The docs say what a score's `score` is. It is the
  creator's score, which for d1 is its median level, while Python's `expected_level` is the mean. The docs also say that
  the Q8_0 files read images through the encoder beside them, with their image answers recorded in docs/accuracy.md. The
  README runs the server in the background.
- **Every failure names its kind** - A mistyped `--reference` or `--requests` file, a bad `--sizes`, or an `--n` below
  `--rounds` exits 3 with `invalid_request` instead of a traceback. A download that fails exits 13 with
  `download_failed`, and a second pull resumes it. A server answers a `Content-Length` that isn't a number with a 400.
  The MCP `decide` tool takes `videos` as HTTP does. A model whose warm-up answers nothing runs in bf16, so ippon never
  chooses fp16 without a measurement. A missing dependency or an MLX that doesn't load exits 14 with
  `runtime_unavailable` and the fix. The error for a damaged checkpoint or tokenizer names the file to repair. A full
  cache disk reports `insufficient_disk`. ippon refuses a bad question builder, batch request or load option before
  anything loads.
- **Send the unsure ones on** - A request's `unsure_below` lists the questions whose most likely option is under that
  value. It works in Python (`result.unsure(0.8)`), the CLI, HTTP and the MCP `decide` tool. Pass those questions to a
  person or a larger model, and settle the rest locally. `Answer.confidence` is the probability of the most likely
  option. Kettei Core, with its least sure third sent to Claude Opus, matched Opus alone at 90.8% on 600 questions.
- **Judges' thresholds in `ippon eval`** - For a yes/no question, `ippon eval` gives the precision, recall and F1 of
  flagging at P(true) >= t, across values of t. It picks the t with the highest F1 that meets `--min-precision` and
  `--min-recall`, and checks that choice on stratified held-out folds (`--folds`). It also gives the Brier score and a
  calibration table of P(true). The probabilities stay the model's own.
- **One-line install** - `curl -fsSL https://sqwish.ai/ippon/install.sh | sh` gets uv if
  it is missing and installs ippon as a tool with its own Python, so a Mac needs no Python first. `IPPON_PACKAGE`
  installs another package spec or a wheel. ippon works in the same terminal, because the installer links it into a
  folder that is already on the PATH and that you can write to, such as `~/.local/bin`. The installer does not replace
  a file to do this.
- **MXFP8 on M1 and M2** - From 224 rows, ippon unpacks an MXFP8 matrix to full precision and multiplies it with
  MLX's matmul. On an M1, d1-2B's MXFP8 build then answers in 17 to 25% less time from 500 words, with the same answers
  to the last digit. Without the unpacking, MLX's MXFP8 matmul took 34% (M1) and 28% (M2 Pro) longer on d1-2B's
  prompts than the same model in fp16, though its 8-bit weights read as many bytes as Q8_0's. `--switch gemm=mlx`
  turns the unpacking off.
- **MXFP8 on M4** - The M4 unpacks MXFP8 matrices from 1,024 rows, instead of the 224 rows of M1 and M2. On a 48 GB M4
  Pro, unpacking took 18 to 28% longer than MLX's matmul at 256 rows and 10 to 14% less time at 4,096 rows.
- **Matmul tiles only where they were measured** - Full-precision matrices use ippon's tiles only on an M5 with the M5
  Pro's 20 GPU cores, and every other M5 runs MLX's matmul for them. On the base M5, the tiles made d3-mini,
  H2O-Lightning-4B, pplx-decider flash and Torchcast take 10 to 24% longer at 1,000 words and more. Quantized builds
  keep ippon's kernels on every M5, and d1-2B's Q8_0 build is up to 28% faster with them on the base M5. With an MLX
  minor version other than 0.32, which the tiles were fitted on, ippon also runs MLX's own matmuls, and `ippon doctor`
  says why.
- **`ippon recommend` on M1 to M4** points an 8-bit build to its family's verified full-precision build when that
  build fits in half the GPU working set, and starts with it (`d1-2b:bf16` on an M4 Pro). On these Macs, 8-bit builds
  take 11% to 34% longer to read prompts (M1, M2 Pro, M4 Pro). This does not apply on an 8 GB M1, where bf16 swaps, or
  on an M5.
- **`ippon decide` reads for people** - In a terminal, it prints a line per question (`refund  yes 99.8%`, `team
  billing 98.8%  (technical 1.2%)`, `urgency  Needs help today (1) 94.1%`). Piped, or with `--json`, it prints the
  full System One response, for scripts and agents.
- **A clearer command line** - `ippon --help` lists the commands in the order a first run meets them, and ends with 3
  lines to start. Each command's own options come before the model-loading options and the server's tuning options,
  and each group has a heading. `ippon COMMAND --help` opens with what the command does and ends with examples, and
  `ippon decide --help` shows a request. If a command doesn't take an option, the error names that option. At a
  terminal, `ippon decide` and `ippon doctor` use the colours of `ippon serve`, and `ippon pull` gives sizes in GB.
  `ippon pull --list` and `ippon recommend` cut a row at the terminal's width, so no row wraps. `ippon
  stop` says how many clients were connected to the server it stopped. `ippon serve` with no model named serves the
  one that setup chose. At a terminal, `ippon serve` and the live `ippon status` open with a small lucky cat beside the
  model's name (Kettei Core) and its address. For status, the cat also sits beside the traffic in 3 short lines, with
  each client's answers as a thin bar under it. The cat has every pixel of the banner's cat where the terminal draws
  Unicode's sextants, and nothing is wider than the terminal from 50 columns up. A licence hint repeats the model as
  it was typed (`ippon pull d1-2b --accept-license`).
- **One reading of a model card** - A loaded model reads its licence from its card's front matter as YAML, as the
  Hub's metadata does for its download. So a comment after the licence, a folded value or a quoted key never names a
  different licence, or asks for an acceptance that the download did not need.
- **An option given is an option asked** - `--truncation creator`, given outright, does not reuse a running server
  that started with `--truncation error`. That server would refuse the long inputs that the client asked to have cut.
  Options left out take the running server's values.
- **A GGUF repository named alone** (`ippon pull LiquidAI/d1-Omni-2B-GGUF`) answers with the build to name
  (`LiquidAI/d1-Omni-2B-GGUF:Q8_0`).
- **One answer to whether a model fits** - Setup, `ippon pull` and a load judge a model by this Mac's memory, the
  same way every time. They never judge by what other apps hold at that moment, so a model that setup says fits always
  loads. ippon refuses a model too big for this Mac and names the precisions that fit. For a named model, it also names
  a smaller published build (`ippon pull d1-2b:q8_0` after d1-2B's bf16 checkpoint), and for a folder it names `ippon
  recommend`.
- **Wiring follows the Mac as it is** - ippon reads what to wire from this Mac's memory when a model loads. It wires
  nothing while macOS reports memory pressure or little memory is available. Otherwise it wires the model and room for
  its passes out of what is free, at most half the Mac. Without these rules, a 9.7 GB model on a 16 GB M5 wired 10.8 GB.
  Wiring and unwiring that every round of a kernel A/B while the Mac swapped ended in a kernel panic in Apple's GPU
  driver (macOS 26.2). A server lets go of what is wired while memory is short, and wires again once there is plenty.
  `Decider.details` records the `wired_limit`.
- **Small Macs** - If a batched pass of several prompts runs out of GPU memory, ippon reads it in halves.
  Torchcast-4B's 3 prompts of 4,335 tokens peak at 13.2 GB, past a 16 GB M2 Pro's 12 GB working set, where `bench`
  failed on the whole pass.
- **Ollama builds load in a third of the time after the first** - ippon checks each blob's sha256 the first time it
  sees that version of the file and remembers the result, as it does for Hub files. With this, d3:4b-mxfp8's weights
  load in 860 ms on an M5 Pro, against 2.59 s when every blob is hashed. If a blob changes after its check (another
  write, size or inode), ippon hashes it again, and a mismatch refuses the load.
- **d1 videos decode in a fifth of the time** - PyAV decodes with FFmpeg's threads, and the frames are the same bit
  for bit as without them. On an M5 Pro, the frames of a 20 s 1280x720 clip decode in 60 ms, against 315 ms without
  them.
- **Videos in `files`** - ippon reads a `data:video/...` entry of llama.cpp's `files` list as a video.
- **Batch size on small Macs** - `ippon bench --scenario concurrent --max-batch N` times the burst with serve's
  `--max-batch` set to N. Use it to choose that value for a Mac with few GPU cores.
- **Your own models** - Classifiers and label-token models run without an adapter (docs/bring-your-own.md). An
  adapter API covers other creators' models (`ippon new-adapter`, docs/adding-a-model.md).
- **Native work around the model** - A small Rust library resizes and lays out images and video as the creators'
  processors do, bit for bit, on every core. That step takes 1.4 ms for a page scan, where numpy took 29.8 ms. It
  takes 15.9 ms for a 20-second 720p clip for d3, where numpy took 638 ms. Whole requests were no faster in a
  fresh-process A/B, with the same answers (d1-2B took 175.2 against 175.0 ms a page and 1,490 against 1,498 ms a
  video). ippon reads a GGUF file's header natively. Against reading it in Python, load time is 30% lower for d1-2B's
  Q8_0 file (620 ms against 880) and 21% lower for H2O-Lightning-4B's (1.33 s against 1.68). These numbers are from an
  Apple M5 Pro.
- **Kernels** - On M5 Macs, matrix multiplies use kernels chosen at load from the chip and the matrix, with no tuning
  step. M1 to M4 Macs run MLX's own kernels. On an M1, M2 or M4, ippon first unpacks an MXFP8 matrix to full
  precision (from 224 rows, 1,024 on the M4), and `ippon suite` A/Bs that there. The M3 has nothing to A/B yet. The M5 kernel choices were
  measured on an M5 Pro, and `ippon suite` collects the same checks on any other Mac (docs/suite.md).

### Install

`pip install ippon` (or `uv add ippon`) gets a wheel for Apple silicon and macOS 11 or later, for every supported
Python. A build from source also needs Rust's cargo. Model weights keep their creators' licences and download from their
repositories. Jeff's weights do not allow commercial use.
