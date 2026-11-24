# Benchmarks

The kit times ippon against other runtimes of the same decision models, with identical generated requests. `results/`
keeps the recorded answers, timings and environment metadata, and [the method and tables](../docs/benchmarks.md)
describe the measured cells. The summaries and charts come from those files, and making them loads no model.

## Recorded comparisons

| Runtime | Model builds | Interface |
|---|---|---|
| llama.cpp b11669 | Clef-Flash's Q8_0 GGUF file, shared by both arms | HTTP |
| Ollama 0.40.9 | its Clef-Flash MXFP8 and d3-mini MXFP8 builds | HTTP |
| oMLX 0.7.2 setup pin | Clef-Flash MLX 8-bit and d3-mini checkpoints, shared by both arms | HTTP |
| Creators' own code, recorded separately on 21 October 2026 | full checkpoints on MPS, while the earlier MLX comparison uses the same MLX builds | In process |

The historical runtime table uses the 2 rounds in `results/m5pro-2026-10-24-rivals/`. The label of each other runtime
comes from the metadata of the selected result. oMLX has the label "setup pin" because its result metadata does not
record its version. HTTP results also do not record the library commit or the MLX version. The older creator results
record `7344347` and MLX 0.32.4. Keep these dates and commits separate from the current source revision.

The comparisons ran on an Apple M5 Pro, 48 GiB, macOS 27.0.1. D3's Ollama builds do not have the nominal
4,000-state-token cell, because the prompt of that cell is longer than their 4,096-token window. The generated tables
show these limits, and for each pair the maximum probability difference and the changed top answers.

## Regenerate the published material

Run these commands from the repository root. They only read recorded results and write documentation.

```sh
python3 -I benchmarks/kit/summarize.py \
  benchmarks/results/m5pro-2026-10-24-rivals/{r1,r2} \
  --out benchmarks/results/m5pro-2026-10-24-rivals/SUMMARY.md \
  --ledger benchmarks/results/m5pro-2026-10-24-rivals/EVIDENCE.json \
  --docs docs/benchmarks.md
python3 -I benchmarks/kit/summarize.py benchmarks/results/m5pro-2026-10-21/{r1,r2,makers} \
  --out benchmarks/results/m5pro-2026-10-21/SUMMARY.md \
  --ledger benchmarks/results/m5pro-2026-10-21/EVIDENCE.json
python3 -I benchmarks/kit/summarize.py benchmarks/results/m5pro-2026-10-21/makers --docs docs/benchmarks.md
# The README: every README campaign recorded so far (summarize.py's README_CAMPAIGNS), each pair from its newest
# complete campaign. A campaign still running, or not recorded yet, is left out until its round is accepted.
python3 -I benchmarks/kit/summarize.py --readme-round --recorded --readme README.md --docs docs/benchmarks.md
python3 -I benchmarks/kit/chart.py --readme-round --recorded
```

`--recorded` reads the accepted rounds of every campaign in `README_CAMPAIGNS` in `summarize.py`, from 28 October to
19 November 2026. It reads the text workload as
the round and the other workloads as `--more`.

The README's 2 hero charts average every model that has a rival in those campaigns. The command adds any folders that
you give in addition to `--recorded`, and a folder that you give and that does not exist is an error. Kettei Core and
Kettei Dot have no rival, so they get no chart. The README gives ippon's times for them in one table. The report lists
new pairs that do not have complete results.

While a round runs, the same 2 commands take its `attempt-r1` folder with `--partial`. `--older FOLDERS` then charts
each pair that the round has not measured, from the one-question, ~1,000-token cell of an older campaign. These pairs
show as a group dated with that campaign and with its own ippon time, and their 50-question cells read "not yet
measured".

For each pair, the kit uses the newest supplied campaign with complete results and pools only the paired rounds of
that campaign. `EVIDENCE.json` records the selected paths, their SHA-256 hashes, metadata, sample counts and unrounded
calculations for every table row and chart bar. To regenerate the summary of a single sibling, run `summarize.py` on
it. `--docs` writes the detailed benchmark blocks, and `--readme-round --readme` writes the speed headline and the
block of model charts. If a campaign ran more than 1 round, include every accepted `rN` folder.

Speed change means `(other median / ippon median − 1) × 100%`, including negative values. It is a throughput
comparison and not a percentage reduction in latency. Short and long summaries are geometric means across the
available shapes, with equal weights. The representative cell of the historical comparison table is the measured
nominal ~1,000-state-token request with one question. If that size is absent, the table names the nearest measured
size and interpolates nothing.

## Run a new comparison

Run these commands to set up the kit and time a pair in 2 rounds, with the order swapped in the second round.

```sh
cd benchmarks
sh kit/setup.sh
python3 -I kit/run.py list
python3 -I kit/run.py run d1-2B-llama-Q8_0 d1-2B-gguf-q8-ippon-serve --tag m5pro/r1/
python3 -I kit/run.py run d1-2B-gguf-q8-ippon-serve d1-2B-llama-Q8_0 --tag m5pro/r2/
python3 -I kit/summarize.py results/m5pro/r1 results/m5pro/r2 --out results/m5pro/SUMMARY.md
```

Perplexity publishes no GGUF or MLX build of `pplx-decider-flash-v1.3`, so its charts compare the full BF16 checkpoint
with its creator's Python code. Models in the README follow the order of JevBench's headline Capability Score (29
October 2026), and releases that are not on that board follow without a rank.

Each runtime runs in its own process and environment. It loads the model, warms up on 2 requests per shape, then times
10 requests by default. The request generator makes seeded support tickets with unique state identifiers. Caller-side
timing includes prompt rendering, tokenization and, for servers, the local HTTP round trip. A decision call retries
HTTP socket failures at most twice, with 50 ms between attempts, inside the same timer. The request row records
`transport_retries` and the cause of each `transport_errors` entry, and its configuration records the retry limit and
delay.

HTTP errors, refused connections, timeouts and model errors stay failures. A swap of the order between rounds shows
drift. Run one runtime at a time, and keep unrelated work idle.

The kit's defaults cover nominal state sizes 100, 300, 1,000, 2,000 and 4,000 with 1 and 3 questions. The October 24
campaign selected 100, 1,000 and 4,000. Options include `--models`, `--envs`, `--llama-server`, `--ollama`, `--sizes`,
`--questions`, `--n`, `--warmup` and `--cooldown`.

llama.cpp uses every layer on the GPU, one shared KV cache, 16 slots, a batch large enough for the entire prompt, and
an 8 GB prompt cache (`--cache-ram 8192`), which b11960 leaves off for decision models. The re-sent and grown-context
comparisons were timed again with the cache on in `results/m5pro-2026-10-31-llama-cache` (1 round, 10 timed requests).
ippon's lead did not change, because llama.cpp still processes every token of each request. The creator and community
adapters run the code of their repositories, so read that code before you run it. ippon itself runs no
model-repository code.

Setup pins the runtime releases and model revisions that the kit uses. Record the installed versions and source
revision with each new measurement.

## One-question and 50-question HTTP round

`kit/readme_round.py` runs both workloads through HTTP. One workload is nominal 1,000-token text with one yes/no
question, and the other is the same text with 50 different mixed questions. ippon runs as installed and chooses its
kernels from the chip (`choices.py`, with no measurement first). The campaign records the installed library commit and
its kernel rule. Run the process under the GPU lock, with unrelated work idle.

Native servers use their own decision endpoints, and this includes the community Clef build and the d3 packages. The
default non-HTTP kit mode is still available. Libraries without a server use `kit/http_adapter.py`, a standard-library
HTTP server that calls their existing adapter on the thread that loaded the model. The timing includes sending JSON,
the model's normal question handling and receiving decoded answers. This wrapper adds no cross-request batching.

Every result identifies its transport, and all ippon arms use `ippon serve`.

The runner takes `--models`, `--envs`, `--llama-server`, `--ollama`, a local readiness script (`--gate`), an optional
interruption script (`--guard`, which prints `stop`), an empty scratch `--cache` and `--out`. `--resume` continues a
stopped campaign in the same `--out` and `--cache` and keeps its accepted runs. It refuses to continue if the installed
library, the request generator, a model build or the sample counts changed. So a resumed campaign can change only the
kit, and a pair dropped from the kit does not run again.

`--smoke` checks every payload once. Smoke observations are diagnostic, and the kit does not publish them as speed
measurements. Every request carries a unique leading identifier, so no request can reuse the work of another.

Each round loads every runtime once and runs all of its workloads against it, with one result file per workload. Each
shape (state size and question count) gets 2 warm-ups, then timed requests until its median settles. A median settles
when its 95% interval, from the order statistics alone, is within ±3% of it (`--precision`), after at least 6 requests
(`--min-n`) and at most 20 (`--n`), or after 90 s (`--cell-seconds`). The kit always sends the first requests of the
generator in order, so both arms answer the same requests as far as both went.

With 8 clients at once, each client sends 6 timed requests, one window of 48. A second round, in the other order, runs
only the pairs that the first round left within 10% of each other or with overlapping intervals (`--close`). Each pair
pools the rounds it ran.

A CPU sampler watches competing processes during each run. The kit judges each timed request by the readings from its
start to the first sample after it ends. If a timed request overlapped another busy process, the kit keeps it in the
result's `set_aside` list and times the next generated request of that shape in its place. Both arms of a pair thus
still answer requests from the same sequence. After n replacements in one shape, further busy requests stay in the run
and the summary counts them. Only a complete round becomes `rN/`.

`summarize.py --readme-round --partial` summarizes the finished pairs in the `attempt-rN/` folder of an unfinished
round, with the label partial. `ATTEMPTS.json` records why each attempt was accepted or rejected. Published records
replace process names with their CPU shares.

`--workloads text,typical,concurrent,image,video,audio,repeat,grow,burst,pictures,image-again,video-again` runs the
README's other workloads beside `text`
(the default). `concurrent` sends 8 clients at once to each runtime's own server. Each client sends its next
~1,000-token one-question request as soon as its last request is answered. `burst` sends 16 clients with short
requests of about 300 tokens. The creators' libraries have no server, and wrapped in-process libraries do not run
`concurrent` or `burst`. Ollama gets `OLLAMA_NUM_PARALLEL` set to the largest number of clients that any workload in
the round sends at once.

`typical` sends 1 client with about 500 state tokens and 3 questions per call. Models without a rival also run every
non-media workload. `image` and `video` ask 3 questions and `audio` asks one, about ~100 tokens of text, with a new
1024×768 image, 4-second video or 10-second clip in every request. `pictures` grows from 1 to 8 images, and
`image-again` and `video-again` ask a new question about media that was sent before. The kit derives the bytes of the media from the request, and these workloads run only
for the pairs whose 2 arms both read that medium.

`repeat` and `grow` send one ~1,000-token state again and again, as an agent sends its context again at each step.
`repeat` asks it a new question each time, and `grow` adds a customer message of ~60 tokens before each request. The 2
warm-ups are the first sends of the same state, so a prompt cache may have kept every timed request. Their results are
`<engine>.typical.json`, `.c8.json`, `.image.json`, `.video.json`, `.audio.json`, `.repeat.json`, `.grow.json`,
`.c16s.json`, `.pictures.json`, `.image-again.json` and `.video-again.json`,
beside the text workload's `<engine>.json`, and `--more` reads them from the same rounds.

Generate the new report and charts from accepted folders with `summarize.py --readme-round` and
`chart.py --readme-round`. The first chart averages every model and workload (`docs/assets/readme-hero-*.svg`). Each
other chart shows 1 model (`docs/assets/readme-<model>-light.svg` and `-dark.svg`), in horizontal bars with a block for
each measured workload, each block on its own scale. The blocks are the typical call, the same context again, a grown
context, 16 clients at once, an image, growing images, the same image again, a video, the same video again and an
audio clip (`README_PANELS` in `summarize.py`). docs/benchmarks.md has every workload.

Each ippon bar sits above every rival measured against that ippon measurement on the same build, and a workload with
several builds names each build. Every bar shows its value. In 1 column on the right, every rival shows how ippon
compares, as a percentage or as "Nx" beyond 100% (2x, 13x). With several clients at once, the bars and the comparison
use requests per second, and the docs tables give the median and 90th-percentile waits. If a rival's median wait is
lower than ippon's, 1 sentence under its chart gives both percentiles.

The charts leave out unmeasured workloads, failed rivals and models without results.
`summarize.py --readme-round --readme README.md` writes the blocks that show those charts. It writes the speed
headline in docs/benchmarks.md, and in the README too if the README has that block. The headline gives the agent's loop and the typical call against Ollama, Clef-Flash's Q8_0 file against
llama.cpp, and the Mac and dates behind the charts.

This mode requires the declared pairs of each selected campaign, both complete workloads and valid distributions. Each
campaign must have one library commit, one measurement date range and one machine, and every selected pair keeps its
own provenance. For each pair the latest complete campaign wins, and the kit does not pool across campaigns. Request
counts can vary when a median settles early. A changed generator source is accepted only if its regenerated text
payloads still match the request hashes of the archive.

An unavailable comparison needs an explicit `--exclude-pair ENGINE=REASON`, which appears in the summary, and the
generator never drops a comparison silently. Metric A is the median end-to-end time in milliseconds. Metric B also
reports `50,000 / median milliseconds` questions/s.

## Final practical campaigns

The 30 October campaigns are among the README's campaigns (`README_CAMPAIGNS`), and each also has its own summary.
The commands below write only those summaries, and they do not update the README or its charts. The campaign names
keep the new text and concurrent arms separate from the practical workloads. In Ollama
0.40.13, `d1:2b` and `d1-omni:2b` resolve to the same manifests as their explicit `2b-mlx-bf16` tags, and `setup.sh`
pins both names with their stored digests. Ollama adds the original registry manifest as a provenance layer, so the
hash of its stored manifest is different from the registry hash. Each campaign uses 1 round, up to 10 requests and 2
warm-ups per shape. The manifests record the actual measurement dates, the installed library revision and the kit
commit.

Run these commands to summarize the campaigns.

```sh
python3 -I benchmarks/kit/summarize.py benchmarks/results/m5pro-2026-10-30-new/r1 \
  --readme-round --more benchmarks/results/m5pro-2026-10-30-new/r1 \
  --out benchmarks/results/m5pro-2026-10-30-new/SUMMARY.md \
  --ledger benchmarks/results/m5pro-2026-10-30-new/EVIDENCE.json
python3 -I benchmarks/kit/summarize.py benchmarks/results/m5pro-2026-10-30-practical/r1 \
  --more-only --out benchmarks/results/m5pro-2026-10-30-practical/SUMMARY.md \
  --ledger benchmarks/results/m5pro-2026-10-30-practical/EVIDENCE.json
```
