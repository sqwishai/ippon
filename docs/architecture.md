# Architecture

ippon is a runtime for one workload. It reads a prompt and turns the model's output at a few positions into
probabilities over a few options. This page describes that workload, the layers of the stack built for it, and the
rules that every layer follows.

## Model names, downloads and provenance

Every model-loading command and `Decider.load` accept a short name, a Hub ID, a local directory, a supported GGUF
file or build, or an `ollama://` build. `ippon pull --list` lists the named builds. A name has the form `FAMILY:TAG`.
`bf16`, `fp16` or `fp32` identifies the creator's stored checkpoint precision, `8bit` identifies MLX builds, `q8_0`
identifies GGUF and `mxfp8` identifies local Ollama builds. These tags describe the stored weights and do not set the
compute dtype. For example, `kettei-core:bf16` uses automatic compute precision unless `--precision` overrides it.

A precision in place of a build's tag (`kettei-core:q8`, `:fp16` or `:fp32`) loads the family's recommended
full-precision build. ippon then runs that build at that precision, as `--precision` does. If you give both, they
must agree. `FAMILY:q8` first takes the family's own verified 8-bit build where it has one, so `d1-2b:q8` loads the
creator's Q8_0 file (`names.own_eight_bit`). That file also keeps its build tag (`d1-2b:q8_0`).

A bare family name first keeps the verified builds whose answers are verified about the most kinds of media. Among
those it chooses the smallest. It prefers a verified safetensors build when that build is at most 1.5 times that
size. With no verified build, it chooses the smallest build whose prompts are checked (`names.recommended`). It
never chooses a build below 8 bits.
The rule picks a build, and it does not rank models by accuracy or speed. If you need a particular build, use an
explicit tag.

ippon reuses a complete recorded Hub download at its recorded commit without contacting the Hub, also offline.
`ippon pull org/name` checks for a newer revision. A short name stays at the commit its answers were checked at,
and an explicitly pinned `org/name@COMMIT` stays pinned. Incomplete or missing files can need a download. The Ollama
importer reads an existing local cache and never downloads.

`verified.py` is the one catalogue of what ippon loads by name. For each model, it records the licence, the builds,
where setup lists the model, whether it is a starter and what it holds in memory. For each build, it records the
status, the reference, the distance of the build past its bound and, for a GGUF build, the pinned file. The short
names (`names.py`), the GGUF builds (`gguf/builds.py`), the models of `ippon suite`, setup, `ippon recommend` and the
models table all read it. `licences.py` resolves a model's licence for a pull and for provenance alike. It gives the
licence name, its link at the revision that runs and whether it is accepted.

`locate.py` is the one place that reads a reference. A load, `ippon check`, `ippon pull` and the "is it downloaded"
question of the MCP server all use it. It opens the model's files as a `ModelDir` (`model_dir.py`). The source of a
`ModelDir` answers for the files in the format they were published in. That format is a folder of safetensors, a
GGUF build (`gguf/source.py`) or an Ollama build (`ollama.py`), so adapters and the loader never branch on the format.
For a check and a pull, `locate.py` also returns the typed list of what a pull downloads.

None of these modules loads MLX. A source imports MLX only to read weights.

`result.diagnostics["build"]` identifies the build behind an answer. It gives the short name where known, the Hub
repository and commit, the GGUF filename and SHA-256, the Ollama checkpoint identity, or the local path. Where
applicable, it also gives the pinned base of a LoRA. Over HTTP, `?diagnostics=1` adds diagnostics, and `/v1/models`
includes the details of each loaded model. [Formats](formats.md) explains packed builds and the limits of their
verification.

### Pulling and licences

`ippon pull NAME` reports a model's creator, compatibility, licence link, download size, free disk space and memory
fit before it fetches weights. To get these, it resolves the build and inspects its configuration and model-card
metadata. A Hub ID that no adapter claims can still load as a generic classifier or label-token model. A known
architecture is not sufficient on its own. The preflight says whether a model is a verified build, has another
catalog status, runs in a generic format or is unsupported. ippon refuses a model that cannot run or fit before any
weights download.

A model under a permissive open-source licence (Apache-2.0, MIT, BSD) downloads without a question. Every other
licence, such as the Kettei Community Model License or CC BY-NC 4.0, needs your acceptance once before ippon
downloads the weights. ippon remembers the acceptance. It never asks again for a later model under the same licence
or for a model already on your Mac.

- At a terminal, ippon shows the licence and its link and asks you. If you press Enter alone, the answer is no.
- In a script, an agent or CI, ippon never asks. Give `--accept-license` with `pull`, `decide`, `serve` or `setup`.
  To accept every licence that a download needs, set `IPPON_ACCEPT_LICENSE=1` in the environment.
- In Python, pass `accept_license=True`, as in `Decider.load(MODEL, accept_license=True)`. A library never asks.
- Without acceptance, the command stops with exit 11. It names the licence, its link and the way to accept it.

`--json` never prompts or prints progress, and it writes one final JSON result or error. ippon stores acceptance in
`~/Library/Application Support/ippon/licenses.json`, or in `IPPON_CONFIG`, separately from caches. After you accept a
named licence, it covers every later model under it.

An unidentified or custom licence is accepted only for its published terms or repository revision. A model without a
licence still needs acceptance. Dependent checkpoints and optional downloaded builds need their own licences too.

Downloads show the overall completed bytes, the total, the speed and the estimated time left. When output is
redirected, ippon writes periodic plain lines on stderr. Ctrl-C leaves partial files in the Hub cache, reports that
they can be resumed and returns 130. To resume, repeat the command. ippon records a completed download only when all
its files are present.

Disk admission counts the files already cached and the partial data already allocated. It refuses the download when
the remaining bytes exceed the free space on the cache disk (exit 12). Another program can still consume disk space
during a transfer.

Local integrations use `ippon.pull.prepare(MODEL)` and the `run(accept_license=..., approve=..., progress=...)` of
the returned plan. The approval callback receives a licence, and the progress callback receives aggregate download
events. The CLI uses the same flow. It downloads only the inspected revisions and leaves model execution to the
caller after completion.

Transfers use pinned HTTP byte ranges and keep partial files under the Hub blob lock. They check the content hash
before they publish a completed file. The Hub client supplies metadata, authentication and HTTP configuration. In
version 2.2, its own transfer function discards partial files on interruption. Every download ends in
`hub.complete`, whether it is for a load, a pull or the file of a GGUF build. `hub.complete` checks the SHA-256 of
each pinned file and records the download.

`ippon remove all --models` deletes a revision only when its download brought files that the Hub cache did not hold. It also keeps a revision
that holds a file ippon did not download, such as another app's quantization of the same commit, and says so.

## Python API

`ippon` exposes a direct Python API (`Decider.load`, `YesNo`, `Choice`, `Score`) for System One requests, typed results,
precision and settings selection, classifiers, and batches of states. It also answers Sqwish's decide API shape via
`model.sqwish(body)` (and `POST /v1/decide` on `ippon serve`).

```python
from ippon import Choice, Decider, Score, YesNo

with Decider.load("kettei-core:q8", accept_license=True) as model:
    result = model.decide(
        state="I was charged twice. Please reverse the second charge by Friday.",
        questions={
            "refund": YesNo("Is the customer asking for a refund?"),
            "team": Choice("Which team should handle it?", ["billing", "shipping", "other"]),
            "mood": Score("How upset is the customer?", ["calm", "annoyed", "furious"]),
        },
    )
    print(result["refund"].p_yes, result["team"].top, result["mood"].expected_level)
    print(result.response)
```

`systemone(body)` accepts and returns System One dictionaries. `decide_many(bodies)` returns typed results, and
`systemone_many(bodies)` returns response dictionaries. Compatible requests share passes. `close()` releases a
model's weights and prefixes, and the context manager closes the model automatically. Further decisions after
closing raise `DecisionError`. Direct Python loads own their weights in that process, while HTTP clients share the
model pool of a server.

## Command line

A request is a JSON file, JSON text, or `-` for stdin. `decide` reads and validates the request before it loads the
model. Results go to stdout, and diagnostics go to stderr.

In a terminal, `decide` prints a readable summary. When it is piped or given `--json`, it prints JSON. `prompt`
returns JSON, and `decide --states` returns JSON lines. Commands with `--json` give machine-readable reports and typed
failures, as [errors.md](errors.md) describes.

| Command | |
|---|---|
| `ippon doctor` | this Mac (chip, profile and its code paths, installed adapters), then setup checks and suggested fixes |
| `ippon recommend [text\|images\|audio\|videos]` | which builds this Mac can run for that input, and the one to start with. Before any download, it shows its size, what is checked of it and whether it fits |
| `ippon pull NAME` | download a model by its short name, a family (`kettei-core` gets its recommended build) or one build (`kettei-core:bf16`). It reports the size and whether the model fits this Mac, then answers a sample request with it. If you run it again, it fetches the newest revision. `--list` shows every family and build |
| `ippon pull MODEL --dry-run` | what runs it, the memory each precision needs here, what is verified and its licence, with no download. It reads the newest revision, the file sizes and the licence from the Hub. Offline, `ippon check MODEL` reports a complete cached download |
| `ippon decide MODEL request.json` | a System One request in, its response out (`run` is the same command) |
| `ippon decide MODEL request.json --states FILE.jsonl` | the request's questions asked about the state on each line, in batches. Each input line gives one JSON output line that holds its response or its error |
| `ippon lint request.json [--model MODEL]` | what to change in a request (vague or doubled questions, overlapping options, no catch-all). With a model, its tokens and what would be cut, and with `--prompt` the text the model reads |
| `ippon eval DATA.jsonl --model A [--model B]` | accuracy, macro-F1, confusion, confidence thresholds and where models disagree, on your labelled examples ([evaluating.md](evaluating.md)) |
| `ippon serve [MODEL] [NAME=MODEL ...] --port 8000` | POST /v1/decide (Sqwish's decide API), /v1/systemone, and /v1/decisions in System One's or SGLang's format, on this Mac, for one model or several (`--policy` and `--api-key` are in the [server contract](#http-server)) |
| `ippon serve MODEL --detach` | start a background server and print its address and log location |
| `ippon decide MODEL request.json --server` | use the shared server's loaded models, and start the server if needed |
| `ippon ps` / `ippon stop [MODEL] [--all] [--port PORT]` | inspect the recorded servers or stop one (the only one, a model's, the one on a port, or `--all`), with `--json` for scripts |
| `ippon mcp [MODEL ...]` | stdio MCP tools for coding agents, using the shared HTTP server ([agent guide](agents.md)) |
| `ippon verify MODEL --reference FILE` | the answers against the creator's code in fp32 |
| `ippon verify MODEL --reference FILE --invariance` | the same answers in chunks, kept, batched |
| `ippon verify MODEL --reference FILE --against ARM=PYTHON` | a quantized build against the runtime it was published for |
| `ippon bench MODEL` | 20 requests per shape, with the median, p95 and 95% interval |
| `ippon bench MODEL --compare ARM=PYTHON` | against another runtime of the model, on the same requests |
| `ippon bench MODEL --ab A B` | two configurations in alternating fresh processes |
| `ippon profile MODEL` | where a request's time goes, by operation, or by phase with `--phases` |
| `ippon suite --models DIR --out DIR` | verify, check and time every suite model on this Mac ([suite.md](suite.md)) |
| `ippon new-adapter CREATOR/MODEL` | an adapter package for a new model, with the parts to fill in marked |

## HTTP server

`ippon serve [MODEL] [NAME=MODEL ...]` serves one or more named models. By default, it serves the model that setup
chose. It loads the first model before it accepts requests, and it loads the others when they are first requested.
On a multi-model server, a body selects a model with `model`, and a body without `model` gets the first model. A
single-model HTTP server always uses its sole model. Shared CLI/MCP clients validate their selected model against the
configured list before they submit the request.

The server reports its actual URL. The default port is 8000, and the server falls back to a free port when that port
is occupied. `--detach --json` returns the PID, URL and log path of the background process. `ps --json` inspects
recorded servers. `stop` stops the only server, `stop MODEL` or `--port PORT` selects one, and `--all` stops every
one. Shared CLI/MCP clients reuse that server.

The server has these routes.

- `POST /v1/systemone` takes a System One request and returns its response.
- `POST /v1/decide` takes Sqwish's decide API and checks the request as the API checks it. The API has `context`,
  `decisions` with `kind` single, binary or ordinal, `outcomes`, `rubric`, `policy` and `temperature`. For a model
  that reads images, it has `images` placed with `<image:N>`. On this Mac, it also has `videos` and `audio` placed
  with `<video:N>` and `<audio:N>`, and the anchors stay in the text. The response gives each decision's
  probabilities by outcome, `top`, `p_top`, an ordinal's `expected_level` and a policy's action (`decide_api.py`). It
  also gives `timing_ms`, with the milliseconds that the request's rendering (`render`) and its scoring pass (`score`)
  took, as `Decider.sqwish` gives them in process. Any process or language can send the same request that the Sqwish
  SDK's `decide` takes.
- `POST /v1/decisions` takes System One, or SGLang's decisions format (`input` and a list of questions with IDs).
- `GET /v1/models` lists the configured models in OpenAI and TypeSafe list shapes. For each model, it gives what the
  model loads, its weights (`full`, `q8`, `q6` or `q4`) and the precision it was asked to load at. It also gives the
  load options that change its answers (`reads`), loaded details and provenance.
- `GET /v1/status` reports the process footprint, the memory budget, model and prefix memory, pressure and recent
  releases.
- `GET /health` returns 200 while the model thread runs, and 503 while closing or after it stops.

SGLang questions use `choice`, `score` or `yes_no`. The adapter returns the requested response shape. Each
`label_mass` is 1, because the decision head normalizes over its options. `temperature` divides label
log-probabilities. ippon rejects `chat_template_kwargs`, `return_prompt_token_ids` and remote image URLs. The
`/v1/decisions` route also accepts the System One request shape that Perplexity's Decisions API uses.

Compatible request fields and paths do not imply identical response semantics. ippon reports `confidence` as each
model's own server does. This is the highest option probability, unless the model's creator measures it otherwise.
For example, Perplexity's choice confidence is how far that probability stands above an even spread.

TypeSafe SDK 0.7.5 text calls work with a local base URL, an explicit model and a placeholder API key when the server
has no key configured. The base URL is the server's address without `/v1` (`http://127.0.0.1:8000`), because the SDK
adds `/v1/systemone` itself. A base URL that ends in `/v1` calls `/v1/v1/systemone`. The SDK's newer `assets` field
and `box` questions are not supported. For builds that support media, use the media fields below.

Bodies can use gzip, deflate or zstd compression. `--max-body-mb` limits the compressed body and the decoded body
each. By default, the limit is a 32nd of the Mac's memory, from 64 MiB to 1.5 GiB, which is 256 MiB on an 8 GB Mac.
The reason is that ippon holds a video sent as base64 about 4 times over while it reads it. The same limit applies to
an MCP message.

The default bind is `127.0.0.1`. On a loopback bind, Host must be `localhost`, `127.0.0.1` or `[::1]`, optionally
with a port. The server refuses requests that carry an Origin unless `--allow-origin ORIGIN` permits it. You can
repeat this option, and `*` permits all origins. Allowed origins receive CORS headers and OPTIONS preflights.
`--api-key KEY`, or `IPPON_API_KEY`, requires `Authorization: Bearer KEY` on every request except preflights.

Without a configured key, the server ignores Authorization. `--host 0.0.0.0` exposes the server to the network, and
ippon then needs an API key, which every request must carry. Without the key, other machines on the network could
use the server. A web page could also reach it through DNS rebinding, and the Host check stops that on loopback
only. HTTP is unencrypted, so use an appropriate secured transport when access is not confined to the local machine.
See the [Security policy](../SECURITY.md).

One model thread runs inference. By default, the queue holds at most 256 waiting requests (`--max-queue`), and a
batch answers up to 64 of them (`--max-batch`). `--batch-window-ms` defaults to 0. A request waits as long as the
model takes, unless `--timeout-s` gives it a deadline. The server drops cancelled or timed-out work that has not
begun, and an in-flight pass finishes. Errors fail the affected request or batch, and the model thread continues
([HTTP errors and retries](errors.md)).

The model pool shares its memory budget between loaded weights and kept prefixes. It refuses models over the weight
budget. To load another model, the pool evicts the least recently used models first. `--memory-gb` sets the budget.
This estimate is not a guarantee against transient activation peaks or macOS swapping.

`--policy balanced` keeps loaded models and sends an idle heartbeat for 5 minutes after a request. `--policy latency`
keeps that heartbeat active for the server's lifetime, and `--no-keep-awake` disables it. `--keep-alive 10m`, `0` or
`forever` controls model unloading independently. With these values, a model unloads after that idle duration,
immediately when no work waits, or never. `--keep-awake` and `--keep-awake-window` override the heartbeat timing, and
`ippon serve --help` lists all policy presets. Automatically started CLI/MCP servers use a 30-minute model idle limit.

The memory monitor reads macOS pressure without MLX. Between batches, the model thread reduces kept-prefix budgets,
clears unused MLX buffers and then unloads idle models that have no waiting requests. Under hard pressure, the server
rejects requests for a model that is not loaded with 503 `overloaded` and `Retry-After`. A loaded model continues to
answer. Recovery restores prefix budgets. `/v1/status` records these actions and monitoring errors, and all of them
are memory-management heuristics.

SIGTERM follows the cleanup path of Ctrl-C. The server stops accepting connections and gives queued work a 5-second
grace period. Then it refuses the remaining waiting work as `unavailable`, stops background threads and unloads
models. A running pass can finish before cleanup completes. If its shutdown timeout expires, `ippon stop` escalates
to SIGKILL.

## Images, audio and video

Media support depends on the build, including whether the build includes its creator's media towers. The
full-precision checkpoints of d1-2B, d1-Omni-2B, d3-mini, d3-flash, pplx-decider flash, Torchcast Decision 4B,
H2O-Lightning-4B and elephant-1 have creator-reference checks of their image answers. Those of d1-2B, d1-Omni-2B,
d3-mini and d3-flash have them for video answers too, and d1-Omni-2B's has them for audio answers. Clef-Flash image
prompts are token-checked, but the verification of their image answers is still open. `verified.py` records each
build's media status, and [Model status](models.md) shows these distinctions.

| Request field | Content |
|---|---|
| `images` | base64 strings or `data:image/...;base64,` URIs |
| `files` | images, videos and audio clips, distinguished by their bytes. An audio-only WebM or MP4 needs its `data:audio/...` URI, since its bytes look like a video's |
| state `image_url` parts | inline data URIs, as many as the model's window holds. Remote URLs are refused |
| `videos` | base64 strings or `data:video/...;base64,` URIs, read after the images |
| `audio` | a clip or a list of clips, base64 or data URIs, in WAV, MP3, M4A or AAC, FLAC, Ogg Vorbis, AIFF or CAF, at any rate and channel count, and Opus (a voice note's, or a browser's WebM recording) |

Each decoded medium in a request is limited to 1 GiB, within the body's limit. ippon reads a video of any length or
frame size from its sampled frames, as the model reads it. `ippon decide MODEL request.json --image photo.jpg`
(and `--video`, `--audio`, each repeatable) adds local files to a request from the command line. In Python, `decide(..., images=[...], audio=...)` also
accepts local paths, bytes, supported PIL images and arrays, with no size limit. ippon never loads paths from HTTP
request fields and never fetches URLs. A model refuses unsupported media instead of ignoring it.

## The workload

A decision request is a state plus questions. For the model, this means the following.

- The model reads the prompt and generates nothing. One forward pass reads the prompt, and a readout follows. Speed
  depends on how fast the model reads a prompt, and token generation speed does not matter.
- Many questions ask about one state. Most models give each question its own prompt, and those prompts share
  everything up to the question. Reading the state once for all of them is the largest saving there is.
- The readout is at a few positions. Only the readout rows need the output of the last layer. From those rows, only
  the logits of the label tokens (or the model's own head) are needed.
- There are 2 latency regimes. Reading the weights bounds a short request, and compute bounds a long one.
- Requests have gaps between them. After an idle gap, a Mac's GPU clocks down and macOS can stop keeping the weights
  resident. So a loaded model wires its memory, and the server policy controls its idle heartbeat
  ([below](#between-requests-residencypy)).

## How a model plugs in, by traits and not flags

Decision models differ along 5 traits. An adapter states them once, or they come from the model's config. From the
traits, the runtime picks the fastest correct way to run the model.

| Trait | Values | What the runtime does |
|---|---|---|
| Layers | Qwen3.5 hybrid, LFM2, LFM3 (set attention), ModernBERT, dense decoders (Llama, Mistral, Qwen2, Qwen3, with YaRN and grouped-norm variants), BERT, DeBERTa, Gemma 3 and 4 | the `arch` module and its kernels |
| Attention | causal / non-causal / block / set | Causal reads a shared prefix once. Non-causal reads every prompt whole. Block reads the marked prefix both ways once and each question after it. Set reads every prompt whole. In it, the state reads its questions, and each question reads the state and itself |
| Prompts | one per question / one for all questions | prefix then tails / a single pass |
| Readout | label logits at slots, a code readout (d3, Perplexity), a joint schema head (Clef) | last layer for the read rows only, every row for a whole-sequence head |
| Weights | full, LoRA on a pinned base, separate head files | LoRA merged into the base at load |

## The stack

### Interface (`schema.py`, `questions.py`, `decider.py`, `cli.py`)

The interface speaks the System One schema of the creators' servers. A state and named `noul`, `choice` and `score`
questions go in, and one answer per question comes out. `parse_request()` validates a body once into immutable typed
values. `YesNo`, `Choice` and `Score` build the same questions in Python. `Decider.load()` accepts the model names and
paths described above. `decide()` returns typed answers and the creator's response body, and `systemone()` is JSON
in, JSON out.

Some formats cut input to fit their model's window, as the creator's code does. For example, Clef cuts the end of
the state. Such a format records each cut in `Rendering.cuts`. The response lists the cuts under `truncated`
(question, part, tokens dropped and kept), in the System One, decide API and SGLang forms alike.
`truncation="error"` (`--truncation error`) refuses such a request instead. `Decider.inspect`
(`ippon lint --model MODEL --prompt`) shows the text and tokens of each prompt.

Sometimes the creator's code sets a window below what the model reads, such as a server's context or a training
budget. This window is 40,944 tokens for H2O-Lightning (its server's 40,960-token context less 16 tokens of slack) and 16,384 for Clef-Flash and Torchcast. `window=`
(`--window TOKENS`) replaces it. ippon reads prompts past the creator's window whole, but they are unverified, and a
warning and `details["window"]` say so. The decider refuses a rendered prompt past the window (`context_too_long`), as
it refuses one past the model's positions. So a format that cuts nothing to fit does not need to check the window.

Each `Result` says in `result.diagnostics` how it was answered. It gives the precision it computed in and whether
input was cut. When another path answered, it says why, such as an fp16 overflow answered again in bf16.
`kept_tokens` gives how many input tokens came from an earlier request's reading. Over HTTP, `?diagnostics=1` adds
the diagnostics to the response.

`serve.py` answers the same schema over HTTP (`POST /v1/systemone`, and `/v1/decisions`). When it is free, one model
thread takes every waiting request. The queue is bounded, and a full queue answers 503 with `Retry-After`. If
`--timeout-s` is given, a request past it gets a 504, and a request that still waits is dropped from the queue. The
server skips a cancelled request. No error stops the model thread, and `/health` says whether it runs.

One server can answer for several models (`ippon serve A B C`). A request names its model, and the one model thread
answers the waiting requests of each model together. The first model loads at startup, and the others load on their
first requests. Past the memory budget, the least recently used models unload first.

`pressure.py` samples the process footprint and the macOS memory state without calling MLX. Between batches, the
model thread acts on pressure. It reduces prefix budgets, clears unused MLX buffers, then unloads idle models with no
waiting requests. Hard pressure rejects requests for unloaded models with 503, and recovery restores prefix budgets.
`/v1/status` reports the readings and recent releases. SIGTERM follows the same cleanup path as Ctrl-C.

`served.py` holds what a server and its clients agree on. This is where a local server is reached, and each model as
`GET /v1/models` lists it (`ServedModel`). It also holds the one rule for whether a model answers a name. The server
routes requests by this rule, and its clients find the model by it.

`daemon.py` records local servers. It finds the server that answers a model at a precision and with the load options
a client gives, or it starts one (`daemon.find`, `daemon.ensure`). If a running server has the model loaded
otherwise, another server starts beside it. `daemon.py` also provides the HTTP client that `decide --server`,
`ippon start`, the agent commands and `mcp.py` use.

The MCP process handles stdio messages and delegates inference over HTTP. It never loads model weights itself.
Clients that start together coordinate through a startup lock and reuse the recorded server. Offline model checks
use the complete cached snapshot that `hub.py` records, and `pull` explicitly checks the newest remote revision.

### Adapters (`adapter.py`, `model_dir.py`, `registry.py`, `adapters/`)

An adapter is everything about a model family other than its layers. It names the files the model needs and how to
recognise a download. It also holds the format. The format is how the creator's code renders a request into prompts,
where the answers are read, and the shape of the answers. Adapters register through the `ippon.adapters` entry
points, the built-in ones too. They are checked against the creator's code token for token and probability for
probability.

ippon never imports or runs code from a model repository. `hub.py` restricts downloads to weights,
configuration/tokenizer assets, the model card and its licence (`.safetensors`, `.json`, `.txt`, `.model`, `.jinja`,
`.md`, `LICENSE`). It also fetches a pickle that an adapter names exactly in its `read_safely`, which only `ptfile.py`
reads. The GGUF route downloads the files that `verified.py` pins by SHA-256.

What one model needs stays in its adapter, which declares it. The library reads the declarations without naming a
model. The optional members of an adapter have their defaults on `AdapterBase`, and those of a format have theirs on
`FormatBase`. These members give the pickles the adapter reads safely (`read_safely`, fetched with the first small
download). They give how the tensors of its GGUF builds read and check (`gguf_rules`, `gguf_check`, built on the
architecture tables in `gguf/names.py`). They also give the base of a LoRA, the questions its benchmarks ask and its
compare arms.

The adapter passes in a creator's answer arithmetic as functions. These are a confidence measure, how a wide choice
is split and its rounds combined (`rounds.Wide`), and how the label tokens of an option pool
(`readout.option_probabilities`). `formulas.py` holds the confidence measures that several creators share.

Backbone variants are load options of the `arch` module (`causal=False`, `block`, Qwen3.5's `recurrence`). The media
paths that several models share are in `media/` (`paths.py`, and Qwen's image and video paths in `qwen.py`). Each
decoder there is named for its creator's decoder.

### Planning and running (`plan.py`, `runtime.py`)

ippon makes a request's prompts into a shared prefix and one tail per prompt, with every readout in a tail. The
strategy follows from the prompts and the backbone.

- With one prompt, ippon reads it causally in chunks of 4,096 tokens and keeps the readout rows from the last chunk.
- With several prompts and a causal backbone, ippon reads every chunk of the prefix except the last. Then it reads
  that last chunk and every tail in one pass (`Backbone.branch`). So a request whose prefix fits in one chunk reads
  the weights once. A pass reads every weight, and for a short request that costs as much as its tokens.
- With a non-causal backbone, ippon reads every prompt whole, together in right-padded batches of up to 16,384
  tokens. A set model's prompts (several questions packed after their state, `Prompt.groups`) carry the question of
  each token. The model's full attention layers read by it.
- `strategy="reference"` reads every prompt whole and alone. The other paths are tested against this path.

ippon reads several requests at once as few plans as it can (`runtime.read_requests`, which `decide_many` and a
server's batches use). Requests whose prompts start alike are read as one plan. Single prompts about different states
that each fit a chunk are read in one padded batch, which reads the weights once for all of them. ippon plans each
request once. Where the creator's code answers a choice wider than one pass in rounds, ippon does the same
(`rounds.py`).

### Kept prefixes (`prefixes.py`)

Agents send the same state again and again, as a request per step or as other questions about one state. After ippon
reads a causal prefix, the decider keeps an exact-size copy of its keys and values and of the recurrent state. It
keeps them at each chunk boundary, at the mark that the adapter set, and at the end. The mark is a few tokens before
the end of the state.

A mark splits the recurrence inside the pass, and it never splits the pass. Each prompt has one mark, because each
mark costs every recurrent layer a split and a copy of its state. A request that resumes from a mark reads the few
tokens after it again.

A later request can resume when its prefix starts with the same tokens up to one of those positions. It copies the
keys and values before that position and reads on from there. A hybrid model has a recurrent state only where ippon
saved it, so only those positions can be resumed from. Beyond a budget, entries go least recently used first. The
budget is 1/16 of memory, at most 4 GB, and `Decider.load(..., prefix_budget=0)` or `--prefix-budget 0` turns it off.
Non-causal models keep nothing.

A head that reads every position, such as Clef's, also needs the final hidden states of the prefix. So a lone
prompt's plan can end its prefix at the last mark of the state, even past readouts, because readouts count back from
the tail. When the request read all the rows of the prefix, the entry keeps the rows of every prefix position. A
request that needs rows inside its prefix resumes only from an entry that has them. With the same 4,000-word state
and other questions, Clef-Flash went from 2.48 s to 233 ms on an M5 Pro. It took 241 ms for a grown state.

### Kept answers (`answers.py`)

Agents also send the very same request again. Examples are a command check for `git status`, a file read twice, or
the same question about a transcript that has not changed. A model gives the same answer to the same prompt. So a
decider keeps the answers to its last 1,024 requests and answers an identical request without reading it.

The key is what the model reads. This is the tokens and readouts of each prompt as rendered, with an image or a clip
by the digest of its placeholders. The key also includes the id, type and options of each question, and the chunk
and strategy of the decider, because these round differently.

On Kettei Dot through `ippon serve`, the README's Quick start request sent again took a median 0.5 ms, against 15 ms
when read again (30 sends each, M5 Pro). The answer cache is on by default. `--answer-cache 0` and `Decider.load(..., answer_cache=0)` keep no answers,
so ippon reads every request.

A choice answered in rounds is read each time. The answers are cleared when the model moves to bf16 and when it
unloads. The warm-up, `bench`, `profile`, `verify` and the invariance check read every request even with the cache on
(`Decider.uncached()`).

### Between requests (`residency.py`)

A model loads, answers warm-up requests and wires its memory. The warm-up requests evaluate its arrays and initialize
the paths those requests use. Otherwise, macOS stops keeping the memory wired after a few idle seconds, and the next
request waits while the driver wires it back. On an M5 Pro, a 500-token request after 5 s idle took 227 ms instead of
141 ms.

A decider also sends the GPU a trivial command every second for 5 minutes after each answer (`keep_awake`, on by
default, None for none). It sends the command from a thread of its own that waits for its lock. An agent's calls come
seconds apart. In a program of its own, Kettei Dot answered a short call 5 to 15 s after the last one. With the
command, it took 30 ms, and without it, 109 to 115 ms. The command does not prevent the GPU's clock from falling in a pause.
So a 500-token, 3-question call 2 s after the last one took 61 ms, against 52 ms back to back.

`ippon serve` sends that command from its model thread instead, between passes. So with several models loaded, the
command never lands in another model's pass. The server sends it for 5 minutes after the last request (`--policy`,
`--keep-awake-window`).

The wired limit and MLX's buffer cache scale with the Mac. The cache keeps 10% of the GPU working set, at most 1 GiB
(about 0.5 GiB on an 8 GB Mac). ippon reads the wired limit from the Mac as it is at that moment. The limit is
nothing while macOS reports memory pressure. It is also nothing while the available memory is under the line at
which a server counts memory as plenty.

Otherwise, the wired limit is the model and what its passes need, from the memory available above that line. It
stays within 90% of the working set and half the Mac's memory. A server lets go of wired memory while memory is
short, and it wires again once memory is plenty. The GPU's clock still falls when idle, which costs the next request
about 12 ms, and only real load holds it up.

### Layers (`arch/`)

Each family's layers sit behind `arch.base.Backbone` and its 3 passes. `prefill` reads tokens after a cache,
causally. `branch` reads the end of a shared prefix and then several tails after it, in one pass. Each tail sees the
cache, that prefix and its own earlier tokens. `batch` reads whole prompts, right-padded. All 3 passes return the
final-normed hidden states of the requested rows only, and the last layer skips everything else.

`arch/common.py` holds what the families share. This is the pass layouts (`Rows`), RoPE at each row's position, the
masks, attention for the read rows, and the `Passes` mixin. The mixin implements the 3 passes over an architecture's
`_forward`. At the marks of a pass, recurrent layers keep their state in `cache.kept`. They split the recurrence
there and do not split the pass.

A `_forward` runs its layers through `Passes._layers`, each layer's `_layer` in turn. With the compile switch, each
layer runs as an `mx.compile` trace per layer kind and pass shape. The inputs of the trace are the layer's weights,
the pass's arrays and the layer's cache block. So Python builds one layer of each kind, and the others replay it.
Compilation applies to eligible GPU layers with plain array weights. Packed-matrix layers, custom measuring Ops and
`attention=bounded` run eagerly.

`arch/lfm2.py` follows Liquid's LFM2 (gated short convolutions and GQA attention, causal or bidirectional).
`arch/lfm3.py` follows Liquid's LFM3 as d1-2B reads it (LFM2's convolutions, gated sliding and full attention,
sandwich norms, the full layers under set attention). `arch/modernbert.py` follows Hugging Face's ModernBERT (global
and sliding-window attention, an encoder).

`arch/qwen35.py` follows Hugging Face's Qwen3_5 text model. It has zero-centred RMSNorm and gated full attention with
per-head q/k norms and partial RoPE (causal or not). Its Gated DeltaNet layers run through MLX's chunked
`mx.fast.gated_delta_update`, which runs on the M5's neural accelerators. Its SwiGLU MLP has its activation in fp32.
In a branch pass, attention layers see the tails through a tree mask over one key/value buffer, with no copy of the
prefix's keys. Recurrent layers run the tails as a batch whose initial state is the prefix's state, broadcast.

Matrices that read the same input are merged at load. In `arch/cache.py`, key/value buffers are preallocated and
written in place. The recurrent state and the convolution window have a fixed size at any length.

### Readout (`readout.py`)

Label probabilities come from the label rows of the output matrix only. This is exact because normalizing over the
vocabulary first cancels out of a softmax over the options.

### Operations and kernels (`arch/ops.py`, `kernels/`)

Metal bodies and helpers are in [`kernels/metal/`](../src/ippon/kernels/metal/README.md). The Python wrappers select
variants, assemble compiler inputs and configure launches. A cached package-resource reader loads the sources from an
installed wheel as well as from a checkout. The reader does not depend on the working directory or import MLX. Shared
activations and quantization loaders have their own source files. The package CI check compares the installed
wheel's sources with the checkout, and GPU tests exercise the assembled kernels against MLX.

The architectures share the elementwise work around the matrix multiplies, and `arch/ops.py` computes it. This work
is norms (RMSNorm, LayerNorm), the residual add with the norm after it, and gated linear units (SwiGLU, GeGLU). It is
also the hybrids' short convolutions (Gated DeltaNet's inputs with SiLU, LFM2's C · conv(B · x), causal or centred).
ippon sets `arch/ops.py` on the backbone at load from the chip profile's switches. The MLX compositions are the
reference.

On every Apple GPU, the fused Metal kernels (`kernels/norm.py`, `kernels/glu.py`, `kernels/conv.py`) compute the
same numbers in one read and one write. The compositions make up to 6 passes over the MLP's widest activation. Each
layer carries its MLP output forward, so the residual add and the next norm are one kernel.
`IPPON_SWITCHES="add_norm=mlx,glu=mlx"` turns the kernels off, for comparisons in fresh processes. The kernels are
tested against the compositions for every dtype and width. Whole models are tested against the numpy models with
exact fp32 kernels.

### Chip profiles (`device.py`, `profiles.py`)

ippon reads the chip from MLX's device info. This is the GPU family, where `applegpu_g13` is M1 and `g17` is M5, and
the suffix is the tier. A profile, m1 to m5, picks each operation's code path once at load. You can force a lower
profile with `profile=`, `IPPON_PROFILE` or `--profile`. ippon refuses a profile above the chip.

You can set the path of any single operation over the profile's path with `switches="glu=mlx,gemm=mlx"`,
`IPPON_SWITCHES` or `--switch`. `profiles.Switches` lists the switches, and "mlx" is always MLX's own operation. This
is how every fused or chosen path is A/B'd.

| Switch | Paths | Every profile runs |
|---|---|---|
| `add_norm`, `glu`, `conv` | `kernel`, `mlx` | the fused kernels (`kernels/norm.py`, `glu.py`, `conv.py`) |
| `gemm` | `kernel`, `mlx` | the kernels that `choices.py` picks at load for each matrix shape and band of rows from the chip ([kernels.md](kernels.md)). On m5, these are the neural accelerators' tile library and the quantized kernel (`kernels/gemm.py`, `kernels/qgemm.py`). The tile library fuses the GLU into the gate/up matmul, and it also runs MLX's thin kernel as tiles, which MLX itself runs only up to 64 rows. On m1 to m4, MLX's matmul, except that m1, m2 and m4 unpack MXFP8 matrices to full precision from 224, 224 and 1,024 rows |
| `gated_delta` | `mlx`, `half` | MLX's chunked kernel with FP32 inputs (`ops.gated_delta`). `half` is an opt-in precision experiment |
| `delta_inputs` | `kernel`, `mlx` | fused preparation of DeltaNet inputs |
| `norm_rope` | `mlx`, `kernel` | MLX norm/RoPE compositions |
| `pipeline` | `layers`, `early`, `off` | `layers` hands completed layer work to the GPU while later graphs are built |
| `compile` | `layers`, `off` | each layer as one `mx.compile` trace per layer kind and pass shape, with its weights, the pass's arrays and its cache block as the inputs. The trace replays for every layer of its kind, so eligible layers replay the same graph with their own weights. Packed-matrix layers and explicit bounded attention run eagerly (`arch/common.py`) |
| `attention` | `mlx`, `bounded` | `mlx` is the default. `bounded` evaluates MLX attention in at most 128 query rows at a time, each against all keys, for unmasked head-size-256 calls whose scores would occupy at least 512 MiB. This explicit memory mode runs layers eagerly, including requests that never need splitting |

`choices.py` picks a kernel for each band of input rows of each matrix. It does this at load and for the shapes that
the loaded model has. On an M5, quantized matrices take ippon's quantized kernel from 16 rows. On an M5 with 20 GPU
cores (the M5 Pro), full-precision matrices take MLX's thin kernel to 160 rows, a 64 × 32 tile to 384 and MLX's
matmul above. Other M5s use MLX's matmul for them. From 65 rows, MLX's fixed
tile leaves GPU cores idle or rereads the weights.

`choices.py` reads the chip's generation and GPU core count, and it measures nothing on the Mac.
[kernels.md](kernels.md) has the rule and the measurements it was fitted to. On m1, m2 and m4, the only choice is where MXFP8
matrices are unpacked to full precision, from 224, 224 and 1,024 rows. Nothing else is chosen on m1 to m4.

### Precision and memory (`quantization.py`, `precision.py`, `admission.py`)

`quantization.py` holds the one table of precisions (`PRECISIONS`). It gives auto, bf16, fp16, fp32, q8, q6 and q4.
Each precision has its dtype, its bits, whether it changes the weights and whether it is below 8 bits. The table also has the
formats ippon reads, the group size it quantizes to and the names of a packed matrix's companions. It imports no MLX,
so `ippon check`, setup, short names and admission read it too. `precision.py` maps its dtypes to MLX's dtypes and
measures auto.

The default, `precision="auto"`, computes in fp16 when the model's activations stay below 2/3 of fp16's range.
ippon measures the activations on the warm-up requests. Otherwise, it computes in bf16. In the recorded M5 Pro d1-2B
checks, FP16 probabilities were within 0.0012 of creator FP32, against 0.0160 for BF16. [accuracy.md](accuracy.md)
records the other builds and limits. A later non-finite FP16 answer triggers a BF16 reload and retry.

Quantized weights (q8) choose their compute dtype the same way. Before anything loads, `admission.py` estimates the
model's resident size from its safetensors headers. The estimate leaves out the vocabulary tables, because a load
reads their rows from the file. `admission.py` compares the estimate with the GPU's working set.

`safetensors.py` reads and checks those headers, and it imports no MLX either. It holds the one table of stored
dtypes and their widths. Admission, locating a model, `quantization.inspect` and the vocabulary tables read from
their files (`host_rows.py`) share this table.

ippon refuses a precision that does not fit, auto included, and names the precisions that do. Setup and `ippon pull`
use the same rule (`recommend.fit`). The rule judges by the Mac's memory and never by what other apps hold at that
moment, so a model that setup says fits always loads. ippon never quantizes a model on its own. Because q8 changes
the model's weights, it runs only when asked for.

A forced profile switches ippon's own paths, and MLX keeps choosing kernels by the chip. To have MLX take the older
chip's kernels too, set `IPPON_PROFILE=m4` (for example) before Python starts. `_emulate.py` then sets MLX's
undocumented `MLX_METAL_GPU_ARCH` to that chip's family before MLX loads. `Profile.emulated` and `ippon doctor`
report this. This setting exercises M1 to M4 paths on an M5, and it does not measure those chips.

### Verification and timing (`verify.py`, `bench.py`, `references/`)

`verify.py` compares a model with its creator's reference file. `references/<adapter>/make_reference.py` makes that
file, and it runs the creator's own code in fp32 in its own environment. `verify.py` checks prompt tokens exactly,
probabilities within the model's bound, top answers and token counts.

`bench.py` times a model the way every published number is measured. A `Spec` says which requests to send.
`compare.py` sends the same requests to other runtimes, and `ab.py` sends them to 2 settings in alternating fresh
processes (`processes.py`). `stats.py` gives the medians, percentiles and bootstrap intervals, and `machine.py` gives
the Mac they were measured on. What ippon writes on this Mac (file hashes, download records, running servers) is
under `storage.cache_dir()`.

## Rules every layer follows

- Answers come first. Every faster path is tested against the simple one. It must match exactly where the arithmetic
  is the same, and within a stated bound where it is not.
- Validate at the boundary, then trust. ippon checks requests once. It raises wrong output, such as probabilities
  that do not sum to 1 or a missing readout, and never returns it.
- ippon chooses one path per machine at load.
- ippon runs no code from model repositories, and it reads weights from safetensors and GGUF only. The one exception
  is a small head that an adapter names exactly (Kev's `head.pt`). `ptfile.py` reads it through a restricted
  unpickler, which runs nothing from the file.
- Every claim is measured.
