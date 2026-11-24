# The suite of everything ippon measures, on one Mac

`ippon suite` is the one command to run on a Mac ippon hasn't measured yet (an M1, M2, M3 or M4). It
records the machine (`doctor`, `info`), then runs these steps for every model of the suite found in the models folder.

| step | what |
|---|---|
| `ab-kernels` | ippon's kernel choices against MLX's own matmuls (`--switch gemm=mlx`), at 100 to 4,000 words |
| `verify` | its answers against its creator's code in fp32 (references/) |
| `invariance` | the same answers in other chunks, from kept and grown states, batched (skipped with `--quick`) |
| `bench` | fresh states of 300 to 4,000 words, 1 and 5 questions, 20 requests each |
| `bench-short`, `bench-repeat` | 20 to 100 words, then a state asked again |
| `compare`, `compare-repeat` | the same requests through another runtime (a community port), where one fits, with fresh states and then one state asked again |

Runtime depends on the Mac and selected models. `--quick` uses fewer shapes and requests and skips invariance.
Each step runs in its own process. If a step fails, the suite records the failure and goes on.

## On a fresh Mac

1. Keep the Mac otherwise idle, because the results record competing activity. Do not run tests, builds or another
   benchmark during timing.
2. Clone ippon (`git clone https://github.com/sqwishai/ippon`) and install
   [uv](https://docs.astral.sh/uv/getting-started/installation/).
3. Run `sh scripts/suite.sh setup` to set up the ippon environment, and mlx-vlm's (pinned) for the comparison
   arm, in `~/.ippon-arms`.
4. Run `sh scripts/suite.sh models ~/models` to download the suite's Hub models.
   GGUF builds download on their first load.
   Check model sizes before starting.
5. Run `sh scripts/suite.sh run ~/models` (add `--quick` for a first look, `--only d1-2b,d3-mini` for some models).
6. Read `suite-<chip>/summary.md`, with `suite.json` and the per-step JSON and logs beside it.
7. To share the results, start a [discussion](https://github.com/sqwishai/ippon/discussions) with the folder zipped.
   If a step failed, attach the zipped folder to an issue.

## On smaller Macs

On 8 and 16 GB Macs, close everything else first. ippon refuses a model that does not fit the GPU's working set before
anything loads (`admission.py`), and names the precisions that would fit. ippon never quantizes a model on its own, so
the suite records the steps of that model as failed and goes on. Clef-Flash's Q8_0 GGUF and 8-bit build, and
Perplexity flash, need a 16 GB Mac or more.

## The models

`ippon.suite.SUITE` holds every named build in [models.md](models.md) that downloads from Hugging Face and has its
creator's reference answers in `references/`, as the catalogue (`src/ippon/verified.py`) lists them. It also holds
the video checks of d1-2B and d1-Omni-2B. Each build has a short name, such as `d1-2b-bf16` or `d1-2b-q8_0`.
`--only` takes these names or the name of a family, such as `--only d1-2b` for all of d1-2B's builds.

The Hub folders download at the revisions they were checked at, each into a folder named as its repository. The GGUF
builds download on their first load. Clef-Flash's reference is tokens only, because its fp32 model is 38 GB. The suite
matches a build that is past its bound as published against its recorded distance.
