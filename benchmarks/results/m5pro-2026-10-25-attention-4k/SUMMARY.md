# M5 Pro fused-attention gate through 4,096 tokens

The broad `attention=kernel` policy is **not promoted**. The default remains `attention=mlx`.
The clean Perplexity Flash run's largest median gain is 0.89%, below the required 3%.
Numerical checks passing does not override that performance requirement.

## Method

- Apple M5 Pro, 48 GiB unified memory, macOS 27.0.1, MLX 0.32.4.
- A: `--switch attention=mlx`; B: `--switch attention=kernel`. Other settings are identical.
- Automatic precision selected FP16. A separate empty tuning cache supplies the untuned defaults.
- Four rounds, alternating fresh processes A/B and B/A. Each shape has three untimed warm-up requests
  per process. Loading and warm-up are excluded; rendering, tokenization and the complete decision are timed.
- Fresh states, five sizes from 100 words to the model-specific 4k limit, one and three questions.
  D3-lite, D3-mini NVFP4 and Perplexity use 40 timed requests per cell in total; Clef and Nimble use 20.
- `scripts/bench_attention_gate.py` renders every actual warm-up and timed request before benchmarking.
  **Every individual prompt is at most 4,096 tokens, including its template.** A multi-question request may
  contain several such prompts, so its aggregate input-token count can exceed 4,096.
- Every arm uses the ordinary `ippon bench --ab` machinery. Evidence records source hashes,
  build/dtype, power state, prompt lengths, requests, probabilities, timings and competing-process flags.
  Campaign commands ran serially under the shared GPU lock; we launched no tests or builds alongside them.
- Promotion requires at least one cell with a median gain of 3% and its ratio confidence interval below 1,
  no significant median or p95 regression above 1%, and the model's numerical gate to pass.
  A significant p95 regression now takes precedence over a faster median; a regression test covers this.

## Completed timing runs

These are the final campaign's pooled raw results. A run with competing-process flags cannot qualify for
promotion, irrespective of its raw verdict. No noisy measurements are presented as release evidence.

| Build | Timed requests/cell | Maximum prompt | Best raw median gain | Raw verdict | Environment qualifies |
|---|---:|---:|---:|---|---|
| [D3-lite](d3-lite-kernel-idle/ab.json) | 40 | 4,096 | 0.68% | reject | no |
| [D3-mini NVFP4](d3-mini-nvfp4-kernel-idle/ab.json) | 40 | 4,096 | 0.81% | reject | no |
| [Perplexity Flash v1.2](pplx-flash-kernel-idle/ab.json) | 40 | 4,096 | 0.89% | no difference | yes |
| [Clef Flash 8-bit](clef-flash-8bit-kernel-idle/ab.json) | 20 | 4,096 | 0.53% | no difference | no |
| [Nimble 9B merged](nimble-9b-kernel-idle/ab.json) | 20 | 4,096 | 0.35% | no difference | no |

Perplexity's clean, single-question 4,096-token cell is **1,223.15 ms to 1,212.29 ms**;
B/A is 0.9911 with a 95% interval of 0.9900–0.9918. This is a measured small gain, but not a promotion.
Its three-question largest-state cell is 3,639.31 ms to 3,638.73 ms (B/A 0.9998).
All ten cells are below the 3% gain requirement.

The contaminated final runs retained 3/4 clean round pairs for D3-lite, 1/4 for NVFP4, 2/4 for Clef and 3/4 for Nimble.
They are retained in full, rather than relabelled as clean gates after dropping inconvenient samples.
Earlier D3 attempts and the interrupted initial NVFP4 attempt also remain in this directory;
`promotion-status.json` records their qualification separately.
The original plan to replace contaminated pairs was not pursued after the clean Perplexity result independently
ruled out broad promotion. The other four builds' performance gates remain unqualified, not passed or silently
waived. None of the five final-campaign runs showed a raw median gain of 3%. The earlier noisy, mixed D3 attempt
was followed by another complete four-round run; there is no clean mixed verdict requiring additional p95-only
rounds to decide this campaign.

## Numerical evidence

Fresh checks on the same inference source use automatic FP16, no saved tuning and normal loader memory
residency. All 51 source files recorded by the verifier match the corresponding benchmark's preflight hashes.
The original probability bounds are unchanged.

| Build | Reference cases | Longest prompt | Worst probability difference | Fixed bound | Result |
|---|---:|---:|---:|---:|---|
| D3-lite | 5 | 4,096 | 0.002596 | 0.0231 | pass |
| D3-mini NVFP4 | 5 | 4,096 | 0.000533 | 0.01 | pass |
| Perplexity Flash v1.2 | 5 | 4,088 | 0.000345 | 0.0119 | pass |
| Nimble 9B merged | 24 | 2,697 | 0.006978 | 0.1459 | pass |

[Full verification reports](../../../references/attention/validation/)
record tokens, probability comparisons, actual dtypes and custom attention graph shapes.
D3/NVFP4 and Perplexity use independent maker-reference cases. Nimble's existing maker reference was filtered
before inference to exclude its single 4,222-token case; its 24 retained cases do not establish a 4k numerical
boundary result. The derived reference preserves original outputs, bounds, provenance and the exclusion.
Clef Flash 8-bit's existing reference is tokens-only: its A/B probability agreement is not a maker-reference
accuracy pass. These limits independently prevent treating this campaign as universal build coverage.

## Decision and remaining scope

The custom attention policies remain explicit options. On this M5, MLX already fuses eligible masked
head-256 attention from 1,024 query tokens; the custom path covers calls MLX leaves unfused.
Kernel-only improvements therefore do not imply the required whole-request improvement.

No default, kernel arithmetic, tiles or tuning-version changes were made for this performance campaign.
The only production-code change is the A/B verdict fix described above. A future promotion needs a changed
policy or kernel with fresh qualifying performance and numerical evidence for every build it reaches.
Newly tuned builds, additional precisions/layouts and physical M1–M4/8–16 GiB validation remain separate work.

## Regression checks

After all timing processes exited: `ruff format .` left all 260 files unchanged; `ruff check .` and `mypy`
passed, including a separate strict type check of the benchmark helper. The full `pytest -q` run on this M5
passed **1,021 tests in 31.87 seconds**, with the one existing skip for unavailable `coremltools` and no new
skips or failures. `git diff --check` also passes. The full suite includes the real-Metal kernel regressions.

## Reproduce

With the Mac idle and exclusive access to its GPU:

```sh
IPPON_CACHE=/path/to/empty-cache HF_HUB_OFFLINE=1 PYTHONPATH=src \
  uv run python scripts/bench_attention_gate.py MODEL --out /path/to/new-results --rounds 4 --n 40
```

Use `--n 20` for this campaign's Clef/Nimble sample counts. For the Ollama build, set `OLLAMA_MODELS` to
the existing local model store and use `ollama://d3:4b-nvfp4`. The script records the exact invoked command
in `preflight.json`, writes raw `ab.json`, and refuses to qualify a changed source or a noisy environment.
