# Clef-Flash concurrent requests

Kernel rule `m5-1` made Clef-Flash MXFP8 slow when several requests arrived at once. `8a69e93` caps the wide MXFP8
tile at 24 MiB of input, and `9abe2ba` unpacks MXFP8 weights at 4,096 rows before it multiplies them with the
full-precision kernels. The measurements here confirm that these 2 commits fix it, and they are the evidence for rule
`m5-4`. For FP16 MXFP8 matrices with a 2,560-wide input or output, `m5-4` uses the wide tile from 192 rows. This
recovers d3-mini's short-request loss and does not change the tile of the smaller requests. The 24 MiB cap, the
4,096-row unpack threshold, the batching policy and the weights stay the same.

All measurements are from 29 October 2026 on an Apple M5 Pro with 20 GPU cores and 48 GiB, macOS 27.0.1 and
MLX 0.32.4, in High Power and on AC, with one model run at a time. `idle.py` at first required three GPU-utilization
readings at or below 2%. Then a steady 13 to 14% desktop baseline was found. After that, interleaved fresh-process
runs used `idle.py --ab`, which requires less than 20% over at least 20 seconds and no more than three percentage
points between the readings. Ordinary non-interleaved runs keep the original near-zero guard. CPU activity is still
recorded. No tests or builds ran beside timing, and SIGTERM stops the servers. Heavy jobs go in one at a time. When
2 or more jobs waited on the GPU job lock at once, they were stopped and restarted one after another.

## Reproduction and cause

`profile_batches.py` renders the kit's one-question requests and selects seven complete 1,344-token prompts.
`interleaved_profile.py` measures seven single backbone passes against one batch in four fresh processes, in the
order singles, batch, batch, singles, and keeps every creator readout row. Each process warms its layout and records
its ordinary wall time. Then it attributes eager operations with `TimedOps` and subtracts the synchronization floor.
The operation times show where the work is, but they do not add up to the compiled wall time. No prefix cache is
used. The final probability calculation of the head is outside this backbone profile, and the native A/Bs and HTTP
tests time complete requests.

| Build and rule | Seven singles, ms | Batch of seven, ms |
|---|---:|---:|
| Clef-Flash MXFP8, m5-1, full readout diagnostic | 5,432.02 | 7,772.13 |
| Clef-Flash MXFP8, m5-3, interleaved | 5,157.65 | 4,924.83 |
| Clef-Flash BF16 checkpoint, explicit BF16, interleaved | 4,520.57 | 4,554.94 |

The old full-readout MXFP8 batch loses in the matmuls. Gate/up takes 3,387 ms against 2,189 ms as seven singles,
and down takes 1,710 against 1,240 ms. Attention takes 43 ms and the recurrence 144 ms, both below their
single-pass totals. With the current rule, batched gate/up and down take 2,022 and 1,083 ms, attention takes 43 ms
and the recurrence 143 ms. The current batch is 4.5% below seven singles. BF16 stays within 0.8% between layouts, so
the full-precision control does not reproduce the large old MXFP8 loss. m5-4 does not change Clef's matrix choices.

The initial `*-baseline.json` and `clef-mxfp8-m5-1.json` profiles kept only the last hidden row (`--last-row`).
They stay as earlier diagnostics, and the full-readout results above replace them.

The historical arm is an archive of `fef0d6a` with only `src/ippon/choices.py` replaced by the file from `d872509`,
the source revision of the reported kit result. The m5-3 baseline is `fef0d6a`. Archives have no `.git`, so their
recorded machine revision is null. These comparisons isolate the kernel rule. All FP16 choices of the measured m5-4
candidate match the final code, and the final code also leaves explicit BF16 on the prior rule.

## Fresh-process A/B

The A/B ran four alternating rounds with `--n 20 --scenario concurrent --clients 1,8 --sizes 100,1000 --questions 1`.
Native bench sizes are words, while the kit uses nominal token targets. Clef's measured prompts here have 256 and
1,156 tokens. The eight-client cells contain 32 timed requests across four rounds, because each round gives every
client at least one request. Both arms use the same requests per round.

The verdict for Clef-Flash MXFP8 is **promote**. At eight clients and 1,000 words, median latency falls from 7,256.6
to 4,878.1 ms (32.8% lower, 95% interval 32.2 to 33.3% lower), and p95 falls 32.5%. The promotion rule finds the
other three cells unchanged, and none regresses. The largest paired probability difference is 0.0009.

Against m5-1, the m5-3 rule was **mixed** on d3-mini MXFP8. Long concurrency improved 32.1%, but a single 203-token
request rose from 69.8 to 71.4 ms (2.2%). A restore of the wide tile from 65 rows was also unsuitable. Against m5-3,
it raised the 104-token median from 40.0 to 44.8 ms (12.0%), so the 65-row restoration did not ship.

The final rule starts that tile at 192 rows on the measured FP16 shapes. `d3-mxfp8-mid-current-ab.json` compares it
with **m5-3** in eight alternating rounds, with 64 requests per cell, sizes of 1, 100, 140 and 1,000 words, one
question, and one or eight clients. The verdict is **promote**.

| Tokens | Clients | m5-3 median, ms | m5-4 median, ms | Time change | Verdict |
|---|---:|---:|---:|---:|---|
| 104 | 1 | 40.0 | 40.0 | +0.1% | same |
| 104 | 8 | 261.1 | 260.7 | -0.2% | same |
| 203 | 1 | 71.5 | 70.0 | -2.1% | same |
| 203 | 8 | 497.4 | 495.4 | -0.4% | same |
| 243 | 1 | 79.3 | 76.8 | -3.2% | faster |
| 243 | 8 | 585.5 | 583.7 | -0.3% | same |
| 1,103 | 1 | 334.4 | 334.5 | +0.0% | same |
| 1,103 | 8 | 2,499.9 | 2,515.2 | +0.6% | same |

The interval of the 243-token median is 3.1 to 3.3% lower, and its p95 is 5.3% lower. No cell regresses under the
3%/1% gate, and the largest paired probability difference is 0.0002. The run records indexing-service CPU activity,
and the JSON keeps the full observations and flags. The earlier 65-row candidate records browser activity. It stays
here as a rejected diagnostic and was not the final promotion gate.

## Eight clients over HTTP

This test runs the kit's concurrent workload unchanged, with 1,000 nominal tokens, one question, eight clients, two
warm-ups and ten timed requests per client. The actual prompt length is 1,344 tokens. Both runs completed all 80
requests without errors and recorded High Power and AC at both ends.

| Run | Requests/s | Median latency, ms |
|---|---:|---:|
| Reported historical ippon result | 0.943 | 8,492.03 |
| Current, first run | 1.386 | 5,757.81 |
| Current, repeat | 1.385 | 5,760.89 |

These first runs exceeded the target but recorded desktop and compiler activity, so they stay as preliminary
observations. The final HTTP comparison uses the same kit workload in four fresh servers in the order A, B, B, A.
A is m5-1 and B is the current rule, with two warm-ups and ten timed requests per client in each process. The steady
GPU baseline was 16 to 17% for at least 20 seconds, within the updated criterion for interleaved A/Bs. All four runs
recorded High Power and AC at both ends. No errors or skipped sizes occurred.

| Fresh server | Rule | Requests/s | Median latency, ms |
|---|---|---:|---:|
| A1 | m5-1 | 0.907 | 8,827.15 |
| B1 | m5-4 | 1.305 | 6,125.42 |
| B2 | m5-4 | 1.292 | 6,209.46 |
| A2 | m5-1 | 0.915 | 8,746.22 |

The two current servers together served 160 requests at 1.298 requests/s, with a median of 6,140.32 ms. The two
historical servers together gave 0.911 requests/s, with a median of 8,767.20 ms. Both B runs exceed 1.24
requests/s, and the pooled median latency is 30.0% lower. The raw reports keep the CPU activity flags, including
brief desktop activity. This is an interleaved comparison under the accepted steady baseline, and the Mac was not
free of other activity. `interleaved_http.py` imports the kit unchanged and uses SIGTERM to stop only its own
servers.

## Accuracy

Rule m5-4 passes the existing same-checkpoint FP32 dequantization reference at 0.00593518 against 0.01. That
reference came from stock MLX dequantization and the shared architecture, so it does **not** establish the
creator's original FP32 quality. The original reference file, outside this repository, had SHA256
`9099c5276eeb31317a318d442a66d01c89b1d01a9c54b9b85c52df613712cc02`. The migrated copy is
`d3-mini-mxfp8-dequantized.json.gz` beside this report. Only its legacy schema tag changed to `ippon-reference/1`,
and the requests and probabilities stayed the same.

`references/d3/d3-mini.json.gz` is the creator's original CPU FP32 reference. Against it, m5-3 and m5-4 give
**identical request reports**, with a maximum of 0.06239356, past 0.0291 on `40-questions`, `choice-10` and
`choice-11`, and with matching prompts, counts and top answers. The quantized build stays experimental, and no bound
was widened. The catalog now records this measured gap and no longer calls it unmeasured.

## Short complete requests

This control uses seven distinct complete 100-token prompts, automatic FP16 and seven repetitions.

| Model | Seven singles, ms | Batch of seven, ms | Batch time reduction |
|---|---:|---:|---:|
| d1-2B | 111.84 | 71.33 | 36.2% |
| d3-mini checkpoint | 292.24 | 187.55 | 35.8% |

These controls show that batching still wins at the requested short shape. The new rule keeps this batching policy
and the tile of the smaller requests.

## Commands

Run these commands from the repository root, with the existing environment and local models. For the historical arm,
make an isolated source copy, which never changes the working tree.

```sh
mkdir -p /tmp/ippon-clef-rule-m5-1
git archive fef0d6a src | tar -x -C /tmp/ippon-clef-rule-m5-1
git show d872509:src/ippon/choices.py > /tmp/ippon-clef-rule-m5-1/src/ippon/choices.py
export OLLAMA_MODELS="$HOME/models/ollama"
.venv/bin/python references/batching/idle.py \
  .venv/bin/python references/batching/profile_batches.py ollama://clef-flash:9b-mxfp8 \
  --out /tmp/clef-profile.json
.venv/bin/python references/batching/idle.py \
  .venv/bin/ippon bench ollama://clef-flash:9b-mxfp8 \
  --ab 'PYTHONPATH=/tmp/ippon-clef-rule-m5-1/src' "PYTHONPATH=$PWD/src" \
  --scenario concurrent --clients 1,8 --sizes 100,1000 --questions 1 --n 20 --rounds 4 \
  --out /tmp/clef-ab.json
.venv/bin/python references/batching/idle.py \
  .venv/bin/python references/batching/serve_workload.py --ippon "$PWD/.venv/bin/ippon" \
  --model ollama://clef-flash:9b-mxfp8 --engine clef-mxfp8 \
  --sizes 1000 --questions 1 --clients 8 --n 10 --warmup 2 --max-input 16384 \
  --out /tmp/clef.c8.json
```

For the old-rule profile, put `env PYTHONPATH=/tmp/ippon-clef-rule-m5-1/src` before the profile command. For BF16,
use the local `~/models/clef-flash` and `--precision bf16`. The short controls use the local `d1-2B` and
`d3-mini` with `--tokens 100 --repeat 7 --wall-only`. Their complete prompts include the question and its original
readout position, without truncation.

For the final profiles, run `interleaved_profile.py` in place of `profile_batches.py` behind `idle.py --ab`. For the
final server comparison, run `interleaved_http.py` with
`--a /tmp/ippon-clef-rule-m5-1/src --b "$PWD/src" --out /tmp/clef-http.json`. The d3 promotion gate uses an
unmodified archive of `fef0d6a` as A and the working source as B, with
`--sizes 1,100,140,1000 --clients 1,8 --questions 1 --scenario concurrent --n 64 --rounds 8`, through the same lock
and `idle.py --ab`.
