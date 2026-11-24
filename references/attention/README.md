# Attention accuracy checks

These references are numerical checks of the attention paths on the M5 Pro. They do not measure latency and give no
evidence about other Macs. `attention=mlx` is the default. `bounded` is a memory mode beside it, which reads MLX's
attention in query blocks. ippon's own fused attention kernels, once checked here as `large` and `kernel`, were
removed on 2 November 2026 because they were never faster than MLX's attention where measured.

The primary acceptance fixture caps each fully rendered prompt at 4,096 tokens, including the question template.
It includes yes/no, 11-option choice and three-level score questions. The earlier extended checks at roughly 4k, 8k
and 12k tokens stay here as additional evidence, but they are not the current acceptance limit. The JSON reports
record the actual creator token counts. The same `verify_model` implementation as `ippon verify` checks token
identity, probability bounds and top-answer agreement.

See [the complete result tables](../../docs/accuracy.md#attention-at-head-size-256-switch-attention). The checks of
d3-mini's NVFP4 build were removed with that format, because ippon loads decision models at full precision or 8-bit
only.

## Independent references

Run each creator in its own Torch environment, outside ippon. The model directories below are downloaded creator
checkpoints, and `OLLAMA_STORE` is a local Ollama model store. Normal library loading changes no model weights.

```sh
# D3's own runtime, FP32 on CPU. Retain the existing d3-lite bound.
python -I references/d3/make_reference.py --model /path/to/d3-lite \
  --requests references/attention/4k-requests.json --fp32-only --bound 0.0231 \
  --out references/attention/d3-lite-4k.json.gz

# Perplexity's own runtime, with its existing bound.
python -I references/perplexity/make_reference.py --model /path/to/pplx-decider-flash-v1.2 \
  --repo perplexity-ai/pplx-decider-flash-v1.2 --revision 9ff7cf9838e59df461c276b7699d56b244d2af58 \
  --requests references/attention/4k-requests.json --fp32-only --bound 0.0119 \
  --out references/attention/pplx-flash-4k.json.gz
```

`requests.py` generates the extended request JSON with the creator's tokenizer. `4k-requests.json` uses the same
state construction with 3,833 archive tokens. Its rendered D3 lengths are 4,000/4,096/4,005, and its Perplexity
lengths are 3,992/4,088/3,997. `expand_reference.py` keeps each mixed request and adds its three single-question
cases and an equal-length pair. This gives 5 cases for the 4k acceptance set and 15 for the three extended requests.
The expansion reuses the creator's independently evaluated per-question outputs without changing them. It applies
only to D3 and Perplexity references that explicitly evaluate one question per sequence, and each derived file
records its source hash. Mixed-length batches alone do not give enough coverage. Their padding mask can select MLX's
fused kernel, while singletons and equal-length batches reach the unmasked path that `bounded` splits.

```sh
python references/attention/expand_reference.py references/attention/pplx-flash-4k.json.gz \
  --out references/attention/pplx-flash-4k-expanded.json.gz
```

## Verification and allocation evidence

Use a fresh process for each switch and an empty scratch tuning cache. Run GPU work one job at a time with any other
agents. The reports use no stored tunings.

The helper records the actual custom/blocked graph shapes, the source and reference hashes, the machine, the MLX
version and the peak active MLX allocations, including loading and warm-up. Compiled replay does not call Python
again, so graph counts are not counts of GPU launches. The allocation figure is not process RSS or total system
memory.

Long-run reports use `pin_memory=true`, the normal `Decider.load` policy, and include the active and cached
allocations after each request. The underlying `verify_model` defaults to `pin_memory=false`, which also skips the
usual free-buffer cache cap. An initial post-fix long run with that policy was stopped after its Metal footprint grew
to about 49 GiB while CPU reference generation also ran. That footprint is not an active-memory reading, and it gives
no evidence that the FP32 product itself needs 49 GiB. The completed long reports use the ordinary loader policy. The
uncapped short reports are the ones without `load_options` and `request_memory`.

```sh
IPPON_CACHE=/tmp/ippon-empty-cache PYTHONPATH=src python scripts/verify_attention.py \
  perplexity-ai/pplx-decider-flash-v1.2 --reference references/attention/pplx-flash-4k-expanded.json.gz \
  --switch attention=bounded --out /tmp/pplx-bounded.json
```

Repeat with `attention=mlx`, and with the reference of each model. The output path must be new, so earlier evidence
is never overwritten. Reports under `validation/` keep failures as well as passes.

`bounded` splits only eligible unmasked head-256 calls whose full score matrix would occupy at least 512 MiB.
Each block has at most 128 query rows and still reads every key and value. It uses MLX's full-key softmax and
evaluates each block before it allocates the next. When you select this explicit memory mode, it disables compiled
layers, also on requests whose attention does not need splitting. Packed-weight models already run layers eagerly.
The mode is a memory and accuracy option with additional synchronization and Python work, and it has no demonstrated
speedup. Under compiled execution, dependencies between lazy blocks did not give a reliable allocation bound.

The [25 October performance campaign](../../benchmarks/results/m5pro-2026-10-25-attention-4k/SUMMARY.md), which
also capped prompts at 4,096 tokens, timed the removed fused kernels. With them, Perplexity Flash's clean run gained
at most 0.89%, below the 3% promotion requirement. An M1 later measured no difference at 1,000 or 4,000 words.
