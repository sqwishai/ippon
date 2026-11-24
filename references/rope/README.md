# Rotary accuracy through position 16,384

MLX's rotation stays well within the creator bounds on both checked long references, so the runtime is unchanged.
These correctness experiments ran on an M5 Pro with 48 GB, macOS 27.0.1 and MLX 0.32.4.

`measure.py` compares `mx.fast.rope` with a NumPy FP64 rotation of the exact same dtype-rounded inputs. The inputs are
seed-17 standard-normal vectors at every integer position from 0 through 16,384 inclusive. The JSON includes FP32,
FP16 and BF16 results, maxima through 1,024/4,096/8,192/16,384, RMS errors, and equality checks for explicit
positions and for a chunk that starts at 8,192. All 36 explicit-position and 36 chunk-offset comparisons are
identical. The M-RoPE cases assign distinct t/h/w coordinates through Qwen3.5's interleaved frequency map, in all
three dtypes.

The LFM2, Qwen3.5 and ModernBERT configurations match the locally checked model families. The Dense and Gemma cases
cover the other supported rotary mechanisms with representative widths, bases and scaling parameters, but they do
not cover all possible third-party configs. YaRN starts from its configured frequencies, which intentionally use the
creator's FP32 frequency construction. The separate axial tables of vision towers are outside this text-state audit.
These experiments do not extend the context length of a model, and a rotation measured past the allowed context of a
model does not permit inference there.

`verify_long.py` checks the existing independent creator references at roughly 12k tokens per question, with the
same `verify_model` implementation as `ippon verify`. It selects the `long-12k-mixed` case, with three questions, and
records the SHA256 of the source reference. Run each mode in a fresh process, with no other GPU work running.

- `shipped` uses the normal switches (`norm_rope=mlx`, `compile=layers` by default).
- `mlx` uses `norm_rope=mlx,compile=off` to isolate MLX's rotation.
- `tables` uses the same switches, and inside the experiment it replaces only `RoPE.__call__` with FP64 angles, FP32
  cosine and sine tables, and FP32 rotation arithmetic before the final output cast.

No production code imports these experimental tables. The normal loader memory limits stay on for long prompts.

| Model / longest prompt | Shipped and MLX worst probability error | Tables worst error | Largest MLX-to-table change | Creator bound |
|---|---:|---:|---:|---:|
| d3-lite / 12,263 | 0.001941 | 0.000892 | 0.001223 | 0.0231 |
| pplx-decider flash v1.2 / 12,255 | 0.000191 | 0.000077 | 0.000237 | 0.0119 |

All six verification runs pass, with top-answer agreement under the existing near-tie rule. Shipped-mode invariance
passes for both, with zero probability difference on the applicable checks. Their non-causal attention always reads
a whole prompt, so a change to the chunk setting does not shorten its attention scope.

FP64-angle tables reduce the error on these two cases, but neither original result comes near its existing bound.
This evidence does not require a runtime replacement. Component rotation errors cannot be compared directly with
probability bounds. These experiments establish no end-to-end long-state conclusion for LFM2, ModernBERT, dense
decoders or Gemma. That claim needs additional creator references, especially for third-party scaling
configurations. No latency benefit was measured, and none is claimed.

To reproduce, use a scratch `IPPON_CACHE` and run GPU work one job at a time. In a shared benchmark environment, use
its GPU lock. The library does not require a lock helper.

```sh
python references/rope/measure.py --out /tmp/rope-components.json
python references/rope/verify_long.py "$MODEL" --reference "$REFERENCE" \
  --mode shipped --invariance --out /tmp/rope-shipped.json
python references/rope/verify_long.py "$MODEL" --reference "$REFERENCE" \
  --mode mlx --out /tmp/rope-mlx.json
python references/rope/verify_long.py "$MODEL" --reference "$REFERENCE" \
  --mode tables --out /tmp/rope-tables.json
```

The references are `references/attention/d3-lite-long-expanded.json.gz` and
`references/attention/pplx-flash-long-expanded.json.gz`, each for its downloaded model.
