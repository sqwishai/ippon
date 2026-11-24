# Packed formats and local Ollama models

ippon runs decision models at full precision (BF16, FP16, FP32) or 8-bit by default. It loads 4-bit, 5-bit, 6-bit and
FP4 builds only when you ask for one, and every such load warns how far the model drifts from its creator's answers.
ippon's accuracy claims and speed charts are for full precision and 8-bit only.

## Run a model below 8 bits

Fewer bits move a model's probabilities further from its creator's answers. In the recorded M5 Pro check, elephant-1
at 4-bit moved up to 52 points from its FP32 answers and changed top answers, so ippon never picks a low-bit build
for you. A bare model name, `ippon pull FAMILY`, `ippon recommend` and `ippon setup` always start at a full-precision or 8-bit build. In
`ippon setup`, the left and right arrow keys take the highlighted model down to 6 or 4 bits, and setup then shows the
drift warning.

To run a model below 8 bits, ask for it in one of these ways.

1. Add `:q4` or `:q6` to a model name, as in `kettei-core:q4`. ippon quantizes the full-precision build as it loads.
2. Give `--precision q4` or `--precision q6`, or `precision="q4"` in Python.
3. Give the name of a low-bit build that `ippon pull --list` shows, as in `d3-mini:nvfp4`.
4. Give the folder, Hub id, GGUF file or `ollama://` tag of a low-bit build.

```sh
ippon run kettei-core:q4 request.json
```

Every low-bit load gives the same warning in 4 places.

- Python raises a `ippon.drift.LowBitWarning`, and ippon's log records it.
- The command line prints it once to stderr.
- `details["drift"]` holds the text, on the `Decider` that `ippon.Decider.load` returns.
- Every response carries it in a `warnings` list, in the System One, decide API and SGLang forms.

The warning names the build and its format, as in "kettei-core:q4 runs at 4-bit." Then it tells you in plain words
how far the model's probabilities move, and what to do about it. There are 3 levels, and the table below names them
close, noticeable and a lot.

- "Its answers stay close to full precision." Thresholds you set at full precision still hold.
- "Its answers move a little from full precision. Check your thresholds on a few examples." A threshold can catch
  different cases, and close calls can go the other way.
- "It sometimes gives a different answer than at full precision. Use 8-bit if this Mac has room." Thresholds set at
  full precision may not hold.

ippon measured each build against its model's reference requests on an M5 Pro. A build moved "close" when its largest
probability change was under 10 points and no top answer changed, and "a lot" when it moved over 25 points. A build
that ippon has not measured counts as noticeable at 6 bits and as a lot below, because every 4-bit build moved
16 points or more. Run `ippon verify MODEL --reference R` to measure a build yourself.

| Model | 6-bit | 4-bit |
|---|---|---|
| `d1-2b` | close (4.6 points, top answer changed on 0 of 28) | noticeable (22.7 points, top answer changed on 3 of 28) |
| `d1-omni-2b` | close (3.7 points, top answer changed on 0 of 28) | noticeable (20.5 points, top answer changed on 3 of 28) |
| `d3-flash` | close (8.2 points, top answer changed on 0 of 20) | a lot (39.0 points, top answer changed on 1 of 20) |
| `d3-mini` | close (3.3 points, top answer changed on 0 of 20) | a lot (38.9 points, top answer changed on 1 of 20) |
| `elephant` | noticeable (14.1 points, top answer changed on 0 of 27) | a lot (52.1 points, top answer changed on 2 of 27) |
| `glide-2b` | noticeable (12.9 points, top answer changed on 0 of 20) | noticeable (20.1 points, top answer changed on 0 of 20) |
| `glide-9b` | noticeable (10.8 points, top answer changed on 0 of 20) | noticeable (23.4 points, top answer changed on 4 of 20) |
| `h2o` | noticeable (9.4 points, top answer changed on 1 of 27) | a lot (26.9 points, top answer changed on 3 of 27) |
| `jeff` | noticeable (21.4 points, top answer changed on 1 of 18) | a lot (45.1 points, top answer changed on 4 of 18) |
| `jeff-qwen-4b` | close (7.7 points, top answer changed on 0 of 18) | a lot (77.9 points, top answer changed on 6 of 18) |
| `kettei-core` | noticeable (19.3 points, top answer changed on 0 of 20) | a lot (74.5 points, top answer changed on 3 of 20) |
| `kettei-dot` | noticeable (23.5 points, top answer changed on 2 of 20) | a lot (48.0 points, top answer changed on 4 of 20) |
| `kev-0.8b` | close (8.8 points, top answer changed on 0 of 20) | noticeable (19.5 points, top answer changed on 3 of 20) |
| `kev-4b` | noticeable (5.3 points, top answer changed on 1 of 20) | noticeable (21.6 points, top answer changed on 5 of 20) |
| `kev-9b` | close (3.9 points, top answer changed on 0 of 20) | a lot (29.9 points, top answer changed on 1 of 20) |
| `pplx-flash` | noticeable (6.6 points, top answer changed on 1 of 15) | noticeable (16.0 points, top answer changed on 0 of 15) |
| `torchcast` | close (7.6 points, top answer changed on 0 of 26) | a lot (43.1 points, top answer changed on 3 of 26) |

ippon refuses 2-bit and 3-bit builds, W8A8, and GGUF types other than those in the table below. The error names an
8-bit or full-precision build to use instead.

## Stored formats

ippon keeps packed MLX matrices with their weights, encoded scales and per-tensor format. Compatible projections can
join. Projections in different formats stay separate multiplies, and ippon joins their outputs. Loading never converts
one quantized format into another. Embedding lookups dequantize only the requested rows.

ippon reads GGUF Q8_0, Q4_0,
Q4_K and Q5_K as affine groups of 32, which keeps the stored values. It unpacks Q6_K to FP16, because Q6_K scales every
16 values. A Q4_K_M or Q5_K_M file holds a mix of these types.

| Stored mode | Bits | Group size | Group scales | Quantization bias | Loads |
|---|---:|---:|---|---|---|
| `affine` | 8 | 32, 64, 128 | floating point | required | by default |
| `mxfp8` | 8 | 32 | E8M0 bytes | none | by default |
| `affine` | 4, 5, 6 | 32, 64, 128 | floating point | required | when asked for |
| `nvfp4` | 4 | 16 | E4M3 bytes, and an optional global scale | none | when asked for |
| `mxfp4` | 4 | 32 | E8M0 bytes | none | when asked for |

An NVFP4 build can carry a global scale per tensor or per expert. ippon reads MLX's own convention and the multiplier
that Ollama and NVIDIA ModelOpt store. Projections with different global scales still join, with one multiplier per
row.

These formats use stock [MLX 0.32.4 quantized operators](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html).
On an M5, they also use ippon's own quantized matmul, which ippon chooses at load from the matrix shape and row count. Its
products equal MLX's within the order of fp32 sums (M5 Pro, fp16 and bf16, 3 to 333 rows). `profile` identifies their
formats and the projections that still need separate multiplies. A smaller file is not always faster.

## Load a checkpoint already pulled by Ollama

Give ippon an `ollama://` name to load a checkpoint that Ollama already pulled. ippon needs no Ollama executable, server,
download or cache write for this.

```sh
ippon pull ollama://d3:4b-mxfp8 --dry-run
ippon run ollama://d3:4b-mxfp8 request.json
ippon profile ollama://d3:4b-mxfp8 --size 200 --questions 3
```

```python
from ippon import Decider, YesNo

with Decider.load("ollama://d3:4b-mxfp8") as model:
    result = model.decide(
        "Customer was charged twice",
        {"refund": YesNo("Is a refund required?")},
    )
    print(result["refund"].p_yes)
```

The importer reads `OLLAMA_MODELS`, or `~/.ollama/models` when that variable is unset. A name can include a namespace
and registry, as in `ollama://registry.ollama.ai/library/d3:4b-mxfp8`. The tag defaults to `latest`. If a build is
missing, the error tells you to pull its MLX variant first. ippon refuses a GGUF-only or ambiguous manifest. This route
does not import arbitrary architectures, so an installed decision adapter must recognise the model and its head.

Every load verifies the SHA-256 hashes and lengths of the referenced tensor and JSON blobs before it uses them. `check`
reads tensor headers, checks blob lengths and verifies the small JSON assets, but it does not hash all weight data. It
reports format counts, the checkpoint identity, and an estimate of the memory that the weights need, with a reserve for
execution. The peak at a given context length can be higher. On the `Decider` that `ippon.Decider.load` returns,
`details["checkpoint"]` identifies the imported manifest, and `details["formats"]` lists packed modes.

Integrity checks are automatic, so there is no `verify=True` load argument. The separate `ippon verify` command checks
probabilities. Imported builds refuse precision options that change the weights. You can still select the compute
precision (`auto`, `fp16`, `bf16`, `fp32`), and result diagnostics report it.

## Native safetensors metadata

Ordinary MLX directories and Hub checkpoints use the quantization object in their `config.json`. To override a global
setting for one projection, use the full name of the projection without `.weight`, as below.

```json
{
  "quantization": {
    "mode": "mxfp8", "bits": 8, "group_size": 32,
    "model.layers.0.mlp.gate_proj": {"mode": "affine", "bits": 8, "group_size": 64},
    "model.layers.0.mlp.up_proj": false
  }
}
```

A quantized matrix has a U32 `*.weight` and `*.scales`, and affine also has `*.biases`. Floating-point encodings
require U8 scale bytes. ippon raises an error for missing scales, inconsistent bits, groups or shapes, incompatible
dtypes, and a global scale on any format but NVFP4. A dense projection stays floating point. This loader reads MLX
packed layouts and does not import every NVIDIA ModelOpt or compressed-tensors export layout.

A single stacked expert bank must have one encoding. ippon does not yet support arbitrarily different encodings among
individual experts. Clef's joint head must keep floating-point weights.

## Verification status

The Clef-Flash 9B MXFP8 build is experimental, and its original-checkpoint quality is not yet measured. It
passes against the same checkpoint dequantized in FP32, but it fails against the answers that Ollama 0.40.9 recorded in October 2026, the fixture this check uses.

The check ran on an M5 Pro (48 GB, macOS 27, MLX 0.32.4) with 18 requests and 30 questions. It used 512, 2,048 and
4,096 tokens, plus short prompts of 134 to 292 tokens (the complete schema exceeds 128). Default packed execution
differs from the same checkpoint dequantized in FP32 by at most 0.00113, within 0.01, and all top answers agree.
The independent NumPy FP64 backbone check over 16 tokens has a largest relative error of 0.0000027.

Against Ollama 0.40.9's recorded answers, however, the difference reaches 0.37945, with 3 top-answer
disagreements outside near ties. BF16 compute also fails (0.39561, from the earlier record). Fixtures and reports are
in `references/clef/9b-mxfp8.*`. With Ollama 0.40.13, the version the benchmarks pin, Ollama's own answers are 0.378
from the same weights in FP32 ([accuracy](accuracy.md)).

A re-check on 30 October 2026 on an M5 Pro, 48 GB, macOS 27.0.1 and MLX 0.32.4 gave the same results as above. The
dequantized reference passes at 0.00113, and the Ollama serving reference fails at 0.37945. On the same machine, the
MXFP8 dequantized reference of d3-mini passes at 0.005935 (bound 0.01), unchanged.

The [dequantized full-model reference](../scripts/dequantized_reference.py) shares ippon's architecture, so the
[NumPy backbone](../scripts/check_qwen35_float64.py) and the published Ollama runtime give independent checks of it.
None of them tells how the build compares with the original unquantized model.

`verify --json` also reports probability crossings at the thresholds in the `thresholds` field of a reference. The
fixtures use 0.5, 0.7 and 0.9. These crossings are diagnostics in addition to the probability bound and the top-answer
check. Clef's dequantized comparison had no crossings at those 3 thresholds. Its default Ollama comparison crosses
0.7 once and 0.9 once. These are counts of option probabilities, not counts of distinct requests.

Same-checkpoint Ollama references require the recorded checkpoint identity to match the local build, so a changed
tag cannot silently reuse the reference answers of an older build. The identity is the SHA-256 of the sorted digests of
the layers that an answer depends on. These layers are every tensor blob, and the config, tokenizer, chat template and
other assets that the adapters read. A licence, a params layer, or the registry manifest that Ollama 0.40.13 stores as
provenance is not part of the identity. A re-pull that changes only that metadata keeps the identity.

```sh
ippon verify ollama://clef-flash:9b-mxfp8 \
  --reference references/clef/9b-mxfp8.dequantized.json.gz --json
ippon verify ollama://clef-flash:9b-mxfp8 \
  --reference references/clef/9b-mxfp8.ollama.json.gz --json
```

The second command currently reports the serving failures above. The reference generators and their source pins are
beside the fixtures. They are explicit research commands, and no creator code runs when the library loads or when tests
run.

## Still open

- Resolve the strict full-serving discrepancies before you claim Ollama answer parity. Ollama rounds at different
  points in its norm and recurrent gate (the pinned [norm implementation](https://github.com/ollama/ollama/blob/v0.40.9/mlxrunner/model/qwen3_5/qwen3_5.go)
  and [recurrent-gate implementation](https://github.com/ollama/ollama/blob/v0.40.9/mlx/gated_delta.go)). ippon has
  not yet isolated how much of the gap this explains. Same-checkpoint execution checks pass for Clef MXFP8.
- Measure Clef MXFP8 quality against the original FP32 checkpoint of its creator. A small execution error does not
  remove the quantization trade-off of a build.
- Run real Metal gates on M1 to M4, including 8/16 GiB machines. A run of the M4 kernel family on an M5 for exact
  FP32 arithmetic is not evidence from physical M4 hardware.
- Time ippon's own quantized matmul on these formats against MLX's, per operator and end to end. ippon claims no speed
  for them until that measurement exists.
