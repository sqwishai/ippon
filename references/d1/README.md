# Liquid video evidence

Both Liquid checkpoints pass the new video references at automatic, FP16 and FP32 precision, and explicit BF16
fails only on d1-Omni-2B. All measurements are from 29 October 2026 on an Apple M5 Pro with 48 GiB, macOS 27.0.1 and
MLX 0.32.4. Model runs and full test runs went one at a time, and no benchmarks ran. The creator runs execute each
checkpoint's own code in the d1 environment, first CPU float32 and then bfloat16 on MPS. Neither ippon nor its
architecture supplies reference probabilities.

| Checkpoint | Pinned revision | Creator BF16 distance from CPU FP32 | Bound |
|---|---|---:|---:|
| LiquidAI/d1-2B | cfe5760ff5d5998f91d9ac56bd4b2dbd96996d41 | 0.0077631830 | 0.0100 |
| LiquidAI/d1-Omni-2B | c8d48c08dada0191c6b362820f055cfb49c99426 | 0.0139815212 | 0.0145 |

The bound is `max(0.01, round(maker_bf16_distance + 0.0005, 4))`. These are new video references, and the existing
references and their bounds did not change. Each has 12 requests, and none was refused. All prompt tokens, set
groups, readouts and token counts match. The cases cover three question types, several videos, repeated videos,
images before videos, odd frame counts, tiny and portrait frames, a variable-rate clip with an audio track, and 17
questions in two rows. The eight processor fixtures also cover a 20 s, 1280×720 clip that reaches the whole-video
pixel budget. The creator's sampler is recorded separately at its 768-frame cap. These last two checks run no
language model.

| Model | auto / FP16 | FP32, M4 kernels | BF16 | Bound |
|---|---:|---:|---:|---:|
| d1-2B | 0.0014555254 | 0.0000041523 | 0.0059233887 | 0.0100 |
| d1-Omni-2B | 0.0021049023 | 0.0000025025 | **0.0156162264, fails** | 0.0145 |

Automatic, FP16 and FP32 runs pass both models. Explicit BF16 passes d1-2B but exceeds Omni's bound on
`two-images-two-videos` (0.0146634936) and `video-two-question-rows` (0.0156162264). No top answer changes in any
video verification. The failed BF16 reports stay in `validation/`, and no bound was widened. Quantized video builds
were not verified. The existing automatic text/image references pass again, with d1-2B at 0.0011681322 /
0.0015767245, Omni at 0.0017691033 / 0.0047727525, and Omni's audio reference at 0.0009853263.

## Decoder and processor

Liquid's `media.Reader` passes video paths to the processor of its checkpoint, which uses Transformers' TorchCodec
decoder with `seek_mode="exact"`, `num_ffmpeg_threads=0` and `device="cpu"`. The frame rate is the decoded frame
count divided by the presentation interval. That interval runs from the PTS of the first frame to the PTS of the last
frame plus its duration. Frames are sampled uniformly by rounded `linspace` at 2 fps, with 4 to 768 samples, capped
by the available frames. Resize uses the unpadded sample count and a pixel budget of 4,096 to 25,165,824 over the
clip, with the per-frame cap disabled. Bicubic antialiasing, RGB normalization and 16×16 patches reuse ippon's Qwen
preprocessing.

An odd final frame is repeated only after resize. Each pair becomes one temporal patch, which the existing LFM3
vision tower reads with independent spatial attention and convolutions. Each merged 2×2 spatial block supplies one
text-model row.

The outer vision span of every video contains a timestamp and an inner vision span for each frame pair. A timestamp
is the mean of the source-frame times of the pair, formatted to one decimal. The model's images come before its
videos, followed by standalone audio and state text. The video list keeps its order. Omni's processor can ingest
soundtracks as an option, but the decision API leaves this disabled, so neither model reads the audio track of a
video. Omni still accepts standalone audio clips and spoken questions.

The existing OpenCV decoder is not equivalent. Even with Liquid's sampling ceiling, three synthetic clips differ by
up to 3 RGB levels. On the Commons variable-rate clip, OpenCV reports 262 frames at 29.5833333333 fps, where
TorchCodec reports 253 at 28.5682023487 fps. This changes the sampled indices and the prompt timestamp text.

PyAV 19.0.2 and 18.1.0 both reproduce all **98 sampled frames and 120,342,816 RGB channels exactly**, with identical
indices and clocks, a maximum pixel difference of 0 and a maximum timestamp difference of 0 seconds.
`video-decoder-comparison.json` records the per-clip OpenCV/PyAV comparison through the actual ippon decoder paths.
`video-processor.npz` holds the creator's RGB frames and flattened patches, and the tests compare every pixel and
every patch without a tower. The processor source is byte-identical in both checkpoints. `video-processor.json`
records the source hashes and creator versions.

PyAV (`av>=18.1`) installs with ippon, and ippon imports it when a model first reads a video. ippon adds no
TorchCodec or PyTorch dependency. The existing OpenCV decoder stays the path for d3. For Python 3.11 the selected PyAV
version is 18.1.0, and for Python 3.12 and later it is 19.0.2. All media already exist in `references/media/`. They
are synthetic CC0 files made by `make_videos.py` and the CC0 Commons clip, recorded in
`references/media/LICENSES.md`. No new download or licence entry is necessary.

## Reproduction

In the creator's d1 environment, install TorchCodec for its processor. Then regenerate the requests, which runs no
model.

```sh
python references/requests/make_media_d1_video.py > references/requests/media-d1-video.json
```

Run each creator command on an otherwise idle Mac, with the Python of the d1 environment in place of `D1_PYTHON`.

```sh
D1_PYTHON references/d1/make_video_fixtures.py --model ~/models/d1-2B
D1_PYTHON references/d1/make_reference.py --model ~/models/d1-2B --repo LiquidAI/d1-2B --revision cfe5760ff5d5998f91d9ac56bd4b2dbd96996d41 --requests references/requests/media-d1-video.json --media references/media --out references/d1/d1-2B-video.json.gz --threads 8
D1_PYTHON references/d1/make_reference.py --model ~/models/d1-Omni-2B --repo LiquidAI/d1-Omni-2B --revision c8d48c08dada0191c6b362820f055cfb49c99426 --requests references/requests/media-d1-video.json --media references/media --out references/d1/d1-omni-2B-video.json.gz --threads 8
uv run python references/d1/compare_video_decoders.py
uv run ippon verify ~/models/d1-2B --reference references/d1/d1-2B-video.json.gz --precision auto --json
env IPPON_PROFILE=m4 uv run ippon verify ~/models/d1-2B --reference references/d1/d1-2B-video.json.gz --precision fp32 --json
```

Repeat the verification with the Omni checkpoint and reference, and with `--precision fp16` and `bf16`. The
exact-kernel FP32 run sets `IPPON_PROFILE=m4` before it imports MLX, as `docs/accuracy.md` requires. `validation/`
holds the full per-request reports and the machine and source hashes of the validation.

The checks were Ruff formatting and lint, strict mypy (200 source files) and full pytest (983 passed, no skips or
failures).

## Instructions that are other JSON

Liquid's prompt.py of 29 October (d1-2B 10e7671d44e4b37700020b1eed3f594c5f2af0b4, d1-Omni-2B
431f1d667de5afcc9b0c58b2d9c170a7b38b358b, the same file in both) takes instructions and option descriptions that are
JSON other than text and writes them with Python's `str()`. A falsy description shows its label. Its only other
change since the pinned revisions is to lfm3.py (memory and a transformers 5.22 keyword). The weights, tokenizer and
calibration are the same files, so the answers above still hold. `d1-2B-json.json.gz` holds the creator's rows for
`requests-json.json`, tokens only, made on 31 October 2026 from the code of 10e7671 (no weights read).

```sh
uv run --no-project --python 3.12 --with torch==2.14.1 --with transformers==5.19.0 --with torchvision==0.29.1 \
    --with pillow python -I references/d1/make_reference.py --model D1_2B_AT_10E7671 --repo LiquidAI/d1-2B \
    --revision 10e7671d44e4b37700020b1eed3f594c5f2af0b4 --requests references/d1/requests-json.json \
    --out references/d1/d1-2B-json.json.gz --tokens-only
```
