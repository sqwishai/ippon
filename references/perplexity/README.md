# Perplexity reference fixtures

`make_reference.py` and `make_media_reference.py` run the model creator's bundled `pplx_decider` code in its own
environment (the requirements.txt of the checkpoint). The library never imports that code. Both scripts ask the
questions of a request in the creator's default mode, `separate`, with one sequence per question after the context of
the request, as ippon does.

- `pplx-decider-flash-v1.3.json.gz` holds the creator's FP32 answers on its 15 text requests
  (`pplx-decider-flash-v1.3.requests.json`), and its bound, which is the distance of bf16 from fp32 plus 0.0005.
- `pplx-decider-flash-v1.3-media.json.gz` holds its FP32 answers on the 17 image requests of
  `../requests/media-qwen.json`, with the ids, multimodal positions and pixel hashes of every sequence. Its bound
  comes from a bf16 run of the same requests.
- `overlength.json.gz` holds the creator's FP32 answers under an explicitly configured 256-token limit
  (`overlength.requests.json`). Each question is its own sequence with the complete state. If a question does not
  fit, the whole request is refused before any inference. It keeps the bound of the text reference.
- `overlength-default.tokens.json.gz` checks the 32,768-token default without inference. Two questions of 32,664
  tokens each are read as two sequences, and a 32,835-token question is refused. `overlength_requests.py` writes its
  requests.

To regenerate the fixtures in the creator's environment, set `MODEL` to the downloaded flash v1.3 directory and
`REV` to `d06fb3f2d97e0a52325a0d463c450cf0338f44fe`. Then run these commands.

```sh
python -I references/perplexity/make_reference.py --model "$MODEL" \
  --requests references/perplexity/pplx-decider-flash-v1.3.requests.json \
  --out references/perplexity/pplx-decider-flash-v1.3.json.gz \
  --repo perplexity-ai/pplx-decider-flash-v1.3 --revision "$REV"

python -I references/perplexity/make_media_reference.py --model "$MODEL" \
  --requests references/requests/media-qwen.json --media references/media/images \
  --out references/perplexity/pplx-decider-flash-v1.3-media.json.gz \
  --repo perplexity-ai/pplx-decider-flash-v1.3 --revision "$REV" --bf16

python -I references/perplexity/make_reference.py --model "$MODEL" \
  --requests references/perplexity/overlength.requests.json --out references/perplexity/overlength.json.gz \
  --repo perplexity-ai/pplx-decider-flash-v1.3 --revision "$REV" --max-length 256 --fp32-only --bound 0.0169

python references/perplexity/overlength_requests.py > /tmp/pplx-default-limit.requests.json
python -I references/perplexity/make_reference.py --model "$MODEL" \
  --requests /tmp/pplx-default-limit.requests.json --out references/perplexity/overlength-default.tokens.json.gz \
  --repo perplexity-ai/pplx-decider-flash-v1.3 --revision "$REV" --tokens-only
```

For `ippon verify` on `overlength.json.gz`, use `--settings` with a settings file that contains
`{"max_length": 256}`. The other fixtures need no settings. Repeat the numerical checks with `--invariance`. Use a
scratch `IPPON_CACHE`, and if you share a Mac, run GPU work one job at a time. The reports in `validation/` were
measured on the M5 Pro with 48 GB, macOS 27.0.1 and MLX 0.32.4. `verify` checks an intentionally rejected request,
and `--invariance` skips it, because its comparisons apply only to accepted requests.
