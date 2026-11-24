# Perplexity's Q8_0 preflight

On the creator's Q8_0 GGUF of flash v1.2, ippon answered all 15 reference requests and llama.cpp b11960 answered
none. ippon no longer loads flash v1.2 or its GGUF, because flash v1.3 replaced them. This record stays as measured,
and the v1.2 reference file it names is in the repository's history.

The checks ran on 30 October 2026 on an Apple M5 Pro with 48 GB and macOS 27.0.1, with ippon `0c11bc8` and MLX
0.32.4, and with llama.cpp `0.8.0-dev`, build 11960, commit `8642541b2`, the official macOS binary. They are
accuracy and endpoint checks and do not measure time. Each reference check ran once.

Both runtimes use the creator's unchanged `decider-flash-v1.2-q8_0.gguf` from
`perplexity-ai/pplx-decider-flash-gguf-v1.2` at revision `2514d5f0b7fbbe1ee1bfec9ffb5801a60289b4cc`.

| File | SHA-256 |
|---|---|
| `decider-flash-v1.2-q8_0.gguf` | `ffb27ade8976765b0e6943a0b7f5f3fd0f7042d45ad90256a76d608bcd3942c2` |
| `mmproj-decider-flash-v1.2-f16.gguf` | `b2459e42a5f12c9ecbc81742a91cb93b6bbe5082297b4aa9578d453ddcbecfab` |

The text reference was `references/perplexity/pplx-decider-flash-v1.2.json.gz` at ippon `0c11bc8`, with SHA-256
`1cefed2a2bae42f197bf080a6d1d3b7ca2e65fa4bcdfaefa36967fc36a4ab8c1`, 15 requests and a bound of 0.0119. The tree holds
only the v1.3 reference.

| Runtime | Answered | Worst difference | Past bound | Changed top answers |
|---|---:|---:|---:|---:|
| ippon | 15/15 | 0.0199954771 | 2 | 0 |
| llama.cpp b11960 | 0/15 | unavailable | unavailable | unavailable |

`pplx-flash-q8_0.ippon.json` is the verify output. `pplx-flash-q8_0.llama.json` keeps every refusal, which is
HTTP 501 with `This model is not a decision model`. Zero answered cases is not a passing accuracy check.
`pplx-flash-q8_0.preflight.json` records the initial endpoint probes with the matching F16 mmproj loaded. Text and
image requests to `/v1/systemone` return 501, and `/v1/decisions` returns 404. ippon's GGUF loader reads text only.

The [creator's pinned card](https://huggingface.co/perplexity-ai/pplx-decider-flash-gguf-v1.2/blob/2514d5f0b7fbbe1ee1bfec9ffb5801a60289b4cc/README.md#gguf-files)
says that stock llama-server implements protocol v1, while these files require v3. Protocol v3 needs one joint
prompt, readouts at each `<|fim_suffix|>`, non-causal full attention, causal linear attention, and the code tokens
and temperature of the file. In b11960,
[`common_get_decision_type`](https://github.com/ggml-org/llama.cpp/blob/8642541b2/common/common.cpp#L1205) requires
`<architecture>.decision.type`. The
[decision setup](https://github.com/ggml-org/llama.cpp/blob/8642541b2/tools/server/server-decision.cpp#L39) then
requires a `systemone` template. The creator's file has `pplx_decider.*` metadata instead. No metadata override,
converted weights or server patch was used. A timed pair would not compare 2 working decision runtimes, so no
six-workload smoke or campaign ran.

The checks ran with these commands, from the repository root at ippon `0c11bc8`, with local paths in place of the
examples. They need that commit, which holds the v1.2 reference and loads the v1.2 file.

```sh
ippon verify /path/to/decider-flash-v1.2-q8_0.gguf \
  --reference references/perplexity/pplx-decider-flash-v1.2.json.gz --json
python3 -I references/llama/answers.py --server /path/to/llama-server \
  --gguf /path/to/decider-flash-v1.2-q8_0.gguf \
  --reference references/perplexity/pplx-decider-flash-v1.2.json.gz \
  --out /path/to/answers.json
```

They ran under the GPU job locks of the machine. The answer script uses the same probability conversion and near-tie
rule as `references/ollama/answers.py`. It records the binary version and file hashes, and it uses SIGTERM to stop
only its own server. For the preflight, launch with the same flags as in the recorded answer file, plus
`--mmproj /path/to/mmproj-decider-flash-v1.2-f16.gguf`. Then POST the requests in the preflight record.
