# Correctness evidence for the archived candidate

Correctness passed, but the complete performance gate failed. These checks apply to the experimental source digest
`b710c5b3774bb234c0d3f7cf037871403dd92936a05f90b6069c4a2eac9f48fa`, which is not a shipped feature. The
[unchanged summary](data/correctness-summary.json.gz) records 1,090 passing tests, one existing optional skip, 67
focused policy tests, strict type and static checks, and five killed policy mutations.

## Creator and separate invariance checks

| Build | Check | Worst difference | Bound | Evidence |
|---|---|---:|---:|---|
| d1-3B | creator | 0.00168386897 | 0.0389 | [report](data/correctness/d1-3B-maker.json.gz) |
| d1-3B | invariance | 0.00160628557 | 0.0389 | [report](data/correctness/d1-3B-invariance.json.gz) |
| d3-lite | creator | 0.00174493731 | 0.0231 | [report](data/correctness/d3-lite-maker.json.gz) |
| d3-lite | invariance | 0.00344013826 | 0.0231 | [report](data/correctness/d3-lite-invariance.json.gz) |
| Clef-Flash MLX 8-bit | creator | 0.0033 | 0.0111 | [report](data/correctness/clef-maker.json.gz) |
| Clef-Flash MLX 8-bit | invariance | 0.00197832286 | 0.01 | [report](data/correctness/clef-invariance.json.gz) |

All six checks passed, with no skipped checks and no relaxed bounds. Clef's creator comparison uses its own-build
FP32 loader. The archived [qualification record](data/correctness/qualification.json.gz) shows the actual loaded
configuration, the weight-format signatures, the native m5 operators and the effective `auto` on all three builds.
All accepted final reports and cells directly record A `off` and B `auto` before and after execution.

## Direct capture, reuse and boundary routes

[d1](data/correctness/d1-capture-grown.json.gz) and [Clef](data/correctness/clef-capture-grown.json.gz) kept three
independently owned fresh prefixes. Then they recorded exactly three grown-prefix hits and the expected reused-token
deltas. [Noncausal d3](data/correctness/d3-capture-grown.json.gz) kept no prefix.

Four d1 cases, six d3 cases and four Clef cases were replayed twice without timings, which gives **28
observations**. Actual padding near 18% used ordinary padded execution, and padding near 22% started packing. The
replays covered B3/B7, including d3 B3 three-question batches with nine sequences. They observed the actual backbone
calls as well as the trace events, and they asserted effective `auto` before and after. The token counts matched
capture, and the probabilities stayed within the existing comparison bound. The complete observations stay in the
files for [d1](data/correctness/d1-withheld.json.gz), [d3](data/correctness/d3-withheld.json.gz) and
[Clef](data/correctness/clef-withheld.json.gz).

Standard invariance groups at most three requests. Focused regression tests, and not that standard gate, establish
the oversized-call fallback. This corrects the wording in the historical local correctness note. The five killed
mutations exercised the boundary, the live-capacity bound, stale reuse, operator restoration and the rejection of an
unmeasured quantized layout. The correctness success did not override the failed final performance gate.
