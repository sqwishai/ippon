# Archived packed-prefill experiment

**The candidate failed the complete release gate and was not shipped.** On an Apple M5 Pro (48 GiB) with macOS
27.0.1 and MLX 0.32.4, all three fresh gates rejected it. All three growing gates promoted it, but those gains did
not meet the requirement to pass the complete workload. The candidate was archived without a change to the library
computation, and no clean slow cells were run again to get a pass.

| Build | Fresh gate | Worst raw median time increase | Growing gate |
|---|---|---:|---|
| d1-3B | reject | +1.177% | promote |
| d3-lite | reject | +1.350% | promote |
| Clef-Flash MLX 8-bit | reject | +5.780% | promote |

[All 60 final cells](FINAL.md) keep four accepted paired rounds, confidence intervals and raw samples. There are 156
accepted paired units and nine excluded final attempts. A fresh unit is a complete A/B cell pair, and a growing unit
contains both complete sessions. The cumulative campaign ledger keeps every physical attempt, including earlier
contaminated runs and the interrupted bulk Clef attempt.

- The [final gate](FINAL.md) has every fresh and growing cell, including the first-turn and session-wall views.
- The [threshold sweep](THRESHOLDS.md) has all 96 earlier cells and the actual-group-padding pitfall.
- The [controls](CONTROLS.md) have the eight-pair A/A control and all 14 untraced boundary cases.
- The [correctness](CORRECTNESS.md) page has the creator bounds, invariance, effective mode and direct route checks.
- The [provenance](PROVENANCE.md) page has the frozen sources, recovery-artifact hashes and publication scope.
- The [attempt ledger](attempts.json) has the excluded and interrupted records and the hashes of the retained local
  raw files.

The gzip datasets contain the complete accepted reports. The measurement numbers and booleans are unchanged, and
portable strings and anonymous activity labels replace identifying text. Nonmeasurement environment metadata is
omitted and is outside the numerical-equivalence claim. [The manifest](MANIFEST.json) records the source,
compressed-file and numerical hashes. [The saved audit](AUDIT.json) also records successful direct-source and
relocated-source comparisons.

Audit the published bundle with the standard library.

```sh
python references/prefill/audit.py
```

To compare directly with the retained local experiment, add `--source` followed by its directory. The audit uses the
recorded source prefixes, so if you move that directory, the raw evidence is not rewritten or made invalid.
