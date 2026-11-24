# d3-mini long-cell rerun

M5 Pro 48 GB, macOS 27.0.1, MLX 0.32.4. One fresh-state cell: 4,000 words, three questions, 12,305 input tokens.
A is revision `25d5632` with its original tuning snapshot; B is `29d0305` with the fresh snapshot used in the
reported regression. Both load in FP16 with zero untuned shapes. The historical source modules were extracted
unchanged from git; full commit IDs and source hashes are in `provenance.json`.

The A/B ran under the shared GPU lock, with 12 alternating fresh-process rounds, 60 timed requests per arm
(five per round), and two warm-ups per process. No tests, builds or other measurements ran alongside it.
Four rounds nevertheless recorded another process above 50% CPU. The complete timing and answer arrays remain
in `historical-rerun.json`; its activity lists retain process CPU readings. The primary diagnostic subset excludes
**both** arms of those four rounds solely on the activity flag, independently of their latencies. The eight
remaining pairs supply 40 timed requests per arm. `uncontaminated-pairs.json` lists every inclusion and exclusion.

| Data | A median ms | B median ms | Median change | 95% interval for change | p95 change | Verdict |
|---|---:|---:|---:|---|---:|---|
| All 12 rounds, including competing activity | 3,653.87 | 3,685.19 | 0.86% slower | 0.62% faster to 1.60% slower | 2.32% slower | no difference |
| Eight pairs without recorded competing CPU activity | 3,644.89 | 3,675.65 | 0.84% slower | 0.87% faster to 1.88% slower | 1.74% slower | no difference |

The subset's p95 interval also includes no change: 0.77% faster to 2.76% slower. The largest paired probability
difference is 0.001317 across all rounds, and 0.001080 in the subset. This is agreement between the
two runtimes, not a maker-reference accuracy check.

The original 3% slowdown is not reproduced. A smaller effect remains unresolved; this experiment establishes
neither a speedup nor a persistent regression under the promotion rule. No runtime or tuning policy was changed.
The original snapshots differ in long-row MLP tile choices, but without a reproduced regression this experiment
does not identify any particular tile as a cause. No new arithmetic was shipped, so no new maker/invariance gate
is claimed for this historical timing comparison.

To replay, use isolated checkouts of the two revisions and scratch cache directories containing their respective
`a-tunings.json` and `b-tunings.json`, renamed `tunings.json`. These replay files retain, unchanged, the dense
GEMM/GLU shapes this model uses; the original complete cache hashes are in `provenance.json`. Do not replace a
live tuning store. Set `A_SOURCE`, `B_SOURCE`, `A_CACHE`, `B_CACHE` and `MODEL` to those local directories, then
run the current benchmark driver inside the shared GPU lock:

```sh
ippon bench "$MODEL" --sizes 4000 --questions 3 --rounds 12 --n 60 --warmup 2 \
  --ab "PYTHONPATH=$A_SOURCE/src IPPON_CACHE=$A_CACHE" \
       "PYTHONPATH=$B_SOURCE/src IPPON_CACHE=$B_CACHE" --out /tmp/d3-long-cell.json
```

The current driver's A/B rule evaluates both median and p95 regressions. Recompute the subset with the included
`paired_subset.py`; it rejects a pair whenever either arm's cell records competing process activity.
