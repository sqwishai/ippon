# Controls and withheld boundary timings

These are supporting diagnostics. They do not replace the failed complete fresh gate. Percentages are time changes; negative means B is faster.

## Identical-arm control

The cumulative d1 A/A control retained eight clean pairs. It still recorded +1.664% at B4/q1/1,000 words and +1.332% at B1/q3/1,000 words; the library verdict was reject. These clean shifts were retained. They were not used to relax the final 1% bound.

| clients | q | words | A ms | B ms | time change | p95 change | library |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1 | 100 | 39.658 | 39.706 | +0.122% | -0.246% | same |
| 4 | 1 | 100 | 119.215 | 119.151 | -0.053% | +0.043% | same |
| 8 | 1 | 100 | 206.845 | 207.036 | +0.092% | -0.392% | same |
| 1 | 3 | 100 | 46.084 | 46.126 | +0.092% | +0.413% | same |
| 4 | 3 | 100 | 183.468 | 183.677 | +0.114% | -0.115% | same |
| 8 | 3 | 100 | 368.058 | 368.502 | +0.121% | +0.346% | same |
| 1 | 1 | 1,000 | 187.557 | 188.595 | +0.553% | +0.761% | same |
| 4 | 1 | 1,000 | 726.696 | 738.788 | +1.664% | -0.208% | slower |
| 8 | 1 | 1,000 | 1477.844 | 1485.000 | +0.484% | -0.112% | same |
| 1 | 3 | 1,000 | 203.266 | 205.973 | +1.332% | +0.218% | same |
| 4 | 3 | 1,000 | 813.736 | 810.765 | -0.365% | +0.429% | same |
| 8 | 3 | 1,000 | 1630.161 | 1603.293 | -1.648% | +0.414% | same |

[Complete cumulative A/A samples](data/runs/control-d1-fresh-aa-more.json.gz). The separate idle sampler control recorded 29 valid intervals and 3.97% of one CPU core in the observer; instrumentation cost is not assumed to be zero.

## Untraced withheld cases

All 14 withheld cases ran with tracing disabled: capture (A) versus auto (B), B3/B7 near 18%/22% actual padding, plus B3/q3 for noncausal d3. The library aggregate verdict was promote for each build, but the below-boundary d3 and Clef B7 cells still had raw median increases above 1%. These diagnostic gains did not qualify the final gate.

### d1-3B

[All samples](data/runs/auto-d1-withheld-timed.json.gz).

| case | actual padding | A ms | B ms | time change | p95 change | library |
|---|---:|---:|---:|---:|---:|---|
| withheld-b3-q1-w1000-actual18 | 18.00% to 18.00% | 523.692 | 523.815 | +0.023% | -0.222% | same |
| withheld-b3-q1-w1000-actual22 | 21.99% to 21.99% | 523.535 | 410.199 | -21.648% | -21.669% | faster |
| withheld-b7-q1-w1000-actual18 | 18.02% to 18.02% | 1262.806 | 1255.729 | -0.560% | -1.153% | same |
| withheld-b7-q1-w1000-actual22 | 21.99% to 22.00% | 1275.113 | 979.701 | -23.168% | -24.346% | faster |

### d3-lite

[All samples](data/runs/auto-d3-withheld-timed.json.gz).

| case | actual padding | A ms | B ms | time change | p95 change | library |
|---|---:|---:|---:|---:|---:|---|
| withheld-b3-q1-w1000-actual18 | 17.97% to 18.06% | 175.356 | 175.551 | +0.111% | +0.372% | same |
| withheld-b3-q1-w1000-actual22 | 21.99% to 22.06% | 175.779 | 142.148 | -19.133% | -19.529% | faster |
| withheld-b7-q1-w1000-actual18 | 17.91% to 18.05% | 392.468 | 398.100 | +1.435% | +0.522% | same |
| withheld-b7-q1-w1000-actual22 | 21.87% to 22.02% | 393.918 | 328.814 | -16.527% | -18.043% | faster |
| withheld-b3-q3-w1000-actual18 | 18.10% to 18.21% | 514.352 | 509.738 | -0.897% | -1.723% | same |
| withheld-b3-q3-w1000-actual22 | 22.02% to 22.12% | 507.336 | 427.086 | -15.818% | -15.626% | faster |

### Clef-Flash MLX 8-bit

[All samples](data/runs/auto-clef-withheld-timed.json.gz).

| case | actual padding | A ms | B ms | time change | p95 change | library |
|---|---:|---:|---:|---:|---:|---|
| withheld-b3-q1-w1000-actual18 | 17.97% to 18.09% | 2024.778 | 2036.402 | +0.574% | +0.976% | same |
| withheld-b3-q1-w1000-actual22 | 21.82% to 22.03% | 2016.899 | 1582.180 | -21.554% | -20.780% | faster |
| withheld-b7-q1-w1000-actual18 | 17.96% to 18.03% | 4708.754 | 4772.584 | +1.356% | +1.460% | same |
| withheld-b7-q1-w1000-actual22 | 21.96% to 22.05% | 4758.102 | 3683.343 | -22.588% | -23.827% | faster |

