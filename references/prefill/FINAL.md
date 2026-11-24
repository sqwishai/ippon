# Complete final gate

**Rejected and archived; the candidate was not shipped.** All 60 cells are shown. A is the unchanged baseline; B is the experimental automatic policy. Recorded effective modes are A `off` and B `auto` before/after every accepted report and cell. Tracing was disabled.

Apple M5 Pro, 48 GiB; macOS 27.0.1; MLX 0.32.4. Each cell has four accepted paired rounds. Percentages are time changes: `100 x (B/A - 1)`; negative means B is faster. Median and p95 changes use raw samples. Confidence intervals and cell verdicts retain the original library calculations.

The 36 fresh cells cover 1/4/8 clients x 1/3 questions x 100/1,000 words. The 24 growing rows are four views of each 4/8-client session: all requests, first turn, steady turns and session wall time. These views reuse the same observations and are not independent workloads.

## d1-3B: fresh

Library gate: **reject**; conservative raw-median gate: **unresolved**. [Complete samples and reports](data/runs/auto-d1-fresh-final.json.gz).

| cell | words | q | A median ms | B median ms | median time change | median 95% CI | p95 time change | library | median >1% slower |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 clients | 100 | 1 | 38.906 | 38.905 | -0.004% | -0.31% to +0.21% | +0.113% | same | no |
| 4 clients | 100 | 1 | 117.279 | 117.574 | +0.252% | +0.07% to +0.43% | -0.033% | same | no |
| 8 clients | 100 | 1 | 194.496 | 196.216 | +0.885% | -3.64% to +5.28% | +0.835% | same | no |
| 1 clients | 100 | 3 | 45.196 | 45.168 | -0.061% | -0.18% to +0.09% | -0.180% | same | no |
| 4 clients | 100 | 3 | 180.319 | 178.322 | -1.107% | -1.29% to -0.97% | -1.094% | same | no |
| 8 clients | 100 | 3 | 360.023 | 358.314 | -0.475% | -0.68% to -0.30% | -0.445% | same | no |
| 1 clients | 1,000 | 1 | 184.369 | 184.382 | +0.007% | -0.08% to +0.17% | +0.223% | same | no |
| 4 clients | 1,000 | 1 | 709.972 | 711.364 | +0.196% | -0.03% to +0.30% | +0.253% | same | no |
| 8 clients | 1,000 | 1 | 1432.090 | 1448.939 | +1.177% | +0.83% to +1.57% | +1.010% | slower | yes |
| 1 clients | 1,000 | 3 | 195.293 | 195.433 | +0.072% | -0.05% to +0.21% | +0.309% | same | no |
| 4 clients | 1,000 | 3 | 781.524 | 781.918 | +0.050% | -0.08% to +0.13% | -0.035% | same | no |
| 8 clients | 1,000 | 3 | 1562.531 | 1562.096 | -0.028% | -0.06% to +0.08% | -0.014% | same | no |

## d1-3B: growing

Library gate: **promote**; conservative raw-median gate: **pass**. [Complete samples and reports](data/runs/auto-d1-growing-final.json.gz).

| cell | words | q | A median ms | B median ms | median time change | median 95% CI | p95 time change | library | median >1% slower |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| async-grow-c4-q3 | 100 | 3 | 164.474 | 126.172 | -23.288% | -23.49% to -23.09% | -0.278% | faster | no |
| async-grow-c4-q3/first-turn | 100 | 3 | 283.895 | 282.847 | -0.369% | -0.73% to +0.04% | -0.141% | same | no |
| async-grow-c4-q3/steady | 100 | 3 | 164.305 | 126.122 | -23.239% | -23.48% to -23.11% | -23.717% | faster | no |
| async-grow-c4-q3/session-wall | 100 | 3 | 1436.377 | 1167.292 | -18.734% | -19.07% to -18.57% | -18.825% | faster | no |
| async-grow-c8-q3 | 100 | 3 | 332.430 | 222.871 | -32.957% | -33.43% to -32.74% | -0.569% | faster | no |
| async-grow-c8-q3/first-turn | 100 | 3 | 835.150 | 828.989 | -0.738% | -0.77% to -0.55% | -0.685% | same | no |
| async-grow-c8-q3/steady | 100 | 3 | 331.773 | 221.391 | -33.270% | -33.72% to -32.79% | -33.155% | faster | no |
| async-grow-c8-q3/session-wall | 100 | 3 | 3161.032 | 2380.229 | -24.701% | -26.63% to -24.52% | -24.627% | faster | no |

## d3-lite: fresh

Library gate: **reject**; conservative raw-median gate: **unresolved**. [Complete samples and reports](data/runs/auto-d3-fresh-final.json.gz).

| cell | words | q | A median ms | B median ms | median time change | median 95% CI | p95 time change | library | median >1% slower |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 clients | 100 | 1 | 13.954 | 13.979 | +0.178% | -0.25% to +0.57% | -0.370% | same | no |
| 4 clients | 100 | 1 | 50.200 | 48.865 | -2.660% | -3.99% to -0.12% | -0.535% | same | no |
| 8 clients | 100 | 1 | 90.905 | 91.574 | +0.737% | +0.08% to +1.34% | -1.331% | same | no |
| 1 clients | 100 | 3 | 35.920 | 35.934 | +0.036% | -0.11% to +0.25% | +0.280% | same | no |
| 4 clients | 100 | 3 | 134.646 | 134.389 | -0.191% | -0.53% to +0.06% | -0.054% | same | no |
| 8 clients | 100 | 3 | 258.413 | 261.901 | +1.350% | -12.92% to +16.36% | -1.211% | same | yes |
| 1 clients | 1,000 | 1 | 62.020 | 62.025 | +0.007% | -0.19% to +0.17% | +0.036% | same | no |
| 4 clients | 1,000 | 1 | 240.238 | 240.118 | -0.050% | -1.54% to +0.84% | +0.245% | same | no |
| 8 clients | 1,000 | 1 | 456.793 | 456.411 | -0.084% | -1.32% to +0.00% | -3.731% | same | no |
| 1 clients | 1,000 | 3 | 177.598 | 177.371 | -0.128% | -0.29% to +0.03% | +0.068% | same | no |
| 4 clients | 1,000 | 3 | 685.288 | 687.303 | +0.294% | +0.04% to +0.53% | +1.189% | slower | no |
| 8 clients | 1,000 | 3 | 1363.442 | 1354.000 | -0.692% | -1.30% to -0.23% | -0.028% | same | no |

## d3-lite: growing

Library gate: **promote**; conservative raw-median gate: **pass**. [Complete samples and reports](data/runs/auto-d3-growing-final.json.gz).

| cell | words | q | A median ms | B median ms | median time change | median 95% CI | p95 time change | library | median >1% slower |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| async-grow-c4-q3 | 100 | 3 | 391.381 | 358.544 | -8.390% | -16.20% to +2.44% | -4.295% | same | no |
| async-grow-c4-q3/first-turn | 100 | 3 | 303.266 | 242.894 | -19.907% | -22.01% to -18.06% | -13.643% | faster | no |
| async-grow-c4-q3/steady | 100 | 3 | 408.681 | 389.940 | -4.586% | -13.94% to +3.10% | -3.711% | same | no |
| async-grow-c4-q3/session-wall | 100 | 3 | 3196.343 | 2954.595 | -7.563% | -9.88% to -5.51% | -7.792% | faster | no |
| async-grow-c8-q3 | 100 | 3 | 1160.642 | 947.392 | -18.374% | -22.09% to -16.42% | -16.807% | faster | no |
| async-grow-c8-q3/first-turn | 100 | 3 | 972.621 | 736.093 | -24.319% | -26.45% to -23.68% | -29.054% | faster | no |
| async-grow-c8-q3/steady | 100 | 3 | 1186.561 | 968.213 | -18.402% | -21.24% to -14.85% | -16.807% | faster | no |
| async-grow-c8-q3/session-wall | 100 | 3 | 9509.455 | 7582.504 | -20.264% | -22.37% to -15.22% | -19.696% | faster | no |

## Clef-Flash MLX 8-bit: fresh

Library gate: **reject**; conservative raw-median gate: **unresolved**. [Complete samples and reports](data/runs/auto-clef-fresh-final.json.gz).

| cell | words | q | A median ms | B median ms | median time change | median 95% CI | p95 time change | library | median >1% slower |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 clients | 100 | 1 | 193.114 | 193.457 | +0.177% | +0.10% to +0.31% | +0.421% | same | no |
| 4 clients | 100 | 1 | 673.518 | 684.878 | +1.687% | -0.93% to +5.23% | +2.022% | slower | yes |
| 8 clients | 100 | 1 | 1289.175 | 1277.151 | -0.933% | -10.60% to -0.40% | +0.714% | same | no |
| 1 clients | 100 | 3 | 268.633 | 268.659 | +0.010% | -0.10% to +0.19% | +0.483% | same | no |
| 4 clients | 100 | 3 | 1002.905 | 1024.087 | +2.112% | +1.50% to +2.27% | +2.064% | slower | yes |
| 8 clients | 100 | 3 | 1965.716 | 1795.655 | -8.651% | -10.18% to +2.64% | -0.532% | same | no |
| 1 clients | 1,000 | 1 | 714.604 | 721.923 | +1.024% | -0.66% to +2.44% | +1.249% | same | yes |
| 4 clients | 1,000 | 1 | 2862.740 | 2918.198 | +1.937% | -0.41% to +3.05% | +4.349% | slower | yes |
| 8 clients | 1,000 | 1 | 5784.719 | 5512.517 | -4.706% | -10.21% to +1.73% | -0.697% | same | no |
| 1 clients | 1,000 | 3 | 804.873 | 805.259 | +0.048% | -0.70% to +0.63% | -0.084% | same | no |
| 4 clients | 1,000 | 3 | 3169.912 | 3230.965 | +1.926% | +1.31% to +3.03% | +4.612% | slower | yes |
| 8 clients | 1,000 | 3 | 6087.057 | 6438.876 | +5.780% | -10.48% to +12.23% | +6.425% | slower | yes |

## Clef-Flash MLX 8-bit: growing

Library gate: **promote**; conservative raw-median gate: **pass**. [Complete samples and reports](data/runs/auto-clef-growing-final.json.gz).

| cell | words | q | A median ms | B median ms | median time change | median 95% CI | p95 time change | library | median >1% slower |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| async-grow-c4-q3 | 100 | 3 | 1761.838 | 778.895 | -55.791% | -56.41% to -54.09% | -26.207% | faster | no |
| async-grow-c4-q3/first-turn | 100 | 3 | 1575.849 | 1365.212 | -13.367% | -19.94% to +0.04% | -3.208% | same | no |
| async-grow-c4-q3/steady | 100 | 3 | 1784.594 | 778.756 | -56.362% | -57.96% to -55.77% | -61.190% | faster | no |
| async-grow-c4-q3/session-wall | 100 | 3 | 14004.741 | 6906.907 | -50.682% | -53.09% to -49.47% | -51.756% | faster | no |
| async-grow-c8-q3 | 100 | 3 | 5403.494 | 1512.577 | -72.007% | -72.61% to -71.71% | -40.793% | faster | no |
| async-grow-c8-q3/first-turn | 100 | 3 | 5001.344 | 3545.400 | -29.111% | -32.48% to -29.02% | -32.808% | faster | no |
| async-grow-c8-q3/steady | 100 | 3 | 5522.064 | 1510.588 | -72.644% | -72.67% to -72.04% | -74.633% | faster | no |
| async-grow-c8-q3/session-wall | 100 | 3 | 43261.218 | 14114.231 | -67.374% | -68.51% to -65.71% | -67.083% | faster | no |

