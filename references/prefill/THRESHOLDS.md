# Threshold sweep

Diagnostic forced packing versus padded capture: 96 complete cells, two accepted pairs and eight timed group calls per arm/cell. This earlier candidate is identified by its own source hashes; it is not the final automatic-policy gate.

All percentages are time changes; negative means forced packing is faster. Actual padding ranges cover all accepted iterations after ordinary sorted/budgeted grouping. A dash means no ordinary padded group exists.

A conservative 20% actual-padding cut selected 39 cells (10 d1, 17 d3, 12 Clef), all faster in this diagnostic comparison. Later correctness replays confirmed the intended routes around this cut; the final fresh gate still failed. d1's 16 three-question branch cells have undefined padded-group waste and cannot be selected from this scalar alone.

The grouping pitfall is visible in d3 B8/q3/1,000 words: nominal 50%/75% mixtures split into two ordinary budget groups, leaving only about 0.2 to 0.6% padding and causing +6.08%/+7.40% time increases. A global longest-prompt calculation would select the wrong path.

The original d1 stage's three contaminated pairs, and bulk Clef's three contaminated pairs plus interrupted fourth, remain in [the exclusion ledger](attempts.json). They contribute no clean samples. Each Clef cell's initial exclusion also remains recorded. The early indexing hypothesis was an inference from generic activity categories, not a reason to waive contamination.

## d1-3B

| B | q | words | target padding | actual padding | A ms | B ms | time change | p95 change | library |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| [4](data/runs/padding-d1-stage2.json.gz) | 1 | 100 | 0% | 0.00% to 0.18% | 105.233 | 106.056 | +0.782% | +1.096% | slower |
| [4](data/runs/padding-d1-stage2.json.gz) | 1 | 100 | 25% | 18.07% to 18.25% | 105.329 | 85.543 | -18.785% | -18.543% | faster |
| [4](data/runs/padding-d1-stage2.json.gz) | 1 | 100 | 50% | 36.68% to 36.86% | 105.228 | 70.294 | -33.198% | -32.853% | faster |
| [4](data/runs/padding-d1-stage2.json.gz) | 1 | 100 | 75% | 54.20% to 54.38% | 105.172 | 50.282 | -52.191% | -52.020% | faster |
| [4](data/runs/padding-d1-stage2.json.gz) | 1 | 1,000 | 0% | 0.00% to 0.02% | 726.449 | 733.831 | +1.016% | +1.614% | slower |
| [4](data/runs/padding-d1-stage2.json.gz) | 1 | 1,000 | 25% | 24.08% to 24.08% | 726.246 | 539.406 | -25.727% | -25.152% | faster |
| [4](data/runs/padding-d1-stage2.json.gz) | 1 | 1,000 | 50% | 48.24% to 48.24% | 727.149 | 371.902 | -48.855% | -48.867% | faster |
| [4](data/runs/padding-d1-stage2.json.gz) | 1 | 1,000 | 75% | 72.25% to 72.28% | 726.015 | 203.030 | -72.035% | -71.962% | faster |
| [4](data/runs/padding-d1-stage2.json.gz) | 3 | 100 | 0% | — | 183.181 | 151.630 | -17.224% | -16.971% | faster |
| [4](data/runs/padding-d1-stage2.json.gz) | 3 | 100 | 25% | — | 175.719 | 129.889 | -26.082% | -24.874% | faster |
| [4](data/runs/padding-d1-stage2.json.gz) | 3 | 100 | 50% | — | 165.969 | 116.353 | -29.895% | -30.046% | faster |
| [4](data/runs/padding-d1-stage2.json.gz) | 3 | 100 | 75% | — | 150.809 | 95.296 | -36.810% | -36.716% | faster |
| [4](data/runs/padding-d1-stage2.json.gz) | 3 | 1,000 | 0% | — | 790.290 | 803.738 | +1.702% | +1.404% | slower |
| [4](data/runs/padding-d1-stage2.json.gz) | 3 | 1,000 | 25% | — | 603.485 | 604.082 | +0.099% | -0.237% | same |
| [4](data/runs/padding-d1-stage2.json.gz) | 3 | 1,000 | 50% | — | 447.450 | 421.530 | -5.793% | -5.696% | faster |
| [4](data/runs/padding-d1-stage2.json.gz) | 3 | 1,000 | 75% | — | 304.442 | 253.011 | -16.893% | -16.803% | faster |
| [8](data/runs/padding-d1-stage2.json.gz) | 1 | 100 | 0% | 0.00% to 0.18% | 196.989 | 196.344 | -0.328% | -0.538% | same |
| [8](data/runs/padding-d1-stage2.json.gz) | 1 | 100 | 25% | 18.52% to 18.70% | 197.706 | 161.141 | -18.495% | -18.630% | faster |
| [8](data/runs/padding-d1-stage2.json.gz) | 1 | 100 | 50% | 36.41% to 36.50% | 197.477 | 126.868 | -35.756% | -35.928% | faster |
| [8](data/runs/padding-d1-stage2.json.gz) | 1 | 100 | 75% | 54.93% to 55.11% | 196.362 | 92.372 | -52.958% | -53.045% | faster |
| [8](data/runs/padding-d1-stage2.json.gz) | 1 | 1,000 | 0% | 0.00% to 0.01% | 1459.959 | 1452.588 | -0.505% | -0.774% | same |
| [8](data/runs/padding-d1-stage2.json.gz) | 1 | 1,000 | 25% | 24.10% to 24.16% | 1463.189 | 1102.375 | -24.659% | -24.773% | faster |
| [8](data/runs/padding-d1-stage2.json.gz) | 1 | 1,000 | 50% | 48.18% to 48.19% | 1464.704 | 754.825 | -48.466% | -48.851% | faster |
| [8](data/runs/padding-d1-stage2.json.gz) | 1 | 1,000 | 75% | 72.30% to 72.34% | 1461.913 | 394.358 | -73.025% | -73.109% | faster |
| [8](data/runs/padding-d1-stage2.json.gz) | 3 | 100 | 0% | — | 365.058 | 287.335 | -21.291% | -21.015% | faster |
| [8](data/runs/padding-d1-stage2.json.gz) | 3 | 100 | 25% | — | 350.583 | 254.159 | -27.504% | -27.328% | faster |
| [8](data/runs/padding-d1-stage2.json.gz) | 3 | 100 | 50% | — | 328.681 | 218.327 | -33.575% | -33.313% | faster |
| [8](data/runs/padding-d1-stage2.json.gz) | 3 | 100 | 75% | — | 294.466 | 183.982 | -37.520% | -37.509% | faster |
| [8](data/runs/padding-d1-stage2.json.gz) | 3 | 1,000 | 0% | — | 1580.442 | 1562.593 | -1.129% | -1.153% | same |
| [8](data/runs/padding-d1-stage2.json.gz) | 3 | 1,000 | 25% | — | 1243.094 | 1229.216 | -1.116% | -0.714% | same |
| [8](data/runs/padding-d1-stage2.json.gz) | 3 | 1,000 | 50% | — | 924.082 | 865.231 | -6.369% | -6.085% | faster |
| [8](data/runs/padding-d1-stage2.json.gz) | 3 | 1,000 | 75% | — | 550.312 | 492.669 | -10.475% | -10.116% | faster |

## d3-lite

| B | q | words | target padding | actual padding | A ms | B ms | time change | p95 change | library |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| [4](data/runs/padding-d3-stage1.json.gz) | 1 | 100 | 0% | 0.00% to 0.58% | 45.915 | 49.443 | +7.684% | +6.013% | slower |
| [4](data/runs/padding-d3-stage1.json.gz) | 1 | 100 | 25% | 11.21% to 11.98% | 46.364 | 42.925 | -7.417% | -8.724% | faster |
| [4](data/runs/padding-d3-stage1.json.gz) | 1 | 100 | 50% | 23.25% to 23.84% | 45.804 | 39.690 | -13.348% | -11.729% | faster |
| [4](data/runs/padding-d3-stage1.json.gz) | 1 | 100 | 75% | 34.46% to 34.88% | 45.726 | 35.041 | -23.368% | -22.952% | faster |
| [4](data/runs/padding-d3-stage1.json.gz) | 1 | 1,000 | 0% | 0.04% to 0.09% | 229.899 | 248.273 | +7.992% | +5.757% | slower |
| [4](data/runs/padding-d3-stage1.json.gz) | 1 | 1,000 | 25% | 22.37% to 22.42% | 233.945 | 191.664 | -18.073% | -17.581% | faster |
| [4](data/runs/padding-d3-stage1.json.gz) | 1 | 1,000 | 50% | 44.88% to 44.95% | 229.997 | 135.879 | -40.921% | -41.671% | faster |
| [4](data/runs/padding-d3-stage1.json.gz) | 1 | 1,000 | 75% | 67.21% to 67.26% | 229.531 | 83.381 | -63.673% | -63.850% | faster |
| [4](data/runs/padding-d3-stage1.json.gz) | 3 | 100 | 0% | 0.62% to 0.97% | 130.653 | 143.382 | +9.743% | +10.580% | slower |
| [4](data/runs/padding-d3-stage1.json.gz) | 3 | 100 | 25% | 11.84% to 12.36% | 130.386 | 126.707 | -2.822% | -2.388% | same |
| [4](data/runs/padding-d3-stage1.json.gz) | 3 | 100 | 50% | 23.87% to 24.34% | 130.985 | 111.756 | -14.680% | -15.370% | faster |
| [4](data/runs/padding-d3-stage1.json.gz) | 3 | 100 | 75% | 34.97% to 35.39% | 130.750 | 97.001 | -25.812% | -26.560% | faster |
| [4](data/runs/padding-d3-stage1.json.gz) | 3 | 1,000 | 0% | 0.14% to 0.21% | 680.131 | 700.723 | +3.028% | +3.298% | slower |
| [4](data/runs/padding-d3-stage1.json.gz) | 3 | 1,000 | 25% | 22.49% to 22.59% | 685.398 | 560.999 | -18.150% | -18.772% | faster |
| [4](data/runs/padding-d3-stage1.json.gz) | 3 | 1,000 | 50% | 44.95% to 45.05% | 684.644 | 404.611 | -40.902% | -41.217% | faster |
| [4](data/runs/padding-d3-stage1.json.gz) | 3 | 1,000 | 75% | 67.31% to 67.43% | 686.977 | 246.212 | -64.160% | -64.619% | faster |
| [8](data/runs/padding-d3-stage1.json.gz) | 1 | 100 | 0% | 0.17% to 0.41% | 90.873 | 97.606 | +7.409% | +5.925% | slower |
| [8](data/runs/padding-d3-stage1.json.gz) | 1 | 100 | 25% | 11.62% to 12.15% | 90.371 | 87.349 | -3.345% | -4.746% | faster |
| [8](data/runs/padding-d3-stage1.json.gz) | 1 | 100 | 50% | 22.96% to 23.49% | 89.631 | 77.958 | -13.023% | -12.759% | faster |
| [8](data/runs/padding-d3-stage1.json.gz) | 1 | 100 | 75% | 34.93% to 35.29% | 89.181 | 66.323 | -25.631% | -25.422% | faster |
| [8](data/runs/padding-d3-stage1.json.gz) | 1 | 1,000 | 0% | 0.02% to 0.09% | 460.109 | 497.000 | +8.018% | +7.447% | slower |
| [8](data/runs/padding-d3-stage1.json.gz) | 1 | 1,000 | 25% | 22.41% to 22.49% | 464.092 | 379.422 | -18.244% | -18.273% | faster |
| [8](data/runs/padding-d3-stage1.json.gz) | 1 | 1,000 | 50% | 44.84% to 44.87% | 463.167 | 268.483 | -42.033% | -41.987% | faster |
| [8](data/runs/padding-d3-stage1.json.gz) | 1 | 1,000 | 75% | 67.27% to 67.32% | 460.007 | 162.452 | -64.685% | -65.478% | faster |
| [8](data/runs/padding-d3-stage1.json.gz) | 3 | 100 | 0% | 0.85% to 1.03% | 259.935 | 280.719 | +7.996% | +7.914% | slower |
| [8](data/runs/padding-d3-stage1.json.gz) | 3 | 100 | 25% | 12.01% to 12.77% | 259.364 | 249.711 | -3.722% | -5.423% | faster |
| [8](data/runs/padding-d3-stage1.json.gz) | 3 | 100 | 50% | 23.45% to 24.11% | 259.189 | 219.293 | -15.393% | -15.667% | faster |
| [8](data/runs/padding-d3-stage1.json.gz) | 3 | 100 | 75% | 35.31% to 35.97% | 258.730 | 186.322 | -27.986% | -27.913% | faster |
| [8](data/runs/padding-d3-stage1.json.gz) | 3 | 1,000 | 0% | 0.05% to 0.08% | 1372.178 | 1397.574 | +1.851% | +2.078% | slower |
| [8](data/runs/padding-d3-stage1.json.gz) | 3 | 1,000 | 25% | 2.83% to 2.90% | 1096.743 | 1091.844 | -0.447% | -0.812% | same |
| [8](data/runs/padding-d3-stage1.json.gz) | 3 | 1,000 | 50% | 0.24% to 0.40% | 747.973 | 793.464 | +6.082% | +6.167% | slower |
| [8](data/runs/padding-d3-stage1.json.gz) | 3 | 1,000 | 75% | 0.40% to 0.57% | 445.394 | 478.345 | +7.398% | +7.641% | slower |

## Clef-Flash MLX 8-bit

| B | q | words | target padding | actual padding | A ms | B ms | time change | p95 change | library |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| [4](data/runs/clef-cells-padding-b4-q1-w100-p0.json.gz) | 1 | 100 | 0% | 0.00% to 0.47% | 642.582 | 631.731 | -1.689% | -1.701% | same |
| [4](data/runs/clef-cells-padding-b4-q1-w100-p25.json.gz) | 1 | 100 | 25% | 8.99% to 9.61% | 646.206 | 590.037 | -8.692% | -8.870% | faster |
| [4](data/runs/clef-cells-padding-b4-q1-w100-p50.json.gz) | 1 | 100 | 50% | 18.42% to 19.03% | 643.620 | 520.748 | -19.091% | -19.542% | faster |
| [4](data/runs/clef-cells-padding-b4-q1-w100-p75.json.gz) | 1 | 100 | 75% | 27.62% to 27.89% | 647.411 | 484.078 | -25.229% | -25.630% | faster |
| [4](data/runs/clef-cells-padding-b4-q1-w1000-p0.json.gz) | 1 | 1,000 | 0% | 0.00% to 0.06% | 2819.788 | 2801.004 | -0.666% | +0.224% | same |
| [4](data/runs/clef-cells-padding-b4-q1-w1000-p25.json.gz) | 1 | 1,000 | 25% | 21.40% to 21.49% | 2836.772 | 2216.753 | -21.856% | -21.817% | faster |
| [4](data/runs/clef-cells-padding-b4-q1-w1000-p50.json.gz) | 1 | 1,000 | 50% | 42.80% to 42.87% | 2923.231 | 1676.109 | -42.662% | -45.345% | faster |
| [4](data/runs/clef-cells-padding-b4-q1-w1000-p75.json.gz) | 1 | 1,000 | 75% | 64.13% to 64.20% | 2837.305 | 1026.828 | -63.810% | -65.387% | faster |
| [4](data/runs/clef-cells-padding-b4-q3-w100-p0.json.gz) | 3 | 100 | 0% | 0.00% to 0.24% | 1013.787 | 990.128 | -2.334% | -3.657% | same |
| [4](data/runs/clef-cells-padding-b4-q3-w100-p25.json.gz) | 3 | 100 | 25% | 5.76% to 5.98% | 1017.389 | 923.522 | -9.226% | -10.996% | faster |
| [4](data/runs/clef-cells-padding-b4-q3-w100-p50.json.gz) | 3 | 100 | 50% | 11.88% to 12.11% | 1017.881 | 880.731 | -13.474% | -15.282% | faster |
| [4](data/runs/clef-cells-padding-b4-q3-w100-p75.json.gz) | 3 | 100 | 75% | 17.52% to 17.65% | 1015.989 | 812.292 | -20.049% | -21.849% | faster |
| [4](data/runs/clef-cells-padding-b4-q3-w1000-p0.json.gz) | 3 | 1,000 | 0% | 0.02% to 0.06% | 3250.958 | 3326.659 | +2.329% | +0.167% | same |
| [4](data/runs/clef-cells-padding-b4-q3-w1000-p25.json.gz) | 3 | 1,000 | 25% | 18.81% to 18.95% | 3223.766 | 2619.428 | -18.746% | -21.221% | faster |
| [4](data/runs/clef-cells-padding-b4-q3-w1000-p50.json.gz) | 3 | 1,000 | 50% | 37.84% to 37.92% | 3105.168 | 1910.975 | -38.458% | -38.735% | faster |
| [4](data/runs/clef-cells-padding-b4-q3-w1000-p75.json.gz) | 3 | 1,000 | 75% | 56.65% to 56.71% | 3205.194 | 1373.282 | -57.154% | -58.728% | faster |
| [8](data/runs/clef-cells-padding-b8-q1-w100-p0.json.gz) | 1 | 100 | 0% | 0.05% to 0.23% | 1277.882 | 1258.867 | -1.488% | -3.119% | same |
| [8](data/runs/clef-cells-padding-b8-q1-w100-p25.json.gz) | 1 | 100 | 25% | 9.32% to 9.65% | 1286.102 | 1146.196 | -10.878% | -12.738% | faster |
| [8](data/runs/clef-cells-padding-b8-q1-w100-p50.json.gz) | 1 | 100 | 50% | 18.49% to 18.89% | 1295.724 | 1037.566 | -19.924% | -21.827% | faster |
| [8](data/runs/clef-cells-padding-b8-q1-w100-p75.json.gz) | 1 | 100 | 75% | 27.87% to 28.31% | 1294.663 | 921.306 | -28.838% | -30.518% | faster |
| [8](data/runs/clef-cells-padding-b8-q1-w1000-p0.json.gz) | 1 | 1,000 | 0% | 0.03% to 0.07% | 5461.809 | 5558.843 | +1.777% | +1.492% | slower |
| [8](data/runs/clef-cells-padding-b8-q1-w1000-p25.json.gz) | 1 | 1,000 | 25% | 21.38% to 21.47% | 5542.245 | 4301.766 | -22.382% | -20.255% | faster |
| [8](data/runs/clef-cells-padding-b8-q1-w1000-p50.json.gz) | 1 | 1,000 | 50% | 42.77% to 42.82% | 5490.266 | 3151.917 | -42.591% | -42.829% | faster |
| [8](data/runs/clef-cells-padding-b8-q1-w1000-p75.json.gz) | 1 | 1,000 | 75% | 64.21% to 64.27% | 5552.983 | 1979.457 | -64.353% | -65.482% | faster |
| [8](data/runs/clef-cells-padding-b8-q3-w100-p0.json.gz) | 3 | 100 | 0% | 0.03% to 0.27% | 2016.715 | 1972.779 | -2.179% | -4.203% | same |
| [8](data/runs/clef-cells-padding-b8-q3-w100-p25.json.gz) | 3 | 100 | 25% | 5.88% to 6.16% | 2043.207 | 1896.416 | -7.184% | -10.140% | faster |
| [8](data/runs/clef-cells-padding-b8-q3-w100-p50.json.gz) | 3 | 100 | 50% | 11.76% to 12.06% | 2039.900 | 1791.630 | -12.171% | -15.344% | faster |
| [8](data/runs/clef-cells-padding-b8-q3-w100-p75.json.gz) | 3 | 100 | 75% | 17.71% to 17.95% | 2045.990 | 1672.013 | -18.279% | -21.140% | faster |
| [8](data/runs/clef-cells-padding-b8-q3-w1000-p0.json.gz) | 3 | 1,000 | 0% | 0.02% to 0.06% | 6126.579 | 6195.081 | +1.118% | -0.000% | same |
| [8](data/runs/clef-cells-padding-b8-q3-w1000-p25.json.gz) | 3 | 1,000 | 25% | 18.86% to 18.98% | 6147.788 | 4998.182 | -18.700% | -18.185% | faster |
| [8](data/runs/clef-cells-padding-b8-q3-w1000-p50.json.gz) | 3 | 1,000 | 50% | 37.78% to 37.85% | 6206.166 | 3847.178 | -38.010% | -37.600% | faster |
| [8](data/runs/clef-cells-padding-b8-q3-w1000-p75.json.gz) | 3 | 1,000 | 75% | 56.71% to 56.77% | 6394.035 | 2767.859 | -56.712% | -56.768% | faster |

