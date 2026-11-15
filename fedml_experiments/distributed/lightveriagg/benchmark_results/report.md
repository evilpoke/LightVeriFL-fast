# LightVeriAgg vs. LightVeriFL + SecAgg

One FL iteration (training excluded), times per iteration with parallel clients (every protocol step counts with its slowest party), V.1 measured once and counted with 1/L. Communication per iteration; '+ comm.' adds the transfer time of every message type for the slowest client at 10 Mbit/s up / 50 Mbit/s down (server bandwidth unlimited). See ../benchmark_vs_lightverifl_secagg.py. Machine: 4 vCPUs, two benchmark processes in parallel (critical-path times fluctuate by up to ~20-50 %; the component table below averages over parties and is more stable). Commands: `--N 10 20 30 40 --dropout 0.1 0.3 0.5 --d 10000 --alpha_bits 0 64 --tag N_scaling`, `--N 10 --dropout 0.3 --d 10000 30730 100000 --alpha_bits 0 64 --tag d_scaling`, `--N 10 --dropout 0.3 --d 1000000 --alpha_bits 0 --tag d_scaling_1M`.

## N_scaling

| protocol | N | d | dropout | aggregation | V.0 decommitting | V.1 batch check (1/L) | verification | total | + comm. (10/50 Mbit/s) | busiest client | server | upload/client | download/client | server in |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LightVeriAgg (alpha 0 bit) | 10 | 10000 | 10% | 476.1 ms | 7.3 ms | 61.5 ms | 68.8 ms | 544.9 ms | 647.6 ms | 1.06 s | 6.1 ms | 107.91 kB | 102.18 kB | 1.08 MB |
| LightVeriAgg (alpha 64 bit) | 10 | 10000 | 10% | 488.6 ms | 7.5 ms | 445.3 ms | 452.8 ms | 941.4 ms | 1.04 s | 4.89 s | 6.0 ms | 107.91 kB | 102.18 kB | 1.08 MB |
| LightVeriFL+SecAgg | 10 | 10000 | 10% | 492.0 ms | 12.8 ms | 61.2 ms | 74.0 ms | 566.0 ms | 605.5 ms | 1.07 s | 14.3 ms | 41.02 kB | 41.44 kB | 409.60 kB |
| LightVeriAgg (alpha 0 bit) | 10 | 10000 | 30% | 438.0 ms | 11.1 ms | 58.6 ms | 69.8 ms | 507.7 ms | 610.4 ms | 1.02 s | 9.8 ms | 107.91 kB | 102.18 kB | 1.08 MB |
| LightVeriAgg (alpha 64 bit) | 10 | 10000 | 30% | 676.4 ms | 11.2 ms | 422.0 ms | 433.2 ms | 1.11 s | 1.21 s | 4.67 s | 10.1 ms | 107.91 kB | 102.18 kB | 1.08 MB |
| LightVeriFL+SecAgg | 10 | 10000 | 30% | 444.3 ms | 15.4 ms | 58.5 ms | 73.9 ms | 518.2 ms | 557.6 ms | 990.9 ms | 16.9 ms | 41.02 kB | 41.44 kB | 409.60 kB |
| LightVeriAgg (alpha 0 bit) | 10 | 10000 | 50% | 507.9 ms | 9.4 ms | 57.4 ms | 66.9 ms | 574.8 ms | 677.5 ms | 1.03 s | 8.2 ms | 107.91 kB | 102.18 kB | 1.08 MB |
| LightVeriAgg (alpha 64 bit) | 10 | 10000 | 50% | 462.9 ms | 11.1 ms | 434.0 ms | 445.1 ms | 908.0 ms | 1.01 s | 4.77 s | 8.6 ms | 107.91 kB | 102.18 kB | 1.08 MB |
| LightVeriFL+SecAgg | 10 | 10000 | 50% | 457.8 ms | 12.0 ms | 62.3 ms | 74.4 ms | 532.2 ms | 571.6 ms | 1.04 s | 13.5 ms | 41.02 kB | 41.44 kB | 409.60 kB |
| LightVeriAgg (alpha 0 bit) | 20 | 10000 | 10% | 628.1 ms | 12.3 ms | 65.2 ms | 77.5 ms | 705.7 ms | 815.8 ms | 1.22 s | 13.0 ms | 115.01 kB | 113.60 kB | 2.30 MB |
| LightVeriAgg (alpha 64 bit) | 20 | 10000 | 10% | 589.7 ms | 13.1 ms | 460.0 ms | 473.1 ms | 1.06 s | 1.17 s | 5.15 s | 12.6 ms | 115.01 kB | 113.60 kB | 2.30 MB |
| LightVeriFL+SecAgg | 20 | 10000 | 10% | 646.5 ms | 26.5 ms | 65.1 ms | 91.6 ms | 738.1 ms | 778.4 ms | 1.23 s | 29.9 ms | 41.78 kB | 42.88 kB | 834.40 kB |
| LightVeriAgg (alpha 0 bit) | 20 | 10000 | 30% | 625.0 ms | 16.8 ms | 71.6 ms | 88.4 ms | 713.3 ms | 823.5 ms | 1.27 s | 16.5 ms | 115.01 kB | 113.60 kB | 2.30 MB |
| LightVeriAgg (alpha 64 bit) | 20 | 10000 | 30% | 591.3 ms | 16.9 ms | 452.1 ms | 469.0 ms | 1.06 s | 1.17 s | 5.09 s | 16.2 ms | 115.01 kB | 113.60 kB | 2.30 MB |
| LightVeriFL+SecAgg | 20 | 10000 | 30% | 698.4 ms | 25.1 ms | 68.2 ms | 93.2 ms | 791.7 ms | 831.9 ms | 1.28 s | 27.9 ms | 41.78 kB | 42.88 kB | 834.40 kB |
| LightVeriAgg (alpha 0 bit) | 20 | 10000 | 50% | 619.9 ms | 19.2 ms | 62.8 ms | 82.0 ms | 701.9 ms | 812.1 ms | 1.21 s | 19.7 ms | 115.01 kB | 113.60 kB | 2.30 MB |
| LightVeriAgg (alpha 64 bit) | 20 | 10000 | 50% | 626.1 ms | 19.2 ms | 439.4 ms | 458.6 ms | 1.08 s | 1.19 s | 4.94 s | 19.7 ms | 115.01 kB | 113.60 kB | 2.30 MB |
| LightVeriFL+SecAgg | 20 | 10000 | 50% | 626.5 ms | 30.0 ms | 66.4 ms | 96.3 ms | 722.9 ms | 763.2 ms | 1.22 s | 34.7 ms | 41.78 kB | 42.88 kB | 834.40 kB |
| LightVeriAgg (alpha 0 bit) | 30 | 10000 | 10% | 847.3 ms | 34.8 ms | 70.0 ms | 104.7 ms | 952.0 ms | 1.07 s | 1.51 s | 24.8 ms | 118.19 kB | 119.21 kB | 3.55 MB |
| LightVeriAgg (alpha 64 bit) | 30 | 10000 | 10% | 851.4 ms | 23.4 ms | 447.8 ms | 471.3 ms | 1.32 s | 1.44 s | 5.26 s | 24.3 ms | 118.19 kB | 119.21 kB | 3.55 MB |
| LightVeriFL+SecAgg | 30 | 10000 | 10% | 958.4 ms | 45.2 ms | 76.3 ms | 121.6 ms | 1.08 s | 1.12 s | 1.55 s | 52.1 ms | 42.54 kB | 44.32 kB | 1.27 MB |
| LightVeriAgg (alpha 0 bit) | 30 | 10000 | 30% | 811.2 ms | 25.3 ms | 67.3 ms | 92.6 ms | 903.9 ms | 1.02 s | 1.42 s | 28.3 ms | 118.19 kB | 119.21 kB | 3.54 MB |
| LightVeriAgg (alpha 64 bit) | 30 | 10000 | 30% | 888.4 ms | 27.6 ms | 450.1 ms | 477.7 ms | 1.37 s | 1.48 s | 5.26 s | 29.6 ms | 118.19 kB | 119.21 kB | 3.54 MB |
| LightVeriFL+SecAgg | 30 | 10000 | 30% | 924.5 ms | 56.0 ms | 70.5 ms | 126.4 ms | 1.05 s | 1.09 s | 1.51 s | 65.2 ms | 42.54 kB | 44.32 kB | 1.27 MB |
| LightVeriAgg (alpha 0 bit) | 30 | 10000 | 50% | 1.27 s | 23.1 ms | 68.2 ms | 91.3 ms | 1.36 s | 1.48 s | 1.54 s | 25.9 ms | 118.19 kB | 119.21 kB | 3.54 MB |
| LightVeriAgg (alpha 64 bit) | 30 | 10000 | 50% | 877.7 ms | 24.7 ms | 457.0 ms | 481.7 ms | 1.36 s | 1.47 s | 5.34 s | 25.5 ms | 118.19 kB | 119.21 kB | 3.54 MB |
| LightVeriFL+SecAgg | 30 | 10000 | 50% | 926.2 ms | 37.5 ms | 69.8 ms | 107.3 ms | 1.03 s | 1.07 s | 1.45 s | 44.8 ms | 42.54 kB | 44.32 kB | 1.27 MB |
| LightVeriAgg (alpha 0 bit) | 40 | 10000 | 10% | 1.28 s | 30.3 ms | 73.9 ms | 104.1 ms | 1.39 s | 1.50 s | 1.94 s | 36.1 ms | 120.49 kB | 123.39 kB | 4.82 MB |
| LightVeriAgg (alpha 64 bit) | 40 | 10000 | 10% | 1.30 s | 30.6 ms | 592.5 ms | 623.0 ms | 1.92 s | 2.04 s | 7.07 s | 34.3 ms | 120.49 kB | 123.39 kB | 4.82 MB |
| LightVeriFL+SecAgg | 40 | 10000 | 10% | 1.28 s | 59.1 ms | 68.4 ms | 127.5 ms | 1.41 s | 1.45 s | 1.86 s | 71.5 ms | 43.30 kB | 45.76 kB | 1.73 MB |
| LightVeriAgg (alpha 0 bit) | 40 | 10000 | 30% | 1.18 s | 33.7 ms | 69.4 ms | 103.1 ms | 1.28 s | 1.40 s | 1.83 s | 39.5 ms | 120.49 kB | 123.39 kB | 4.82 MB |
| LightVeriAgg (alpha 64 bit) | 40 | 10000 | 30% | 1.31 s | 34.0 ms | 449.6 ms | 483.6 ms | 1.79 s | 1.91 s | 5.60 s | 41.4 ms | 120.49 kB | 123.39 kB | 4.82 MB |
| LightVeriFL+SecAgg | 40 | 10000 | 30% | 1.30 s | 60.5 ms | 72.6 ms | 133.1 ms | 1.43 s | 1.47 s | 1.87 s | 72.6 ms | 43.30 kB | 45.76 kB | 1.73 MB |
| LightVeriAgg (alpha 0 bit) | 40 | 10000 | 50% | 1.60 s | 34.4 ms | 73.2 ms | 107.6 ms | 1.71 s | 1.82 s | 2.05 s | 41.5 ms | 120.49 kB | 123.39 kB | 4.82 MB |
| LightVeriAgg (alpha 64 bit) | 40 | 10000 | 50% | 1.23 s | 35.6 ms | 434.8 ms | 470.5 ms | 1.70 s | 1.81 s | 5.44 s | 40.9 ms | 120.49 kB | 123.39 kB | 4.82 MB |
| LightVeriFL+SecAgg | 40 | 10000 | 50% | 1.24 s | 69.9 ms | 77.7 ms | 147.5 ms | 1.39 s | 1.43 s | 1.86 s | 89.4 ms | 43.30 kB | 45.76 kB | 1.73 MB |

## d_scaling

| protocol | N | d | dropout | aggregation | V.0 decommitting | V.1 batch check (1/L) | verification | total | + comm. (10/50 Mbit/s) | busiest client | server | upload/client | download/client | server in |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LightVeriAgg (alpha 0 bit) | 10 | 10000 | 30% | 444.5 ms | 11.1 ms | 57.2 ms | 68.3 ms | 512.8 ms | 615.5 ms | 989.5 ms | 10.0 ms | 107.91 kB | 102.18 kB | 1.08 MB |
| LightVeriAgg (alpha 64 bit) | 10 | 10000 | 30% | 443.0 ms | 11.8 ms | 469.9 ms | 481.6 ms | 924.7 ms | 1.03 s | 5.10 s | 9.7 ms | 107.91 kB | 102.18 kB | 1.08 MB |
| LightVeriFL+SecAgg | 10 | 10000 | 30% | 531.3 ms | 15.3 ms | 58.1 ms | 73.4 ms | 604.6 ms | 644.1 ms | 1.08 s | 16.8 ms | 41.02 kB | 41.44 kB | 409.60 kB |
| LightVeriAgg (alpha 0 bit) | 10 | 30730 | 30% | 1.03 s | 11.4 ms | 164.5 ms | 175.9 ms | 1.21 s | 1.52 s | 2.66 s | 12.8 ms | 329.03 kB | 309.48 kB | 3.29 MB |
| LightVeriAgg (alpha 64 bit) | 10 | 30730 | 30% | 1.09 s | 11.2 ms | 1.34 s | 1.35 s | 2.44 s | 2.75 s | 14.41 s | 12.7 ms | 329.03 kB | 309.48 kB | 3.29 MB |
| LightVeriFL+SecAgg | 10 | 30730 | 30% | 1.12 s | 15.9 ms | 157.6 ms | 173.4 ms | 1.29 s | 1.41 s | 2.67 s | 20.8 ms | 123.94 kB | 124.36 kB | 1.24 MB |
| LightVeriAgg (alpha 0 bit) | 10 | 100000 | 30% | 2.91 s | 10.9 ms | 472.3 ms | 483.2 ms | 3.40 s | 4.41 s | 7.62 s | 24.4 ms | 1.07 MB | 1.00 MB | 10.68 MB |
| LightVeriAgg (alpha 64 bit) | 10 | 100000 | 30% | 2.97 s | 10.7 ms | 4.07 s | 4.08 s | 7.05 s | 8.07 s | 43.65 s | 23.2 ms | 1.07 MB | 1.00 MB | 10.68 MB |
| LightVeriFL+SecAgg | 10 | 100000 | 30% | 2.99 s | 14.9 ms | 469.8 ms | 484.7 ms | 3.47 s | 3.86 s | 7.64 s | 29.8 ms | 401.02 kB | 401.44 kB | 4.01 MB |

## d_scaling_1M

| protocol | N | d | dropout | aggregation | V.0 decommitting | V.1 batch check (1/L) | verification | total | + comm. (10/50 Mbit/s) | busiest client | server | upload/client | download/client | server in |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LightVeriAgg (alpha 0 bit) | 10 | 1000000 | 30% | 20.40 s | 10.9 ms | 3.72 s | 3.73 s | 24.12 s | 34.26 s | 56.78 s | 267.5 ms | 10.67 MB | 10.00 MB | 106.68 MB |
| LightVeriFL+SecAgg | 10 | 1000000 | 30% | 20.69 s | 15.2 ms | 3.88 s | 3.90 s | 24.59 s | 28.43 s | 58.42 s | 208.3 ms | 4.00 MB | 4.00 MB | 40.01 MB |

figures: scaling_N.png, scaling_d.png

## Components (mean per party and call, less sensitive to load noise than the critical path)

**N = 10, d = 10000, dropout 10%**

| component | LightVeriFL+SecAgg | LightVeriAgg (alpha 0 bit) | LightVeriAgg (alpha 64 bit) |
|---|---|---|---|
| model masks (secure aggregation), client | 995 us | 1.8 ms | 1.9 ms |
| verification masks: generate + encode, client | 40.8 ms | 41.1 ms | 41.7 ms |
| hash h_i, client | 379.9 ms | 389.3 ms | 380.2 ms |
| commitment + masking of h_i / r_i, client | 1.7 ms | 3.0 ms | 3.0 ms |
| V.0 aggregate masks + verify commitments, client | 272 us | 3.3 ms | 3.5 ms |
| V.1 batch check (per batch), client | 569.1 ms | 569.3 ms | 4.20 s |
| model recovery, server | 1.9 ms | 2.5 ms | 2.6 ms |
| V.0 decode + unmask, server | 12.3 ms | 3.6 ms | 3.5 ms |

**N = 10, d = 10000, dropout 30%**

| component | LightVeriFL+SecAgg | LightVeriAgg (alpha 0 bit) | LightVeriAgg (alpha 64 bit) |
|---|---|---|---|
| model masks (secure aggregation), client | 983 us | 1.7 ms | 1.9 ms |
| verification masks: generate + encode, client | 38.7 ms | 38.5 ms | 40.6 ms |
| hash h_i, client | 373.1 ms | 366.6 ms | 390.7 ms |
| commitment + masking of h_i / r_i, client | 1.6 ms | 2.9 ms | 2.9 ms |
| V.0 aggregate masks + verify commitments, client | 258 us | 3.3 ms | 3.4 ms |
| V.1 batch check (per batch), client | 567.1 ms | 563.4 ms | 4.18 s |
| model recovery, server | 1.9 ms | 2.3 ms | 2.4 ms |
| V.0 decode + unmask, server | 15.1 ms | 7.6 ms | 7.5 ms |

**N = 10, d = 10000, dropout 50%**

| component | LightVeriFL+SecAgg | LightVeriAgg (alpha 0 bit) | LightVeriAgg (alpha 64 bit) |
|---|---|---|---|
| model masks (secure aggregation), client | 944 us | 1.6 ms | 1.7 ms |
| verification masks: generate + encode, client | 39.9 ms | 38.7 ms | 39.3 ms |
| hash h_i, client | 378.1 ms | 405.8 ms | 370.4 ms |
| commitment + masking of h_i / r_i, client | 1.6 ms | 3.6 ms | 2.9 ms |
| V.0 aggregate masks + verify commitments, client | 241 us | 3.3 ms | 3.7 ms |
| V.1 batch check (per batch), client | 588.7 ms | 569.1 ms | 4.23 s |
| model recovery, server | 1.7 ms | 2.1 ms | 2.5 ms |
| V.0 decode + unmask, server | 11.7 ms | 6.1 ms | 6.1 ms |

**N = 10, d = 30730, dropout 30%**

| component | LightVeriFL+SecAgg | LightVeriAgg (alpha 0 bit) | LightVeriAgg (alpha 64 bit) |
|---|---|---|---|
| model masks (secure aggregation), client | 2.9 ms | 3.8 ms | 4.0 ms |
| verification masks: generate + encode, client | 37.6 ms | 36.8 ms | 38.6 ms |
| hash h_i, client | 1.01 s | 963.6 ms | 979.4 ms |
| commitment + masking of h_i / r_i, client | 1.5 ms | 3.0 ms | 2.9 ms |
| V.0 aggregate masks + verify commitments, client | 271 us | 3.4 ms | 3.3 ms |
| V.1 batch check (per batch), client | 1.55 s | 1.58 s | 12.73 s |
| model recovery, server | 5.3 ms | 5.4 ms | 5.4 ms |
| V.0 decode + unmask, server | 15.5 ms | 7.4 ms | 7.3 ms |

**N = 10, d = 100000, dropout 30%**

| component | LightVeriFL+SecAgg | LightVeriAgg (alpha 0 bit) | LightVeriAgg (alpha 64 bit) |
|---|---|---|---|
| model masks (secure aggregation), client | 10.5 ms | 11.8 ms | 11.9 ms |
| verification masks: generate + encode, client | 37.5 ms | 36.0 ms | 37.4 ms |
| hash h_i, client | 2.77 s | 2.73 s | 2.75 s |
| commitment + masking of h_i / r_i, client | 1.5 ms | 3.0 ms | 2.9 ms |
| V.0 aggregate masks + verify commitments, client | 256 us | 3.3 ms | 3.3 ms |
| V.1 batch check (per batch), client | 4.60 s | 4.60 s | 39.28 s |
| model recovery, server | 15.2 ms | 17.0 ms | 16.0 ms |
| V.0 decode + unmask, server | 14.6 ms | 7.4 ms | 7.2 ms |

**N = 10, d = 1000000, dropout 30%**

| component | LightVeriFL+SecAgg | LightVeriAgg (alpha 0 bit) |
|---|---|---|
| model masks (secure aggregation), client | 121.5 ms | 160.6 ms |
| verification masks: generate + encode, client | 36.3 ms | 37.2 ms |
| hash h_i, client | 19.43 s | 19.23 s |
| commitment + masking of h_i / r_i, client | 1.6 ms | 2.9 ms |
| V.0 aggregate masks + verify commitments, client | 245 us | 3.4 ms |
| V.1 batch check (per batch), client | 37.16 s | 36.73 s |
| model recovery, server | 193.4 ms | 260.2 ms |
| V.0 decode + unmask, server | 14.9 ms | 7.4 ms |

**N = 20, d = 10000, dropout 10%**

| component | LightVeriFL+SecAgg | LightVeriAgg (alpha 0 bit) | LightVeriAgg (alpha 64 bit) |
|---|---|---|---|
| model masks (secure aggregation), client | 2.2 ms | 3.5 ms | 3.6 ms |
| verification masks: generate + encode, client | 174.7 ms | 175.6 ms | 173.6 ms |
| hash h_i, client | 371.9 ms | 380.4 ms | 362.4 ms |
| commitment + masking of h_i / r_i, client | 1.6 ms | 3.1 ms | 2.9 ms |
| V.0 aggregate masks + verify commitments, client | 450 us | 3.6 ms | 3.7 ms |
| V.1 batch check (per batch), client | 620.6 ms | 621.8 ms | 4.21 s |
| model recovery, server | 4.1 ms | 4.5 ms | 4.5 ms |
| V.0 decode + unmask, server | 25.8 ms | 8.5 ms | 8.0 ms |

**N = 20, d = 10000, dropout 30%**

| component | LightVeriFL+SecAgg | LightVeriAgg (alpha 0 bit) | LightVeriAgg (alpha 64 bit) |
|---|---|---|---|
| model masks (secure aggregation), client | 2.0 ms | 3.6 ms | 3.5 ms |
| verification masks: generate + encode, client | 177.6 ms | 173.8 ms | 177.3 ms |
| hash h_i, client | 385.9 ms | 376.2 ms | 370.3 ms |
| commitment + masking of h_i / r_i, client | 3.6 ms | 3.1 ms | 3.0 ms |
| V.0 aggregate masks + verify commitments, client | 503 us | 3.7 ms | 3.7 ms |
| V.1 batch check (per batch), client | 627.4 ms | 632.4 ms | 4.27 s |
| model recovery, server | 4.1 ms | 4.2 ms | 4.3 ms |
| V.0 decode + unmask, server | 23.8 ms | 12.3 ms | 12.0 ms |

**N = 20, d = 10000, dropout 50%**

| component | LightVeriFL+SecAgg | LightVeriAgg (alpha 0 bit) | LightVeriAgg (alpha 64 bit) |
|---|---|---|---|
| model masks (secure aggregation), client | 1.9 ms | 3.4 ms | 3.8 ms |
| verification masks: generate + encode, client | 179.1 ms | 171.7 ms | 176.0 ms |
| hash h_i, client | 375.4 ms | 371.6 ms | 370.5 ms |
| commitment + masking of h_i / r_i, client | 1.5 ms | 3.0 ms | 2.9 ms |
| V.0 aggregate masks + verify commitments, client | 472 us | 3.8 ms | 3.7 ms |
| V.1 batch check (per batch), client | 634.7 ms | 617.0 ms | 4.28 s |
| model recovery, server | 5.3 ms | 4.7 ms | 4.5 ms |
| V.0 decode + unmask, server | 29.4 ms | 15.1 ms | 15.2 ms |

**N = 30, d = 10000, dropout 10%**

| component | LightVeriFL+SecAgg | LightVeriAgg (alpha 0 bit) | LightVeriAgg (alpha 64 bit) |
|---|---|---|---|
| model masks (secure aggregation), client | 2.8 ms | 6.0 ms | 6.0 ms |
| verification masks: generate + encode, client | 381.2 ms | 379.2 ms | 383.5 ms |
| hash h_i, client | 388.3 ms | 365.9 ms | 377.0 ms |
| commitment + masking of h_i / r_i, client | 4.2 ms | 3.0 ms | 3.0 ms |
| V.0 aggregate masks + verify commitments, client | 618 us | 4.4 ms | 4.0 ms |
| V.1 batch check (per batch), client | 660.3 ms | 657.4 ms | 4.28 s |
| model recovery, server | 7.6 ms | 7.3 ms | 6.8 ms |
| V.0 decode + unmask, server | 44.5 ms | 17.5 ms | 17.5 ms |

**N = 30, d = 10000, dropout 30%**

| component | LightVeriFL+SecAgg | LightVeriAgg (alpha 0 bit) | LightVeriAgg (alpha 64 bit) |
|---|---|---|---|
| model masks (secure aggregation), client | 2.8 ms | 5.9 ms | 6.1 ms |
| verification masks: generate + encode, client | 381.2 ms | 369.5 ms | 378.2 ms |
| hash h_i, client | 366.7 ms | 358.6 ms | 365.9 ms |
| commitment + masking of h_i / r_i, client | 1.6 ms | 2.9 ms | 2.9 ms |
| V.0 aggregate masks + verify commitments, client | 805 us | 3.9 ms | 4.0 ms |
| V.1 batch check (per batch), client | 642.3 ms | 641.4 ms | 4.29 s |
| model recovery, server | 10.2 ms | 7.1 ms | 7.2 ms |
| V.0 decode + unmask, server | 55.0 ms | 21.2 ms | 22.4 ms |

**N = 30, d = 10000, dropout 50%**

| component | LightVeriFL+SecAgg | LightVeriAgg (alpha 0 bit) | LightVeriAgg (alpha 64 bit) |
|---|---|---|---|
| model masks (secure aggregation), client | 2.7 ms | 6.0 ms | 6.0 ms |
| verification masks: generate + encode, client | 379.5 ms | 392.2 ms | 379.0 ms |
| hash h_i, client | 380.1 ms | 410.2 ms | 371.8 ms |
| commitment + masking of h_i / r_i, client | 1.6 ms | 8.9 ms | 2.9 ms |
| V.0 aggregate masks + verify commitments, client | 640 us | 3.9 ms | 4.0 ms |
| V.1 batch check (per batch), client | 656.4 ms | 651.7 ms | 4.28 s |
| model recovery, server | 8.1 ms | 6.9 ms | 6.8 ms |
| V.0 decode + unmask, server | 36.8 ms | 19.0 ms | 18.7 ms |

**N = 40, d = 10000, dropout 10%**

| component | LightVeriFL+SecAgg | LightVeriAgg (alpha 0 bit) | LightVeriAgg (alpha 64 bit) |
|---|---|---|---|
| model masks (secure aggregation), client | 4.0 ms | 9.7 ms | 9.9 ms |
| verification masks: generate + encode, client | 738.8 ms | 746.0 ms | 746.8 ms |
| hash h_i, client | 376.5 ms | 367.8 ms | 377.9 ms |
| commitment + masking of h_i / r_i, client | 1.6 ms | 2.9 ms | 3.0 ms |
| V.0 aggregate masks + verify commitments, client | 850 us | 4.2 ms | 4.6 ms |
| V.1 batch check (per batch), client | 663.3 ms | 682.5 ms | 4.60 s |
| model recovery, server | 13.7 ms | 10.5 ms | 10.4 ms |
| V.0 decode + unmask, server | 57.8 ms | 25.6 ms | 23.9 ms |

**N = 40, d = 10000, dropout 30%**

| component | LightVeriFL+SecAgg | LightVeriAgg (alpha 0 bit) | LightVeriAgg (alpha 64 bit) |
|---|---|---|---|
| model masks (secure aggregation), client | 4.0 ms | 9.0 ms | 9.5 ms |
| verification masks: generate + encode, client | 736.1 ms | 722.3 ms | 740.9 ms |
| hash h_i, client | 376.0 ms | 367.0 ms | 373.4 ms |
| commitment + masking of h_i / r_i, client | 1.6 ms | 2.9 ms | 3.0 ms |
| V.0 aggregate masks + verify commitments, client | 846 us | 4.2 ms | 4.4 ms |
| V.1 batch check (per batch), client | 685.1 ms | 680.4 ms | 4.31 s |
| model recovery, server | 13.2 ms | 10.4 ms | 12.4 ms |
| V.0 decode + unmask, server | 59.4 ms | 29.1 ms | 29.1 ms |

**N = 40, d = 10000, dropout 50%**

| component | LightVeriFL+SecAgg | LightVeriAgg (alpha 0 bit) | LightVeriAgg (alpha 64 bit) |
|---|---|---|---|
| model masks (secure aggregation), client | 3.7 ms | 10.0 ms | 9.1 ms |
| verification masks: generate + encode, client | 728.4 ms | 743.1 ms | 727.2 ms |
| hash h_i, client | 372.0 ms | 399.0 ms | 358.2 ms |
| commitment + masking of h_i / r_i, client | 1.5 ms | 7.5 ms | 2.9 ms |
| V.0 aggregate masks + verify commitments, client | 1.2 ms | 4.2 ms | 4.4 ms |
| V.1 batch check (per batch), client | 688.8 ms | 678.6 ms | 4.22 s |
| model recovery, server | 21.0 ms | 11.6 ms | 10.6 ms |
| V.0 decode + unmask, server | 68.4 ms | 29.9 ms | 30.3 ms |


## Security differences (tests in fedml_api/distributed/utils/tests.py)

| | LightVeriFL + SecAgg (paper implementation) | LightVeriAgg |
|---|---|---|
| commitment | c = noise + g + h + l (points): **not binding** | El Gamal (rG, h + rH), H hash-to-curve: binding, hiding (DDH) |
| server forges y and shifts the reconstructed hash by h(delta), noise by -h(delta) | **undetected** (`test_pedersen_variant_lets_a_server_forge_but_el_gamal_does_not`) | detected by the commitment check |
| masks of hash / commitment randomness | the same EC point z_i: server + 1 colluding user compute every h_i (`test_shared_mask_of_the_baseline_leaks_individual_hashes`) | independent EC point zh_i and integer zr_i in Z_n |
| batch check | plain sum (alpha_l = 1): +delta / -delta in two rounds of a batch is undetected | private random alpha_l (64 bit; `test_cross_round_forgery_needs_random_batch_coefficients`) |
| hash bases | alpha_j G with public alpha_j (MPI script): collisions are easy, e.g. y'_1 = y_1 + alpha_2, y'_2 = y_2 - alpha_1 (the benchmark port uses LightVeriAgg's bases) | hash-to-curve, no known discrete-log relations |

