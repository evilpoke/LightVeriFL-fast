# ezkl benchmark: CIFAR-10 models

Machine: Intel(R) Xeon(R) Processor @ 2.30GHz, 4 logical CPUs, 15.7 GB RAM, Linux-6.18.44-fc-v553-x86_64-with-glibc2.39  
ezkl 23.0.5, Python 3.13.18, 2026-11-17 21:55:58  
Visibility: params **private**, input **public**, output **public**; calibration target `resources`; threads: all; repeats of witness/prove/verify: 3  
The SRS is downloaded once per circuit size and cached, so later models of the same size show only the time to find it.

## Circuits

| model | params | logrows | rows used | input / param scale | proving key | verification key | proof | SRS |
|---|---:|---:|---:|---|---:|---:|---:|---|
| cnn6k | 5,810 | 21 | 1,277,272 | 13 / 13 | - | - | - | - |
| cnn15k | 14,666 | 22 | 2,256,900 | 13 / 13 | - | - | - | - |
| lr | 30,730 | 18 | 31,065 | 11 / 11 | 1.22 GB | 578 KB | 475.6 KB | downloaded (ezkl public KZG setup) |
| lenet5 | 62,006 | 20 | 617,842 | 13 / 13 | 8.56 GB | 2.4 MB | - | downloaded (ezkl public KZG setup) |
| mlp99k | 98,666 | 17 | 66,020 | 13 / 13 | 528 MB | 257 KB | 478.8 KB | downloaded (ezkl public KZG setup) |

## CPU and RAM per phase

*wall* = elapsed time, *CPU* = CPU time over all threads, *cores* = CPU / wall (average number of busy cores, '-' when the phase is too short to tell), *peak RAM* = highest resident memory of the process during the phase, *phase RAM* = peak RAM minus the memory the process already held before the phase started (Python + ezkl, about 30-60 MB).

### import ONNX

| model | params | wall | CPU | cores | peak RAM | phase RAM |
|---|---:|---:|---:|---:|---:|---:|
| cnn6k | 5,810 | 625 ms | 1.57 s | 2.5 | 106 MB | 65 MB |
| cnn15k | 14,666 | 864 ms | 2.24 s | 2.6 | 127 MB | 85 MB |
| lr | 30,730 | 78 ms | 269 ms | 3.5 | 71 MB | 29 MB |
| lenet5 | 62,006 | 373 ms | 939 ms | 2.5 | 80 MB | 39 MB |
| mlp99k | 98,666 | 153 ms | 384 ms | 2.5 | 110 MB | 69 MB |

### quantize model

| model | params | wall | CPU | cores | peak RAM | phase RAM |
|---|---:|---:|---:|---:|---:|---:|
| cnn6k | 5,810 | 88.13 s | 214 s | 2.4 | 217 MB | 176 MB |
| cnn15k | 14,666 | 144 s | 359 s | 2.5 | 239 MB | 198 MB |
| lr | 30,730 | 9.81 s | 18.13 s | 1.8 | 271 MB | 230 MB |
| lenet5 | 62,006 | 43.14 s | 104 s | 2.4 | 188 MB | 147 MB |
| mlp99k | 98,666 | 11.27 s | 20.30 s | 1.8 | 216 MB | 175 MB |

### quantize input

| model | params | wall | CPU | cores | peak RAM | phase RAM |
|---|---:|---:|---:|---:|---:|---:|
| cnn6k | 5,810 | 5 ms | 17 ms | - | 44 MB | 2.4 MB |
| cnn15k | 14,666 | 6 ms | 30 ms | - | 44 MB | 3.3 MB |
| lr | 30,730 | 3 ms | 15 ms | - | 42 MB | 1.2 MB |
| lenet5 | 62,006 | 4 ms | 16 ms | - | 43 MB | 1.4 MB |
| mlp99k | 98,666 | 3 ms | 17 ms | - | 42 MB | 1.2 MB |

### generate witness

| model | params | wall | CPU | cores | peak RAM | phase RAM |
|---|---:|---:|---:|---:|---:|---:|
| cnn6k | 5,810 | 701 ms | 1.73 s | 2.5 | 105 MB | 64 MB |
| cnn15k | 14,666 | 1.06 s | 2.71 s | 2.6 | 120 MB | 79 MB |
| lr | 30,730 | 86 ms | 295 ms | 3.4 | 71 MB | 30 MB |
| lenet5 | 62,006 | 338 ms | 914 ms | 2.7 | 79 MB | 38 MB |
| mlp99k | 98,666 | 162 ms | 410 ms | 2.5 | 105 MB | 64 MB |

### prove

| model | params | wall | CPU | cores | peak RAM | phase RAM |
|---|---:|---:|---:|---:|---:|---:|
| cnn6k | 5,810 | skipped: prove: needs setup | | | | |
| cnn15k | 14,666 | skipped: prove: needs setup | | | | |
| lr | 30,730 | 17.85 s | 61.79 s | 3.5 | 2.96 GB | 2.92 GB |
| lenet5 | 62,006 | failed: prove: killed (SIGKILL), most likely out of memory | | | | |
| mlp99k | 98,666 | 8.23 s | 27.58 s | 3.4 | 1.31 GB | 1.27 GB |

### verify

| model | params | wall | CPU | cores | peak RAM | phase RAM |
|---|---:|---:|---:|---:|---:|---:|
| cnn6k | 5,810 | skipped: verify: needs prove | | | | |
| cnn15k | 14,666 | skipped: verify: needs prove | | | | |
| lr | 30,730 | 176 ms | 465 ms | 2.6 | 223 MB | 181 MB |
| lenet5 | 62,006 | skipped: verify: needs prove | | | | |
| mlp99k | 98,666 | 115 ms | 345 ms | 3.0 | 134 MB | 93 MB |

### one-time setup (SRS + keys)

| model | params | wall | CPU | cores | peak RAM | phase RAM |
|---|---:|---:|---:|---:|---:|---:|
| cnn6k | 5,810 | skipped: srs: circuit 2^21 > --max-prove-logrows 20; setup: needs srs | | | | |
| cnn15k | 14,666 | skipped: srs: circuit 2^22 > --max-prove-logrows 20; setup: needs srs | | | | |
| lr | 30,730 | 17.17 s | 44.04 s | 2.6 | 1.76 GB | 1.72 GB |
| lenet5 | 62,006 | 135 s | 357 s | 2.7 | 11.86 GB | 11.82 GB |
| mlp99k | 98,666 | 8.02 s | 20.53 s | 2.6 | 896 MB | 855 MB |

## Accuracy lost to quantization

Float = the original ONNX model run by onnxruntime; ezkl = the quantized circuit (output of gen_witness, i.e. exactly what is proven). Measured on held-out test images (cifar10, models trained 2 epochs on 10000 images).

| model | images | float accuracy | ezkl accuracy | accuracy drop | same prediction | tied top score | mean abs. output error | max abs. output error | relative output error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cnn6k | 100 | 32.0 % | 32.0 % | +0.0 pp | 100.0 % (0 changed) | 0.0 % | 5.04e-04 (4.1 steps) | 1.74e-03 | 0.053 % |
| cnn15k | 100 | 38.0 % | 38.0 % | +0.0 pp | 100.0 % (0 changed) | 0.0 % | 3.64e-04 (3.0 steps) | 2.05e-03 | 0.027 % |
| lr | 100 | 36.0 % | 37.0 % | -1.0 pp | 87.0 % (13 changed) | 22.0 % | 5.30e-04 (1.1 steps) | 9.74e-03 | 0.154 % |
| lenet5 | 100 | 37.0 % | 37.0 % | +0.0 pp | 100.0 % (0 changed) | 0.0 % | 6.16e-04 (5.0 steps) | 3.38e-03 | 0.057 % |
| mlp99k | 100 | 37.0 % | 37.0 % | +0.0 pp | 100.0 % (0 changed) | 0.0 % | 7.69e-04 (6.3 steps) | 4.07e-03 | 0.063 % |

- **float / ezkl accuracy**: share of test images classified correctly by each version.
- **accuracy drop**: float accuracy minus ezkl accuracy, in percentage points (positive = quantization hurt).
- **same prediction**: share of images where ezkl picks the same class as the float model, whether right or wrong. 100 % means quantization never changed a decision.
- **tied top score**: share of images where two or more classes get exactly the same highest ezkl output, so the circuit's answer is ambiguous (argmax then takes the lowest class index). Typical for sigmoid outputs that saturate: 0.99998 and 0.99999 both round to 1.0.
- **mean / max abs. output error**: |ezkl output - float output| over all 10 outputs of all images; *steps* = that error in units of the output quantization step 2^-scale.
- **relative output error**: length of the error vector / length of the float output vector, per image, averaged.
