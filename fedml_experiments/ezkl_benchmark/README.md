# ezkl benchmark for the CIFAR-10 models

CPU time and RAM of every [ezkl](https://github.com/zkonduit/ezkl) phase, and the accuracy lost to
quantization, for the five models of `fedml_api/model/cv/cifar10_models.py`:

| model  | layers                                                   | parameters |
|--------|----------------------------------------------------------|-----------:|
| cnn6k  | 3 x [conv3x3, ReLU, maxpool] (12, 16, 24) + GAP + FC     |      5,810 |
| cnn15k | 3 x [conv3x3, ReLU, maxpool] (16, 32, 32) + GAP + FC     |     14,666 |
| lr     | logistic regression on the 3072 pixels (sigmoid output)  |     30,730 |
| lenet5 | LeNet-5                                                  |     62,006 |
| mlp99k | FC 3072 -> 32, ReLU, FC 32 -> 10                         |     98,666 |

## Run it

```bash
cd fedml_experiments/ezkl_benchmark
pip install -r requirements.txt
./run_benchmark.sh                     # = python export_models.py && python bench_ezkl.py
```

The results are in `results/report.md` (tables) and in `summary.csv`, `phases.csv`, `accuracy.csv` and `results.json`.
The console output is saved to `bench.log`. `example_results/` holds a complete run from a 4-core / 16 GB Linux machine.

Step by step:

```bash
python export_models.py                # torch: build, train briefly (2 epochs, 10k images), export ONNX -> artifacts/
python bench_ezkl.py                   # ezkl: import the ONNX files, measure every phase -> results/
```

Useful options:

| option | meaning |
|---|---|
| `export_models.py --data synthetic` | no CIFAR-10 download (synthetic images); the default downloads CIFAR-10 once (170 MB) to `<repo>/data/cifar10`, or reuses an existing torchvision download there |
| `export_models.py --epochs 0` | keep the random initial weights |
| `bench_ezkl.py --models lr mlp99k` | only some models, in this order (default: all, by parameter count) |
| `bench_ezkl.py --repeats 3` | run quantize_input / witness / prove / verify 3 times, report the median |
| `bench_ezkl.py --threads 4` | limit ezkl to 4 threads (default: all cores) |
| `bench_ezkl.py --n-eval 100` | test images for the accuracy comparison (default 100; 0 skips it) |
| `bench_ezkl.py --calibration-target accuracy` | let ezkl choose larger scales: more precision, bigger circuit |
| `bench_ezkl.py --param-visibility hashed/public` | hide the weights but publish their Poseidon hash (see below) |
| `bench_ezkl.py --max-prove-logrows 19` | skip SRS / key setup / prove / verify for circuits above 2^19 rows (import, quantization, witness and accuracy still run). Use this if the machine has too little RAM for the big circuits, so it does not start swapping |
| `bench_ezkl.py --phase-timeout 3600` | give up on a phase after an hour |
| `bench_ezkl.py --keep-keys` | keep the proving keys (deleted by default: up to many GB each) |

## What is measured

Visibility: **model parameters private** (hidden from the verifier), **input public**, **output public**.

Each phase runs in its own new Python process, so the CPU time and peak RAM of one phase cannot be inflated
by another one:

| phase (report)      | ezkl call(s)                         | what happens |
|---------------------|--------------------------------------|---|
| import ONNX         | `gen_settings`                        | the ONNX graph is parsed and laid out as a circuit |
| quantize model      | `calibrate_settings` + `compile_circuit` | fixed-point scales are chosen on calibration images, the weights are quantized and stored in the compiled circuit |
| quantize input      | `float_to_felt` per input value       | the float image becomes field elements at the input scale (checked to equal what `gen_witness` produces) |
| generate witness    | `gen_witness`                         | the quantized model runs on the image inside the circuit; this gives the public output |
| prove               | `prove`                               | the zk-SNARK proof (halo2, KZG) |
| verify              | `verify`                              | the proof is checked against the public input and output |
| one-time setup      | `get_srs` + `setup`                   | reference string (downloaded once per circuit size, then cached) and proving / verification keys; once per model, not per proof |

Per phase: **wall** time, **CPU** time over all threads, **cores** = CPU / wall (how many cores were busy on
average), **peak RAM** of the process (Linux: `VmHWM`, reset when the phase starts; macOS: `ru_maxrss`;
Windows: peak working set) and **phase RAM** = peak RAM minus what the process held before the phase began
(Python + ezkl, about 40 MB).

The accuracy comparison runs the float ONNX model (onnxruntime) and the ezkl circuit (`gen_witness`, i.e.
exactly the values that are proven) on the same held-out test images:

- **float / ezkl accuracy**: share of images classified correctly by each version.
- **accuracy drop**: float accuracy minus ezkl accuracy, in percentage points.
- **same prediction**: share of images where ezkl chooses the same class as the float model (right or wrong).
- **mean / max abs. output error**: |ezkl output - float output| over all outputs, also in units of the
  quantization step 2^-scale.
- **tied top score**: share of images where several classes get exactly the same highest ezkl output
  (e.g. saturated sigmoids that all round to 1.0), so the circuit cannot say which class it predicts.
- **relative output error**: |error vector| / |float output vector| per image, averaged, in %.

## Things to know

- **Circuit size grows with computation, not parameter count.** A convolution weight is used at every
  pixel position, so cnn6k (5.8k parameters) needs 2^21 rows and cnn15k 2^22, while mlp99k (99k parameters)
  needs only 2^17. lr is 2^18 because of the lookup table for its sigmoid, not because of its matmul.
  Proving RAM follows the circuit size, roughly doubling per extra logrow. Measured on 4 cores / 16 GB
  (`example_results/report.md`):

  | circuit | model  | setup: peak RAM, proving key | prove: wall, peak RAM |
  |---------|--------|------------------------------|-----------------------|
  | 2^17    | mlp99k | 0.9 GB, 0.5 GB               | 8 s, 1.3 GB           |
  | 2^18    | lr     | 1.8 GB, 1.2 GB               | 18 s, 3.0 GB          |
  | 2^20    | lenet5 | 11.9 GB, 8.6 GB              | out of memory at 16 GB |

  Extrapolated: lenet5 needs about 20 GB to prove, cnn6k (2^21) about 40 GB and cnn15k (2^22) about 80 GB,
  plus disk for the proving key (about 17 GB and 35 GB). Import, quantization, witness generation and verification
  stay below 300 MB for all five models.
- **Private parameters are not bound to a particular model.** With `private`, the verifier learns
  nothing about the weights, but the proof also does not say *which* weights were used. To prove that the
  output came from a specific committed model while keeping the weights hidden, use
  `--param-visibility hashed/public` (Poseidon hash of the weights is a public output) or `polycommit` (KZG
  commitment). Both make the circuit larger.
- If a phase runs out of memory, the report says so ("killed (SIGKILL), most likely out of memory") and the
  benchmark continues with the next model.
- `--srs auto` (default) downloads ezkl's public KZG reference string and, only if that fails, generates
  an insecure local one (the report says which). This does not change the prove / verify cost.
