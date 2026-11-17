#!/usr/bin/env bash
# Export the five CIFAR-10 models to ONNX, then benchmark them in ezkl.
# Extra arguments go to bench_ezkl.py, e.g.:  ./run_benchmark.sh --models lr mlp99k --repeats 3
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

if [ ! -f artifacts/manifest.json ]; then
    "$PY" export_models.py            # add --data synthetic to skip the CIFAR-10 download
fi
"$PY" bench_ezkl.py "$@" 2>&1 | tee bench.log
