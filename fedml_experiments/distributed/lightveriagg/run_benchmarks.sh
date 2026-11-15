#!/usr/bin/env bash
# Benchmarks of LightVeriAgg on THIS machine -- without parallelism:
#  * every run is single-process / single-thread (server and clients in one process, the clients one after
#    another, torch/numpy restricted to one thread), and
#  * the runs are executed one after another,
# so that nothing that runs independently in a real deployment competes for the CPU.
#
#   bash run_benchmarks.sh          quick:  ~25 min on a 4-core laptop/VM
#   bash run_benchmarks.sh full     as in benchmark_results/report.md (N up to 40, d up to 1M): ~1.5 h
#
# Output: results/benchmarks_<date>/
#   1_lightveriagg_cifar10/<run>/   the setup of run_lightsecagg_distributed_pytorch.sh (LR on CIFAR-10, 5 clients),
#                                   a few rounds: report.txt, communication.png, computation.png, bench.json, metrics.json
#   2_vs_lightverifl_secagg/        LightVeriAgg vs. LightVeriFL + SecAgg as in the LightVeriFL paper:
#                                   report.md, scaling_N.png, scaling_d.png, results_*.csv (+ every run in runs/)
# Redraw the figures / report of 2. from the csv files:
#   python3 benchmark_vs_lightverifl_secagg.py --report_only --out results/benchmarks_<date>/2_vs_lightverifl_secagg
set -e
cd "$(dirname "$0")"
PRESET=${1:-quick}
BENCH_OUT=${BENCH_OUT:-./results/benchmarks_$(date +%Y%m%d-%H%M%S)}
mkdir -p "$BENCH_OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONWARNINGS=ignore

if [ "$PRESET" = "full" ]; then
  ROUNDS=10; N_LIST="10 20 30 40"; D_LIST="10000 30730 100000"; D_LARGE="1000000"
else
  ROUNDS=3;  N_LIST="5 10 20";     D_LIST="10000 30730 100000"; D_LARGE=""
fi

echo "== 1. LightVeriAgg, setup of run_lightsecagg_distributed_pytorch.sh, $ROUNDS rounds, single process"
LOG=/dev/stdout ROUND=$ROUNDS VERI_BATCH_SIZE=$ROUNDS SINGLE_PROCESS=1 RESULTS_DIR="$BENCH_OUT/1_lightveriagg_cifar10" \
  bash run_lightsecagg_distributed_pytorch.sh > "$BENCH_OUT/1_lightveriagg_cifar10.log" 2>&1
grep -A 3 "^VERIFICATION" "$BENCH_OUT/1_lightveriagg_cifar10.log" || true

CMP="$BENCH_OUT/2_vs_lightverifl_secagg"
echo "== 2a. LightVeriAgg vs. LightVeriFL + SecAgg: N = $N_LIST, dropout 10/30/50 %, d = 10000"
python3 -u benchmark_vs_lightverifl_secagg.py --N $N_LIST --dropout 0.1 0.3 0.5 --d 10000 \
  --alpha_bits 0 64 --tag N_scaling --out "$CMP" 2>&1 | grep -E "^LightVeri|results:" | tee "$BENCH_OUT/2a_N_scaling.log"
echo "== 2b. LightVeriAgg vs. LightVeriFL + SecAgg: d = $D_LIST $D_LARGE, N = 10, 30 % dropout"
python3 -u benchmark_vs_lightverifl_secagg.py --N 10 --dropout 0.3 --d $D_LIST \
  --alpha_bits 0 64 --tag d_scaling --out "$CMP" 2>&1 | grep -E "^LightVeri|results:" | tee "$BENCH_OUT/2b_d_scaling.log"
if [ -n "$D_LARGE" ]; then
  python3 -u benchmark_vs_lightverifl_secagg.py --N 10 --dropout 0.3 --d $D_LARGE \
    --alpha_bits 0 --tag d_scaling_1M --out "$CMP" 2>&1 | grep -E "^LightVeri|results:" | tee -a "$BENCH_OUT/2b_d_scaling.log"
fi
python3 benchmark_vs_lightverifl_secagg.py --report_only --out "$CMP"
echo "done: $BENCH_OUT"
