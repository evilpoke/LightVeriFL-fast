#!/usr/bin/env bash


CLIENT_NUM=${CLIENT_NUM:-5}               # N: number of clients
WORKER_NUM=${WORKER_NUM:-$CLIENT_NUM}     # clients per round (= N)
PRIVACY_GUARANTEE=${PRIVACY_GUARANTEE:-2} # T: privacy against T colluding users (T < N; paper: T = (N-1)/2)
# CIFAR-10 models (d = number of parameters, see fedml_api/model/cv/cifar10_models.py):
#   cnn6k (5,810)  cnn15k (14,666)  lr (30,730)  lenet5 (62,006)  mlp99k (98,666)
MODEL=${MODEL:-lr}
DISTRIBUTION=homo
ROUND=${ROUND:-100}
EPOCH=${EPOCH:-2}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-0.001}        # plain SGD; LR=0.05 converges much faster for all models
DATASET=cifar10
DATA_DIR=./../../../data/cifar10
CLIENT_OPTIMIZER=sgd
CI=0

# verification (LightVeriFL, El Gamal commitment)
VERI_BATCH_SIZE=${VERI_BATCH_SIZE:-10}        # L: the aggregations of the last L rounds are verified at once (amortized)
VERI_ALPHA_BITS=${VERI_ALPHA_BITS:-64}        # bits of the private random coefficients of the batch check (0: plain sum)
VERI_DROPOUT_RATE=${VERI_DROPOUT_RATE:-0.0}     # simulated fraction of users that drop during the verification phase

# 1: run the server and all clients in ONE process (single thread, no MPI): the clients run one after
#    another, nothing competes for the CPU. Benchmarks communication (size of every virtually transmitted
#    object) and computation (timers). Results (report.txt, bench.json, metrics.json, communication.png,
#    computation_per_step.png, computation_per_round.png, convergence.png and the CSV logs run.csv,
#    rounds.csv, computation.csv, communication.csv, verification.csv) go to RESULTS_DIR/<run>/.
#    Compare runs over N: python plot_runs.py RESULTS_DIR/* --y time_protocol_s
# (every setting written as X=${X:-...} can be overridden from the environment,
#  e.g. CLIENT_NUM=10 ROUND=3 SINGLE_PROCESS=1 bash run_lightsecagg_distributed_pytorch.sh)
SINGLE_PROCESS=${SINGLE_PROCESS:-0}
RESULTS_DIR=${RESULTS_DIR:-./results}



#CLIENT_NUM=$1   5  #=$1
#WORKER_NUM=$2   5
#MODEL=$3        resnet
#DISTRIBUTION=$4 homo
#ROUND=$5        100 ???
#EPOCH=$6        4
#BATCH_SIZE=$7   64
#LR=$8           0.001
#DATASET=$9      cifar10
#DATA_DIR=${10}  ./../../../data/cifar10
#CLIENT_OPTIMIZER=${11}     adam
#CI=${12}        0


PROCESS_NUM=`expr $WORKER_NUM + 1`

# log to the container's stdout (docker) or to $LOG; fall back to this shell's stdout where that is not possible
OUT=${LOG:-/proc/1/fd/1}
if ! (echo -n > $OUT) 2>/dev/null; then OUT=/dev/stdout; fi
echo $PROCESS_NUM > $OUT

ARGS="--model $MODEL \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --partition_method $DISTRIBUTION  \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --privacy_guarantee $PRIVACY_GUARANTEE \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --client_optimizer $CLIENT_OPTIMIZER \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --ci $CI \
  --veri_batch_size $VERI_BATCH_SIZE \
  --veri_alpha_bits $VERI_ALPHA_BITS \
  --veri_dropout_rate $VERI_DROPOUT_RATE \
  --results_dir $RESULTS_DIR"

if [ "$SINGLE_PROCESS" = "1" ]; then
  python3 -u ./main_lightveriagg.py $ARGS --single_process 1 --backend "SP" > $OUT
else
  mpirun --tag-output -x PATH -x LD_LIBRARY_PATH -np $PROCESS_NUM -hostfile ./mpi_host_file python3 -u ./main_lightveriagg.py \
    $ARGS \
    --backend "MPI" > $OUT
fi
