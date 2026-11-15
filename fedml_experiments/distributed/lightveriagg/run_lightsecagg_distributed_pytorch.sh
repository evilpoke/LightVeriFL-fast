#!/usr/bin/env bash


CLIENT_NUM=5
WORKER_NUM=5
MODEL=lr
DISTRIBUTION=homo
ROUND=100
EPOCH=2
BATCH_SIZE=64
LR=0.001
DATASET=cifar10
DATA_DIR=./../../../data/cifar10
CLIENT_OPTIMIZER=sgd
CI=0

# verification (LightVeriFL, El Gamal commitment)
VERI_BATCH_SIZE=10        # L: the aggregations of the last L rounds are verified at once (amortized)
VERI_ALPHA_BITS=64        # bits of the private random coefficients of the batch check (0: plain sum)
VERI_DROPOUT_RATE=0.0     # simulated fraction of users that drop during the verification phase

# 1: run the server and all clients in ONE process (single thread, no MPI) and benchmark
#    communication (size of every virtually transmitted object) and computation (timers).
#    Results (report.txt, bench.json, metrics.json, communication.png, computation.png) go to RESULTS_DIR.
SINGLE_PROCESS=0
RESULTS_DIR=./results



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

# log to the container's stdout (docker); fall back to this shell's stdout where that is not possible
OUT=/proc/1/fd/1
if ! (echo -n > $OUT) 2>/dev/null; then OUT=/dev/stdout; fi
echo $PROCESS_NUM > $OUT

ARGS="--model $MODEL \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --partition_method $DISTRIBUTION  \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
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
