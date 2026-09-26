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
echo $PROCESS_NUM > /proc/1/fd/1

mpirun --tag-output -x PATH -x LD_LIBRARY_PATH -np $PROCESS_NUM -hostfile ./mpi_host_file python3 -u ./main_lightveriagg.py \
  --model $MODEL \
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
  --backend "MPI" > /proc/1/fd/1


