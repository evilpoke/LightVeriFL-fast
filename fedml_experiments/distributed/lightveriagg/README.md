## LightVeriAgg: LightSecAgg (aggregation) + LightVeriFL (verification, El Gamal commitment)

The protocol (message types, order, payloads) is defined in
`fedml_api/distributed/lightveriagg/message_define.py`. In short, per round:
LightSecAgg masks z_i (MDS coded, one-shot recovery of the aggregate mask), and for the verification
a hash h_i of the local model (linearly homomorphic, on P-256), an El Gamal commitment
c_i = (r_i G, h_i + r_i H) and two separate masks: an EC point zh_i for h_i and an integer zr_i in Z_n
for r_i, both coded with an (N, T+1) MDS code so that the server recovers sum h_i and sum r_i in
one shot from any T+1 users. Users check the commitments every round and the aggregate models
(amortized, every `VERI_BATCH_SIZE` rounds) with private random coefficients alpha_l.

### Install
```
apt-get install openmpi-bin libopenmpi-dev libgmp-dev
pip install -r ../../../requirements.txt        # note: fastecdsa < 4, protobuf == 3.20.3
```
CIFAR-10 is downloaded by torchvision into `DATA_DIR` (`./../../../data/cifar10`) on the first run.

### Run
All settings are at the top of `run_lightsecagg_distributed_pytorch.sh` (5 clients, logistic
regression on CIFAR-10, 100 rounds):
```
bash run_lightsecagg_distributed_pytorch.sh
```
* `SINGLE_PROCESS=0` (default): one MPI process per party (`mpirun -np WORKER_NUM+1`). As root,
  Open MPI additionally needs `export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1`.
  The run ends with FedML's usual `MPI_ABORT ... errorcode 0` after all clients verified the last round.
* `SINGLE_PROCESS=1`: the server and all clients run in ONE process (single thread, no MPI) with the
  same protocol code; every virtually transmitted object is sized (communication) and every protocol
  step is timed (computation). `RESULTS_DIR` then contains `report.txt` (tables as in the flcourse
  framework, estimated round latency with concurrent clients), `bench.json`, `metrics.json`,
  `communication.png`, `computation.png`.
* `VERI_BATCH_SIZE` (L), `VERI_ALPHA_BITS` (0 = plain sum as in the LightVeriFL implementation, which
  lets a server shift a forgery between rounds), `VERI_DROPOUT_RATE` (simulated verification dropouts;
  at least T+1 users must survive).
* LightSecAgg/LightVeriFL parameters: `--privacy_guarantee` (T, default 2),
  `--targeted_number_active_clients` (U, set to N), `--prime_number` (p = 2^31-1),
  `--precision_parameter` (quantization bits, 10).

### Tests
```
cd ../../..   # repository root
python -m pytest -q fedml_api/distributed/utils/tests.py
```

### Benchmark against LightVeriFL + SecAgg
`benchmark_vs_lightverifl_secagg.py` (see its docstring) runs both protocols on the same quantized
local models in the single-process mode (the baseline is `ServerVerification/LightVeriFL_SecAgg_single_process.py`,
a single-process port of the paper's MPI code).

## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Run Experiments
```

nohup sh run_lightsecagg_distributed_pytorch.sh 8 8 resnet56 homo 100 20 64 0.001 cifar10 ./../../../data/cifar10 adam 0 > ./lightsecagg_lr001.log 2>&1 &

nohup sh run_lightsecagg_distributed_pytorch.sh 8 8 resnet56 homo 100 20 64 0.01 cifar10 ./../../../data/cifar10 adam 0 > ./lightsecagg_lr01.log 2>&1 &

nohup sh run_lightsecagg_distributed_pytorch.sh 8 8 resnet56 homo 100 20 64 0.1 cifar10 ./../../../data/cifar10 adam 0 > ./lightsecagg_lr1.log 2>&1 &
```

Stefan Schärdginer

RUN TASK:

**reproduce** lightsegacc:  trying to get mpirun to run on a cpu correctly and stabile with a reliable kill signal


3 ways:

1. recode this to sequential -> ist i think doable [on hold]
2. just use cpu with mpirun [failed] -> see docker readme 
3. update to current OS and current libraries. I can then use the single_processor feature somehow.  [done]
   1. requires to use FedML (the current api)
   2. A lot of work

Question: what does "communication rounds" in the bash script do?
need to downgrade training to a completedly selfcontrolled NN setup!


future task maybe: upgrade to fedml recent