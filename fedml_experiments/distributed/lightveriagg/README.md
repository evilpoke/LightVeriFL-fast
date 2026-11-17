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
  `mpi_host_file` allows 8 processes (`slots=8`), i.e. N <= 7; raise it for more clients.
* `SINGLE_PROCESS=1`: the server and all clients run in ONE process (single thread, no MPI) with the
  same protocol code; every virtually transmitted object is sized (communication) and every protocol
  step is timed (computation). `RESULTS_DIR` then contains `report.txt` (tables as in the flcourse
  framework, estimated round latency with concurrent clients), `bench.json`, `metrics.json`,
  `communication.png`, `computation_per_step.png`, `computation_per_round.png`, `convergence.png`
  (test loss / accuracy of the global model per round). The stacked
  per-round charts are colour-blind safe: five colours without a red-green pair (stark blue, red,
  orange, light blue, black), each repeated with stripes and dots, so no series relies on colour alone.
* `MODEL` (CIFAR-10, d = number of parameters, `fedml_api/model/cv/cifar10_models.py`): `cnn6k`
  (5,810), `cnn15k` (14,666), `lr` (logistic regression, 30,730, default), `lenet5` (62,006),
  `mlp99k` (98,666). No batch norm, so the state_dict holds only parameters. E.g.
  `MODEL=lenet5 LR=0.05 bash run_lightsecagg_distributed_pytorch.sh`. Plain SGD at the default
  `LR=0.001` is very slow; use `LR=0.05`. Convergence with 5 clients, `LR=0.05`, 2 local epochs,
  30 rounds (`benchmark_results/convergence_lr0.05/`, all rounds verified), final test accuracy:
  `cnn6k` 53.9 %, `cnn15k` 57.7 %, `lr` 34.7 %, `lenet5` 61.0 %, `mlp99k` 45.4 %. `lr` and `mlp99k`
  have plateaued, `lenet5` is flattening (+2.6 points in the last 10 rounds), the two GAP-CNNs still
  gain ~0.5 points per round. Compare runs with `python compare_convergence.py <run dir> ...`.
  Each run folder also has communication.png (exact), computation_per_step.png and
  computation_per_round.png; the five runs shared 4 cores, so their times are 1.3-1.5x too high (lr: training 6.1 s vs 4.6 s alone).
* `VERI_BATCH_SIZE` (L), `VERI_ALPHA_BITS` (0 = plain sum as in the LightVeriFL implementation, which
  lets a server shift a forgery between rounds), `VERI_DROPOUT_RATE` (simulated verification dropouts;
  at least T+1 users must survive).
* `CLIENT_NUM` (N, default 5), `PRIVACY_GUARANTEE` (T, default 2, T < N), `EPOCH`, `BATCH_SIZE`.
  Other LightSecAgg/LightVeriFL parameters: `--targeted_number_active_clients` (U, set to N),
  `--prime_number` (p = 2^31-1), `--precision_parameter` (quantization bits, 10).
  Every setting written as `X=${X:-...}` in the script can be set from the environment.

### CSV logs and comparing runs over N
With `SINGLE_PROCESS=1` every run also writes its records as CSV (one header line each) into its
run folder `RESULTS_DIR/<run>/` (the MPI mode measures nothing, so it writes none):

| file | rows | columns |
|---|---|---|
| `run.csv` | 1 | parameters (model, N, T, U, d, L, alpha_bits, dropout, lr, epochs, ...) and results (final_test_acc / loss, all_verified, wall clock, total communication) |
| `rounds.csv` | 1 per round | test_acc, test_loss, batch_check (0/1), `time_<phase>_s` and `comm_<phase>_bits` per phase, the sums `*_verification_*`, `*_protocol_*`, `time_iteration_s`, `comm_iteration_bits`, upload/download per client |
| `computation.csv` | 1 per timed step | round, phase, party (server or client rank), step, seconds |
| `communication.csv` | 1 per message | round, phase, message type, src, dst, bits |
| `verification.csv` | 1 per client and round | the results of the commitment and batch checks, dropouts |

Phases (from `message_define.py`): `setup` (message 1), `offline` ([A.1] masks: generate, encode,
exchange), `training` (local training, model update), `upload` ([A.2] hash, El Gamal commitment,
masking, uploads), `aggregation` ([A.3] mask aggregation, decoding, recovery of y), `commitment_check`
([V.0]), `batch_check` ([V.1], every L rounds), `evaluation` (server test). Sums: verification =
commitment_check + batch_check; protocol = offline + upload + aggregation + verification; iteration =
all phases (time) or all except setup (communication). A time is the critical path: per step the
slowest party (= clients running concurrently), steps one after another.

Several runs, one after another (each call blocks until its run has finished), then one figure:
```
for N in 5 10 20; do
  for p in 0.1 0.3; do
    CLIENT_NUM=$N PRIVACY_GUARANTEE=$(( (N-1)/2 )) VERI_DROPOUT_RATE=$p MODEL=lr LR=0.05 ROUND=20 \
      SINGLE_PROCESS=1 RESULTS_DIR=./results/sweep LOG=./results/log_N${N}_p${p}.txt \
      bash run_lightsecagg_distributed_pytorch.sh
  done
done
python plot_runs.py results/sweep/* --y time_protocol_s                # total protocol time per iteration
python plot_runs.py results/sweep/* --y time_verification_s --where dropout=0.3
python plot_runs.py results/sweep/* --y comm_offline_bits             # communication of one phase
```
`plot_runs.py` puts N on the x-axis, averages the `--y` column of `rounds.csv` over the rounds (so the
batch check is amortized over L rounds) and draws one line per combination of the parameters that
differ between the runs (model, lr, dropout, L, ...); `--where` selects a single combination.

### Architecture: one protocol, interchangeable transports
* protocol layer -- `LightVeriAggServerManager` / `LightVeriAggClientManager`: only message handlers
  (`msg_params.get(MyMessage.MSG_ARG_KEY_...)`) and `send_message(Message)`; no MPI, threads, ports.
* transport layer -- FedML's `BaseCommunicationManager` (strategy; the managers are its observers),
  chosen with `--backend`:
  * `MPI` (one process per party with send/receive threads), FedML's `GRPC` / `MQTT` (ports): deployment;
  * `SP` (`fedml_core/distributed/communication/local`): an in-process message broker (`LocalBus`,
    an event loop) that delivers one message at a time -- all parties in one thread, the clients compute
    one after another, no waiting processes drain the CPU. Messages are copied with pickle like MPI does.
  The end of a run (`barrier()`, `finish()`) is part of FedML's managers as well (MPI: barrier + abort).
* benchmarking (cross-cutting): message sizes are measured at the bus, computation with `timer()`
  (a no-op unless a `Bench` is active, i.e. in the single-process mode).

### Benchmarks on your machine
```
bash run_benchmarks.sh          # quick, ~35 min
bash run_benchmarks.sh full     # as in benchmark_results/report.md, ~2 h
```
Everything runs single-process (clients one after another, one torch/BLAS thread) and the runs one
after another. Output in `results/benchmarks_<date>/`: (1) the setup of the run script (LR on CIFAR-10,
5 clients, many more rounds than L) with `report.txt`, `communication.png`, `computation_per_step.png`,
`computation_per_round.png` (the amortized batch check makes every L-th round expensive, the other
rounds of a batch are cheap); (2) the comparison with LightVeriFL + SecAgg with `report.md`,
`scaling_N.png`, `scaling_d.png`. Example: `benchmark_results/lightveriagg_cifar10_100rounds_L10/`
(100 rounds, L = 10: ~6-9 s per round, 21-24.5 s in the rounds 9, 19, ..., 99 with the batch check,
8.5 s per round on average; all 500 commitment checks and 50 batch checks ok).

Does the batch check get more expensive over the rounds? No:
```
python analyze_batch_check.py <run dir> [<run dir> ...]   # e.g. benchmark_results/lightveriagg_cifar10_*
```
compares its time with the local training (identical work every round = the speed of the machine) and
with its machine-independent work, the sum of the bit lengths of the scalars of h(sum_l alpha_l y(l))
(recorded in metrics.json). Over 100 rounds the batch check takes 14.6-15.7 s per client without a
trend and its work is constant (~70 bits per model entry: 64-bit alpha_l + the bits of the quantized
aggregate y(l) + log2 L). It cannot grow with the rounds: only the last L rounds enter, and y(l) is in
F_p (|y| < p/2), so at most ~98 bits per entry. The rise in the 40-round run (16.3 -> 17.4 s) was the
machine: its local training slowed down as well (`benchmark_results/batch_check_runs.png`).

### Tests
```
cd ../../..   # repository root
python -m pytest -q fedml_api/distributed/utils/tests.py
```

### Benchmark against LightVeriFL + SecAgg
`benchmark_vs_lightverifl_secagg.py` (see its docstring) runs both protocols on the same quantized
local models in the single-process mode (the baseline is `ServerVerification/LightVeriFL_SecAgg_single_process.py`,
a single-process port of the paper's MPI code). Results: `benchmark_results/report.md` (+ csv, figures).

Findings (one iteration, T = floor((N-1)/2), L = 10, N <= 40, d <= 1M, see the report):
* computation: with alpha_l = 1 LightVeriAgg costs the same as LightVeriFL + SecAgg (e.g. N = 10,
  d = 100K: 3.40 s vs 3.47 s per iteration with parallel clients). Both are dominated by the shared
  LightVeriFL parts (hash of the d-dimensional model, offline EC mask encoding ~ N(T+1) point
  multiplications); LightSecAgg vs SecAgg and El Gamal vs the Pedersen variant differ by milliseconds.
  The verification time does not grow with the dropout rate (one-shot decoding in both).
* the batch check with private 64-bit coefficients alpha_l (needed against cross-round forgeries)
  costs ~8x the plain-sum batch check (scalar sizes grow from ~10 to ~75 bits): +0.4 s per iteration
  for d = 10K, +3.6 s for d = 100K (amortized over L = 10).
* communication: LightSecAgg's coded mask pieces make the per-client upload/download ~2.5x larger
  (d = 100K: 1.07 MB / 1.00 MB vs 0.40 MB / 0.40 MB).
* security: the baseline's commitment c = noise + g + h + l is not binding (a server can forge the
  aggregate undetected) and its shared mask leaks every individual hash to the server plus one colluding
  user; LightVeriAgg detects the forgery and uses independent masks (tests in fedml_api/distributed/utils/tests.py).

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