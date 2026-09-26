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