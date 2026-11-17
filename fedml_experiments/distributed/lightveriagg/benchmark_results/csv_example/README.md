CSV logs of six short example runs (cnn6k, LR=0.05, 1 epoch, 2 rounds, L=2, T=1, N in {3, 5, 7},
verification dropout 0 % / 30 %), made with

    for N in 3 5 7; do for p in 0.0 0.3; do
      CLIENT_NUM=$N PRIVACY_GUARANTEE=1 VERI_DROPOUT_RATE=$p MODEL=cnn6k LR=0.05 EPOCH=1 ROUND=2 \
        VERI_BATCH_SIZE=2 SINGLE_PROCESS=1 RESULTS_DIR=./results/example bash run_lightsecagg_distributed_pytorch.sh
    done; done

and plotted with

    python plot_runs.py benchmark_results/csv_example/* --y time_protocol_s
    python plot_runs.py benchmark_results/csv_example/* --y comm_offline_bits

Only 2 rounds per run, so the times are noisy; the runs only show the format.
