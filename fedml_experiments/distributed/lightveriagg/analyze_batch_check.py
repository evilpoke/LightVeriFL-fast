"""Does the batch check (V.1) get more expensive over the rounds? -- trend analysis of runs.

    python analyze_batch_check.py <run dir> [<run dir> ...]

(run dir: output of the single-process mode with bench.json and metrics.json, e.g. the runs in
benchmark_results/). For every run it compares, relative to the beginning of the run,
* the batch-check time (mean over the clients, every L rounds),
* the local training time (identical work every round: a pure measure of the machine's speed),
* the hash h_i of the local model (every round),
* the machine-independent work of the hashes: the sum of the bit lengths of all scalars that are
  multiplied with the hash bases (a scalar multiplication costs ~ the bit length of the scalar),
writes batch_check_trend.png into the run dir and prints the linear trends.
With several runs (same setup) it also draws their batch-check and local-training times into one
figure, batch_check_runs.png next to the first run dir: is a difference real or the machine?
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def per_round_mean(bench, label_part):
    per = defaultdict(list)
    for r in bench["time"]:
        if label_part in r["label"]:
            per[r["round"]].append(r["seconds"])
    return {k: float(np.mean(v)) for k, v in sorted(per.items())}


def per_round_record_mean(verification, key):
    per = defaultdict(list)
    for rec in verification:
        if rec.get(key) is not None:
            per[rec["round"]].append(rec[key])
    return {k: float(np.mean(v)) for k, v in sorted(per.items())}


def trend(series):
    """relative change per 100 rounds of a linear fit (in % of the fitted start value)"""
    if len(series) < 2:
        return float("nan")
    x = np.array(list(series.keys()), dtype=float)
    y = np.array(list(series.values()), dtype=float)
    slope, icpt = np.polyfit(x, y, 1)
    return 100.0 * slope * 100 / (icpt + slope * x[0])


def relative(series, n_first):
    keys = list(series)
    base = np.mean([series[k] for k in keys[:n_first]])
    return keys, [series[k] / base for k in keys]


def load(run_dir):
    run_dir = Path(run_dir)
    bench = json.loads((run_dir / "bench.json").read_text())
    verification = json.loads((run_dir / "metrics.json").read_text())["verification"]
    return bench, verification


def compare_runs(run_dirs):
    """batch check and local training of several runs in one figure (absolute seconds)"""
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for run_dir, color in zip(run_dirs, colors):
        bench, _ = load(run_dir)
        for ax, label, marker in [(axes[0], "[V.1]", "o"), (axes[1], "local training", None)]:
            series = per_round_mean(bench, label)
            ax.plot(list(series), list(series.values()), color=color, marker=marker, ms=6, lw=1.5,
                    label=Path(run_dir).name)
    axes[0].set_title("(a) batch check V.1 (every L rounds)")
    axes[1].set_title("(b) local training (identical work every round = speed of the machine)")
    for ax in axes:
        ax.set_xlabel("round")
        ax.set_ylabel("seconds per client (mean over the clients)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(run_dirs[0]).parent / "batch_check_runs.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def analyze(run_dir):
    run_dir = Path(run_dir)
    bench, verification = load(run_dir)
    training = per_round_mean(bench, "local training")
    hashing = per_round_mean(bench, "hash h_i")
    batch = per_round_mean(bench, "[V.1]")
    batch_bits = per_round_record_mean(verification, "batch_scalar_bits")
    hash_bits = per_round_record_mean(verification, "hash_scalar_bits")

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    ax = axes[0]
    for series, label, style, n_first in [(training, "local training (same work every round)", "-", 10),
                                          (hashing, "hash h_i (every round)", "-", 10),
                                          (batch, "batch check V.1 (every L rounds)", "-o", 1)]:
        if series:
            xs, ys = relative(series, n_first)
            ax.plot(xs, ys, style, ms=4, lw=1.2, label=label)
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_xlabel("round")
    ax.set_ylabel("time relative to the start of the run")
    ax.set_title("(a) measured times")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    if batch_bits or hash_bits:
        for series, label, style, n_first in [(hash_bits, "hash h_i: sum of scalar bits", "-", 10),
                                              (batch_bits, "batch check: sum of scalar bits", "-o", 1)]:
            if series:
                xs, ys = relative(series, n_first)
                ax.plot(xs, ys, style, ms=4, lw=1.2, label=label)
        if batch and batch_bits:
            per_bit = {k: batch[k] / batch_bits[k] for k in batch if k in batch_bits}
            xs, ys = relative(per_bit, 1)
            ax.plot(xs, ys, "--s", ms=4, lw=1, label="batch check: seconds per scalar bit (machine speed)")
        ax.axhline(1.0, color="gray", lw=0.8)
        ax.set_title("(b) machine-independent work of the hashes")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no scalar-bit records in metrics.json\n(run recorded before they were added)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(b) machine-independent work of the hashes")
    ax.set_xlabel("round")
    ax.set_ylabel("relative to the start of the run")
    ax.grid(alpha=0.3)
    fig.suptitle("Batch-check cost over the rounds: %s" % run_dir.name, fontsize=10)
    fig.tight_layout()
    out = run_dir / "batch_check_trend.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)

    print("linear trend per 100 rounds (relative to the start):")
    for name, series in [("local training", training), ("hash h_i", hashing), ("batch check", batch),
                         ("hash h_i scalar bits", hash_bits), ("batch-check scalar bits", batch_bits)]:
        if series:
            print("  %-26s %+6.1f %%   (%s)" % (name, trend(series), ", ".join("%.4g" % v for v in list(series.values())[:3]) + ", ..."))
    if batch:
        print("batch check per batch [s]:", ", ".join("%.2f" % v for v in batch.values()))
    if batch_bits:
        print("batch-check scalar bits per batch (mean over clients):", ", ".join("%.0f" % v for v in batch_bits.values()))
    print("figure:", out)


def main(run_dirs):
    for run_dir in run_dirs:
        print("== %s" % run_dir)
        analyze(run_dir)
    if len(run_dirs) > 1:
        print("comparison of the runs:", compare_runs(run_dirs))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
