"""Plot ONE quantity over the number of clients N for two or more runs (from their CSV logs).

    python plot_runs.py <run dir> <run dir> ... --y time_protocol_s
    python plot_runs.py results/* --y comm_offline_bits --where model=lr dropout=0.3 --out offline.png

A run dir is the output of one run with SINGLE_PROCESS=1 (results/<run>/). It contains
    run.csv     one row: the parameters of the run (model, N, T, d, L, dropout, lr, ...) and its results
    rounds.csv  one row per round: test_acc, test_loss, time_<phase>_s, comm_<phase>_bits, ...

--y      a column of rounds.csv, averaged over the rounds (= per iteration; the batch check of every
         L-th round is thereby amortized over the L rounds), or a column of run.csv (e.g. final_test_acc).
         Phases: setup, offline, training, upload, aggregation, commitment_check, batch_check, evaluation;
         sums: verification (= commitment_check + batch_check), protocol (all except training,
         evaluation and setup), iteration (all). Examples:
             time_iteration_s      total time per iteration (incl. training and evaluation)
             time_protocol_s       total time of the protocol per iteration
             time_verification_s   verification phase
             time_<phase>_s        any other phase
             comm_iteration_bits   communication per iteration (all messages), comm_<phase>_bits per phase
             upload_per_client_bits / download_per_client_bits
         Times are the critical path (per step the slowest party = concurrent clients), in seconds;
         communication is shown in MB.
--where  keeps only the runs with these parameter values (e.g. --where model=lr dropout=0.3).

Every line connects the runs that agree in the GROUP_BY parameters below (N is the x-axis).
Runs with the same parameters and the same N are averaged.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# parameters (columns of run.csv) that tell the lines apart; only those that differ between the runs
# appear in the legend. (T, U and d are left out on purpose: they may change together with N or the model.)
GROUP_BY = ["model", "lr", "dropout", "L", "alpha_bits", "epochs", "batch_size", "rounds"]
COLORS = ["#0050E6", "#D80000", "#FF8C00", "#8CCBFF", "#000000"]   # colour-blind safe (see bench.py)
LINESTYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D", "v"]


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def value_of_run(run_dir, y):
    """(parameters of the run, value of y): mean over the rounds of a rounds.csv column, or a run.csv column"""
    run = read_csv(run_dir / "run.csv")[0]
    if y in run:
        return run, float(run[y])
    rounds = read_csv(run_dir / "rounds.csv")
    if y not in rounds[0]:
        raise SystemExit("unknown --y %r. Columns of rounds.csv: %s\nColumns of run.csv: %s"
                         % (y, ", ".join(rounds[0]), ", ".join(run)))
    values = [float(r[y]) for r in rounds if r[y] != ""]
    return run, sum(values) / len(values)


def same(a, b):
    """parameter values compare as numbers if possible (0.3 == 0.30), else as text"""
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def unit(y):
    if y.endswith("_bits"):
        return 1 / 8e6, "MB"
    if "acc" in y:
        return 100.0, "%"
    return 1.0, ("s" if y.endswith("_s") else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run dirs (each with run.csv and rounds.csv)")
    ap.add_argument("--y", default="time_protocol_s", help="what to plot (see above)")
    ap.add_argument("--where", nargs="*", default=[], help="keep only runs with these values, e.g. model=lr")
    ap.add_argument("--out", default=None, help="output file (default: plot_<y>.png)")
    a = ap.parse_args()
    where = dict(w.split("=", 1) for w in a.where)

    # 1. one point per run: (parameters, N, value)
    points = []
    for run_dir in map(Path, a.runs):
        if not (run_dir / "run.csv").exists():
            continue                                   # not a run dir (e.g. another file in results/)
        run, value = value_of_run(run_dir, a.y)
        if all(same(run.get(k), v) for k, v in where.items()):
            points.append((run, int(run["N"]), value))
    if not points:
        raise SystemExit("no run (with run.csv) matches")

    # 2. group the points into lines: one line per combination of the GROUP_BY parameters
    varying = [k for k in GROUP_BY if len({p[0].get(k) for p in points}) > 1]
    fixed = [k for k in GROUP_BY if k not in varying]
    lines = defaultdict(lambda: defaultdict(list))    # legend label -> N -> values
    for run, N, value in points:
        label = ", ".join("%s=%s" % (k, run.get(k)) for k in varying) or run.get("model", "runs")
        lines[label][N].append(value)

    # 3. plot
    scale, u = unit(a.y)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for i, (label, by_N) in enumerate(sorted(lines.items())):
        Ns = sorted(by_N)
        ys = [scale * sum(by_N[n]) / len(by_N[n]) for n in Ns]
        thinner = min(i, 3)                            # later lines thinner: lines on top of each other stay visible
        ax.plot(Ns, ys, color=COLORS[i % len(COLORS)], ls=LINESTYLES[i // len(COLORS) % len(LINESTYLES)],
                marker=MARKERS[i % len(MARKERS)], ms=9 - 1.5 * thinner, lw=3.5 - 0.7 * thinner, label=label)
    ax.set_xlabel("number of clients N")
    ax.set_xticks(sorted({N for _, N, _ in points}))
    per = "" if a.y in points[0][0] else " (per iteration, mean over the rounds)"
    ax.set_ylabel("%s [%s]" % (a.y, u) if u else a.y)
    ax.set_title("%s%s\n%s" % (a.y, per, ", ".join("%s=%s" % (k, points[0][0].get(k)) for k in fixed)), fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    if len(lines) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(a.out or "plot_%s.png" % a.y)
    fig.savefig(out, dpi=130)
    print("%d runs, %d line(s) -> %s" % (len(points), len(lines), out))


if __name__ == "__main__":
    main()
