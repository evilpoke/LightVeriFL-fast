"""Test loss / accuracy of the global model per round for several runs in one figure.

    python compare_convergence.py <run dir> [<run dir> ...]

(run dir: output of the single-process mode, with metrics.json). Writes convergence_models.png into
the parent directory of the first run dir and prints the final accuracy of every run.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from fedml_api.distributed.utils.bench import SERIES_COLORS, _pyplot, _save  # noqa: E402

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":", "-"]


def main(run_dirs):
    plt = _pyplot()
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 4.6))
    for i, run_dir in enumerate(map(Path, run_dirs)):
        m = json.loads((run_dir / "metrics.json").read_text())
        cfg, metrics = m["config"], m["global"]
        d = re.search(r"\bd=(\d+)", (run_dir / "report.txt").read_text())  # number of parameters
        d = "{:,}".format(int(d.group(1))) if d else ""
        label = "%s (d = %s)" % (cfg["model"], d) if d else cfg["model"]
        style = dict(color=SERIES_COLORS[i % len(SERIES_COLORS)], marker=MARKERS[i % len(MARKERS)], ms=4,
                     ls=LINESTYLES[i % len(LINESTYLES)], lw=1.8, label=label)
        rounds = [r["round"] for r in metrics]
        ax_loss.plot(rounds, [r["test_loss"] for r in metrics], **style)
        ax_acc.plot(rounds, [100.0 * r["test_acc"] for r in metrics], **style)
        print("%-8s rounds %3d  final test accuracy %.1f %%  test loss %.3f" % (
            cfg["model"], len(metrics), 100.0 * metrics[-1]["test_acc"], metrics[-1]["test_loss"]))
    ax_loss.set_title("test loss (global test set)")
    ax_loss.set_ylabel("cross-entropy")
    ax_acc.set_title("test accuracy (global test set)")
    ax_acc.set_ylabel("accuracy [%]")
    for ax in (ax_loss, ax_acc):
        ax.set_xlabel("round")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    out = _save(fig, Path(run_dirs[0]).parent / "convergence_models.png",
                "LightVeriAgg, N = %d clients, %s, lr = %g, %d local epochs per round" % (
                    cfg["client_num_per_round"], cfg["client_optimizer"], cfg["lr"], cfg["epochs"]))
    print("figure:", out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
