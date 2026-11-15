"""Benchmark: LightVeriAgg (LightSecAgg + LightVeriFL with El Gamal) vs. LightVeriFL + SecAgg.

As in the experiments of the LightVeriFL paper (Fig. 3, Tables 1 and 4): one FL iteration without
model training, N users, T = floor((N-1)/2), a fraction `dropout` of the users drops in the
verification phase, amortized verification with batch size L (the batch check is measured once and
counted with 1/L per iteration), model dimension d.

* LightVeriAgg runs through the real protocol code (FedML managers, messages of message_define.py)
  in the single-process mode; a synthetic logistic regression with d parameters is trained on random
  data (training and evaluation are excluded from the times).
* LightVeriFL + SecAgg runs in ServerVerification/LightVeriFL_SecAgg_single_process.py on exactly
  the same quantized local models and with the same verification dropouts.

Both are timed with the same `timer`/`measure` instrumentation. Times are "parallel": every protocol
step counts with its slowest party (= concurrent clients), the steps are added up.

    cd fedml_experiments/distributed/lightveriagg
    python benchmark_vs_lightverifl_secagg.py --N 10 20 30 40 --dropout 0.1 0.3 0.5 --d 10000
    python benchmark_vs_lightverifl_secagg.py --N 10 --dropout 0.3 --d 10000 100000 --tag d_scaling

Output (--out): results.csv, report.md, scaling_N.png / scaling_d.png and the per-run reports.
"""
import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fedml_api.distributed.lightveriagg.LightVeriAggClientManager import LightVeriAggClientManager  # noqa: E402
from fedml_api.distributed.lightveriagg.mpc_function import verification_dropouts  # noqa: E402
from fedml_api.distributed.lightveriagg.single_process import FedML_lightveriagg_single_process  # noqa: E402
from fedml_api.distributed.lightveriagg.utils import flatten_finite_model  # noqa: E402
from fedml_api.distributed.utils.EC import get_hash_bases  # noqa: E402
from fedml_api.distributed.utils.bench import Bench, fmt_bits, fmt_seconds, kind  # noqa: E402
from fedml_api.model.linear.lr import LogisticRegression  # noqa: E402
from ServerVerification.LightVeriFL_SecAgg_single_process import LightVeriFLSecAggSimulation  # noqa: E402

P_MODEL = 2 ** 31 - 1
UPLINK_MBPS, DOWNLINK_MBPS = 10.0, 50.0                                  # for the communication-time estimate
Q_BITS = 10
NOT_PROTOCOL = ("local training", "evaluate", "update local model")      # FL work, not part of the protocols


class RecordingClient(LightVeriAggClientManager):
    """remembers the quantized local models, so that the baseline runs on exactly the same inputs"""
    uploaded = dict()

    def send_model_to_server(self, receive_id, weights, local_sample_num):
        RecordingClient.uploaded[(self.round_idx, self.client_idx)] = flatten_finite_model(self.stored_model, P_MODEL) % P_MODEL
        super().send_model_to_server(receive_id, weights, local_sample_num)


def make_args(N, T, d, rounds, dropout, alpha_bits, out_dir, seed):
    return argparse.Namespace(
        model="lr", dataset="synthetic", data_dir="", partition_method="homo", partition_alpha=0.5,
        client_num_in_total=N, client_num_per_round=N, batch_size=32, client_optimizer="sgd", backend="SP",
        lr=0.1, wd=0.0, epochs=1, comm_round=rounds, is_mobile=0, frequency_of_the_test=1, ci=0,
        targeted_number_active_clients=N, privacy_guarantee=T, prime_number=P_MODEL, precision_parameter=Q_BITS,
        veri_batch_size=rounds, veri_alpha_bits=alpha_bits, veri_dropout_rate=dropout, veri_dropout_seed=seed,
        single_process=1, results_dir=str(out_dir), bench_name="lightveriagg_a%d_p%g" % (alpha_bits, dropout),
        uplink_mbps=10.0, downlink_mbps=50.0)


def run_lightveriagg(N, T, d, rounds, dropout, alpha_bits, out_dir, seed=0, samples=32):
    assert d % 10 == 0, "synthetic logistic regression with 10 classes: d = 10 * (inputs + 1)"
    torch.manual_seed(seed)
    in_dim = d // 10 - 1
    X = torch.randn(N * samples, in_dim)
    y = (X @ torch.randn(in_dim, 10)).argmax(1)
    train_local = {i: DataLoader(TensorDataset(X[i * samples:(i + 1) * samples], y[i * samples:(i + 1) * samples]),
                                 batch_size=samples) for i in range(N)}
    test_global = DataLoader(TensorDataset(X, y), batch_size=256)
    args = make_args(N, T, d, rounds, dropout, alpha_bits, out_dir, seed)
    RecordingClient.uploaded.clear()
    result = FedML_lightveriagg_single_process(
        torch.device("cpu"), LogisticRegression(in_dim, 10), N * samples, test_global, test_global,
        {i: samples for i in range(N)}, train_local, {i: test_global for i in range(N)}, args,
        client_manager_class=RecordingClient, write_results=True)
    assert result["summary"]["all_rounds_verified"], result["summary"]
    return result, dict(RecordingClient.uploaded)


def run_baseline(N, T, d, rounds, dropout, xs, seed=0):
    sim = LightVeriFLSecAggSimulation(N, T, d, L=rounds, seed=seed)
    bench = Bench("lightverifl_secagg").activate()
    try:
        for rnd in range(rounds):
            dropped = verification_dropouts(rnd, range(N), dropout, T, seed)
            surviving = [i for i in range(N) if i not in dropped]
            sim.run_round(rnd, [xs[(rnd, i)] for i in range(N)], surviving)
    finally:
        bench.deactivate()
    log = sim.verification_log
    assert all(r["commitments_ok"] and r["aggregate_correct"] and r["hash_correct"] for r in log), log
    assert all(r["batch_ok"] in (True, None) for r in log), log
    return bench


def costs(bench, rounds, L_paper):
    """per iteration: critical-path times of the phases, busiest client, server, communication"""
    phase_round = defaultdict(lambda: defaultdict(float))
    by_round_label = defaultdict(lambda: defaultdict(float))
    client_time = defaultdict(lambda: defaultdict(float))
    server_time = defaultdict(float)
    for r in bench.time:
        if any(s in r["label"] for s in NOT_PROTOCOL):
            continue
        by_round_label[(r["round"], r["label"])][str(r["who"])] += r["seconds"]
        if kind(r["who"]) == "client":
            client_time[r["round"]][r["who"]] += r["seconds"]
        else:
            server_time[r["round"]] += r["seconds"]
    batch_checks = 0
    for (rnd, label), parties in by_round_label.items():
        phase = "A" if "[A." in label else ("V0" if "[V.0]" in label else ("V1" if "[V.1]" in label else "other"))
        phase_round[phase][rnd] += max(parties.values())
        if phase == "V1":
            batch_checks += 1
    out = {
        "A_s": sum(phase_round["A"].values()) / rounds,
        "V0_s": sum(phase_round["V0"].values()) / rounds,
        # one batch check verifies L iterations -> counted with 1/L per iteration (as in the paper)
        "V1_measured_s": sum(phase_round["V1"].values()) / max(batch_checks, 1),
    }
    out["V1_s"] = out["V1_measured_s"] / L_paper
    out["verification_s"] = out["V0_s"] + out["V1_s"]
    out["total_s"] = out["A_s"] + out["verification_s"]
    out["client_max_s"] = float(np.mean([max(v.values()) for v in client_time.values()]))
    out["server_s"] = float(np.mean(list(server_time.values()))) if server_time else 0.0
    up, down = defaultdict(int), defaultdict(int)
    server_in = server_out = 0
    for r in bench.comm:
        if "INIT_CONFIG" in r["label"]:      # one-time download of the initial model (setup, not part of an iteration)
            continue
        if kind(r["src"]) == "client":
            up[r["src"]] += r["bits"]
            server_in += r["bits"]
        if kind(r["dst"]) == "client":
            down[r["dst"]] += r["bits"]
            server_out += r["bits"]
    # communication time with concurrent clients: per message type the slowest client, types one after another
    comm_by = defaultdict(lambda: defaultdict(float))
    for r in bench.comm:
        if "INIT_CONFIG" in r["label"]:
            continue
        if kind(r["src"]) == "client":
            comm_by[(r["round"], r["label"])][r["src"]] += r["bits"] / (UPLINK_MBPS * 1e6)
        if kind(r["dst"]) == "client":
            comm_by[(r["round"], r["label"])][r["dst"]] += r["bits"] / (DOWNLINK_MBPS * 1e6)
    out["comm_s"] = sum(max(v.values()) for v in comm_by.values()) / rounds
    out["total_with_comm_s"] = out["total_s"] + out["comm_s"]
    out["client_upload_bytes"] = max(up.values()) / 8 / rounds
    out["client_download_bytes"] = max(down.values()) / 8 / rounds
    out["server_received_bytes"] = server_in / 8 / rounds
    out["server_sent_bytes"] = server_out / 8 / rounds
    return out


# ------------------------------------------------------------------------------------------------
# figures + report (from all results_*.csv in the output directory)
# ------------------------------------------------------------------------------------------------
def short(protocol):
    if protocol.startswith("LightVeriFL + SecAgg"):
        return "LightVeriFL+SecAgg"
    return "LightVeriAgg (alpha %s bit)" % protocol.split("alpha ")[1].split(" bit")[0]


COLORS = {"LightVeriFL+SecAgg": "#E45756", "LightVeriAgg (alpha 0 bit)": "#4C78A8", "LightVeriAgg (alpha 64 bit)": "#54A24B"}
STYLES = {0.1: ":", 0.3: "--", 0.5: "-"}


def read_rows(out):
    rows = []
    for path in sorted(Path(out).glob("results_*.csv")):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                r = {k: (v if k == "protocol" else float(v)) for k, v in r.items()}
                r["tag"] = path.stem[len("results_"):]
                r["short"] = short(r["protocol"])
                rows.append(r)
    return rows


def plot_N_scaling(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in rows if r["tag"] == "N_scaling"]
    if not rows:
        return None
    d = int(rows[0]["d"])
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
    panels = [("total_s", "(a) total time per iteration [s]"), ("verification_s", "(b) verification phase [s]"),
              ("A_s", "(c) aggregation phase [s]"), ("client_upload_bytes", "(d) upload per client and iteration [kB]")]
    for ax, (key, title) in zip(axes, panels):
        for proto in COLORS:
            for p in sorted({r["dropout"] for r in rows}):
                pts = sorted((r["N"], r[key] / (1e3 if "bytes" in key else 1)) for r in rows
                             if r["short"] == proto and r["dropout"] == p)
                if pts and (key != "client_upload_bytes" or p == 0.3):
                    ax.plot(*zip(*pts), STYLES.get(p, "-"), marker="o" if "SecAgg" in proto else "^", ms=4,
                            color=COLORS[proto], label="%s, %d%% dropout" % (proto, round(100 * p)))
        ax.set_xlabel("number of users N")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("LightVeriAgg vs. LightVeriFL + SecAgg, one iteration, d = %d, L = %d, T = floor((N-1)/2); "
                 "parallel clients (slowest party per step)" % (d, int(rows[0]["L"])), fontsize=10)
    fig.tight_layout()
    path = Path(out) / "scaling_N.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_d_scaling(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in rows if r["tag"].startswith("d_scaling")]
    if not rows:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    panels = [("total_s", "(a) total time per iteration [s]"), ("verification_s", "(b) verification phase [s]"),
              ("client_upload_bytes", "(c) upload per client and iteration [MB]")]
    for ax, (key, title) in zip(axes, panels):
        for proto in COLORS:
            pts = sorted((r["d"], r[key] / (1e6 if "bytes" in key else 1)) for r in rows if r["short"] == proto)
            if pts:
                ax.plot(*zip(*pts), "-o", ms=4, color=COLORS[proto], label=proto)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("model dimension d")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7)
    fig.suptitle("N = %d, %d%% dropout, L = %d" % (int(rows[0]["N"]), round(100 * rows[0]["dropout"]), int(rows[0]["L"])),
                 fontsize=10)
    fig.tight_layout()
    path = Path(out) / "scaling_d.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


SECURITY_NOTES = """## Security differences (tests in fedml_api/distributed/utils/tests.py)

| | LightVeriFL + SecAgg (paper implementation) | LightVeriAgg |
|---|---|---|
| commitment | c = noise + g + h + l (points): **not binding** | El Gamal (rG, h + rH), H hash-to-curve: binding, hiding (DDH) |
| server forges y and shifts the reconstructed hash by h(delta), noise by -h(delta) | **undetected** (`test_pedersen_variant_lets_a_server_forge_but_el_gamal_does_not`) | detected by the commitment check |
| masks of hash / commitment randomness | the same EC point z_i: server + 1 colluding user compute every h_i (`test_shared_mask_of_the_baseline_leaks_individual_hashes`) | independent EC point zh_i and integer zr_i in Z_n |
| batch check | plain sum (alpha_l = 1): +delta / -delta in two rounds of a batch is undetected | private random alpha_l (64 bit; `test_cross_round_forgery_needs_random_batch_coefficients`) |
| hash bases | alpha_j G with public alpha_j (MPI script): collisions are easy, e.g. y'_1 = y_1 + alpha_2, y'_2 = y_2 - alpha_1 (the benchmark port uses LightVeriAgg's bases) | hash-to-curve, no known discrete-log relations |
"""


def table(rows, keys):
    header = ["protocol"] + [k for k, _ in keys]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join([r["short"]] + [f(r) for _, f in keys]) + " |")
    return "\n".join(lines)


def write_report(out):
    rows = read_rows(out)
    figs = [plot_N_scaling(rows, out), plot_d_scaling(rows, out)]
    t = lambda k: (lambda r: fmt_seconds(r[k]))
    b = lambda k: (lambda r: fmt_bits(8 * r[k]))
    keys = [("N", lambda r: "%d" % r["N"]), ("d", lambda r: "%d" % r["d"]), ("dropout", lambda r: "%d%%" % round(100 * r["dropout"])),
            ("aggregation", t("A_s")), ("V.0 decommitting", t("V0_s")), ("V.1 batch check (1/L)", t("V1_s")),
            ("verification", t("verification_s")), ("total", t("total_s")),
            ("+ comm. (10/50 Mbit/s)", t("total_with_comm_s")), ("busiest client", t("client_max_s")),
            ("server", t("server_s")), ("upload/client", b("client_upload_bytes")),
            ("download/client", b("client_download_bytes")), ("server in", b("server_received_bytes"))]
    lines = ["# LightVeriAgg vs. LightVeriFL + SecAgg", "",
             "One FL iteration (training excluded), times per iteration with parallel clients (every protocol step "
             "counts with its slowest party), V.1 measured once and counted with 1/L. Communication per iteration; "
             "'+ comm.' adds the transfer time of every message type for the slowest client at 10 Mbit/s up / "
             "50 Mbit/s down (server bandwidth unlimited). "
             "See benchmark_vs_lightverifl_secagg.py.", ""]
    for tag in sorted({r["tag"] for r in rows}):
        sub = sorted([r for r in rows if r["tag"] == tag], key=lambda r: (r["d"], r["N"], r["dropout"], r["short"]))
        lines += ["## %s" % tag, "", table(sub, keys), ""]
    lines += ["figures: " + ", ".join(p.name for p in figs if p is not None), "", SECURITY_NOTES]
    (Path(out) / "report.md").write_text("\n".join(lines) + "\n")
    return Path(out) / "report.md"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, nargs="+", default=[10, 20, 30, 40])
    ap.add_argument("--dropout", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    ap.add_argument("--d", type=int, nargs="+", default=[10000])
    ap.add_argument("--L", type=int, default=10, help="batch size of the amortized verification (paper: 10)")
    ap.add_argument("--rounds", type=int, default=1, help="measured iterations per configuration")
    ap.add_argument("--alpha_bits", type=int, nargs="+", default=[0, 64],
                    help="LightVeriAgg batch-check coefficients (0: plain sum as the baseline, 64: random)")
    ap.add_argument("--out", type=str, default="./results/benchmark_vs_lightverifl_secagg")
    ap.add_argument("--tag", type=str, default="N_scaling")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report_only", action="store_true", help="only redraw figures / report.md from the csv files")
    a = ap.parse_args()
    if a.report_only:
        print("report:", write_report(a.out))
        return
    logging.basicConfig(level=logging.WARNING)
    out = Path(a.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    rows = []
    csv_path = out / ("results_%s.csv" % a.tag)
    for d in a.d:
        t0 = time.time()
        get_hash_bases(d)                     # public setup, excluded from all timings
        print("hash bases for d=%d: %.1f s (setup)" % (d, time.time() - t0), flush=True)
        for N in a.N:
            T = (N - 1) // 2
            for p in a.dropout:
                base = None
                for alpha_bits in a.alpha_bits:
                    t0 = time.time()
                    result, xs = run_lightveriagg(N, T, d, a.rounds, p, alpha_bits, out / "runs", a.seed)
                    row = dict(protocol="LightVeriAgg (LightSecAgg + LightVeriFL, El Gamal, alpha %d bit)" % alpha_bits,
                               N=N, T=T, d=d, dropout=p, dropped=len(verification_dropouts(0, range(N), p, T, a.seed)),
                               L=a.L, **costs(result["bench"], a.rounds, a.L))
                    rows.append(row)
                    print("%-70s N=%3d p=%.1f d=%7d  A %8s  V0 %8s  V1 %8s  total %8s  up %10s  (%.0f s)" % (
                        row["protocol"], N, p, d, fmt_seconds(row["A_s"]), fmt_seconds(row["V0_s"]),
                        fmt_seconds(row["V1_s"]), fmt_seconds(row["total_s"]), fmt_bits(8 * row["client_upload_bytes"]),
                        time.time() - t0), flush=True)
                    if base is None:
                        t0 = time.time()
                        bench = run_baseline(N, T, d, a.rounds, p, xs, a.seed)
                        bench.save(out / "runs" / ("lightverifl_secagg_N%d_d%d_p%g.json" % (N, d, p)))
                        base = dict(protocol="LightVeriFL + SecAgg (baseline)", N=N, T=T, d=d, dropout=p,
                                    dropped=row["dropped"], L=a.L, **costs(bench, a.rounds, a.L))
                        rows.append(base)
                        print("%-70s N=%3d p=%.1f d=%7d  A %8s  V0 %8s  V1 %8s  total %8s  up %10s  (%.0f s)" % (
                            base["protocol"], N, p, d, fmt_seconds(base["A_s"]), fmt_seconds(base["V0_s"]),
                            fmt_seconds(base["V1_s"]), fmt_seconds(base["total_s"]),
                            fmt_bits(8 * base["client_upload_bytes"]), time.time() - t0), flush=True)
                    with open(csv_path, "w", newline="") as fh:
                        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                        writer.writeheader()
                        writer.writerows(rows)
    print("results:", csv_path)
    print("report:", write_report(out))


if __name__ == "__main__":
    main()
