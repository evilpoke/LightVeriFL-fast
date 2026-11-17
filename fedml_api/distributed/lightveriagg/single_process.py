"""LightVeriAgg with the server and all clients in ONE process (no MPI) -- with benchmarking.

Switched on with ``--single_process 1`` (``SINGLE_PROCESS=1`` in
fedml_experiments/distributed/lightveriagg/run_lightsecagg_distributed_pytorch.sh).

The protocol code is exactly the one of the MPI mode (same managers, same messages of
message_define.py); only the transport is replaced by an in-memory FIFO bus (backend "SP").
While running

* every (virtually) transmitted message is sized with ``measure`` -- label = message type,
  payload = all message parameters except the header (type / sender / receiver),
* the protocol steps of clients and server are timed with ``timer`` (inside the managers).

Output (``--results_dir``, default ./results): report.txt, metrics.json, bench.json,
communication.png, computation_per_step.png, computation_per_round.png -- as in the flcourse framework --
and convergence.png (test loss / accuracy of the global model per round); everything also as CSV:
run.csv, rounds.csv, computation.csv, communication.csv, verification.csv (see write_csv).
"""
import copy
import csv
import json
import logging
import os
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from fedml_core.distributed.communication.local import LocalBus
from fedml_core.distributed.communication.message import Message
from fedml_api.distributed.utils.bench import Bench, fmt_bits, fmt_seconds, kind, measure, plot_communication, \
    plot_computation_per_round, plot_computation_per_step, STARK_BLUE, STARK_ORANGE, _pyplot, _save

from .LightVeriAggAggregator import LightVeriAggAggregator
from .LightVeriAggClientManager import LightVeriAggClientManager, VerificationError
from .LightVeriAggServerManager import LightVeriAggServerManager
from .LightVeriAggTrainer import LightVeriAggTrainer
from .MyModelTrainer import MyModelTrainer
from .message_define import MyMessage

_HEADER_KEYS = {Message.MSG_ARG_KEY_TYPE, Message.MSG_ARG_KEY_SENDER, Message.MSG_ARG_KEY_RECEIVER}
_MSG_NAMES = {v: k[len("MSG_TYPE_"):] for k, v in vars(MyMessage).items() if k.startswith("MSG_TYPE_")}


def message_label(msg_type):
    return "%2d %s" % (msg_type, _MSG_NAMES.get(msg_type, "?"))


# ---- phases of one FL iteration (protocol of message_define.py) ---------------------------------
# every timed step and every message belongs to exactly one phase
PHASES = ("setup",             # message 1: initial model / config (once, not part of an iteration)
          "offline",           # [A.1] masks z_i, zh_i, zr_i: generate, encode, exchange (messages 2, 3, 12)
          "training",          # local training and the update of the local model with y / |U_1| (FL work)
          "upload",            # [A.2] hash h_i, El Gamal commitment c_i, masking; messages 4, 13, 5, 6
          "aggregation",       # [A.3] U_1, aggregate encoded masks, decode, recover y; messages 7, 8, 9
          "commitment_check",  # [V.0] verification masks: aggregate, decode, unmask sum h / sum r, check
                               #       the commitments; messages 10, 11
          "batch_check",       # [V.1] amortized check of the aggregates (every L rounds), no messages
          "evaluation",        # server: test of the global model (FL work)
          "other")             # anything not listed here (normally empty)
PROTOCOL_PHASES = ("offline", "upload", "aggregation", "commitment_check", "batch_check")
VERIFICATION_PHASES = ("commitment_check", "batch_check")
_MESSAGE_PHASE = {
    MyMessage.MSG_TYPE_S2C_INIT_CONFIG: "setup",
    MyMessage.MSG_TYPE_C2S_SEND_ENCODED_MASK_TO_SERVER: "offline",
    MyMessage.MSG_TYPE_C2S_SEND_VERI_ENCODED_MASK_TO_SERVER: "offline",
    MyMessage.MSG_TYPE_S2C_BOTH_ENCODED_MASK_TO_CLIENT_FWD: "offline",
    MyMessage.MSG_TYPE_C2S_VERI_COMMITMENT_TO_SERVER: "upload",
    MyMessage.MSG_TYPE_S2C_VERI_COMMITMENT_TO_CLIENT_FWD: "upload",
    MyMessage.MSG_TYPE_C2S_VERI_MASKED_HASH_R: "upload",
    MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER: "upload",
    MyMessage.MSG_TYPE_S2C_SEND_TO_ACTIVE_CLIENT: "aggregation",
    MyMessage.MSG_TYPE_C2S_SEND_MASK_TO_SERVER: "aggregation",
    MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT: "aggregation",
    MyMessage.MSG_TYPE_C2S_VERI_SEND_MASK_TO_SERVER: "commitment_check",
    MyMessage.MSG_TYPE_S2C_SYNC_H_R_TO_CLIENT: "commitment_check",
}


def step_phase(label):
    """phase of a timed step (label of timer())"""
    if "local training" in label or "update local model" in label:
        return "training"
    if "evaluate" in label:
        return "evaluation"
    for tag, phase in (("[A.1]", "offline"), ("[A.2]", "upload"), ("[A.3]", "aggregation"),
                       ("[V.0]", "commitment_check"), ("[V.1]", "batch_check")):
        if tag in label:
            return phase
    return "other"


def message_phase(label):
    """phase of a message (label of message_label())"""
    return _MESSAGE_PHASE.get(int(label.split()[0]), "other")


def party(rank):
    return "server" if rank == 0 else int(rank)


def FedML_lightveriagg_single_process(
    device,
    model,
    train_data_num,
    train_data_global,
    test_data_global,
    train_data_local_num_dict,
    train_data_local_dict,
    test_data_local_dict,
    args,
    model_trainer=None,
    server_manager_class=LightVeriAggServerManager,
    client_manager_class=LightVeriAggClientManager,
    write_results=True,
):
    """Runs args.comm_round rounds with N = args.client_num_per_round clients; returns a summary dict."""
    N = int(args.client_num_per_round)
    size = N + 1
    name = getattr(args, "bench_name", "") or "lightveriagg"
    bench = Bench(name).activate()

    parties = dict()

    def on_send(params, sender, receiver):
        # the transmitted objects = the values of all message parameters except the header
        payload = [v for k, v in params.items() if k not in _HEADER_KEYS]
        rnd = getattr(parties.get(sender), "round_idx", None)
        measure(payload, message_label(params[Message.MSG_ARG_KEY_TYPE]), src=party(sender), dst=party(receiver),
                rnd=rnd)

    bus = LocalBus(on_send=on_send)

    # ---- server (rank 0) -------------------------------------------------------------------------
    server_trainer = MyModelTrainer(copy.deepcopy(model)) if model_trainer is None else model_trainer
    server_trainer.set_id(-1)
    aggregator = LightVeriAggAggregator(
        train_data_global, test_data_global, train_data_num, train_data_local_dict, test_data_local_dict,
        train_data_local_num_dict, N, device, args, server_trainer,
    )
    server = server_manager_class(args, aggregator, bus, 0, size, "SP")
    parties[0] = server

    # ---- clients (rank 1..N): every client owns its copy of the model --------------------------------
    clients = []
    for rank in range(1, size):
        client_trainer = MyModelTrainer(copy.deepcopy(model))
        client_trainer.set_id(rank - 1)
        trainer = LightVeriAggTrainer(rank - 1, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                                      train_data_num, device, args, client_trainer)
        client = client_manager_class(args, trainer, bus, rank, size, "SP")
        parties[rank] = client
        clients.append(client)

    for p in parties.values():
        p.run()  # registers the message handlers (the bus drives the delivery)

    t0 = time.perf_counter()
    error = None
    server.send_init_msg()
    try:
        bus.run()
    except VerificationError as exc:
        error = str(exc)
        logging.error("VERIFICATION FAILED: %s" % exc)
    wall = time.perf_counter() - t0
    bench.deactivate()

    verification = [rec for c in clients for rec in c.verification_log]
    summary = {
        "name": name,
        "rounds_completed": int(server.round_idx),
        "rounds_requested": int(args.comm_round),
        "verification_error": error,
        "all_rounds_verified": error is None and server.round_idx == args.comm_round
                               and all(rec.get("commitments_ok", True) and rec.get("batch_ok", True) for rec in verification),
        "wall_clock_s": wall,
        "messages_delivered": bus.delivered,
        "total_communication_bits": int(sum(r["bits"] for r in bench.comm)),
        "final": server.metrics[-1] if server.metrics else {},
    }
    result = {"summary": summary, "bench": bench, "server": server, "clients": clients}
    if write_results:
        result["output_dir"] = write_report(args, bench, summary, server, clients)
    return result


def plot_convergence(metrics, path, title=""):
    """test loss and test accuracy of the global model (server-side evaluation on the global test set)
    per round, in two panels"""
    if not metrics:
        return None
    plt = _pyplot()
    from matplotlib.ticker import MaxNLocator

    rounds = [m["round"] for m in metrics]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax_loss.plot(rounds, [m["test_loss"] for m in metrics], color=STARK_ORANGE, lw=2, marker="o", ms=4)
    ax_loss.set_title("test loss (global test set)")
    ax_loss.set_ylabel("cross-entropy")
    ax_acc.plot(rounds, [100.0 * m["test_acc"] for m in metrics], color=STARK_BLUE, lw=2, marker="o", ms=4)
    ax_acc.set_title("test accuracy (global test set), final %.1f %%" % (100.0 * metrics[-1]["test_acc"]))
    ax_acc.set_ylabel("accuracy [%]")
    for ax in (ax_loss, ax_acc):
        ax.set_xlabel("round")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
    return _save(fig, path, title)


def _write_csv(path, rows, fieldnames=None):
    fieldnames = fieldnames or list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_csv(out, args, bench, summary, server, clients):
    """CSV logs of one run (every file has one header line):

    run.csv            1 row: the parameters of the run (model, N, T, U, d, L, dropout, lr, ...) and its results
    rounds.csv         1 row per round: test accuracy / loss of the global model; per phase (PHASES) the time
                       (time_<phase>_s: per step the slowest party = concurrent clients, steps one after another)
                       and the communication (comm_<phase>_bits: all messages of the phase); sums:
                       *_verification_* = commitment_check + batch_check, *_protocol_* = the PROTOCOL_PHASES,
                       time_iteration_s = all phases, comm_iteration_bits = all phases except setup;
                       upload/download_per_client_bits (mean over the clients, without setup)
    computation.csv    every timed step: round, phase, party (server / client rank), step, seconds
    communication.csv  every message: round, phase, message, src, dst, bits
    verification.csv   every client and round: the results of the checks
    """
    N = int(args.client_num_per_round)
    d = int(server.aggregator.total_dimension)
    metrics = {m["round"]: m for m in server.metrics}
    final = server.metrics[-1] if server.metrics else {}
    _write_csv(out / "run.csv", [{
        "run": out.name, "model": args.model, "dataset": args.dataset, "N": N, "T": args.privacy_guarantee,
        "U": args.targeted_number_active_clients, "d": d, "L": getattr(args, "veri_batch_size", ""),
        "alpha_bits": getattr(args, "veri_alpha_bits", ""), "dropout": getattr(args, "veri_dropout_rate", ""),
        "dropout_seed": getattr(args, "veri_dropout_seed", ""), "lr": args.lr, "optimizer": args.client_optimizer,
        "epochs": args.epochs, "batch_size": args.batch_size, "q_bits": args.precision_parameter,
        "p": args.prime_number, "rounds": args.comm_round, "rounds_completed": summary["rounds_completed"],
        "all_verified": int(bool(summary["all_rounds_verified"])), "final_test_acc": final.get("test_acc", ""),
        "final_test_loss": final.get("test_loss", ""), "wall_clock_s": summary["wall_clock_s"],
        "messages": summary["messages_delivered"], "total_comm_bits": summary["total_communication_bits"]}])

    # per round and phase: time (critical path) and communication
    step_parties = defaultdict(lambda: defaultdict(float))
    for r in bench.time:
        step_parties[(r["round"], r["label"])][str(r["who"])] += r["seconds"]
    time_phase = defaultdict(lambda: defaultdict(float))
    for (rnd, label), parties in step_parties.items():
        time_phase[rnd][step_phase(label)] += max(parties.values())
    comm_phase = defaultdict(lambda: defaultdict(int))
    up, down = defaultdict(int), defaultdict(int)
    for r in bench.comm:
        phase = message_phase(r["label"])
        comm_phase[r["round"]][phase] += r["bits"]
        if phase != "setup":
            if kind(r["src"]) == "client":
                up[r["round"]] += r["bits"]
            if kind(r["dst"]) == "client":
                down[r["round"]] += r["bits"]
    batch_rounds = {rec["round"] for c in clients for rec in c.verification_log if "batch_ok" in rec}
    rows = []
    for rnd in sorted({k for k in list(time_phase) + list(comm_phase) + list(metrics) if k is not None}):
        m = metrics.get(rnd, {})
        row = {"round": rnd, "test_acc": m.get("test_acc", ""), "test_loss": m.get("test_loss", ""),
               "batch_check": int(rnd in batch_rounds)}
        t, c = time_phase[rnd], comm_phase[rnd]
        for ph in PHASES:
            row["time_%s_s" % ph] = t.get(ph, 0.0)
        row["time_verification_s"] = sum(t.get(ph, 0.0) for ph in VERIFICATION_PHASES)
        row["time_protocol_s"] = sum(t.get(ph, 0.0) for ph in PROTOCOL_PHASES)
        row["time_iteration_s"] = sum(t.get(ph, 0.0) for ph in PHASES)
        for ph in PHASES:
            row["comm_%s_bits" % ph] = c.get(ph, 0)
        row["comm_verification_bits"] = sum(c.get(ph, 0) for ph in VERIFICATION_PHASES)
        row["comm_protocol_bits"] = sum(c.get(ph, 0) for ph in PROTOCOL_PHASES)
        row["comm_iteration_bits"] = sum(c.get(ph, 0) for ph in PHASES if ph != "setup")
        row["upload_per_client_bits"] = up[rnd] / N
        row["download_per_client_bits"] = down[rnd] / N
        rows.append(row)
    _write_csv(out / "rounds.csv", rows)

    _write_csv(out / "computation.csv", [
        {"round": r["round"], "phase": step_phase(r["label"]), "party": r["who"], "step": r["label"],
         "seconds": r["seconds"]} for r in bench.time], ["round", "phase", "party", "step", "seconds"])
    _write_csv(out / "communication.csv", [
        {"round": r["round"], "phase": message_phase(r["label"]), "message": r["label"].strip(), "src": r["src"],
         "dst": r["dst"], "bits": r["bits"]} for r in bench.comm], ["round", "phase", "message", "src", "dst", "bits"])
    verification = [dict(rec, batch_checked_rounds=" ".join(map(str, rec["batch_checked_rounds"])))
                    if "batch_checked_rounds" in rec else dict(rec) for c in clients for rec in c.verification_log]
    _write_csv(out / "verification.csv", sorted(verification, key=lambda r: (r["round"], r["client"])))
    return ["run.csv", "rounds.csv", "computation.csv", "communication.csv", "verification.csv"]


def write_report(args, bench, summary, server, clients):
    up, down = float(getattr(args, "uplink_mbps", 10.0)), float(getattr(args, "downlink_mbps", 50.0))
    N = int(args.client_num_per_round)
    d = server.aggregator.total_dimension
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(getattr(args, "results_dir", "./results")) / ("%s_N%d_d%d_%s" % (summary["name"], N, d, stamp))
    out.mkdir(parents=True, exist_ok=True)

    title = "LightVeriAgg (LightSecAgg + LightVeriFL, El Gamal)  N=%d  T=%d  U=%d  d=%d  L=%d" % (
        N, args.privacy_guarantee, args.targeted_number_active_clients, d, getattr(args, "veri_batch_size", 0))
    figures = [plot_communication(bench, out / "communication.png", title),
               plot_computation_per_step(bench, out / "computation_per_step.png", title),
               plot_computation_per_round(bench, out / "computation_per_round.png", title),
               plot_convergence(server.metrics, out / "convergence.png", "%s  model=%s  lr=%g" % (
                   title, args.model, args.lr))]
    latency = bench.latency_estimate(up, down)
    summary = dict(summary, output_dir=str(out), estimated_round_latency=latency)

    verification = [rec for c in clients for rec in c.verification_log]
    with open(out / "metrics.json", "w") as fh:
        json.dump({"summary": summary, "config": vars(args), "global": server.metrics, "verification": verification},
                  fh, indent=1, default=str)
    bench.save(out / "bench.json")
    csv_files = write_csv(out, args, bench, summary, server, clients)

    totals = bench.per_party_totals()
    client_keys = [str(c.rank) for c in clients]
    up_bits = [totals["sent_bits"].get(k, 0) for k in client_keys]
    down_bits = [totals["received_bits"].get(k, 0) for k in client_keys]
    secs = [totals["seconds"].get(k, 0.0) for k in client_keys]
    rounds = max(summary["rounds_completed"], 1)

    lines = [
        "EXPERIMENT  %s" % title,
        "output      %s" % out,
        "machine     %s, %d CPU(s)" % (platform.processor() or platform.machine(), os.cpu_count() or 1),
        "wall clock  %s (sequential simulation of %d clients)" % (fmt_seconds(summary["wall_clock_s"]), N),
        "",
        "CONFIG      model=%s dataset=%s rounds=%d epochs=%d batch=%d lr=%g  p=%d q_bits=%d" % (
            args.model, args.dataset, args.comm_round, args.epochs, args.batch_size, args.lr, args.prime_number,
            args.precision_parameter),
        "            T=%d U=%d  veri_batch_size L=%d  alpha bits=%d  verification dropout rate=%g" % (
            args.privacy_guarantee, args.targeted_number_active_clients, getattr(args, "veri_batch_size", 0),
            getattr(args, "veri_alpha_bits", 0), getattr(args, "veri_dropout_rate", 0.0)),
        "",
        "VERIFICATION  rounds completed %d / %d, all verified: %s%s" % (
            summary["rounds_completed"], summary["rounds_requested"], summary["all_rounds_verified"],
            ("   ERROR: " + summary["verification_error"]) if summary["verification_error"] else ""),
        "  commitment checks: %d ok / %d,  batch checks: %d ok / %d,  simulated verification dropouts: %d" % (
            sum(1 for r in verification if r.get("commitments_ok")), sum(1 for r in verification if "commitments_ok" in r),
            sum(1 for r in verification if r.get("batch_ok")), sum(1 for r in verification if "batch_ok" in r),
            sum(1 for r in verification if r.get("dropped_in_verification"))),
        "",
    ]
    if server.metrics:
        lines.append("GLOBAL MODEL (server-side evaluation)")
        lines += ["  round %3d  test_acc %.4f  test_loss %.4f" % (m["round"], m["test_acc"], m["test_loss"])
                  for m in server.metrics]
        lines.append("")
    per_round = bench.per_round_compute()
    if per_round:
        lines.append("COMPUTATION PER ROUND (slowest party per step, steps one after another; * = batch check V.1)")
        for rnd, steps in per_round.items():
            batch = any("[V.1]" in label for label in steps)
            lines.append("  round %3d%s %10s" % (rnd, "*" if batch else " ", fmt_seconds(sum(steps.values()))))
        lines.append("")
    lines += [
        "PER CLIENT (sum over %d rounds; mean / max over clients)" % rounds,
        "  upload    %s / %s   (per round %s / %s)" % (fmt_bits(np.mean(up_bits)), fmt_bits(max(up_bits)),
                                                       fmt_bits(np.mean(up_bits) / rounds), fmt_bits(max(up_bits) / rounds)),
        "  download  %s / %s   (per round %s / %s)" % (fmt_bits(np.mean(down_bits)), fmt_bits(max(down_bits)),
                                                       fmt_bits(np.mean(down_bits) / rounds), fmt_bits(max(down_bits) / rounds)),
        "  compute   %s / %s   (per round %s / %s)" % (fmt_seconds(np.mean(secs)), fmt_seconds(max(secs)),
                                                       fmt_seconds(np.mean(secs) / rounds), fmt_seconds(max(secs) / rounds)),
        "SERVER      compute %s (per round %s)" % (fmt_seconds(totals["seconds"].get("server", 0.0)),
                                                   fmt_seconds(totals["seconds"].get("server", 0.0) / rounds)),
        "",
        bench.report(up, down),
        "",
        "figures: " + ", ".join(p.name for p in figures if p is not None),
        "csv:     " + ", ".join(csv_files) + "   (plot several runs over N: python plot_runs.py <run dirs> --y ...)",
    ]
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print("\n" + text, flush=True)
    return out
