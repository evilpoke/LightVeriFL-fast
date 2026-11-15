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
communication.png, computation.png -- as in the flcourse framework.
"""
import copy
import json
import logging
import os
import platform
import time
from pathlib import Path

import numpy as np

from fedml_core.distributed.communication.local import LocalBus
from fedml_core.distributed.communication.message import Message
from fedml_api.distributed.utils.bench import Bench, fmt_bits, fmt_seconds, measure, plot_communication, \
    plot_computation

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
               plot_computation(bench, out / "computation.png", title)]
    latency = bench.latency_estimate(up, down)
    summary = dict(summary, output_dir=str(out), estimated_round_latency=latency)

    verification = [rec for c in clients for rec in c.verification_log]
    with open(out / "metrics.json", "w") as fh:
        json.dump({"summary": summary, "config": vars(args), "global": server.metrics, "verification": verification},
                  fh, indent=1, default=str)
    bench.save(out / "bench.json")

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
    ]
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print("\n" + text, flush=True)
    return out
