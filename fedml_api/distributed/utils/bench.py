"""Benchmarking: communication (bits) and computation (seconds) with two one-liners.

Ported from the flcourse framework (``flcourse/bench.py``) and extended by the objects of
LightVeriAgg (elliptic curve points, big field elements).

    from fedml_api.distributed.utils.bench import measure, timer

    measure(payload, "upload/masked model", src=client_rank, dst="server")   # -> number of bits
    with timer("client: [A.2] hash h_i", who=client_rank):
        ...

* :func:`measure` returns the estimated number of bits needed to serialise *any* (nested)
  object -- numpy arrays, tensors, EC points, (big) ints, dicts, lists, ... -- and records it in
  the active :class:`Bench`.
* :func:`timer` is a context manager that records wall-clock time.

Both record into the *active* :class:`Bench` together with the current round number. If no bench
is active (e.g. in the MPI mode) they still work: measure returns the bits, timer just times.

``who`` / ``src`` / ``dst`` are ``"server"`` or a client id (int). :meth:`Bench.latency_estimate`
estimates the duration of a round *as if all clients ran concurrently*: for every label it takes
the maximum over clients (instead of the sum) and converts bits into seconds with the configured
up/down link bandwidth.

Sizes: EC points are counted as their two 256-bit coordinates (64 B, uncompressed, no SEC1 prefix;
the identity as 1 B), Python ints with the minimal number of whole bytes, integer arrays with the
smallest of 8/16/32/64 bits per entry that holds all their values (field elements of
F_{2^31-1} -> 32 bit), float arrays/tensors with their dtype.
"""
from __future__ import annotations

import dataclasses
import json
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from fastecdsa.point import Point as _ECPoint
except ImportError:  # pragma: no cover
    _ECPoint = None

_ACTIVE: Optional["Bench"] = None
EC_POINT_BITS = 2 * 256


# ----------------------------------------------------------------------------------
# bit estimation
# ----------------------------------------------------------------------------------
def _int_array_bits(arr: np.ndarray) -> int:
    if arr.size == 0:
        return 0
    lo, hi = int(arr.min()), int(arr.max())
    for bits in (8, 16, 32, 64):
        if (lo >= 0 and hi < 2 ** bits) or (lo >= -(2 ** (bits - 1)) and hi < 2 ** (bits - 1)):
            return bits
    return arr.dtype.itemsize * 8


def num_bits(obj: Any, bits_per_entry: Optional[int] = None) -> int:
    """Estimated size in bits of a serialised object (payload only, no framing)."""
    if obj is None:
        return 0
    if isinstance(obj, bool):
        return 8
    if isinstance(obj, (int, np.integer)) and not isinstance(obj, bool):
        v = int(obj)
        return 8 * max(1, math.ceil((v.bit_length() + (1 if v < 0 else 0)) / 8))
    if isinstance(obj, float):
        return 64
    if isinstance(obj, str):
        return 8 * len(obj.encode("utf-8"))
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return 8 * len(obj)
    if _ECPoint is not None and isinstance(obj, _ECPoint):
        return 8 if (obj.curve is None and obj.x == 0 and obj.y == 0) else EC_POINT_BITS
    if torch is not None and isinstance(obj, torch.Tensor):
        per = bits_per_entry if bits_per_entry is not None else obj.element_size() * 8
        return int(obj.numel()) * int(per)
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return sum(num_bits(v, bits_per_entry) for v in obj.reshape(-1))
        if bits_per_entry is not None:
            per = bits_per_entry
        elif np.issubdtype(obj.dtype, np.integer):
            per = _int_array_bits(obj)
        else:
            per = obj.dtype.itemsize * 8
        return int(obj.size) * int(per)
    if isinstance(obj, np.generic):
        return obj.dtype.itemsize * 8
    if torch is not None and isinstance(obj, torch.nn.Module):
        return sum(num_bits(t, bits_per_entry) for t in obj.state_dict().values())
    if hasattr(obj, "__num_bits__"):
        return int(obj.__num_bits__())
    if isinstance(obj, dict):
        return sum(num_bits(k) + num_bits(v, bits_per_entry) for k, v in obj.items())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return sum(num_bits(v, bits_per_entry) for v in obj)
    if dataclasses.is_dataclass(obj):
        return sum(num_bits(getattr(obj, f.name), bits_per_entry) for f in dataclasses.fields(obj))
    raise TypeError(f"measure(): don't know how to size an object of type {type(obj).__name__}")


def measure(obj: Any, label: str = "data", src: Any = None, dst: Any = None,
            bits_per_entry: Optional[int] = None, rnd: Optional[int] = None) -> int:
    """Return the estimated bits to serialise ``obj`` and record them in the active Bench.

    ``rnd`` overrides the bench's current round (parties may be in different rounds).
    """
    bits = num_bits(obj, bits_per_entry)
    if _ACTIVE is not None:
        _ACTIVE.add_comm(label, bits, src, dst, rnd)
    return bits


@contextmanager
def timer(label: str, who: Any = "server", rnd: Optional[int] = None):
    """``with timer("server: aggregate"): ...`` -- record the elapsed wall-clock time."""
    start = time.perf_counter()
    try:
        yield
    finally:
        if _ACTIVE is not None:
            _ACTIVE.add_time(label, who, time.perf_counter() - start, rnd)


def active_bench() -> Optional["Bench"]:
    return _ACTIVE


def set_round(rnd: int) -> None:
    """Tell the active bench (if any) which FL round is running."""
    if _ACTIVE is not None:
        _ACTIVE.set_round(rnd)


# ----------------------------------------------------------------------------------
# formatting helpers
# ----------------------------------------------------------------------------------
def fmt_bits(bits: float) -> str:
    """Human readable size: bits are shown as bytes (B, kB, MB, GB; 1 kB = 1000 B)."""
    b = bits / 8.0
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if abs(b) < 1000 or unit == "TB":
            return f"{b:.0f} {unit}" if unit == "B" else f"{b:.2f} {unit}"
        b /= 1000.0
    return f"{b:.2f} TB"


def fmt_seconds(s: float) -> str:
    if s < 1e-3:
        return f"{s * 1e6:.0f} us"
    if s < 1:
        return f"{s * 1e3:.1f} ms"
    if s < 120:
        return f"{s:.2f} s"
    return f"{s / 60:.1f} min"


def kind(party: Any) -> str:
    """'server', 'client' or another named party."""
    if party is None or party == "server":
        return "server"
    if isinstance(party, (int, np.integer)):
        return "client"
    if isinstance(party, str) and party.startswith("client"):
        return "client"
    return str(party)


def _table(rows: List[List[str]], header: List[str]) -> str:
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)] if rows else [len(h) for h in header]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    out = [line, "-" * len(line)]
    for row in rows:
        out.append("  ".join(str(x).ljust(w) if i == 0 else str(x).rjust(w)
                             for i, (x, w) in enumerate(zip(row, widths))))
    return "\n".join(out)


# ----------------------------------------------------------------------------------
# the recorder
# ----------------------------------------------------------------------------------
class Bench:
    """Collects communication and timing records of one experiment."""

    def __init__(self, name: str = ""):
        self.name = name
        self.round = 0
        self.comm: List[dict] = []
        self.time: List[dict] = []

    # -- lifecycle ---------------------------------------------------------------------
    def activate(self) -> "Bench":
        global _ACTIVE
        _ACTIVE = self
        return self

    def deactivate(self) -> None:
        global _ACTIVE
        if _ACTIVE is self:
            _ACTIVE = None

    def __enter__(self):
        return self.activate()

    def __exit__(self, *exc):
        self.deactivate()

    def set_round(self, rnd: int) -> None:
        self.round = int(rnd)

    # -- recording ---------------------------------------------------------------------
    def add_comm(self, label: str, bits: int, src: Any = None, dst: Any = None, rnd: Optional[int] = None) -> None:
        self.comm.append({"round": self.round if rnd is None else int(rnd), "label": label, "bits": int(bits),
                          "src": _plain(src), "dst": _plain(dst)})

    def add_time(self, label: str, who: Any, seconds: float, rnd: Optional[int] = None) -> None:
        self.time.append({"round": self.round if rnd is None else int(rnd), "label": label, "who": _plain(who),
                          "seconds": float(seconds)})

    # -- summaries ---------------------------------------------------------------------
    @property
    def num_rounds(self) -> int:
        rounds = {r["round"] for r in self.comm} | {r["round"] for r in self.time}
        return max(len(rounds), 1)

    def comm_summary(self) -> List[dict]:
        """One row per (label, direction): messages, total bits, bits per message."""
        groups: Dict[tuple, List[dict]] = defaultdict(list)
        for r in self.comm:
            groups[(r["label"], f"{kind(r['src'])}->{kind(r['dst'])}")].append(r)
        rows = []
        for (label, direction), recs in groups.items():
            total = sum(r["bits"] for r in recs)
            rounds = {r["round"] for r in recs}
            rows.append({"label": label, "direction": direction, "messages": len(recs), "bits": total,
                         "bits_per_message": total / len(recs), "bits_per_round": total / max(len(rounds), 1)})
        return sorted(rows, key=lambda r: r["label"])

    def time_summary(self) -> List[dict]:
        """One row per label: total time, and per round the *average* and *max* over parties."""
        groups: Dict[str, List[dict]] = defaultdict(list)
        for r in self.time:
            groups[r["label"]].append(r)
        rows = []
        for label, recs in groups.items():
            per_round: Dict[int, Dict[Any, float]] = defaultdict(lambda: defaultdict(float))
            for r in recs:
                per_round[r["round"]][str(r["who"])] += r["seconds"]
            avgs = [float(np.mean(list(d.values()))) for d in per_round.values()]
            maxs = [max(d.values()) for d in per_round.values()]
            kinds = {kind(r["who"]) for r in recs}
            rows.append({"label": label, "who": "/".join(sorted(kinds)), "calls": len(recs),
                         "total_s": sum(r["seconds"] for r in recs),
                         "mean_call_s": float(np.mean([r["seconds"] for r in recs])),
                         "avg_per_round_s": float(np.mean(avgs)), "parallel_per_round_s": float(np.mean(maxs))})
        return sorted(rows, key=lambda r: r["label"])

    def per_party_totals(self) -> dict:
        """Per party: bits sent / received and seconds of computation (summed over all rounds)."""
        sent, recv, secs = defaultdict(int), defaultdict(int), defaultdict(float)
        for r in self.comm:
            sent[str(r["src"])] += r["bits"]
            recv[str(r["dst"])] += r["bits"]
        for r in self.time:
            secs[str(r["who"])] += r["seconds"]
        return {"sent_bits": dict(sent), "received_bits": dict(recv), "seconds": dict(secs)}

    def latency_estimate(self, uplink_mbps: float = 10.0, downlink_mbps: float = 50.0) -> dict:
        """Estimated duration of a round if all clients ran concurrently.

        Every label is treated as one protocol phase; within a phase clients work in parallel
        (max over clients), phases are sequential (sum over labels). Communication time of a
        client = bits sent / uplink + bits received / downlink. The server has unlimited bandwidth.
        """
        up, down = uplink_mbps * 1e6, downlink_mbps * 1e6
        per_round: Dict[int, Dict[str, float]] = defaultdict(lambda: {"compute": 0.0, "comm": 0.0})

        by_round_label: Dict[tuple, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for r in self.time:
            by_round_label[(r["round"], r["label"])][str(r["who"])] += r["seconds"]
        for (rnd, _), parties in by_round_label.items():
            per_round[rnd]["compute"] += max(parties.values())

        comm_by: Dict[tuple, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for r in self.comm:
            key = (r["round"], r["label"])
            if kind(r["src"]) == "client":
                comm_by[key][f"c{r['src']}"] += r["bits"] / up
            if kind(r["dst"]) == "client":
                comm_by[key][f"c{r['dst']}"] += r["bits"] / down
        for (rnd, _), parties in comm_by.items():
            per_round[rnd]["comm"] += max(parties.values())

        if not per_round:
            return {"compute_s": 0.0, "comm_s": 0.0, "total_s": 0.0, "rounds": 0}
        comp = float(np.mean([v["compute"] for v in per_round.values()]))
        comm = float(np.mean([v["comm"] for v in per_round.values()]))
        return {"compute_s": comp, "comm_s": comm, "total_s": comp + comm, "rounds": len(per_round),
                "uplink_mbps": uplink_mbps, "downlink_mbps": downlink_mbps}

    def per_round_comm(self) -> Dict[int, int]:
        out: Dict[int, int] = defaultdict(int)
        for r in self.comm:
            out[r["round"]] += r["bits"]
        return dict(sorted(out.items()))

    # -- text report ---------------------------------------------------------------------
    def report(self, uplink_mbps: float = 10.0, downlink_mbps: float = 50.0) -> str:
        lines = []
        comm_rows = self.comm_summary()
        if comm_rows:
            total = sum(r["bits"] for r in comm_rows)
            lines.append(f"COMMUNICATION   (total {fmt_bits(total)} over {self.num_rounds} round(s))")
            rows = [[r["label"], r["direction"], r["messages"], fmt_bits(r["bits"]),
                     fmt_bits(r["bits_per_round"]), fmt_bits(r["bits_per_message"])] for r in comm_rows]
            lines.append(_table(rows, ["label", "direction", "msgs", "total", "per round", "per msg"]))
        time_rows = self.time_summary()
        if time_rows:
            lines.append("")
            lines.append("COMPUTATION   ('avg/round' = average over parties, "
                         "'parallel/round' = max over parties = concurrent clients)")
            rows = [[r["label"], r["who"], r["calls"], fmt_seconds(r["total_s"]), fmt_seconds(r["mean_call_s"]),
                     fmt_seconds(r["avg_per_round_s"]), fmt_seconds(r["parallel_per_round_s"])] for r in time_rows]
            lines.append(_table(rows, ["label", "who", "calls", "total", "per call", "avg/round", "parallel/round"]))
        if comm_rows or time_rows:
            est = self.latency_estimate(uplink_mbps, downlink_mbps)
            lines.append("")
            lines.append(f"ESTIMATED ROUND LATENCY with concurrent clients (uplink {uplink_mbps:g} Mbit/s, "
                         f"downlink {downlink_mbps:g} Mbit/s): compute {fmt_seconds(est['compute_s'])} + "
                         f"communication {fmt_seconds(est['comm_s'])} = {fmt_seconds(est['total_s'])} per round")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"name": self.name, "comm": self.comm, "time": self.time}

    def save(self, path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), default=_json_default))
        return path

    @classmethod
    def load(cls, path) -> "Bench":
        raw = json.loads(Path(path).read_text())
        bench = cls(raw.get("name", ""))
        bench.comm, bench.time = raw["comm"], raw["time"]
        return bench


def _plain(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ----------------------------------------------------------------------------------
# figures (matplotlib, PNG)
# ----------------------------------------------------------------------------------
def plot_communication(bench: Bench, path, title: str = "") -> Optional[Path]:
    """Total bytes per message type (left) and bytes per round stacked by message type (right)."""
    rows = bench.comm_summary()
    if not rows:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, 2, figsize=(15, max(3.2, 0.42 * len(rows) + 1.5)))
    colors = {"client->server": "#4C78A8", "server->client": "#F58518", "server->server": "#999999"}
    labels = [f"{r['label']}  [{r['direction']}]" for r in rows]
    vals = [r["bits"] / 8 for r in rows]
    ax = axes[0]
    bars = ax.barh(range(len(rows)), vals, color=[colors.get(r["direction"], "#72B7B2") for r in rows])
    ax.set_yticks(range(len(rows)), labels, fontsize=8)
    ax.invert_yaxis()
    if max(vals) > 50 * max(min(vals), 1):
        ax.set_xscale("log")
        ax.set_xlabel("total bytes (log scale)")
    else:
        ax.set_xlabel("total bytes")
        ax.set_xlim(0, max(vals) * 1.25)
    ax.set_title("total communication per message type")
    for b, r in zip(bars, rows):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, " " + fmt_bits(r["bits"]), va="center", fontsize=7)
    ax.grid(alpha=0.3, axis="x")

    ax = axes[1]
    per: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for rec in bench.comm:
        per[rec["label"]][rec["round"]] += rec["bits"] / 8 / 1e6
    rounds = sorted({rec["round"] for rec in bench.comm})
    bottom = np.zeros(len(rounds))
    cmap = plt.get_cmap("tab20")
    for i, (label, d) in enumerate(sorted(per.items())):
        v = np.array([d.get(rd, 0.0) for rd in rounds])
        ax.bar(rounds, v, bottom=bottom, label=label, color=cmap(i % 20))
        bottom += v
    ax.set_xlabel("round")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("MB")
    ax.set_title("communication per round")
    ax.legend(fontsize=7, ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(alpha=0.3, axis="y")
    if title:
        fig.suptitle(title, fontsize=11)
    path = Path(path)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_computation(bench: Bench, path, title: str = "") -> Optional[Path]:
    """Seconds per round and step: average over parties vs. max over parties (= concurrent clients)."""
    rows = bench.time_summary()
    if not rows:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, max(3.0, 0.45 * len(rows) + 1.2)))
    y = np.arange(len(rows))
    ax.barh(y - 0.2, [max(r["avg_per_round_s"], 1e-7) for r in rows], height=0.4, color="#BAB0AC",
            label="average over parties")
    ax.barh(y + 0.2, [max(r["parallel_per_round_s"], 1e-7) for r in rows], height=0.4, color="#4C78A8",
            label="max over parties (concurrent clients)")
    ax.set_yticks(y, [f"{r['label']} [{r['who']}]" for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("seconds per round (log scale)")
    ax.set_title(title or "computation time per round")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    path = Path(path)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
