"""Step 2 of the ezkl benchmark: CPU and RAM of every ezkl phase, and the accuracy lost to quantization.

Imports the ONNX files written by export_models.py into ezkl (https://github.com/zkonduit/ezkl) with

    model parameters  private   (hidden from the verifier)
    input             public
    output            public

and runs, for every model, each ezkl phase in its own fresh Python process so that CPU time and peak
RAM of a phase are not mixed up with any other phase:

    phase             ezkl call                         what it does
    import            gen_settings                      parse the ONNX graph, lay out the circuit
    calibrate         calibrate_settings                choose the fixed-point scales (quantization) on calibration images
    compile           compile_circuit                   quantize the weights, store the circuit
    quantize_input    float_to_felt                     turn the float input image into field elements
    witness           gen_witness                       run the quantized model inside the circuit
    srs               get_srs (or gen_srs)              fetch the KZG reference string (one-time, not per proof)
    setup             setup                             proving key + verification key (one-time per model)
    prove             prove                             create the zk proof
    verify            verify                            check the proof

"quantize model" = calibrate + compile. Afterwards the quantized circuit is run (gen_witness) on --n-eval test
images and compared with the float ONNX model (onnxruntime) to measure the accuracy lost to quantization.
If a phase fails (e.g. out of memory), only the phases that need its output are skipped.

Usage:
    python export_models.py                    # once: writes artifacts/
    python bench_ezkl.py                       # all models, results in results/
    python bench_ezkl.py --models lr mlp99k --repeats 3 --threads 4

Results: results/report.md (readable tables), summary.csv, phases.csv, accuracy.csv, results.json, logs/.
"""
import argparse
import asyncio
import csv
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from importlib import metadata

import numpy as np
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
PHASES = ["import", "calibrate", "compile", "quantize_input", "witness", "srs", "setup", "prove", "verify"]
# a phase runs only if the phases it needs succeeded (e.g. the witness does not need the keys)
DEPENDS = {"import": [], "calibrate": ["import"], "compile": ["calibrate"], "quantize_input": ["calibrate"],
           "witness": ["compile"], "srs": ["calibrate"], "setup": ["compile", "srs"], "prove": ["witness", "setup"],
           "verify": ["prove"]}
PROVING = {"srs", "setup", "prove", "verify"}          # the memory-hungry part, see --max-prove-logrows
REPEATABLE = {"quantize_input", "witness", "prove", "verify"}
# the phases asked for, each made of one or more ezkl phases (wall/CPU are summed, peak RAM is the max)
USER_PHASES = [
    ("import ONNX", ["import"]),
    ("quantize model", ["calibrate", "compile"]),
    ("quantize input", ["quantize_input"]),
    ("generate witness", ["witness"]),
    ("prove", ["prove"]),
    ("verify", ["verify"]),
    ("one-time setup (SRS + keys)", ["srs", "setup"]),
]
MB = 1024 * 1024


# ----------------------------------------------------------------------------------------------- measuring

def _peak_rss_bytes():
    """High-water mark of this process' resident memory."""
    if sys.platform.startswith("linux"):
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    if sys.platform == "darwin":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss          # bytes on macOS
    mem = psutil.Process().memory_info()
    return getattr(mem, "peak_wset", mem.rss)                               # Windows


def _reset_peak_rss():
    """Linux only: restart the high-water mark at the current RSS, so the import of ezkl does not count."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def _call(fn, *args, **kwargs):
    """ezkl functions are partly async (get_srs, calibrate_settings, ...), depending on the version."""
    async def go():
        r = fn(*args, **kwargs)
        return (await r) if inspect.isawaitable(r) else r
    return asyncio.run(go())


# ----------------------------------------------------------------------------------------------- phases
# Each runs in a worker process; cfg holds the paths of the model's files.

def _run_args(ezkl, cfg):
    ra = ezkl.PyRunArgs()
    ra.param_visibility = cfg["param_visibility"]
    ra.input_visibility = cfg["input_visibility"]
    ra.output_visibility = cfg["output_visibility"]
    if cfg.get("num_inner_cols"):
        ra.num_inner_cols = cfg["num_inner_cols"]
    return ra


def _needs_kzg_in_witness(cfg):
    return any("polycommit" in cfg[k] or "kzg" in cfg[k].lower()
               for k in ("param_visibility", "input_visibility", "output_visibility"))


def phase_import(ezkl, cfg):
    assert _call(ezkl.gen_settings, cfg["onnx"], cfg["settings"], py_run_args=_run_args(ezkl, cfg))
    return {}


def phase_calibrate(ezkl, cfg):
    kwargs = {}
    if cfg.get("max_logrows"):
        kwargs["max_logrows"] = cfg["max_logrows"]
    assert _call(ezkl.calibrate_settings, cfg["calib_json"], cfg["onnx"], cfg["settings"],
                 cfg["calibration_target"], **kwargs)
    return {}


def phase_compile(ezkl, cfg):
    assert _call(ezkl.compile_circuit, cfg["onnx"], cfg["compiled"], cfg["settings"])
    return {}


def phase_srs(ezkl, cfg):
    logrows = json.load(open(cfg["settings"]))["run_args"]["logrows"]
    srs = cfg["srs"]
    if os.path.exists(srs):
        return {"srs_source": "cached", "logrows": logrows}
    mode = cfg["srs_mode"]
    if mode in ("download", "auto"):
        try:
            assert _call(ezkl.get_srs, cfg["settings"], srs_path=srs + ".part")
            os.replace(srs + ".part", srs)
            return {"srs_source": "downloaded (ezkl public KZG setup)", "logrows": logrows}
        except Exception as e:  # noqa: BLE001
            if mode == "download":
                raise
            print(f"get_srs failed ({e}); generating a local SRS instead", flush=True)
    _call(ezkl.gen_srs, srs, logrows)
    return {"srs_source": "generated locally (insecure, benchmark only)", "logrows": logrows}


def phase_setup(ezkl, cfg):
    assert _call(ezkl.setup, cfg["compiled"], cfg["vk"], cfg["pk"], srs_path=cfg["srs"])
    return {"pk_mb": os.path.getsize(cfg["pk"]) / MB, "vk_mb": os.path.getsize(cfg["vk"]) / MB}


def phase_quantize_input(ezkl, cfg):
    scale = json.load(open(cfg["settings"]))["model_input_scales"][0]
    values = json.load(open(cfg["input_json"]))["input_data"][0]
    felts = [ezkl.float_to_felt(float(v), scale, ezkl.PyInputType.F32) for v in values]
    with open(cfg["input_quantized"], "w") as f:
        json.dump(felts, f)
    return {"input_elements": len(felts), "input_scale": scale}


def phase_witness(ezkl, cfg):
    kw = {"vk_path": cfg["vk"], "srs_path": cfg["srs"]} if _needs_kzg_in_witness(cfg) else {}
    w = _call(ezkl.gen_witness, cfg["input_json"], cfg["compiled"], cfg["witness"], **kw)
    if w is None:
        w = json.load(open(cfg["witness"]))
    extra = {}
    if os.path.exists(cfg["input_quantized"]):     # sanity check of the quantize_input phase (not timed)
        extra["quantized_input_matches_witness"] = json.load(open(cfg["input_quantized"])) == w["inputs"][0]
    return extra


def phase_prove(ezkl, cfg):
    _call(ezkl.prove, cfg["witness"], cfg["compiled"], cfg["pk"], cfg["proof"], srs_path=cfg["srs"])
    return {"proof_kb": os.path.getsize(cfg["proof"]) / 1024}


def phase_verify(ezkl, cfg):
    ok = _call(ezkl.verify, cfg["proof"], cfg["settings"], cfg["vk"], srs_path=cfg["srs"])
    if not ok:
        raise RuntimeError("proof did not verify")
    return {"verified": True}


def phase_accuracy(ezkl, cfg):
    """Not a benchmarked phase: float ONNX vs ezkl circuit outputs on the evaluation images."""
    import onnxruntime as ort
    data = np.load(cfg["dataset"])
    x, y = data["eval_x"][:cfg["n_eval"]], data["eval_y"][:cfg["n_eval"]]
    settings = json.load(open(cfg["settings"]))
    out_scale = settings["model_output_scales"][0]
    sess = ort.InferenceSession(cfg["onnx"], providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    tmp_in, tmp_w = cfg["witness"] + ".acc_in.json", cfg["witness"] + ".acc_w.json"
    kw = {"vk_path": cfg["vk"], "srs_path": cfg["srs"]} if _needs_kzg_in_witness(cfg) else {}
    float_out, ezkl_out = [], []
    t0 = time.time()
    for i in range(len(x)):
        xi = x[i:i + 1].astype(np.float32)
        float_out.append(sess.run(None, {in_name: xi})[0].reshape(-1))
        with open(tmp_in, "w") as f:
            json.dump({"input_data": [xi.reshape(-1).tolist()]}, f)
        w = _call(ezkl.gen_witness, tmp_in, cfg["compiled"], tmp_w, **kw)
        if w is None:
            w = json.load(open(tmp_w))
        ezkl_out.append([ezkl.felt_to_float(v, out_scale) for v in w["outputs"][0]])
        if (i + 1) % max(1, len(x) // 10) == 0:
            print(f"  accuracy: {i + 1}/{len(x)} images ({time.time() - t0:.0f}s)", flush=True)
    for p in (tmp_in, tmp_w):
        if os.path.exists(p):
            os.remove(p)
    np.savez(cfg["accuracy_npz"], float_out=np.array(float_out), ezkl_out=np.array(ezkl_out), labels=y,
             out_scale=out_scale)
    return {"n_eval": int(len(x))}


PHASE_FNS = {name: globals()["phase_" + name] for name in PHASES + ["accuracy"]}


def worker_main(phase, cfg_path, result_path):
    cfg = json.load(open(cfg_path))
    import ezkl
    proc = psutil.Process()
    peak_reset = _reset_peak_rss()
    rss_before = proc.memory_info().rss
    cpu0, pt0, t0 = proc.cpu_times(), time.process_time(), time.perf_counter()
    extra = PHASE_FNS[phase](ezkl, cfg)
    wall = time.perf_counter() - t0
    cpu = time.process_time() - pt0         # all threads; finer-grained than psutil's user/system split below
    cpu1 = proc.cpu_times()
    peak = _peak_rss_bytes()
    res = {"status": "ok", "wall_s": wall, "cpu_s": cpu,
           "cpu_user_s": cpu1.user - cpu0.user, "cpu_sys_s": cpu1.system - cpu0.system,
           "avg_cores": cpu / wall if wall > 0 else 0.0,
           "peak_rss_mb": peak / MB, "rss_before_mb": rss_before / MB,
           "phase_ram_mb": max(0.0, (peak - rss_before) / MB), "peak_is_phase_only": peak_reset, **extra}
    with open(result_path, "w") as f:
        json.dump(res, f)


# ----------------------------------------------------------------------------------------------- driver

def run_phase(phase, cfg, log_dir, args, run=0):
    """Run one phase in a fresh process and return its measurements."""
    tag = phase if run == 0 else f"{phase}.{run}"
    cfg_path = os.path.join(log_dir, f"{tag}.cfg.json")
    res_path = os.path.join(log_dir, f"{tag}.result.json")
    log_path = os.path.join(log_dir, f"{tag}.log")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=1)
    if os.path.exists(res_path):
        os.remove(res_path)
    env = dict(os.environ)
    if args.threads:
        env["RAYON_NUM_THREADS"] = str(args.threads)
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        try:
            rc = subprocess.run([sys.executable, os.path.abspath(__file__), "_worker", phase, cfg_path, res_path],
                                stdout=log, stderr=subprocess.STDOUT, env=env,
                                timeout=args.phase_timeout or None).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    if rc == 0 and os.path.exists(res_path):
        return json.load(open(res_path))
    if rc == "timeout":
        why = f"timed out after {args.phase_timeout}s"
    elif isinstance(rc, int) and rc in (-9, 137):
        why = "killed (SIGKILL), most likely out of memory"
    else:
        tail = open(log_path, errors="replace").read().strip().splitlines()[-3:]
        why = f"exit code {rc}: " + " | ".join(tail)
    return {"status": "failed", "error": why, "wall_s": time.perf_counter() - t0}


def aggregate(runs):
    """Several runs of one phase: median wall and CPU time, max peak RAM."""
    ok = [r for r in runs if r["status"] == "ok"]
    if not ok:
        return dict(runs[-1])
    agg = dict(ok[-1])
    for k in ("wall_s", "cpu_s", "cpu_user_s", "cpu_sys_s", "avg_cores"):
        agg[k] = float(np.median([r[k] for r in ok]))
    for k in ("peak_rss_mb", "phase_ram_mb"):
        agg[k] = max(r[k] for r in ok)
    agg["runs"] = len(ok)
    return agg


def combine(phases, names):
    parts = [phases.get(n) for n in names]
    if any(p is None or p["status"] != "ok" for p in parts):
        bad = [(n, p) for n, p in zip(names, parts) if p is not None and p["status"] != "ok"]
        status = "failed" if any(p["status"] == "failed" for _, p in bad) else bad[0][1]["status"] if bad else "not run"
        return {"status": status, "error": "; ".join(f"{n}: {p['error']}" for n, p in bad)}
    wall = sum(p["wall_s"] for p in parts)
    cpu = sum(p["cpu_s"] for p in parts)
    return {"status": "ok", "wall_s": wall, "cpu_s": cpu, "avg_cores": cpu / wall if wall else 0.0,
            "peak_rss_mb": max(p["peak_rss_mb"] for p in parts),
            "phase_ram_mb": max(p["phase_ram_mb"] for p in parts)}


def accuracy_metrics(npz_path):
    d = np.load(npz_path)
    f, q, y = d["float_out"], d["ezkl_out"], d["labels"]
    pf, pq = f.argmax(1), q.argmax(1)
    err = np.abs(q - f)
    step = 2.0 ** -int(d["out_scale"])
    return {
        "n_images": int(len(y)),
        "acc_float_pct": 100 * float((pf == y).mean()),
        "acc_ezkl_pct": 100 * float((pq == y).mean()),
        "acc_drop_pp": 100 * float((pf == y).mean() - (pq == y).mean()),
        "agreement_pct": 100 * float((pf == pq).mean()),
        "changed_predictions": int((pf != pq).sum()),
        # several classes share the highest quantized output (e.g. saturated sigmoids that all round to 1.0):
        # the circuit cannot tell them apart and argmax falls back to the lowest class index
        "tied_top_pct": 100 * float(((q == q.max(1, keepdims=True)).sum(1) > 1).mean()),
        "mean_abs_err": float(err.mean()),
        "max_abs_err": float(err.max()),
        "rel_err_pct": 100 * float(np.mean(np.linalg.norm(q - f, axis=1) / np.maximum(np.linalg.norm(f, axis=1), 1e-12))),
        "quant_step": step,
        "mean_abs_err_in_steps": float(err.mean() / step),
    }


def machine_info():
    cpu_model = platform.processor()
    if sys.platform.startswith("linux"):
        try:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            cpu_model = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:  # noqa: BLE001
            pass
    ver = lambda pkg: metadata.version(pkg) if _installed(pkg) else None
    return {"platform": platform.platform(), "cpu": cpu_model, "logical_cpus": psutil.cpu_count(),
            "physical_cpus": psutil.cpu_count(logical=False), "ram_gb": psutil.virtual_memory().total / 1024 ** 3,
            "python": platform.python_version(), "ezkl": ver("ezkl"), "onnxruntime": ver("onnxruntime"),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")}


def _installed(pkg):
    try:
        metadata.version(pkg)
        return True
    except metadata.PackageNotFoundError:
        return False


def fmt_mb(mb):
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.0f} MB" if mb >= 10 else f"{mb:.1f} MB" if mb >= 1 else f"{mb * 1024:.0f} KB"


def fmt_s(s):
    return f"{s * 1000:.0f} ms" if s < 1 else f"{s:.2f} s" if s < 100 else f"{s:.0f} s"


def write_outputs(results, out_dir):
    models = results["models"]
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(out_dir, "phases.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "params", "phase", "status", "runs", "wall_s", "cpu_s", "avg_cores",
                    "peak_rss_mb", "phase_ram_mb", "error"])
        for name, m in models.items():
            for ph, r in m["phases"].items():
                w.writerow([name, m["params"], ph, r["status"], r.get("runs", 1), r.get("wall_s"), r.get("cpu_s"),
                            r.get("avg_cores"), r.get("peak_rss_mb"), r.get("phase_ram_mb"), r.get("error", "")])

    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "params", "logrows", "phase", "status", "wall_s", "cpu_s", "avg_cores",
                    "peak_rss_mb", "phase_ram_mb"])
        for name, m in models.items():
            for label, r in m["summary"].items():
                w.writerow([name, m["params"], m.get("circuit", {}).get("logrows"), label, r["status"],
                            r.get("wall_s"), r.get("cpu_s"), r.get("avg_cores"), r.get("peak_rss_mb"),
                            r.get("phase_ram_mb")])

    acc_keys = ["n_images", "acc_float_pct", "acc_ezkl_pct", "acc_drop_pp", "agreement_pct", "changed_predictions",
                "tied_top_pct", "mean_abs_err", "max_abs_err", "rel_err_pct", "quant_step", "mean_abs_err_in_steps"]
    with open(os.path.join(out_dir, "accuracy.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "params"] + acc_keys)
        for name, m in models.items():
            if m.get("accuracy"):
                w.writerow([name, m["params"]] + [m["accuracy"][k] for k in acc_keys])

    md = [report_markdown(results)]
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(md))


def report_markdown(results):
    mi, cfg, models = results["machine"], results["config"], results["models"]
    L = ["# ezkl benchmark: CIFAR-10 models", "",
         f"Machine: {mi['cpu']}, {mi['logical_cpus']} logical CPUs, {mi['ram_gb']:.1f} GB RAM, {mi['platform']}  ",
         f"ezkl {mi['ezkl']}, Python {mi['python']}, {mi['date']}  ",
         f"Visibility: params **{cfg['param_visibility']}**, input **{cfg['input_visibility']}**, "
         f"output **{cfg['output_visibility']}**; calibration target `{cfg['calibration_target']}`; "
         f"threads: {cfg['threads'] or 'all'}; repeats of witness/prove/verify: {cfg['repeats']}  ",
         "The SRS is downloaded once per circuit size and cached, so later models of the same size show only "
         "the time to find it.", "",
         "## Circuits", "",
         "| model | params | logrows | rows used | input / param scale | proving key | verification key | proof | SRS |",
         "|---|---:|---:|---:|---|---:|---:|---:|---|"]
    L += [_circuit_row(name, m) for name, m in models.items()]
    L += ["", "## CPU and RAM per phase", "",
          "*wall* = elapsed time, *CPU* = CPU time over all threads, *cores* = CPU / wall (average number of busy "
          "cores, '-' when the phase is too short to tell), *peak RAM* = highest resident memory of the process during the phase, *phase RAM* = peak RAM "
          "minus the memory the process already held before the phase started (Python + ezkl, about 30-60 MB).", ""]
    for label, _ in USER_PHASES:
        L += [f"### {label}", "", "| model | params | wall | CPU | cores | peak RAM | phase RAM |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for name, m in models.items():
            r = m["summary"].get(label, {"status": "not run"})
            if r["status"] == "ok":
                cores = f"{r['avg_cores']:.1f}" if r["wall_s"] >= 0.05 else "-"
                L.append(f"| {name} | {m['params']:,} | {fmt_s(r['wall_s'])} | {fmt_s(r['cpu_s'])} | "
                         f"{cores} | {fmt_mb(r['peak_rss_mb'])} | {fmt_mb(r['phase_ram_mb'])} |")
            else:
                L.append(f"| {name} | {m['params']:,} | {r['status']}: {r.get('error', '')} | | | | |")
        L.append("")
    L += ["## Accuracy lost to quantization", "",
          f"Float = the original ONNX model run by onnxruntime; ezkl = the quantized circuit (output of gen_witness, "
          f"i.e. exactly what is proven). Measured on held-out test images ({results['dataset']}).", "",
          "| model | images | float accuracy | ezkl accuracy | accuracy drop | same prediction | tied top score "
          "| mean abs. output error | max abs. output error | relative output error |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in models.items():
        a = m.get("accuracy")
        if a:
            L.append(f"| {name} | {a['n_images']} | {a['acc_float_pct']:.1f} % | {a['acc_ezkl_pct']:.1f} % | "
                     f"{a['acc_drop_pp']:+.1f} pp | {a['agreement_pct']:.1f} % ({a['changed_predictions']} changed) | "
                     f"{a['tied_top_pct']:.1f} % | "
                     f"{a['mean_abs_err']:.2e} ({a['mean_abs_err_in_steps']:.1f} steps) | {a['max_abs_err']:.2e} | "
                     f"{a['rel_err_pct']:.3f} % |")
        else:
            L.append(f"| {name} | {m.get('accuracy_error', 'not run')} | | | | | | | | |")
    L += ["", "- **float / ezkl accuracy**: share of test images classified correctly by each version.",
          "- **accuracy drop**: float accuracy minus ezkl accuracy, in percentage points (positive = quantization hurt).",
          "- **same prediction**: share of images where ezkl picks the same class as the float model, whether right "
          "or wrong. 100 % means quantization never changed a decision.",
          "- **tied top score**: share of images where two or more classes get exactly the same highest ezkl output, "
          "so the circuit's answer is ambiguous (argmax then takes the lowest class index). Typical for sigmoid "
          "outputs that saturate: 0.99998 and 0.99999 both round to 1.0.",
          "- **mean / max abs. output error**: |ezkl output - float output| over all 10 outputs of all images; "
          "*steps* = that error in units of the output quantization step 2^-scale.",
          "- **relative output error**: length of the error vector / length of the float output vector, per image, "
          "averaged.", ""]
    return "\n".join(L)


def _circuit_row(name, m):
    c = m.get("circuit", {})
    g = lambda k, f: f(c[k]) if k in c else "-"
    return (f"| {name} | {m['params']:,} | {g('logrows', str)} | {g('num_rows', lambda v: f'{v:,}')} | "
            f"{g('input_scale', str)} / {g('param_scale', str)} | {g('pk_mb', fmt_mb)} | {g('vk_mb', fmt_mb)} | "
            f"{g('proof_kb', lambda v: f'{v:.1f} KB')} | {c.get('srs_source', '-')} |")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifacts", default=os.path.join(HERE, "artifacts"), help="output of export_models.py")
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    p.add_argument("--models", nargs="+", help="default: every model in the manifest, by parameter count")
    p.add_argument("--param-visibility", default="private",
                   help="private = weights hidden (default); others: fixed, public, hashed/public, polycommit")
    p.add_argument("--input-visibility", default="public")
    p.add_argument("--output-visibility", default="public")
    p.add_argument("--calibration-target", choices=["resources", "accuracy"], default="resources",
                   help="ezkl calibration: smallest circuit (resources) or highest precision (accuracy)")
    p.add_argument("--max-logrows", type=int, help="upper bound for the circuit size 2^logrows during calibration")
    p.add_argument("--num-inner-cols", type=int, help="ezkl columns per layout (more columns = fewer rows)")
    p.add_argument("--n-calib", type=int, help="calibration images (default: all in dataset.npz)")
    p.add_argument("--n-eval", type=int, default=100, help="test images for the accuracy comparison (0 = skip)")
    p.add_argument("--repeats", type=int, default=1, help="runs of quantize_input/witness/prove/verify; median reported")
    p.add_argument("--threads", type=int, help="limit ezkl's thread pool (RAYON_NUM_THREADS); default: all cores")
    p.add_argument("--srs", choices=["auto", "download", "local"], default="auto",
                   help="KZG SRS: download ezkl's public one, generate an insecure local one, or auto (download, else local)")
    p.add_argument("--max-prove-logrows", type=int,
                   help="skip SRS, key setup, prove and verify for circuits bigger than 2^N rows (they need a lot of "
                        "RAM: about 12 GB for setup at 2^20 and roughly double per extra row); default: never skip")
    p.add_argument("--phase-timeout", type=int, default=0, help="seconds before a phase is abandoned (0 = none)")
    p.add_argument("--keep-keys", action="store_true", help="keep proving keys (can be several GB each)")
    p.add_argument("--report-only", action="store_true",
                   help="do not run anything: rebuild report.md and the CSVs from <out>/results.json")
    args = p.parse_args()

    if args.report_only:
        results = json.load(open(os.path.join(args.out, "results.json")))
        for name, m in results["models"].items():
            npz = os.path.join(args.out, name, "accuracy.npz")
            if m.get("accuracy") and os.path.exists(npz):
                m["accuracy"] = accuracy_metrics(npz)
        write_outputs(results, args.out)
        print(report_markdown(results))
        return

    manifest = json.load(open(os.path.join(args.artifacts, "manifest.json")))
    names = args.models or sorted(manifest["models"], key=lambda n: manifest["models"][n]["params"])
    unknown = [n for n in names if n not in manifest["models"]]
    if unknown:
        sys.exit(f"not in {args.artifacts}/manifest.json: {unknown} (run export_models.py --models ...)")
    dataset = os.path.join(args.artifacts, "dataset.npz")
    data = np.load(dataset)
    calib_x = data["calib_x"][:args.n_calib] if args.n_calib else data["calib_x"]
    n_eval = min(args.n_eval, len(data["eval_y"]))

    os.makedirs(args.out, exist_ok=True)
    srs_dir = os.path.join(args.out, "srs")
    os.makedirs(srs_dir, exist_ok=True)
    results = {"machine": machine_info(),
               "config": {k: getattr(args, k) for k in ("param_visibility", "input_visibility", "output_visibility",
                                                        "calibration_target", "max_logrows", "max_prove_logrows", "num_inner_cols",
                                                        "repeats", "threads", "srs")} | {"n_calib": int(len(calib_x)),
                                                                                         "n_eval": n_eval},
               "dataset": f"{manifest['data']}, models trained {manifest['epochs']} epochs on "
                          f"{manifest['train_samples']} images" if manifest["epochs"] else
                          f"{manifest['data']}, untrained (random) weights",
               "models": {}}
    print(f"ezkl {results['machine']['ezkl']} on {results['machine']['cpu']} "
          f"({results['machine']['logical_cpus']} CPUs, {results['machine']['ram_gb']:.1f} GB RAM)")

    for name in names:
        mdir = os.path.join(args.out, name)
        log_dir = os.path.join(args.out, "logs", name)
        os.makedirs(mdir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        f = lambda fn: os.path.join(mdir, fn)
        cfg = {"onnx": os.path.join(args.artifacts, manifest["models"][name]["onnx"]),
               "settings": f("settings.json"), "compiled": f("network.compiled"), "calib_json": f("calibration.json"),
               "input_json": f("input.json"), "input_quantized": f("input_quantized.json"), "witness": f("witness.json"),
               "pk": f("pk.key"), "vk": f("vk.key"), "proof": f("proof.json"), "srs": None,
               "srs_mode": args.srs,
               "param_visibility": args.param_visibility, "input_visibility": args.input_visibility,
               "output_visibility": args.output_visibility, "calibration_target": args.calibration_target,
               "max_logrows": args.max_logrows, "num_inner_cols": args.num_inner_cols,
               "dataset": dataset, "n_eval": n_eval, "accuracy_npz": f("accuracy.npz")}
        with open(cfg["calib_json"], "w") as fh:          # all calibration images, one after another
            json.dump({"input_data": [calib_x.reshape(-1).tolist()]}, fh)
        with open(cfg["input_json"], "w") as fh:          # the image that is proven: first evaluation image
            json.dump({"input_data": [data["eval_x"][0].reshape(-1).tolist()]}, fh)

        m = {"params": manifest["models"][name]["params"], "phases": {}, "circuit": {}}
        results["models"][name] = m
        print(f"\n=== {name} ({m['params']:,} parameters) ===")
        for phase in PHASES:
            missing = [d for d in DEPENDS[phase] if m["phases"].get(d, {}).get("status") != "ok"]
            logrows = m["circuit"].get("logrows")
            if missing:
                m["phases"][phase] = {"status": "skipped", "error": "needs " + ", ".join(missing)}
            elif phase in PROVING and args.max_prove_logrows and logrows > args.max_prove_logrows:
                m["phases"][phase] = {"status": "skipped",
                                      "error": f"circuit 2^{logrows} > --max-prove-logrows {args.max_prove_logrows}"}
            if phase in m["phases"]:
                print(f"  {phase:15s} skipped ({m['phases'][phase]['error']})", flush=True)
                continue
            if phase == "srs":
                logrows = json.load(open(cfg["settings"]))["run_args"]["logrows"]
                cfg["srs"] = os.path.join(srs_dir, f"kzg{logrows}.srs")
            runs = []
            for r in range(args.repeats if phase in REPEATABLE else 1):
                res = run_phase(phase, cfg, log_dir, args, run=r)
                runs.append(res)
                if res["status"] != "ok":
                    break
            agg = aggregate(runs)
            m["phases"][phase] = agg
            if agg["status"] == "ok":
                cores = f"{agg['avg_cores']:4.1f}" if agg["wall_s"] >= 0.05 else "   -"
                print(f"  {phase:15s} wall {fmt_s(agg['wall_s']):>9s}  CPU {fmt_s(agg['cpu_s']):>9s}  "
                      f"cores {cores}  peak RAM {fmt_mb(agg['peak_rss_mb']):>9s}", flush=True)
            else:
                print(f"  {phase:15s} FAILED: {agg['error']}  (log: {log_dir})", flush=True)
            if phase == "calibrate" and agg["status"] == "ok":
                s = json.load(open(cfg["settings"]))
                m["circuit"].update({"logrows": s["run_args"]["logrows"], "num_rows": s["num_rows"],
                                     "total_assignments": s["total_assignments"],
                                     "input_scale": s["run_args"]["input_scale"],
                                     "param_scale": s["run_args"]["param_scale"],
                                     "output_scale": s["model_output_scales"][0],
                                     "lookups": s.get("required_lookups")})
            for k in ("pk_mb", "vk_mb", "proof_kb", "srs_source"):
                if k in agg:
                    m["circuit"][k] = agg[k]
            if phase == "witness" and agg.get("quantized_input_matches_witness") is False:
                print("  warning: float_to_felt input differs from the witness input")

        # accuracy: needs only the compiled circuit (and keys/SRS only for polycommit visibility)
        if n_eval > 0 and m["phases"].get("compile", {}).get("status") == "ok":
            print(f"  accuracy: running the circuit on {n_eval} test images ...", flush=True)
            res = run_phase("accuracy", cfg, log_dir, args)
            if res["status"] == "ok":
                m["accuracy"] = accuracy_metrics(cfg["accuracy_npz"])
                a = m["accuracy"]
                print(f"  accuracy: float {a['acc_float_pct']:.1f} %, ezkl {a['acc_ezkl_pct']:.1f} %, "
                      f"same prediction {a['agreement_pct']:.1f} %, tied top score {a['tied_top_pct']:.1f} %, "
                      f"mean abs. output error {a['mean_abs_err']:.2e}")
            else:
                m["accuracy_error"] = res["error"]
                print(f"  accuracy FAILED: {res['error']}")

        m["summary"] = {label: combine(m["phases"], parts) for label, parts in USER_PHASES}
        if not args.keep_keys and os.path.exists(cfg["pk"]):
            os.remove(cfg["pk"])
        write_outputs(results, args.out)        # after every model, so partial results survive

    print("\n" + report_markdown(results))
    print(f"results written to {args.out} (report.md, summary.csv, phases.csv, accuracy.csv, results.json)")


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "_worker":
        worker_main(*sys.argv[2:])
    else:
        main()
