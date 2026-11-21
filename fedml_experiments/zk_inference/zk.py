"""Shared by server.py, client.py and user.py: config, ezkl circuit build, the signed statement, signatures.

The statement every client signs describes the model as ezkl proves it:

    {"model": "mlp99k", "param_visibility": "fixed",
     "settings_sha256": <SHA-256 of the circuit settings>,
     "vk_sha256": <SHA-256 of the verification key>,
     "params_hash": <Poseidon hash of the quantized weights>}     # only with param_visibility hashed/public

With "fixed" visibility the quantized weights are fixed columns of the circuit, and the verification key
contains KZG commitments to them: a proof that verifies with the signed key was computed with exactly the
signed weights. With "hashed/public" the Poseidon hash of the weights is a public value of every proof.
The key and settings must be signed in both cases: they define the circuit the proof is checked against.
"""
import asyncio
import contextlib
import hashlib
import inspect
import json
import os
import sys

import numpy as np
import yaml
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------------------------------- config, keys

def load_config():
    """config.yaml next to this file, or the file named by $ZK_CONFIG. Paths in it are relative to the file."""
    path = os.environ.get("ZK_CONFIG", os.path.join(HERE, "config.yaml"))
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = os.path.dirname(os.path.abspath(path))
    for k in ("model_dir", "keys_dir", "work_dir"):
        cfg[k] = os.path.join(base, cfg[k])
    return cfg


def load_public_keys(cfg):
    """{client_id: Ed25519 public key (hex)}: the list everyone has."""
    with open(os.path.join(cfg["keys_dir"], "public_keys.yaml")) as f:
        return yaml.safe_load(f)


def load_private_key(cfg, client_id):
    with open(os.path.join(cfg["keys_dir"], "private_keys.yaml")) as f:
        return yaml.safe_load(f)[client_id]


def canonical(obj):
    """The exact bytes that are hashed and signed: JSON with sorted keys and no spaces."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def sign(private_key_hex, statement):
    return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key_hex)).sign(canonical(statement)).hex()


def signature_ok(public_key_hex, statement, signature_hex):
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex)).verify(
            bytes.fromhex(signature_hex), canonical(statement))
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


# ------------------------------------------------------------------------------------------- ezkl

def ezkl_call(fn, *args, **kwargs):
    """Some ezkl functions are async (get_srs), most are not."""
    async def run():
        r = fn(*args, **kwargs)
        return (await r) if inspect.isawaitable(r) else r
    return asyncio.run(run())


@contextlib.contextmanager
def ezkl_output_to(log_path):
    """ezkl (Rust) prints large tables to stdout; send them to a log file instead."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with open(log_path, "a") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


def paths(cfg, party):
    """Files of one party (server, client1, ...): each builds its own circuit from the shared model files."""
    d = os.path.join(cfg["work_dir"], party)
    return {"dir": d, "log": os.path.join(d, "ezkl.log"),
            "onnx": os.path.join(cfg["model_dir"], cfg["model"], "network.onnx"),
            "dataset": os.path.join(cfg["model_dir"], "dataset.npz"),
            "calibration": os.path.join(d, "calibration.json"), "settings": os.path.join(d, "settings.json"),
            "compiled": os.path.join(d, "network.compiled"), "vk": os.path.join(d, "vk.key"),
            "pk": os.path.join(d, "pk.key")}


def srs_path(cfg, settings_path):
    """ezkl's public KZG parameters (from its trusted setup), downloaded once per circuit size."""
    import ezkl
    logrows = json.load(open(settings_path))["run_args"]["logrows"]
    path = os.path.join(cfg["work_dir"], "srs", f"kzg{logrows}.srs")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ezkl_call(ezkl.get_srs, settings_path, srs_path=path + ".part")
        os.replace(path + ".part", path)
    return path


def canonicalize_settings(path):
    """Make the settings byte-identical for every party.

    ezkl's quantization is deterministic, but it fills two lists from Rust hash sets (random order per process)
    and stamps the current time. The order changes the circuit layout and thereby the keys, so two parties
    with the same model would get different verification keys. Sorted, the keys come out identical.
    """
    with open(path) as f:
        s = json.load(f)
    s["required_range_checks"] = sorted(s["required_range_checks"])
    s["required_lookups"] = sorted(s["required_lookups"], key=lambda v: json.dumps(v, sort_keys=True))
    s["timestamp"] = 0
    with open(path, "wb") as f:
        f.write(canonical(s))


def build_circuit(cfg, party, keep_pk):
    """Quantize the ONNX model with ezkl, compile it and create the keys (once; cached in work/<party>/)."""
    import ezkl
    p = paths(cfg, party)
    if os.path.exists(p["vk"]) and (os.path.exists(p["pk"]) or not keep_pk):
        return p
    os.makedirs(p["dir"], exist_ok=True)
    calib = np.load(p["dataset"])["calib_x"]                     # the same calibration images for everyone
    with open(p["calibration"], "w") as f:
        json.dump({"input_data": [calib.reshape(-1).tolist()]}, f)
    run_args = ezkl.PyRunArgs()
    run_args.param_visibility = cfg["param_visibility"]
    run_args.input_visibility = "public"
    run_args.output_visibility = "public"
    with ezkl_output_to(p["log"]):
        ezkl_call(ezkl.gen_settings, p["onnx"], p["settings"], py_run_args=run_args)            # import ONNX
        ezkl_call(ezkl.calibrate_settings, p["calibration"], p["onnx"], p["settings"], "resources")  # scales
        canonicalize_settings(p["settings"])
        ezkl_call(ezkl.compile_circuit, p["onnx"], p["compiled"], p["settings"])                 # quantize weights
        ezkl_call(ezkl.setup, p["compiled"], p["vk"], p["pk"], srs_path=srs_path(cfg, p["settings"]))
    if not keep_pk:
        os.remove(p["pk"])            # only the server proves; the proving key is large
    return p


def input_size(settings):
    return int(np.prod(settings["model_instance_shapes"][0]))


def params_hash(cfg, p):
    """Poseidon hash of the quantized weights, as ezkl puts it into every proof (hashed/public only)."""
    import ezkl
    settings = json.load(open(p["settings"]))
    inp, wit = os.path.join(p["dir"], "zero_input.json"), os.path.join(p["dir"], "zero_witness.json")
    with open(inp, "w") as f:
        json.dump({"input_data": [[0.0] * input_size(settings)]}, f)     # the hash does not depend on the input
    with ezkl_output_to(p["log"]):
        w = ezkl_call(ezkl.gen_witness, inp, p["compiled"], wit)
    return w["processed_params"]["poseidon_hash"][0]


def make_statement(cfg, p):
    with open(p["settings"], "rb") as f:
        settings_bytes = f.read()
    with open(p["vk"], "rb") as f:
        vk_bytes = f.read()
    statement = {"model": cfg["model"], "param_visibility": cfg["param_visibility"],
                 "settings_sha256": sha256(settings_bytes), "vk_sha256": sha256(vk_bytes)}
    if cfg["param_visibility"].startswith("hashed"):
        statement["params_hash"] = params_hash(cfg, p)
    return statement


def split_instances(proof, settings, param_visibility):
    """The public values of a proof: [weights hash (hashed/public only)] + input + output, as field elements."""
    import ezkl
    flat = proof["instances"][0]
    n_hash = 1 if param_visibility.startswith("hashed") else 0
    n_in = input_size(settings)
    out_scale = settings["model_output_scales"][0]
    return {"params_hash": flat[0] if n_hash else None,
            "input": flat[n_hash:n_hash + n_in],
            "output": [ezkl.felt_to_float(v, out_scale) for v in flat[n_hash + n_in:]]}


def quantize_input(x, settings):
    """The field elements ezkl makes of a float input (what must appear as the proof's public input)."""
    import ezkl
    scale = settings["model_input_scales"][0]
    return [ezkl.float_to_felt(float(v), scale, ezkl.PyInputType.F32) for v in x]


def prove(cfg, job_dir, x):
    """Server side, run in a worker process: witness (= quantized inference) and proof for input x."""
    import ezkl
    p = paths(cfg, "server")
    os.makedirs(job_dir, exist_ok=True)
    inp, wit, proof_path = (os.path.join(job_dir, n) for n in ("input.json", "witness.json", "proof.json"))
    with open(inp, "w") as f:
        json.dump({"input_data": [x]}, f)
    with ezkl_output_to(os.path.join(job_dir, "ezkl.log")):
        ezkl_call(ezkl.gen_witness, inp, p["compiled"], wit)
        ezkl_call(ezkl.prove, wit, p["compiled"], p["pk"], proof_path, srs_path=srs_path(cfg, p["settings"]))
    with open(proof_path) as f:
        proof = json.load(f)
    settings = json.load(open(p["settings"]))
    return {"proof": proof, "output": split_instances(proof, settings, cfg["param_visibility"])["output"]}
