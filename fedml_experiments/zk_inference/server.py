"""Flask server: zk-proven inference with ezkl, plus the clients' signatures on the model.

    GET  /model             statement, circuit settings and verification key (everything needed to verify)
    POST /signatures        a client uploads {"client_id", "statement", "signature"}
    GET  /signatures        all stored signatures
    POST /infer             {"input": [3072 floats]} -> 202 {"job_id", "status_url"}; proving runs in the background
    GET  /infer/<job_id>    {"status": "pending"} until done, then output, proof, statement and signatures

Run: python server.py   (builds the circuit on first start, ~30 s; config.yaml has the address)
"""
import base64
import json
import multiprocessing
import os
import threading
import uuid
from concurrent.futures import ProcessPoolExecutor
from urllib.parse import urlparse

from flask import Flask, jsonify, request

import zk

cfg = zk.load_config()
print(f"building the circuit for {cfg['model']} (params {cfg['param_visibility']}) ...", flush=True)
PATHS = zk.build_circuit(cfg, "server", keep_pk=True)
STATEMENT = zk.make_statement(cfg, PATHS)
SETTINGS_TEXT = open(PATHS["settings"]).read()
VK_B64 = base64.b64encode(open(PATHS["vk"], "rb").read()).decode()
N_INPUT = zk.input_size(json.loads(SETTINGS_TEXT))
PUBLIC_KEYS = zk.load_public_keys(cfg)
print("model statement:", json.dumps(STATEMENT, indent=1), flush=True)

SIG_FILE = os.path.join(PATHS["dir"], "signatures.json")
signatures = json.load(open(SIG_FILE)) if os.path.exists(SIG_FILE) else {}
sig_lock = threading.Lock()

# one proof at a time (proving uses all cores and a lot of RAM); further jobs wait in the queue
pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
jobs = {}

app = Flask(__name__)


@app.get("/model")
def get_model():
    return jsonify(statement=STATEMENT, settings=SETTINGS_TEXT, vk=VK_B64)


@app.post("/signatures")
def add_signature():
    body = request.get_json(force=True, silent=True) or {}
    client_id, statement, signature = body.get("client_id"), body.get("statement"), body.get("signature")
    if client_id not in PUBLIC_KEYS:
        return jsonify(error=f"unknown client {client_id!r}"), 400
    if statement != STATEMENT:
        differ = sorted(k for k in set(STATEMENT) | set(statement or {}) if (statement or {}).get(k) != STATEMENT.get(k))
        return jsonify(error="statement differs from the server's model", fields=differ, server=STATEMENT), 409
    if not zk.signature_ok(PUBLIC_KEYS[client_id], STATEMENT, signature):
        return jsonify(error="signature does not verify with the client's public key"), 400
    with sig_lock:
        signatures[client_id] = signature
        with open(SIG_FILE, "w") as f:
            json.dump(signatures, f, indent=1)
    return jsonify(stored=client_id, signatures=len(signatures)), 201


@app.get("/signatures")
def get_signatures():
    return jsonify(statement=STATEMENT, signatures=signatures)


@app.post("/infer")
def infer():
    x = (request.get_json(force=True, silent=True) or {}).get("input")
    if not isinstance(x, list) or len(x) != N_INPUT or not all(isinstance(v, (int, float)) for v in x):
        return jsonify(error=f"'input' must be a list of {N_INPUT} numbers (the flattened image)"), 400
    job_id = uuid.uuid4().hex
    jobs[job_id] = pool.submit(zk.prove, cfg, os.path.join(PATHS["dir"], "jobs", job_id), x)
    return jsonify(job_id=job_id, status_url=f"/infer/{job_id}"), 202


@app.get("/infer/<job_id>")
def get_job(job_id):
    job = jobs.get(job_id)
    if job is None:
        return jsonify(error="unknown job"), 404
    if not job.done():
        return jsonify(status="pending")
    if job.exception() is not None:
        return jsonify(status="failed", error=repr(job.exception()))
    result = job.result()
    return jsonify(status="done", output=result["output"], proof=result["proof"],
                   statement=STATEMENT, signatures=signatures)


if __name__ == "__main__":
    url = urlparse(cfg["server_url"])
    app.run(host=url.hostname, port=url.port, threaded=True)
