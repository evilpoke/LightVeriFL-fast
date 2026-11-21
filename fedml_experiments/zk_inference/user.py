"""The user: asks the server for a prediction and checks that it was proven with the model the clients signed.

Trusted by the user: keys/public_keys.yaml (the clients' public keys) and ezkl's public SRS. Nothing the
server sends is trusted until it is checked:

  1. enough clients signed the model statement (Ed25519, public keys from the YAML)
  2. the verification key and circuit settings the server hands out are the signed ones (SHA-256)
  3. the zk proof verifies with them
  4. the proof's public input is the user's own input, quantized the way ezkl does it
  5. hashed/public only: the weights hash in the proof is the signed one
  6. the output the server reports is the output in the proof

Run: python user.py [--image N] [--min-signatures K] [--tamper]
"""
import argparse
import base64
import json
import os
import sys
import tempfile
import time

import numpy as np
import requests

import zk


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=int, default=0, help="which test image of model/dataset.npz to classify")
    p.add_argument("--min-signatures", type=int, help="valid client signatures required (default: all clients)")
    p.add_argument("--poll", type=float, default=1.0, help="seconds between status requests")
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--tamper", action="store_true", help="change the proven output before checking (must FAIL)")
    args = p.parse_args()

    cfg = zk.load_config()
    url = cfg["server_url"]
    public_keys = zk.load_public_keys(cfg)
    min_sigs = len(public_keys) if args.min_signatures is None else args.min_signatures
    data = np.load(os.path.join(cfg["model_dir"], "dataset.npz"))
    x, label = data["eval_x"][args.image].reshape(-1), int(data["eval_y"][args.image])

    # --- ask for the prediction, then poll until the proof is ready
    job = requests.post(url + "/infer", json={"input": x.tolist()}, timeout=30)
    job.raise_for_status()
    job = job.json()
    print(f"job {job['job_id']} submitted, waiting for the proof ...", flush=True)
    t0 = time.time()
    while True:
        res = requests.get(url + job["status_url"], timeout=30).json()
        if res["status"] != "pending":
            break
        if time.time() - t0 > args.timeout:
            sys.exit("timed out")
        time.sleep(args.poll)
    if res["status"] != "done":
        sys.exit(f"server failed: {res.get('error')}")
    print(f"proof received after {time.time() - t0:.1f} s\n")
    model = requests.get(url + "/model", timeout=30).json()

    proof, statement = res["proof"], res["statement"]
    if args.tamper:                                  # pretend the server lied about the first output value
        proof["instances"][0][-10] = proof["instances"][0][-9]
    results = []

    def check(ok, text):
        results.append(bool(ok))
        print(f"[{'OK' if ok else 'FAIL'}] {text}")

    print("statement:", json.dumps(statement))
    valid = [c for c, key in public_keys.items() if zk.signature_ok(key, statement, res["signatures"].get(c, ""))]
    missing = [c for c in public_keys if c not in valid]
    check(len(valid) >= min_sigs, f"{len(valid)}/{len(public_keys)} clients signed this statement "
                                  f"(need {min_sigs}; valid: {', '.join(valid) or '-'}"
                                  f"{'; missing or invalid: ' + ', '.join(missing) if missing else ''})")
    vk = base64.b64decode(model["vk"])
    settings_text = model["settings"]
    settings = json.loads(settings_text)
    check(zk.sha256(vk) == statement["vk_sha256"], "verification key is the signed one")
    check(zk.sha256(settings_text.encode()) == statement["settings_sha256"], "circuit settings are the signed ones")

    with tempfile.TemporaryDirectory() as d:
        for name, content in (("proof.json", json.dumps(proof)), ("settings.json", settings_text)):
            with open(os.path.join(d, name), "w") as f:
                f.write(content)
        with open(os.path.join(d, "vk.key"), "wb") as f:
            f.write(vk)
        import ezkl
        srs = zk.srs_path(cfg, os.path.join(d, "settings.json"))     # ezkl's public SRS, not from the server
        try:
            proof_ok = zk.ezkl_call(ezkl.verify, os.path.join(d, "proof.json"), os.path.join(d, "settings.json"),
                                    os.path.join(d, "vk.key"), srs_path=srs)
        except Exception as e:  # noqa: BLE001  (ezkl raises on an invalid proof)
            proof_ok = False
            print(f"       ezkl.verify: {e}")
    check(proof_ok, "zk proof verifies")

    public = zk.split_instances(proof, settings, statement["param_visibility"])
    check(public["input"] == zk.quantize_input(x, settings), f"the proof is about my input ({len(x)} values)")
    if "params_hash" in statement:
        check(public["params_hash"] == statement["params_hash"], "weights hash in the proof is the signed one")
    check(np.allclose(res["output"], public["output"]), "the server's output is the proven output")

    out = np.array(public["output"])
    print(f"\nproven output: {np.array2string(out, precision=3)}")
    print(f"prediction: class {int(out.argmax())}   (true label of test image {args.image}: {label})")
    print("VERIFIED" if all(results) else "NOT VERIFIED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
