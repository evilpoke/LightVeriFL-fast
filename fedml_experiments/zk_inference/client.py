"""A client: quantizes its own copy of the model with ezkl, signs the resulting statement, uploads the signature.

Run: python client.py client1     (the id must be in keys/private_keys.yaml)
"""
import json
import sys
import time

import requests

import zk

if len(sys.argv) != 2:
    sys.exit(__doc__)
client_id = sys.argv[1]
cfg = zk.load_config()
private_key = zk.load_private_key(cfg, client_id)

t0 = time.time()
paths = zk.build_circuit(cfg, client_id, keep_pk=False)      # independent of the server: own work/<client_id>/
statement = zk.make_statement(cfg, paths)
signature = zk.sign(private_key, statement)
print(f"{client_id}: built the circuit in {time.time() - t0:.0f} s, statement {json.dumps(statement)}")

r = requests.post(cfg["server_url"] + "/signatures",
                  json={"client_id": client_id, "statement": statement, "signature": signature}, timeout=30)
print(f"{client_id}: upload -> HTTP {r.status_code} {r.json()}")
sys.exit(0 if r.status_code == 201 else 1)
