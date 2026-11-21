# Verifiable inference with ezkl

A Flask server classifies images with a fixed model and proves every prediction with
[ezkl](https://github.com/zkonduit/ezkl). A group of clients, who all hold the same model, vouch for it by
signing a statement that pins the model as ezkl proves it. A user gets the prediction, the proof and the
signatures, and checks all of them. The user trusts only the clients' public keys and never the server.

| who | has | does |
|---|---|---|
| server (`server.py`) | the model | quantizes it with ezkl, makes the keys, proves predictions, stores signatures |
| client (`client.py client1`) | the same model, `keys/public_keys.yaml`, its private key in `keys/private_keys.yaml` | quantizes the model itself, signs the statement, uploads the signature |
| user (`user.py`) | `keys/public_keys.yaml` (hard-coded list of trusted clients) | sends an image, polls until the proof is ready, checks everything |

Model: **mlp99k** (FC 3072 -> 32 -> 10), trained for 2 epochs on random synthetic images, then fixed. It has the
most parameters of the five models, but the smallest circuit (2^17 rows), so a proof takes about 10 s and 1.5 GB.
The convnets need 2^21+ rows (see `../ezkl_benchmark`).

## Run

```bash
pip install -r requirements.txt
./run_demo.sh
```

This creates the model and 3 client key pairs, starts the server, lets every client sign, and runs the user twice:
once normally (`VERIFIED`) and once with the proven output altered (`NOT VERIFIED`). It takes about 2 minutes,
mostly because every party quantizes the model and creates the keys itself.

By hand, in separate terminals:

```bash
python server.py                 # first start: quantize + keys, ~30 s
python client.py client1         # likewise client2, client3
python user.py --image 3         # classify test image 3 of model/dataset.npz
```

## API

| | request | response |
|---|---|---|
| `GET /model` | | `statement`, `settings` (circuit settings JSON text), `vk` (verification key, base64) |
| `POST /signatures` | `{"client_id", "statement", "signature"}` | 201 stored; 409 if the statement is not the server's model; 400 bad signature / unknown client |
| `GET /signatures` | | `statement`, `signatures` (`{client_id: hex}`) |
| `POST /infer` | `{"input": [3072 floats]}` (flattened 3x32x32 image) | 202 `{"job_id", "status_url"}` |
| `GET /infer/<job_id>` | | `{"status": "pending"}` until done, then `output`, `proof`, `statement`, `signatures` |

Proving takes ~10 s per request, so `/infer` only queues the job (one proof at a time, in a separate
process) and the user polls `/infer/<job_id>` every second.

## What is signed, what the user checks

Every client signs (Ed25519, over the JSON with sorted keys) the statement

```json
{"model": "mlp99k", "param_visibility": "fixed",
 "settings_sha256": "<SHA-256 of the ezkl circuit settings>",
 "vk_sha256": "<SHA-256 of the ezkl verification key>"}
```

The user then checks:

1. enough clients signed the statement (default: all in `public_keys.yaml`, `--min-signatures` to change)
2. the verification key and settings the server hands out have the signed hashes
3. the zk proof verifies with them, using ezkl's public SRS, which the user downloads from ezkl and not from the server
4. the public input in the proof is the user's own image, quantized as ezkl does it
5. (`hashed/public` only) the weights hash in the proof is the signed one
6. the output the server reports is the output in the proof

## Design notes

- **Where the "model hash" is.** ezkl can put a Poseidon hash of the quantized weights into every proof
  (`param_visibility: hashed/public`), but that costs about 41 circuit rows per weight. Every one of the five
  models then needs 2^21 or 2^22 rows: tens of GB of RAM per proof. The default here is `fixed`: the quantized
  weights are fixed columns of the circuit, and the verification key contains KZG commitments to them. A
  proof that verifies with the signed key therefore came from exactly the signed weights. This is the same
  guarantee at no extra cost. `hashed/public` still works (set it in `config.yaml`, delete `work/`). It was
  tested end to end on a small network, and the statement then also contains `params_hash`.
- **Why the key and settings are signed, in both modes.** They define the circuit the proof is checked
  against. If only the weights hash were signed, a dishonest server could hand out a key for a different
  circuit that publishes the right hash but ignores the weights.
- **Determinism.** ezkl's quantization is deterministic (same scales every time), but the settings list the
  required range-check tables in Rust hash-set order, which is random per process. That order changes the
  circuit layout, so two parties got different verification keys. `zk.canonicalize_settings` sorts these
  lists and drops the timestamp. With that, the server and every client get byte-identical settings and keys.
  All parties must use the same ezkl version (`requirements.txt` pins it), the same ONNX file and the
  same calibration images (`model/dataset.npz`).
- **Demo shortcuts.** All private keys are in one file (in practice each client keeps only its own). The
  Flask development server is used. Jobs are kept in memory (signatures are stored in `work/server/signatures.json`).
  If you change `config.yaml`, delete `work/` so the circuits are rebuilt.
