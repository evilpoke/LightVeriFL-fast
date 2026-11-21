"""Create one Ed25519 key pair per client.

    keys/public_keys.yaml    everyone has it (the user's hard-coded list of trusted clients)
    keys/private_keys.yaml   the clients only

Run: python gen_keys.py [--clients 3]
"""
import argparse
import os

import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import zk

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--clients", type=int, default=3)
args = p.parse_args()

raw = dict(encoding=serialization.Encoding.Raw)
public, private = {}, {}
for i in range(1, args.clients + 1):
    key = Ed25519PrivateKey.generate()
    private[f"client{i}"] = key.private_bytes(format=serialization.PrivateFormat.Raw,
                                              encryption_algorithm=serialization.NoEncryption(), **raw).hex()
    public[f"client{i}"] = key.public_key().public_bytes(format=serialization.PublicFormat.Raw, **raw).hex()

keys_dir = zk.load_config()["keys_dir"]
os.makedirs(keys_dir, exist_ok=True)
for name, keys, comment in (("public_keys.yaml", public, "Ed25519 public keys of the clients (hex). Everyone has this file."),
                            ("private_keys.yaml", private, "Ed25519 private keys (hex). Clients only, keep secret.")):
    with open(os.path.join(keys_dir, name), "w") as f:
        f.write(f"# {comment}\n")
        yaml.safe_dump(keys, f, sort_keys=True)
print(f"wrote {args.clients} key pairs to {keys_dir}/public_keys.yaml and private_keys.yaml")
