#!/usr/bin/env bash
# End to end: model -> keys -> server -> every client signs -> the user gets a verified prediction.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
cfg() { $PY -c "import zk; print(zk.load_config()['$1'])"; }
MODEL=$(cfg model); MODEL_DIR=$(cfg model_dir); URL=$(cfg server_url)

# 1. the model: trained briefly on random (synthetic) images, then fixed. Server and clients use these files.
if [ ! -f "$MODEL_DIR/$MODEL/network.onnx" ]; then
    $PY ../ezkl_benchmark/export_models.py --data synthetic --models "$MODEL" --out-dir "$MODEL_DIR"
fi

# 2. one key pair per client
[ -f keys/public_keys.yaml ] || $PY gen_keys.py --clients 3

# 3. the server (quantizes the model and creates the keys on first start)
$PY server.py > server.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
echo "starting the server (log: server.log) ..."
until curl -sf "$URL/model" > /dev/null; do
    kill -0 $SERVER 2>/dev/null || { cat server.log; exit 1; }
    sleep 2
done

# 4. every client quantizes the model itself, signs the statement and uploads the signature
for c in $($PY -c "import zk; print(' '.join(zk.load_public_keys(zk.load_config())))"); do
    $PY client.py "$c"
done

# 5. the user asks for a prediction and checks everything
echo; echo "=== user ==="
$PY user.py --image 0

echo; echo "=== user, with the proven output altered (must be rejected) ==="
$PY user.py --image 1 --tamper || echo "-> rejected, as it should be"
