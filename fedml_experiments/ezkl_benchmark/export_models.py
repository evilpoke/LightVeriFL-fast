"""Step 1 of the ezkl benchmark: build the CIFAR-10 models, train them briefly, export them to ONNX.

The five models come from fedml_api/model/cv/cifar10_models.py (cnn6k, cnn15k, lr, lenet5, mlp99k).
Each one is trained for a short time ("quick and dirty", default 2 epochs on 10k images) so its
predictions mean something, then exported to ONNX with input shape 1 x 3 x 32 x 32.

Writes into --out-dir (default ./artifacts):

    <model>/network.onnx    the network bench_ezkl.py imports into ezkl
    dataset.npz             calibration images (for ezkl's quantization), evaluation images + labels
    manifest.json           parameter counts, float accuracy measured by torch, how the data was made

The benchmark (bench_ezkl.py) needs only these files, not torch.

Usage (from anywhere):
    python export_models.py                       # real CIFAR-10, downloaded once (~170 MB) to <repo>/data/cifar10
    python export_models.py --data synthetic      # no download: random class templates + noise
    python export_models.py --epochs 0            # untrained, randomly initialised weights
"""
import argparse
import json
import os
import pickle
import sys
import tarfile
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO_ROOT)

from fedml_api.model.cv.cifar10_models import CIFAR10_MODELS, IMAGE_SHAPE  # noqa: E402

# same normalisation as fedml_api/data_preprocessing/cifar10/data_loader.py
CIFAR_MEAN = np.array([0.49139968, 0.48215827, 0.44653124], dtype=np.float32).reshape(1, 3, 1, 1)
CIFAR_STD = np.array([0.24703233, 0.24348505, 0.26158768], dtype=np.float32).reshape(1, 3, 1, 1)
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def load_cifar10(data_dir):
    """CIFAR-10 as normalised float32 arrays (N, 3, 32, 32). Reuses a torchvision download in data_dir."""
    batches = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.isdir(batches):
        os.makedirs(data_dir, exist_ok=True)
        archive = os.path.join(data_dir, "cifar-10-python.tar.gz")
        if not os.path.exists(archive):
            print(f"downloading CIFAR-10 to {archive} ...")
            urllib.request.urlretrieve(CIFAR_URL, archive + ".part")
            os.replace(archive + ".part", archive)      # an interrupted download never looks complete
        with tarfile.open(archive) as tar:
            try:
                tar.extractall(data_dir, filter="data")
            except TypeError:          # Python < 3.12 without extraction filters
                tar.extractall(data_dir)

    def read(name):
        with open(os.path.join(batches, name), "rb") as f:
            d = pickle.load(f, encoding="latin1")
        return np.asarray(d["data"], dtype=np.uint8).reshape(-1, 3, 32, 32), np.asarray(d["labels"], dtype=np.int64)

    parts = [read(f"data_batch_{i}") for i in range(1, 6)]
    x_train = np.concatenate([p[0] for p in parts])
    y_train = np.concatenate([p[1] for p in parts])
    x_test, y_test = read("test_batch")
    norm = lambda x: ((x.astype(np.float32) / 255.0) - CIFAR_MEAN) / CIFAR_STD
    return norm(x_train), y_train, norm(x_test), y_test


def make_synthetic(n_train, n_test, rng):
    """10 smooth random 'class template' images; a sample is its class template plus noise."""
    coarse = rng.normal(size=(10, 3, 8, 8)).astype(np.float32)
    templates = coarse.repeat(4, axis=2).repeat(4, axis=3)            # 10 x 3 x 32 x 32, blocky

    def sample(n):
        y = rng.integers(0, 10, size=n)
        x = 0.07 * templates[y] + rng.normal(size=(n, *IMAGE_SHAPE)).astype(np.float32)
        return x.astype(np.float32), y.astype(np.int64)

    x_train, y_train = sample(n_train)
    x_test, y_test = sample(n_test)
    return x_train, y_train, x_test, y_test


def train(model, x, y, epochs, batch_size, lr, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()        # as in FedML's trainer, also for the sigmoid-output 'lr' model
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()
    model.eval()


@torch.no_grad()
def accuracy(model, x, y):
    pred = model(torch.from_numpy(x)).argmax(1).numpy()
    return float((pred == y).mean())


def export_onnx(model, path):
    dummy = torch.zeros(1, *IMAGE_SHAPE)
    kwargs = dict(input_names=["input"], output_names=["output"], opset_version=17, do_constant_folding=True)
    try:   # the TorchScript exporter gives the plainest graphs (what ezkl's importer handles best)
        torch.onnx.export(model, dummy, path, dynamo=False, **kwargs)
    except TypeError:   # torch < 2.5 has no 'dynamo' argument and uses the TorchScript exporter anyway
        torch.onnx.export(model, dummy, path, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=os.path.join(HERE, "artifacts"))
    p.add_argument("--models", nargs="+", default=list(CIFAR10_MODELS), choices=list(CIFAR10_MODELS))
    p.add_argument("--data", choices=["cifar10", "synthetic"], default="cifar10")
    p.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data", "cifar10"),
                   help="CIFAR-10 location (shared with the FedML data loader)")
    p.add_argument("--train-samples", type=int, default=10000, help="training images used for the quick training")
    p.add_argument("--epochs", type=int, default=2, help="0 = keep the random initialisation")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-calib", type=int, default=20, help="images ezkl uses to choose its quantization scales")
    p.add_argument("--n-eval", type=int, default=200, help="test images for the float-vs-ezkl accuracy comparison")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if args.data == "cifar10":
        x_train, y_train, x_test, y_test = load_cifar10(args.data_dir)
    else:
        x_train, y_train, x_test, y_test = make_synthetic(args.train_samples, 2000, rng)

    tr = rng.permutation(len(x_train))[:args.train_samples]
    x_train, y_train = x_train[tr], y_train[tr]
    te = rng.permutation(len(x_test))
    x_calib = x_train[:args.n_calib]                       # calibration from training data, never from the eval set
    x_eval, y_eval = x_test[te[:args.n_eval]], y_test[te[:args.n_eval]]

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.out_dir, "dataset.npz"),
                        calib_x=x_calib, eval_x=x_eval, eval_y=y_eval)

    manifest = {"data": args.data, "train_samples": int(len(x_train)), "epochs": args.epochs,
                "n_calib": int(len(x_calib)), "n_eval": int(len(x_eval)), "input_shape": [1, *IMAGE_SHAPE],
                "classes": CLASSES, "torch_version": torch.__version__, "models": {}}
    for name in args.models:
        torch.manual_seed(args.seed)
        model = CIFAR10_MODELS[name]()
        n_params = sum(p.numel() for p in model.parameters())
        t0 = time.time()
        if args.epochs > 0:
            train(model, x_train, y_train, args.epochs, args.batch_size, args.lr, args.seed)
        model.eval()
        mdir = os.path.join(args.out_dir, name)
        os.makedirs(mdir, exist_ok=True)
        export_onnx(model, os.path.join(mdir, "network.onnx"))
        acc = accuracy(model, x_eval, y_eval)
        manifest["models"][name] = {"params": int(n_params), "onnx": f"{name}/network.onnx",
                                    "torch_eval_accuracy": acc, "train_seconds": round(time.time() - t0, 1)}
        print(f"{name:8s} {n_params:7d} params   float accuracy on {len(x_eval)} eval images: {acc:6.1%}"
              f"   ({time.time() - t0:.1f}s)")

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote ONNX models, dataset.npz and manifest.json to {args.out_dir}")


if __name__ == "__main__":
    main()
