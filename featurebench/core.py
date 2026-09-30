import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import SCHEMA_VERSION


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, sort_keys=True, default=str)
        temp = f.name
    os.replace(temp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=path.parent, delete=False) as f:
        temp = f.name
    try:
        np.savez(temp, **arrays)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def read_cache(path, expected_key, count=None, dim=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as obj:
        key = str(obj["key"].item())
        ids, labels, features = obj["ids"], obj["labels"], obj["features"]
        if key != expected_key or features.ndim != 2 or len(ids) != len(labels) or len(ids) != len(features):
            raise ValueError(f"Invalid cache key or shape: {path}")
        if count is not None and len(ids) != count:
            raise ValueError(f"Invalid sample count: {path}")
        if dim is not None and features.shape[1] != dim:
            raise ValueError(f"Invalid feature dimension: {path}")
        if not np.isfinite(features).all() or len(set(ids.tolist())) != len(ids):
            raise ValueError(f"Invalid values or duplicate IDs: {path}")
        return ids.copy(), labels.copy(), features.astype(np.float32)


def cache_key(config, manifest_hash, split, provenance):
    return digest({"schema": SCHEMA_VERSION, "config": config, "manifest": manifest_hash,
                   "split": split, "provenance": provenance, "feature": "official_rae_encode_spatial_mean"})


def stratified_indices(labels, count, seed):
    labels = np.asarray(labels)
    classes = sorted(np.unique(labels).tolist())
    if count < len(classes):
        raise ValueError("count must include every class")
    rng = np.random.default_rng(seed)
    quotas = {c: count // len(classes) + (j < count % len(classes)) for j, c in enumerate(classes)}
    chosen = []
    for c in classes:
        candidates = np.flatnonzero(labels == c)
        if len(candidates) < quotas[c]:
            raise ValueError("insufficient class samples")
        chosen.extend(rng.choice(candidates, quotas[c], replace=False).tolist())
    return sorted(chosen)


def make_manifest(train_labels, test_labels, profile, dataset_sha256, seed=42):
    n_train, n_val, n_test = profile["train"], profile["validation"], profile["test"]
    selected = stratified_indices(train_labels, n_train + n_val, seed)
    rng = np.random.default_rng(seed + 1)
    train, val = [], []
    labels = np.asarray(train_labels)
    for c in sorted(np.unique(labels).tolist()):
        group = np.asarray([i for i in selected if labels[i] == c])
        rng.shuffle(group)
        train.extend(group[:n_train // 10].tolist())
        val.extend(group[n_train // 10:].tolist())
    if len(train) != n_train or len(val) != n_val:
        raise ValueError("profile requires class-balanced train and validation counts")
    test = stratified_indices(test_labels, n_test, seed + 2)
    splits = {}
    for name, indices, source in (("train", train, "train"), ("validation", val, "train"), ("test", test, "test")):
        splits[name] = [{"id": f"cifar10:{source}:{i:05d}", "index": int(i),
                         "label": int((train_labels if source == "train" else test_labels)[i]),
                         "source": source} for i in sorted(indices)]
    return {"dataset": "CIFAR-10 Python archive", "dataset_sha256": dataset_sha256,
            "split_seed": seed, "splits": splits}


def pool_bchw(z):
    if z.ndim != 4:
        raise ValueError("expected BCHW")
    return z.mean(axis=(2, 3))


def bchw_tokens(z):
    b, c, h, w = z.shape
    return z.transpose(0, 2, 3, 1).reshape(b * h * w, c)


def tokens_bchw(x, shape):
    b, c, h, w = shape
    if x.shape != (b * h * w, c):
        raise ValueError("token shape mismatch")
    return x.reshape(b, h, w, c).transpose(0, 3, 1, 2)


def environment():
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__}
    for name in ("torch", "torchvision", "sklearn", "transformers"):
        try:
            module = __import__(name)
            info[name] = module.__version__
        except (ImportError, OSError):
            info[name] = "unavailable"
    try:
        from threadpoolctl import threadpool_info
        info["threadpools"] = threadpool_info()
    except ImportError:
        info["threadpools"] = "unavailable"
    return info


def git_info(path):
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=path, text=True))
        return {"commit": head, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
