import os
import sys
import time
from pathlib import Path

import numpy as np

from .core import atomic_npz, cache_key, digest, file_hash, git_info, make_manifest, read_cache


def provenance(root, config):
    repo = root / config["rae_root"]
    official = git_info(repo)
    cfg = repo / config["rae_config"]
    decoder = repo / "models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt"
    stats = repo / "models/stats/dinov2/wReg_base/imagenet1k/stat.pt"
    hf_root = Path.home() / ".cache/huggingface/hub/models--facebook--dinov2-with-registers-base"
    weight_files = sorted(hf_root.glob("snapshots/*/model.safetensors")) + sorted(hf_root.glob("snapshots/*/pytorch_model.bin"))
    if not weight_files:
        raise FileNotFoundError(f"Cached official DINOv2-B encoder weights required; no automatic download: {hf_root}")
    return {"official": official, "config_sha256": file_hash(cfg),
            "decoder_sha256": file_hash(decoder), "stats_sha256": file_hash(stats),
            "hf_weight_sha256": file_hash(weight_files[0]),
            "hf_weight_path": str(weight_files[0]),
            "extraction": "RAE.encode output after official latent stats; spatial mean to pooled x",
            "preprocess": config["preprocess"]}


def cifar_datasets(root, config):
    from torchvision import datasets, transforms
    from torchvision.transforms import InterpolationMode

    transform = transforms.Compose([
        transforms.Resize((config["preprocess"]["resize"], config["preprocess"]["resize"]),
                          interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
    ])
    data_dir = root / config["data_dir"]
    train = datasets.CIFAR10(root=str(data_dir), train=True, download=False, transform=transform)
    test = datasets.CIFAR10(root=str(data_dir), train=False, download=False, transform=transform)
    return train, test


def build_manifest(root, config):
    train, test = cifar_datasets(root, config)
    archive = root / config["data_dir"] / "cifar-10-python.tar.gz"
    if not archive.exists():
        raise FileNotFoundError(f"CIFAR archive required for provenance: {archive}")
    return make_manifest(train.targets, test.targets, config["counts"], file_hash(archive), config["split_seed"])


def instantiate_official(root, config, device):
    import torch

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    repo = (root / config["rae_root"]).resolve()
    sys.path.insert(0, str(repo / "src"))
    from utils.model_utils import instantiate_from_config
    from utils.train_utils import parse_configs

    before = Path.cwd()
    try:
        os.chdir(repo)
        cfg, *_ = parse_configs(config["rae_config"])
        model = instantiate_from_config(cfg).to(device).eval()
        model.requires_grad_(False)
        return model
    finally:
        os.chdir(before)


def extract(root, run_dir, config, manifest, manifest_hash, prov, limit_seconds=None):
    import torch
    from torch.utils.data import DataLoader, Dataset

    expected_dim = None
    results = {}
    started = time.monotonic()
    missing = []
    for split, rows in manifest["splits"].items():
        key = cache_key(config, manifest_hash, split, prov)
        path = run_dir / "features" / f"{split}.npz"
        if not path.exists():
            missing.append(split)
            continue
        ids, labels, data = read_cache(path, key, count=len(rows), dim=expected_dim)
        if ids.tolist() != [r["id"] for r in rows] or labels.tolist() != [r["label"] for r in rows]:
            raise ValueError(f"Cache IDs/labels differ: {path}")
        expected_dim = data.shape[1]
        results[split] = {"status": "reused_verified_cache", "seconds": 0, "count": len(ids), "dim": expected_dim}
    if not missing:
        return results

    train, test = cifar_datasets(root, config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Official RAE real extraction requires local CUDA GPU; smoke works on CPU")
    model = instantiate_official(root, config, device)

    class Indexed(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            row = self.rows[i]
            ds = train if row["source"] == "train" else test
            image, label = ds[row["index"]]
            if int(label) != row["label"]:
                raise ValueError(f"CIFAR label mismatch for {row['id']}")
            return image, label, row["id"]

    for split, rows in manifest["splits"].items():
        if split in results:
            continue
        key = cache_key(config, manifest_hash, split, prov)
        path = run_dir / "features" / f"{split}.npz"
        if limit_seconds is not None and time.monotonic() - started >= limit_seconds:
            results[split] = {"status": "not_run_budget"}
            continue
        loader = DataLoader(Indexed(rows), batch_size=config["extract_batch"], shuffle=False, num_workers=0)
        collected, ids, labels = [], [], []
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for images, y, names in loader:
                if limit_seconds is not None and time.monotonic() - started >= limit_seconds:
                    results[split] = {"status": "not_run_budget", "processed": len(ids)}
                    break
                z = model.encode(images.to(device))
                if z.ndim != 4 or z.shape[2:] != (16, 16):
                    raise ValueError(f"Unexpected official RAE output: {tuple(z.shape)}")
                pooled = z.mean(dim=(2, 3)).float().cpu().numpy()
                expected_dim = pooled.shape[1]
                collected.append(pooled)
                ids.extend(names)
                labels.extend(y.tolist())
        if len(ids) != len(rows):
            continue
        torch.cuda.synchronize()
        features = np.concatenate(collected).astype(np.float32)
        atomic_npz(path, key=np.array(key), ids=np.asarray(ids), labels=np.asarray(labels, dtype=np.int64), features=features)
        read_cache(path, key, count=len(rows), dim=expected_dim)
        results[split] = {"status": "actual_measurement", "count": len(ids), "dim": expected_dim,
                          "seconds": time.perf_counter() - t0,
                          "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
                          "device": torch.cuda.get_device_name(0), "decoder_loaded": True}
    return results
