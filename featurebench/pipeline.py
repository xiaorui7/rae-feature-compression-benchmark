import csv
import json
import time
from pathlib import Path

import numpy as np

from .core import atomic_json, atomic_npz, cache_key, digest, environment, git_info, read_cache
from .evaluation import bootstrap_diff, break_even, fit_probe, neighbor_metrics, recommend, reconstruction, search, unit
from .methods import Adapter, FP16, Full, PCAMethod, load_method
from .source import build_manifest, extract, provenance


ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    if config["profile"] not in ("smoke", "pilot", "benchmark_small"):
        raise ValueError("Unknown profile")
    return config


def run_dir_for(config):
    return ROOT / config["output_dir"] / f"{config['profile']}-{digest(config)[:12]}"


def setup(config):
    run_dir = run_dir_for(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "config.json", config)
    atomic_json(run_dir / "run_info.json", {"profile": config["profile"], "status": "in_progress",
                "protocol_id": digest({"schema": 1, "representation": "pooled_official_rae" if config["profile"] != "smoke" else "synthetic",
                                       "config": config["preprocess"]})[:16],
                "code": git_info(ROOT), "official": git_info(ROOT / config["rae_root"]),
                "environment": environment(), "split_seed": config["split_seed"], "started_at": time.time()})
    return run_dir


def extract_stage(config, max_runtime_minutes=120):
    run_dir = setup(config)
    if config["profile"] == "smoke":
        rng = np.random.default_rng(42)
        splits = {}
        for name, count in (("train", 120), ("validation", 30), ("test", 30)):
            labels = np.arange(count) % 3
            data = rng.normal(size=(count, 16)).astype(np.float32) + labels[:, None] * 0.15
            ids = np.array([f"synthetic:{name}:{i:04d}" for i in range(count)])
            splits[name] = [{"id": str(i), "label": int(y)} for i, y in zip(ids, labels)]
            key = digest({"profile": "smoke", "split": name, "schema": 1})
            atomic_npz(run_dir / "features" / f"{name}.npz", key=np.array(key), ids=ids, labels=labels, features=data)
        manifest = {"dataset": "SYNTHETIC_ONLY", "split_seed": 42, "splits": splits}
        atomic_json(run_dir / "manifest.json", manifest)
        atomic_json(run_dir / "extract.json", {"status": "synthetic", "splits": {s: {"count": len(v)} for s, v in splits.items()}})
        return run_dir
    manifest = build_manifest(ROOT, config)
    atomic_json(run_dir / "manifest.json", manifest)
    prov = provenance(ROOT, config)
    atomic_json(run_dir / "provenance.json", prov)
    mh = digest(manifest)
    results = extract(ROOT, run_dir, config, manifest, mh, prov, max_runtime_minutes * 60)
    atomic_json(run_dir / "extract.json", {"status": "actual_measurement" if all(v["status"] in ("actual_measurement", "reused_verified_cache") for v in results.values()) else "partial",
                                           "manifest_hash": mh, "splits": results})
    return run_dir


def load_features(run_dir):
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if config["profile"] == "smoke":
        keys = {s: digest({"profile": "smoke", "split": s, "schema": 1}) for s in manifest["splits"]}
    else:
        prov = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
        keys = {s: cache_key(config, digest(manifest), s, prov) for s in manifest["splits"]}
    out = {}
    for s, rows in manifest["splits"].items():
        ids, labels, features = read_cache(run_dir / "features" / f"{s}.npz", keys[s], len(rows))
        if ids.tolist() != [r["id"] for r in rows] or labels.tolist() != [r["label"] for r in rows]:
            raise ValueError(f"Manifest mismatch in {s}")
        out[s] = {"ids": ids, "labels": labels, "x": features}
    return config, manifest, out


def methods_for(config, xdim):
    dims = config["pca_dims"]
    if max(dims) > xdim:
        raise ValueError("PCA dimensions exceed actual input dimension")
    items = [("full_fp32", Full()), ("full_fp16_storage", FP16())]
    items += [(f"pca{k}", PCAMethod(k, config["split_seed"])) for k in sorted(dims, reverse=True)]
    items += [(f"linear_adapter{k}_s{seed}", Adapter(k, seed, epochs=config["adapter_epochs"],
                batch=config["adapter_batch"], patience=5)) for k in config["adapter_dims"] for seed in config["adapter_seeds"]]
    return items


def write_deployment_states(run_dir, method_id, model):
    folder = run_dir / "models" / "deployment"
    encoder = folder / f"{method_id}_encoder.npz"
    inverse = folder / f"{method_id}_inverse.npz"
    if isinstance(model, PCAMethod):
        atomic_npz(encoder, components=model.components, internal_mean=model.mean)
        atomic_npz(inverse, components=model.components, internal_mean=model.mean)
    elif isinstance(model, Adapter):
        atomic_npz(encoder, down_w=model.weights[0], down_b=model.weights[1])
        atomic_npz(inverse, up_w=model.weights[2], up_b=model.weights[3])
    else:
        atomic_npz(encoder, method=np.array(model.name))
        atomic_npz(inverse, method=np.array("identity"))
    return encoder.stat().st_size, inverse.stat().st_size


def fit_stage(run_dir, max_runtime_minutes=120):
    config, manifest, data = load_features(run_dir)
    train = data["train"]["x"]
    val = data["validation"]["x"]
    mu = train.mean(axis=0).astype(np.float32)
    atomic_npz(run_dir / "models" / "train_mean.npz", mean=mu)
    started = time.monotonic()
    summary = []
    prior = {}
    if (run_dir / "fit.json").exists():
        prior = {r["method_id"]: r for r in json.loads((run_dir / "fit.json").read_text(encoding="utf-8"))["models"]}
    all_items = methods_for(config, train.shape[1])
    shared_pca = None
    for method_id, model in all_items:
        target = run_dir / "models" / f"{method_id}.npz"
        if target.exists():
            existing = load_method(target)
            if hasattr(model, "dim") and existing.metadata()["dim"] != model.dim:
                raise ValueError(f"Existing method dimension mismatch: {target}")
            enc_bytes, inv_bytes = write_deployment_states(run_dir, method_id, existing)
            previous_time = prior.get(method_id, {}).get("fit_seconds")
            timing_type = prior.get(method_id, {}).get(
                "fit_timing_type", "original_fit" if previous_time is not None else "not_recorded"
            )
            summary.append({"method_id": method_id, "status": "reused_verified_model", "metadata": existing.metadata(),
                            "fit_seconds": previous_time, "fit_timing_type": timing_type,
                            "state_bytes": target.stat().st_size, "encoder_state_bytes": enc_bytes,
                            "inverse_state_bytes": inv_bytes})
            if method_id == f"pca{max(config['pca_dims'])}":
                shared_pca = existing
            continue
        if time.monotonic() - started >= max_runtime_minutes * 60:
            summary.append({"method_id": method_id, "status": "not_run_budget"})
            continue
        t0 = time.perf_counter()
        if isinstance(model, PCAMethod) and shared_pca is not None and model.dim < shared_pca.dim:
            model.components = shared_pca.components[:model.dim].copy()
            model.mean = shared_pca.mean.copy()
        else:
            model.fit(train - mu, val - mu)
        if isinstance(model, PCAMethod) and model.dim == max(config["pca_dims"]):
            shared_pca = model
        model.save(target)
        enc_bytes, inv_bytes = write_deployment_states(run_dir, method_id, model)
        summary.append({"method_id": method_id, "status": "synthetic" if config["profile"] == "smoke" else "actual_measurement",
                        "fit_seconds": time.perf_counter() - t0, "metadata": model.metadata(),
                        "fit_timing_type": "original_fit",
                        "state_bytes": target.stat().st_size, "encoder_state_bytes": enc_bytes,
                        "inverse_state_bytes": inv_bytes})
        atomic_json(run_dir / "fit.json", {"models": summary})
    atomic_json(run_dir / "fit.json", {"models": summary})
    return summary


def timed_transform(model, x):
    for _ in range(2):
        model.transform(x[:min(16, len(x))])
    times = []
    for _ in range(5):
        start = time.perf_counter()
        model.transform(x)
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def evaluate_stage(run_dir, max_runtime_minutes=120):
    config, manifest, data = load_features(run_dir)
    with np.load(run_dir / "models" / "train_mean.npz", allow_pickle=False) as obj:
        mu = obj["mean"].copy()
    fit_info = json.loads((run_dir / "fit.json").read_text(encoding="utf-8"))["models"]
    existing = {r["method_id"]: r for r in fit_info if (run_dir / "models" / f"{r['method_id']}.npz").exists()}
    protocol = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))["protocol_id"]
    data_status = "synthetic" if config["profile"] == "smoke" else "pilot" if config["profile"] == "pilot" else "actual_measurement"
    labels_by_id = dict(zip(data["train"]["ids"].tolist(), data["train"]["labels"].tolist()))
    base_neighbors = {}
    all_rows, scalar_records, query_records = [], [], {}
    started = time.monotonic()
    for method_id in [x[0] for x in methods_for(config, data["train"]["x"].shape[1])]:
        if method_id not in existing:
            all_rows.append({"method_id": method_id, "status": "not_run_fit"})
            scalar_records.append({"method": method_id, "dim": None, "dtype": None, "seed": None,
                                   "split": "test", "protocol_id": protocol, "metric": "not_run",
                                   "value": None, "sample_count": len(data["test"]["ids"]), "status": "not_run_fit"})
            continue
        if time.monotonic() - started >= max_runtime_minutes * 60:
            all_rows.append({"method_id": method_id, "status": "not_run_budget"})
            scalar_records.append({"method": method_id, "dim": None, "dtype": None, "seed": None,
                                   "split": "test", "protocol_id": protocol, "metric": "not_run",
                                   "value": None, "sample_count": len(data["test"]["ids"]), "status": "not_run_budget"})
            continue
        model_path = run_dir / "models" / f"{method_id}.npz"
        load_start = time.perf_counter()
        model = load_method(model_path)
        model_load_seconds = time.perf_counter() - load_start
        representations = {s: model.transform(v["x"] - mu) for s, v in data.items()}
        for s, values in representations.items():
            path = run_dir / "representations" / method_id / f"{s}.npz"
            atomic_npz(path, ids=data[s]["ids"], labels=data[s]["labels"], vectors=values)
        gallery_path = run_dir / "representations" / method_id / "train.npz"
        for _ in range(2):
            with np.load(gallery_path, allow_pickle=False) as f:
                f["vectors"].astype(np.float32)
        gallery_load_times = []
        for _ in range(5):
            load_start = time.perf_counter()
            with np.load(gallery_path, allow_pickle=False) as f:
                f["vectors"].astype(np.float32)
            gallery_load_times.append(time.perf_counter() - load_start)
        encoder_path = run_dir / "models" / "deployment" / f"{method_id}_encoder.npz"
        inverse_path = run_dir / "models" / "deployment" / f"{method_id}_inverse.npz"
        if not encoder_path.exists() or not inverse_path.exists():
            write_deployment_states(run_dir, method_id, model)
        deployment_bytes = gallery_path.stat().st_size + encoder_path.stat().st_size + (run_dir / "models" / "train_mean.npz").stat().st_size
        probe = fit_probe(representations["train"], data["train"]["labels"], representations["validation"], data["validation"]["labels"])
        row = {"method_id": method_id, **model.metadata(), "status": data_status,
               "gallery_raw_bytes": representations["train"].nbytes,
               "gallery_file_bytes": gallery_path.stat().st_size,
               "model_state_bytes": model_path.stat().st_size,
               "encoder_state_bytes": encoder_path.stat().st_size,
               "inverse_state_bytes": inverse_path.stat().st_size,
               "normalization_state_bytes": (run_dir / "models" / "train_mean.npz").stat().st_size,
               "deployment_bytes": deployment_bytes,
               "probe_C": probe["C"], "probe_fit_seconds": probe["total_fit_seconds"],
               "model_load_seconds": model_load_seconds,
               "gallery_load_convert_seconds_median": float(np.median(gallery_load_times)),
               "probe_head_raw_bytes": int(probe["model"].coef_.nbytes + probe["model"].intercept_.nbytes),
               "probe_convergence_warning": probe["any_convergence_warning"],
               "transform_seconds_median": timed_transform(model, data["test"]["x"] - mu),
               "encoder_excluded_from_query_timing": True,
               "cpu_threads": 1,
               "search_ram_dtype": "float32"}
        baseline_payload = data["train"]["x"].shape[1] * 4
        method_payload = row["dim"] * (2 if row["dtype"] == "float16" else 4)
        extra_state = max(0, row["encoder_state_bytes"] - (run_dir / "models" / "deployment" / "full_fp32_encoder.npz").stat().st_size)
        row["break_even_gallery_count_estimate"] = break_even(extra_state, baseline_payload, method_payload)
        if row["break_even_gallery_count_estimate"] is not None:
            scalar_records.append({"method": row["method"], "dim": row["dim"], "dtype": row["dtype"],
                                   "seed": row.get("seed"), "split": "gallery", "protocol_id": protocol,
                                   "metric": "break_even_gallery_count", "value": row["break_even_gallery_count_estimate"],
                                   "sample_count": len(data["train"]["ids"]), "status": "analytical_estimate"})
        for split in ("validation", "test"):
            values = data[split]
            encoded = representations[split]
            restored = model.inverse_transform(encoded.astype(np.float32)) + mu
            rec = reconstruction(values["x"], restored)
            search_args = (representations["train"].astype(np.float32), encoded.astype(np.float32),
                           data["train"]["ids"], values["ids"])
            search(*search_args, block=config["search_block"])
            search_times = []
            for _ in range(5):
                neighbors, search_seconds = search(*search_args, block=config["search_block"])
                search_times.append(search_seconds)
            if method_id == "full_fp32":
                base_neighbors[split] = neighbors
            metrics = neighbor_metrics(neighbors, labels_by_id, values["ids"], values["labels"], base_neighbors.get(split))
            row[f"{split}_search_seconds"] = float(np.median(search_times))
            row[f"{split}_probe_accuracy"] = float((probe["model"].predict(unit(encoded.astype(np.float32))) == values["labels"]).mean())
            row.update({f"{split}_{key}": value for key, value in rec.items()})
            row.update({f"{split}_{key}": float(value.mean()) for key, value in metrics.items()})
            query_records[(method_id, split)] = {k: v.tolist() for k, v in metrics.items()}
            for metric, value in {**rec, **{k: float(v.mean()) for k, v in metrics.items()},
                                  "probe_accuracy": row[f"{split}_probe_accuracy"]}.items():
                scalar_records.append({"method": row["method"], "dim": row["dim"], "dtype": row["dtype"],
                                       "seed": row.get("seed"), "split": split, "protocol_id": protocol,
                                       "metric": metric, "value": value, "sample_count": len(values["ids"]),
                                       "status": data_status})
        all_rows.append(row)
        atomic_json(run_dir / "results.json", {"rows": all_rows, "metrics": scalar_records,
                                                "query_metrics": {f"{k[0]}:{k[1]}": v for k, v in query_records.items()}})
    for row in all_rows:
        if row.get("status") not in (data_status,):
            continue
        for metric in ("knn_accuracy", "precision_at_10"):
            key = (row["method_id"], "test")
            if key not in query_records:
                continue
            row[f"test_{metric}_diff_95ci"] = bootstrap_diff(query_records[key][metric],
                query_records[("full_fp32", "test")][metric])
    atomic_json(run_dir / "results.json", {"rows": all_rows, "metrics": scalar_records,
                                            "query_metrics": {f"{k[0]}:{k[1]}": v for k, v in query_records.items()}})
    with open(run_dir / "results.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "dim", "dtype", "seed", "split", "protocol_id", "metric", "value", "sample_count", "status"])
        writer.writeheader()
        writer.writerows(scalar_records)
    return all_rows


def recommend_stage(run_dir, max_drop_pp=2, require_smaller_than_full=False):
    data = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    rows = [r for r in data["rows"] if "validation_precision_at_10" in r]
    selected = recommend([{"method_id": r["method_id"], "method": r["method"],
                           "precision_at_10": r["validation_precision_at_10"],
                           "deployment_bytes": r["deployment_bytes"]} for r in rows],
                         max_drop_pp, require_smaller_than_full)
    selected["selection_split"] = "validation_only"
    selected["max_drop_pp"] = max_drop_pp
    if selected["selected"]:
        chosen = next(r for r in rows if r["method_id"] == selected["selected"])
        selected["test_audit"] = {"precision_at_10": chosen["test_precision_at_10"],
                                  "knn_accuracy": chosen["test_knn_accuracy"],
                                  "met_validation_threshold": chosen["test_precision_at_10"] >= selected["threshold"]}
    atomic_json(run_dir / "recommendation.json", selected)
    return selected
