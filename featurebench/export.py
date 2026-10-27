"""Immutable, checksummed vector exports and an independent float64 oracle."""

import hashlib
import json
import os
import re
import struct
import tempfile
from pathlib import Path

import numpy as np

from .core import atomic_json, digest, file_hash

HEADER = struct.Struct("<8sIIIQ")
MAGIC = b"RAEVEC01"
VERSION = 1
DTYPES = {1: np.dtype("<f4"), 2: np.dtype("<f2")}


def validate_vectors(ids, vectors):
    ids = list(ids)
    if not ids or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("IDs must be nonempty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate IDs")
    if vectors.ndim != 2 or len(vectors) != len(ids) or vectors.shape[1] < 1:
        raise ValueError("Invalid vector shape")
    if vectors.dtype.kind != "f" or vectors.dtype.itemsize not in (2, 4):
        raise ValueError("Only float16 and float32 vectors are supported")
    if not np.isfinite(vectors).all():
        raise ValueError("Non-finite vector values")


def write_vectors(path, vectors):
    """Write a portable 28-byte header and raw values using atomic replacement."""
    path = Path(path)
    code = 1 if vectors.dtype.itemsize == 4 else 2
    values = np.ascontiguousarray(vectors, dtype=DTYPES[code])
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temp = handle.name
            handle.write(HEADER.pack(MAGIC, VERSION, code, values.shape[1], len(values)))
            handle.write(memoryview(values).cast("B"))
        os.replace(temp, path)
    finally:
        if temp and os.path.exists(temp):
            os.unlink(temp)


def read_vectors(path):
    path = Path(path)
    with path.open("rb") as handle:
        header = handle.read(HEADER.size)
        if len(header) != HEADER.size:
            raise ValueError("Truncated vector header")
        magic, version, code, dim, count = HEADER.unpack(header)
        if magic != MAGIC or version != VERSION or code not in DTYPES or not dim or not count:
            raise ValueError("Invalid vector header")
        dtype = DTYPES[code]
        if path.stat().st_size != HEADER.size + count * dim * dtype.itemsize:
            raise ValueError("Vector payload length mismatch")
        values = np.fromfile(handle, dtype=dtype, count=count * dim).reshape(count, dim)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite vector values")
    return values


def oracle(gallery_ids, gallery, query_ids, queries, k=10):
    """Score stored values in float64; bound scratch space by gallery/query blocks."""
    validate_vectors(gallery_ids, gallery)
    validate_vectors(query_ids, queries)
    if queries.shape[1] != gallery.shape[1] or k < 1:
        raise ValueError("Invalid oracle dimensions or k")
    if set(gallery_ids) & set(query_ids):
        raise ValueError("Query/gallery ID overlap")
    order = np.argsort(gallery_ids, kind="stable")
    ids = np.asarray(gallery_ids)[order]
    k = min(k, len(ids))
    results = []
    for start in range(0, len(queries), 32):
        q = queries[start:start + 32].astype(np.float64)
        qnorm = np.linalg.norm(q, axis=1)
        best = [(np.empty(0, dtype=np.int64), np.empty(0)) for _ in q]
        for offset in range(0, len(gallery), 8192):
            g = gallery[order[offset:offset + 8192]].astype(np.float64)
            denominator = qnorm[:, None] * np.linalg.norm(g, axis=1)[None, :]
            dots = q @ g.T
            scores = np.divide(dots, denominator, out=np.zeros_like(dots), where=denominator > 0)
            indices = np.arange(offset, offset + len(g))
            for row, values in enumerate(scores):
                candidates = np.concatenate((best[row][0], indices))
                values = np.concatenate((best[row][1], values))
                selected = np.lexsort((candidates, -values))[:k]
                best[row] = candidates[selected], values[selected]
        for row, (indices, scores) in enumerate(best):
            results.append({"id": query_ids[start + row], "hits": [
                {"id": str(ids[i]), "score": float(score)} for i, score in zip(indices, scores)
            ]})
    return {"k": k, "arithmetic": "float64", "order": "score_desc_id_asc", "results": results}


def _fingerprint(ids, vectors, labels):
    return {"ids": digest(ids), "labels": digest(labels), "shape": list(vectors.shape),
            "dtype": vectors.dtype.name,
            "values": hashlib.sha256(memoryview(np.ascontiguousarray(
                vectors, dtype="<f4" if vectors.dtype.itemsize == 4 else "<f2"
            )).cast("B")).hexdigest()}


def _write_group(folder, prefix, ids, vectors, labels):
    vec = folder / f"{prefix}.vec"
    names = folder / f"{prefix}.ids.json"
    write_vectors(vec, vectors)
    atomic_json(names, ids)
    record = {"vec_file": vec.name, "ids_file": names.name, "count": len(ids),
              "sha256_vec": file_hash(vec), "sha256_ids": file_hash(names)}
    if labels is not None:
        path = folder / f"{prefix}.labels.json"
        atomic_json(path, labels)
        record.update(labels_file=path.name, sha256_labels=file_hash(path))
    return record


def export_arrays(out, source, status, gallery, queries, shards):
    """Publish manifest last. Existing exports are verified, never overwritten."""
    ids, vectors, labels = gallery
    validate_vectors(ids, vectors)
    if not 1 <= shards <= len(ids):
        raise ValueError("Shard count must be between 1 and gallery count")
    if status not in ("actual_measurement", "synthetic") or not queries:
        raise ValueError("Invalid data status or empty query selection")
    groups = {"train": gallery, **queries}
    for name, (names, values, targets) in groups.items():
        if name not in ("train", "validation", "test"):
            raise ValueError("Invalid split name")
        validate_vectors(names, values)
        if values.shape[1] != vectors.shape[1] or values.dtype != vectors.dtype:
            raise ValueError("Split dimension/dtype mismatch")
        if targets is not None and len(targets) != len(names):
            raise ValueError("Label count mismatch")
        if name != "train" and set(ids) & set(names):
            raise ValueError("Query/gallery ID overlap")
    identity = {"schema_version": VERSION, "source": source, "status": status,
                "shards": shards, "oracle_k": 10,
                "content": {s: _fingerprint(*g) for s, g in groups.items()}}
    export_id = digest(identity)
    folder = Path(out).resolve() / f"export-{export_id}"
    folder.parent.mkdir(parents=True, exist_ok=True)
    if folder.exists():
        if not (folder / "manifest.json").exists():
            raise RuntimeError(f"Incomplete export at {folder}: manifest.json is missing. "
                               "Inspect the interrupted export; choose a fresh --out directory "
                               "or remove only this incomplete export before retrying.")
        verify_export(folder)
        return folder
    folder.mkdir()
    manifest = {"schema_version": VERSION, "export_id": export_id, "identity": identity,
                "source": source, "status": status, "dtype": vectors.dtype.name,
                "dim": vectors.shape[1], "total_count": len(ids), "shard_count": shards,
                "sharding_rule": "id_ascending_round_robin", "shards": [], "queries": {}}
    order = np.argsort(ids, kind="stable")
    for shard in range(shards):
        rows = order[shard::shards]
        record = _write_group(folder, f"shard-{shard:04d}", [ids[i] for i in rows], vectors[rows],
                              None if labels is None else [labels[i] for i in rows])
        manifest["shards"].append({"index": shard, **record})
    for split, (names, values, targets) in queries.items():
        record = _write_group(folder, f"queries-{split}", names, values, targets)
        path = folder / f"oracle-{split}.json"
        atomic_json(path, oracle(ids, vectors, names, values))
        record.update(oracle_file=path.name, sha256_oracle=file_hash(path))
        manifest["queries"][split] = record
    atomic_json(folder / "manifest.json", manifest)
    return folder


def verify_export(folder):
    """Reject incomplete exports, corrupted payloads, and unsafe manifest paths."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != VERSION or digest(manifest["identity"]) != manifest["export_id"]:
        raise ValueError("Invalid export identity/version")
    if len(manifest["shards"]) != manifest["shard_count"]:
        raise ValueError("Invalid shard count")
    gallery_ids = []
    query_ids = []
    for record in [*manifest["shards"], *manifest["queries"].values()]:
        for kind in ("vec", "ids", "labels", "oracle"):
            key = f"{kind}_file"
            if key in record:
                name = record[key]
                if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
                    raise ValueError("Unsafe export path")
                if file_hash(folder / name) != record[f"sha256_{kind}"]:
                    raise ValueError(f"Checksum mismatch: {name}")
        values = read_vectors(folder / record["vec_file"])
        names = json.loads((folder / record["ids_file"]).read_text(encoding="utf-8"))
        validate_vectors(names, values)
        if (len(names) != record["count"] or values.shape[1] != manifest["dim"]
                or values.dtype.name != manifest["dtype"]):
            raise ValueError("Manifest vector metadata mismatch")
        if "labels_file" in record:
            labels = json.loads((folder / record["labels_file"]).read_text(encoding="utf-8"))
            if len(labels) != len(names):
                raise ValueError("Label count mismatch")
        if "oracle_file" in record:
            query_ids.extend(names)
        else:
            gallery_ids.extend(names)
    if len(gallery_ids) != manifest["total_count"] or len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError("Invalid gallery coverage")
    if set(query_ids) & set(gallery_ids):
        raise ValueError("Query/gallery ID overlap")
    return manifest


def export_run(run_dir, method, shards, out, query_splits=("validation", "test")):
    run_dir = Path(run_dir).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", method):
        raise ValueError("Invalid method ID")
    if not query_splits or len(set(query_splits)) != len(query_splits):
        raise ValueError("Query splits must be nonempty and unique")
    if any(s not in ("validation", "test") for s in query_splits):
        raise ValueError("Queries must be validation or test")
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    if info["status"] != "complete":
        raise ValueError("Export requires a completed benchmark run")
    original_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    row = next((r for r in results["rows"] if r["method_id"] == method), None)
    if row is None or row.get("status") not in ("actual_measurement", "pilot", "synthetic"):
        raise ValueError("Method has no completed result")
    data = {}
    for split in ("train", *query_splits):
        with np.load(run_dir / "representations" / method / f"{split}.npz", allow_pickle=False) as obj:
            ids, labels, vectors = obj["ids"].tolist(), obj["labels"].tolist(), obj["vectors"].copy()
        expected = original_manifest["splits"][split]
        if ids != [r["id"] for r in expected] or labels != [r["label"] for r in expected]:
            raise ValueError(f"Source manifest mismatch: {split}")
        data[split] = ids, vectors, labels
    status = "synthetic" if info["profile"] == "smoke" else "actual_measurement"
    if (row["status"] == "synthetic") != (status == "synthetic"):
        raise ValueError("Source data status mismatch")
    source = {"run_dir": str(run_dir), "protocol_id": info["protocol_id"], "method_id": method,
              "profile": info["profile"], "result_status": row["status"],
              "results_sha256": file_hash(run_dir / "results.json"),
              "manifest_sha256": file_hash(run_dir / "manifest.json")}
    return export_arrays(out, source, status, data["train"], {s: data[s] for s in query_splits}, shards)


def export_synthetic(count, dim, dtype, shards, seed, out, query_count=100):
    if count < 1 or dim < 1 or query_count < 1 or dtype not in ("float16", "float32"):
        raise ValueError("Invalid synthetic dimensions, counts, or dtype")
    rng = np.random.default_rng(seed)
    gallery = rng.standard_normal((count, dim), dtype=np.float32).astype(dtype)
    queries = rng.standard_normal((query_count, dim), dtype=np.float32).astype(dtype)
    source = {"kind": "synthetic", "seed": seed, "generator": "numpy_PCG64_standard_normal_float32",
              "numpy_version": np.__version__, "query_count": query_count}
    return export_arrays(out, source, "synthetic",
                         ([f"synthetic:gallery:{i:010d}" for i in range(count)], gallery, None),
                         {"test": ([f"synthetic:test:{i:010d}" for i in range(query_count)], queries, None)},
                         shards)
