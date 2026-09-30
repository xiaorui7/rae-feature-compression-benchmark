import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


def search(gallery, query, gallery_ids, query_ids, top=10, block=256):
    gallery, query = unit(gallery), unit(query)
    gallery_ids, query_ids = np.asarray(gallery_ids), np.asarray(query_ids)
    if len(gallery) != len(gallery_ids):
        raise ValueError("gallery IDs mismatch")
    order = np.argsort(gallery_ids, kind="stable")
    gallery, gallery_ids = gallery[order], gallery_ids[order]
    neighbors = []
    start = time.perf_counter()
    for i in range(0, len(query), block):
        scores = query[i:i + block] @ gallery.T
        scores[query_ids[i:i + block, None] == gallery_ids[None, :]] = -np.inf
        for row in scores:
            available = int(np.isfinite(row).sum())
            k = min(top, available)
            # Stable mergesort keeps the pre-sorted sample-ID order for exact ties.
            idx = np.argsort(-row, kind="stable")[:k]
            neighbors.append(gallery_ids[idx].tolist())
    return neighbors, time.perf_counter() - start


def neighbor_metrics(neighbors, labels_by_id, query_ids, query_labels, reference=None):
    votes, precision, overlap = [], [], []
    for j, found in enumerate(neighbors):
        if not found:
            raise ValueError("empty gallery after self exclusion")
        classes = [int(labels_by_id[x]) for x in found]
        first_five = classes[:5]
        counts = {c: first_five.count(c) for c in set(first_five)}
        votes.append(int(min(counts, key=lambda c: (-counts[c], c)) == query_labels[j]))
        precision.append(sum(c == query_labels[j] for c in classes[:10]) / len(classes[:10]))
        if reference is not None:
            overlap.append(len(set(found[:10]) & set(reference[j][:10])) / min(10, len(found), len(reference[j])))
    return {"knn_accuracy": np.asarray(votes, dtype=float),
            "precision_at_10": np.asarray(precision, dtype=float),
            **({"neighbor_overlap_at_10": np.asarray(overlap, dtype=float)} if reference is not None else {})}


def reconstruction(original, restored, eps=1e-12):
    original, restored = np.asarray(original, dtype=np.float32), np.asarray(restored, dtype=np.float32)
    if original.shape != restored.shape:
        raise ValueError("reconstruction shapes differ")
    error = ((original - restored) ** 2).sum(axis=1)
    denom = (original ** 2).sum(axis=1) + eps
    return {"pooled_feature_mse": float(error.sum() / original.size),
            "relative_squared_error": float(error.sum() / denom.sum())}


def bootstrap_diff(a, b, seed=2026, repeats=2000):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired query arrays required")
    rng = np.random.default_rng(seed)
    difference = a - b
    means = np.empty(repeats)
    for i in range(repeats):
        means[i] = difference[rng.integers(0, len(a), len(a))].mean()
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def fit_probe(train_x, train_y, val_x, val_y):
    best = None
    total_seconds = 0.0
    any_convergence_warning = False
    for c in (0.1, 1, 10):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ConvergenceWarning)
            model = LogisticRegression(C=c, solver="lbfgs", max_iter=1000)
            start = time.perf_counter()
            model.fit(unit(train_x), train_y)
            elapsed = time.perf_counter() - start
        total_seconds += elapsed
        any_convergence_warning |= any(issubclass(w.category, ConvergenceWarning) for w in captured)
        acc = float((model.predict(unit(val_x)) == val_y).mean())
        candidate = {"C": c, "model": model, "validation_accuracy": acc,
                     "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in captured),
                     "fit_seconds": elapsed}
        if best is None or acc > best["validation_accuracy"]:
            best = candidate
    best["total_fit_seconds"] = total_seconds
    best["any_convergence_warning"] = any_convergence_warning
    return best


def break_even(extra_state_bytes, full_bytes_per_vector, compressed_bytes_per_vector):
    saving = full_bytes_per_vector - compressed_bytes_per_vector
    return None if saving <= 0 else int(np.ceil(extra_state_bytes / saving))


def recommend(validation_rows, max_drop_pp=2.0, require_smaller_than_full=False):
    baseline = next(r for r in validation_rows if r["method"] == "full_fp32")
    threshold = baseline["precision_at_10"] - max_drop_pp / 100
    feasible = [r for r in validation_rows if r["precision_at_10"] >= threshold]
    if require_smaller_than_full:
        feasible = [r for r in feasible if r["deployment_bytes"] < baseline["deployment_bytes"]]
        if not feasible:
            return {"status": "no_feasible_compression", "selected": None, "threshold": threshold}
    selected = min(feasible, key=lambda r: (r["deployment_bytes"], -r["precision_at_10"], r["method_id"]))
    return {"status": "selected", "selected": selected["method_id"], "threshold": threshold,
            "validation_precision_at_10": selected["precision_at_10"],
            "validation_deployment_bytes": selected["deployment_bytes"]}
