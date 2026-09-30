from pathlib import Path

import numpy as np

from featurebench.core import atomic_json
from featurebench.evaluation import break_even, neighbor_metrics, recommend, search, unit
from featurebench.pipeline import recommend_stage


def test_exact_block_search_is_deterministic_and_excludes_self():
    gallery = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]],
        dtype=np.float32,
    )
    gallery_ids = np.array(["b", "a", "c", "d", "e"])
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    query_ids = np.array(["b", "query"])

    unblocked, _ = search(gallery, queries, gallery_ids, query_ids, top=10, block=10)
    blocked, _ = search(gallery, queries, gallery_ids, query_ids, top=10, block=1)

    assert unblocked == blocked
    assert unblocked[0][0] == "a"
    assert "b" not in unblocked[0]

    labels = {"a": 1, "b": 1, "c": 2, "d": 2, "e": 1}
    metrics = neighbor_metrics(unblocked, labels, query_ids, np.array([1, 2]), unblocked)
    np.testing.assert_array_equal(metrics["knn_accuracy"], [1.0, 0.0])
    np.testing.assert_array_equal(metrics["neighbor_overlap_at_10"], [1.0, 1.0])
    assert metrics["precision_at_10"][0] == 0.5
    assert np.isfinite(unit(np.zeros((1, 3)))).all()
    assert break_even(1000, 3072, 512) == 1


def test_recommendation_uses_validation_metrics_only(tmp_path: Path):
    rows = [
        {
            "method_id": "full",
            "method": "full_fp32",
            "validation_precision_at_10": 0.80,
            "deployment_bytes": 1000,
            "test_precision_at_10": 0.10,
            "test_knn_accuracy": 0.10,
        },
        {
            "method_id": "pca",
            "method": "pca",
            "validation_precision_at_10": 0.79,
            "deployment_bytes": 200,
            "test_precision_at_10": 0.10,
            "test_knn_accuracy": 0.10,
        },
    ]
    atomic_json(tmp_path / "results.json", {"rows": rows})
    selected = recommend_stage(tmp_path)["selected"]

    rows[1]["test_precision_at_10"] = 0.99
    atomic_json(tmp_path / "results.json", {"rows": rows})
    assert recommend_stage(tmp_path)["selected"] == selected == "pca"

    validation_rows = [
        {"method_id": "full", "method": "full_fp32", "precision_at_10": 0.8, "deployment_bytes": 1000},
        {"method_id": "pca", "method": "pca", "precision_at_10": 0.5, "deployment_bytes": 200},
    ]
    result = recommend(validation_rows, require_smaller_than_full=True)
    assert result["status"] == "no_feasible_compression"

