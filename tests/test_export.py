import json
import struct

import numpy as np
import pytest

from featurebench import export as exporting


@pytest.fixture
def arrays():
    return (["c", "a", "b", "e", "d"], np.array(
        [[1, 0], [1, 0], [0, 1], [-1, 0], [0, 0]], dtype=np.float32), [0, 0, 1, 2, 3])


def publish(tmp_path, gallery, shards=2):
    return exporting.export_arrays(tmp_path, {"kind": "synthetic"}, "synthetic", gallery,
                                   {"test": (["q"], np.array([[1, 0]], dtype=gallery[1].dtype), [0])}, shards)


@pytest.mark.parametrize("dtype,code", [("float32", 1), ("float16", 2)])
def test_binary_contract(tmp_path, dtype, code):
    values = np.array([[0, -0.0, 0.5], [2**-24, -3.25, 2]], dtype=dtype)
    path = tmp_path / "vectors.vec"
    exporting.write_vectors(path, values)
    assert path.read_bytes()[:28] == struct.pack("<8sIIIQ", b"RAEVEC01", 1, code, 3, 2)
    assert path.stat().st_size == 28 + values.nbytes
    np.testing.assert_array_equal(exporting.read_vectors(path).view(f"u{values.itemsize}"),
                                  values.view(f"u{values.itemsize}"))


@pytest.mark.parametrize("shards", [1, 2, 4])
def test_balanced_exact_coverage_and_reuse(tmp_path, arrays, shards):
    folder = publish(tmp_path, arrays, shards)
    manifest = exporting.verify_export(folder)
    counts = [r["count"] for r in manifest["shards"]]
    assert max(counts) - min(counts) <= 1
    found = []
    for index, record in enumerate(manifest["shards"]):
        ids = json.loads((folder / record["ids_file"]).read_text())
        assert ids == sorted(arrays[0])[index::shards]
        found.extend(ids)
        values = exporting.read_vectors(folder / record["vec_file"])
        for name, value in zip(ids, values):
            np.testing.assert_array_equal(value, arrays[1][arrays[0].index(name)])
    assert sorted(found) == sorted(arrays[0])
    assert publish(tmp_path, arrays, shards) == folder


@pytest.mark.parametrize("suffix", ["vec", "ids.json", "labels.json"])
def test_checksum_corruption_rejected(tmp_path, arrays, suffix):
    folder = publish(tmp_path, arrays)
    with (folder / f"shard-0000.{suffix}").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        exporting.verify_export(folder)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        publish(tmp_path, arrays)


def test_oracle_ties_zero_and_scores(arrays):
    result = exporting.oracle(arrays[0], arrays[1], ["q", "zero"], np.array([[1, 0], [0, 0]], dtype=np.float32))
    hits = result["results"][0]["hits"]
    assert [h["id"] for h in hits] == ["a", "c", "b", "d", "e"]
    assert [h["score"] for h in hits] == [1, 1, 0, 0, -1]
    assert [h["id"] for h in result["results"][1]["hits"]] == sorted(arrays[0])


def test_oracle_uses_stored_half_values(tmp_path):
    gallery = np.array([[0.21, 1.2], [0.8, -0.5]], dtype=np.float16)
    query = np.array([[0.6, 0.7]], dtype=np.float16)
    result = exporting.oracle(["a", "b"], gallery, ["q"], query)
    g, q = gallery.astype(np.float64), query[0].astype(np.float64)
    scores = (g @ q) / (np.linalg.norm(g, axis=1) * np.linalg.norm(q))
    for hit in result["results"][0]["hits"]:
        assert hit["score"] == pytest.approx(scores[["a", "b"].index(hit["id"])], abs=1e-15)


def test_overlap_rejected_before_writing(tmp_path, arrays):
    with pytest.raises(ValueError, match="overlap"):
        exporting.export_arrays(tmp_path, {}, "synthetic", arrays,
                                {"test": (["a"], arrays[1][:1], [0])}, 2)
    assert not list(tmp_path.iterdir())


def test_failure_never_publishes_manifest(tmp_path, arrays, monkeypatch):
    def fail(*args, **kwargs):
        assert not list(tmp_path.glob("*/manifest.json"))
        raise RuntimeError("simulated oracle failure")

    monkeypatch.setattr(exporting, "oracle", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        publish(tmp_path, arrays)
    folder, = tmp_path.iterdir()
    assert list(folder.glob("shard-*.vec"))
    assert not (folder / "manifest.json").exists()
    with pytest.raises(FileNotFoundError):
        exporting.verify_export(folder)


def test_changed_content_changes_directory(tmp_path, arrays):
    first = publish(tmp_path, arrays)
    arrays[1][0, 0] = 2
    second = publish(tmp_path, arrays)
    assert first != second


@pytest.mark.parametrize("mutation", ["truncated", "trailing", "bad_magic", "nan"])
def test_bad_binary_rejected(tmp_path, mutation):
    path = tmp_path / "bad.vec"
    exporting.write_vectors(path, np.ones((1, 1), dtype=np.float32))
    data = path.read_bytes()
    if mutation == "truncated":
        data = data[:20]
    elif mutation == "trailing":
        data += b"x"
    elif mutation == "bad_magic":
        data = b"BADMAGIC" + data[8:]
    else:
        data = data[:28] + struct.pack("<f", float("nan"))
    path.write_bytes(data)
    with pytest.raises(ValueError):
        exporting.read_vectors(path)


def test_synthetic_reproducible(tmp_path):
    first = exporting.export_synthetic(20, 4, "float16", 3, 42, tmp_path, query_count=4)
    second = exporting.export_synthetic(20, 4, "float16", 3, 42, tmp_path, query_count=4)
    assert first == second
    assert exporting.verify_export(first)["status"] == "synthetic"


def test_oracle_across_gallery_and_query_blocks():
    ids = [f"g{i:05d}" for i in range(8200)][::-1]
    gallery = np.ones((8200, 2), dtype=np.float32)
    query_ids = [f"q{i}" for i in range(33)]
    result = exporting.oracle(ids, gallery, query_ids, np.ones((33, 2), dtype=np.float32))
    assert len(result["results"]) == 33
    for row in result["results"]:
        assert [hit["id"] for hit in row["hits"]] == sorted(ids)[:10]


@pytest.mark.parametrize("shards", [0, -1, 6])
def test_invalid_shard_count(tmp_path, arrays, shards):
    with pytest.raises(ValueError, match="Shard count"):
        publish(tmp_path, arrays, shards)
