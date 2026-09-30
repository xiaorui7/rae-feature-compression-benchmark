from pathlib import Path

import numpy as np
import pytest

from featurebench.core import (
    atomic_npz,
    bchw_tokens,
    cache_key,
    pool_bchw,
    read_cache,
    tokens_bchw,
)


def test_spatial_roundtrip_and_pool():
    spatial = np.arange(2 * 3 * 2 * 4, dtype=np.float32).reshape(2, 3, 2, 4)

    np.testing.assert_array_equal(tokens_bchw(bchw_tokens(spatial), spatial.shape), spatial)
    np.testing.assert_array_equal(pool_bchw(spatial), spatial.mean((2, 3)))


def test_cache_key_changes_with_protocol_inputs(tmp_path: Path):
    config = {"preprocess": {"resize": 256}}
    provenance = {"hash": "abc"}
    key = cache_key(config, "manifest-a", "train", provenance)

    assert key != cache_key({"preprocess": {"resize": 224}}, "manifest-a", "train", provenance)
    assert key != cache_key(config, "manifest-b", "train", provenance)
    assert key != cache_key(config, "manifest-a", "train", {"hash": "def"})

    cache = tmp_path / "cache.npz"
    atomic_npz(
        cache,
        key=np.array(key),
        ids=np.array(["sample-a"]),
        labels=np.array([1]),
        features=np.zeros((1, 3)),
    )
    assert read_cache(cache, key, count=1, dim=3)[2].shape == (1, 3)

    with pytest.raises(ValueError):
        read_cache(cache, "wrong-key")


def test_interrupted_cache_is_rejected(tmp_path: Path):
    cache = tmp_path / "cache.npz"
    cache.write_bytes(b"interrupted")

    with pytest.raises(Exception):
        read_cache(cache, "unused")

