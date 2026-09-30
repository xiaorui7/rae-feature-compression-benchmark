from pathlib import Path

import numpy as np
import pytest

from featurebench.evaluation import reconstruction
from featurebench.methods import Adapter, FP16, Full, PCAMethod, load_method


@pytest.fixture
def low_rank_features():
    rng = np.random.default_rng(1)
    coefficients = rng.normal(size=(40, 2)).astype(np.float32)
    return np.column_stack(
        (
            coefficients,
            coefficients[:, 0] + coefficients[:, 1],
            2 * coefficients[:, 0],
        )
    ).astype(np.float32)


@pytest.mark.parametrize(
    "model",
    [Full(), FP16(), PCAMethod(2), Adapter(2, epochs=2, patience=1)],
)
def test_method_serialization_roundtrip(tmp_path: Path, low_rank_features, model):
    validation = low_rank_features[:8]
    model.fit(low_rank_features, validation)
    checkpoint = tmp_path / f"{model.name}.npz"
    model.save(checkpoint)

    loaded = load_method(checkpoint)
    np.testing.assert_allclose(
        loaded.transform(validation),
        model.transform(validation),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        loaded.inverse_transform(loaded.transform(validation)),
        model.inverse_transform(model.transform(validation)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_pca_recovers_low_rank_features(low_rank_features):
    model = PCAMethod(2).fit(low_rank_features)
    restored = model.inverse_transform(model.transform(low_rank_features))

    assert reconstruction(low_rank_features, restored)["pooled_feature_mse"] < 1e-10

