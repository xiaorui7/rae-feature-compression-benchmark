import pytest
from featurebench.export import export_synthetic


def test_incomplete_destination_has_actionable_error(tmp_path):
    folder = export_synthetic(12, 3, "float32", 2, 42, tmp_path, 2)
    (folder / "manifest.json").unlink()
    with pytest.raises(RuntimeError, match="fresh --out"):
        export_synthetic(12, 3, "float32", 2, 42, tmp_path, 2)
