import json
import subprocess
import sys
from pathlib import Path


def test_smoke_command_runs_end_to_end(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs" / "featurebench_smoke.json").read_text(encoding="utf-8-sig"))
    config["output_dir"] = str(tmp_path / "runs")
    config_path = tmp_path / "smoke.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "featurebench", "smoke", "--config", str(config_path)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    run_dir = Path(result.stdout.strip().splitlines()[-1])

    assert (run_dir / "report.md").exists()
    assert "synthetic" in (run_dir / "report.md").read_text(encoding="utf-8")
    assert json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))["status"] == "complete"


def test_smoke_export_commands(tmp_path: Path):
    from featurebench.export import verify_export
    from featurebench.pipeline import extract_stage, fit_stage, evaluate_stage, load_config
    from featurebench.pipeline import recommend_stage
    from featurebench.reporting import build_report

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/featurebench_smoke.json")
    config["output_dir"] = str(tmp_path / "runs")
    run_dir = extract_stage(config)
    fit_stage(run_dir)
    evaluate_stage(run_dir)
    recommend_stage(run_dir)
    build_report(run_dir)
    for method in ("full_fp32", "full_fp16_storage", "pca4"):
        result = subprocess.run(
            [sys.executable, "-m", "featurebench", "export", "--run-dir", str(run_dir),
             "--method", method, "--shards", "2", "--out", str(tmp_path / "exports")],
            cwd=root, check=True, capture_output=True, text=True,
        )
        manifest = verify_export(Path(result.stdout.strip().splitlines()[-1]))
        assert manifest["status"] == "synthetic"
        assert manifest["total_count"] == 120
        assert set(manifest["queries"]) == {"validation", "test"}
    result = subprocess.run(
        [sys.executable, "-m", "featurebench", "export-synthetic", "--count", "20", "--dim", "4",
         "--dtype", "float16", "--shards", "2", "--out", str(tmp_path / "exports")],
        cwd=root, check=True, capture_output=True, text=True,
    )
    assert verify_export(Path(result.stdout.strip().splitlines()[-1]))["status"] == "synthetic"

