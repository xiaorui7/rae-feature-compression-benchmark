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

