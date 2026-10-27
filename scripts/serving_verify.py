"""Run real gRPC shard verification on existing representations; no model fitting."""

import argparse
import json
import platform
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from threadpoolctl import threadpool_limits

from featurebench.core import atomic_json
from featurebench.export import export_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--methods", default="full_fp32,full_fp16_storage,pca64")
    parser.add_argument("--bin-dir", type=Path, default=Path("serving/bin"))
    parser.add_argument("--exports", type=Path, default=Path("serving_exports"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suffix = ".exe" if platform.system() == "Windows" else ""
    reports = []
    for method in args.methods.split(","):
        with threadpool_limits(limits=1):
            folder = export_run(args.run_dir, method, 1, args.exports)
        with tempfile.TemporaryFile(mode="w+") as log:
            server = subprocess.Popen([str((args.bin_dir / f"shard{suffix}").resolve()),
                                       "--export", str(folder), "--workers", "4"], stdout=log, stderr=log)
            try:
                until = time.monotonic() + 30
                while True:
                    if server.poll() is not None:
                        log.seek(0)
                        raise RuntimeError(log.read())
                    try:
                        with urllib.request.urlopen("http://127.0.0.1:9091/readyz", timeout=1) as response:
                            if response.status == 200:
                                break
                    except (OSError, urllib.error.URLError):
                        pass
                    if time.monotonic() > until:
                        raise TimeoutError("shard readiness timeout")
                    time.sleep(0.1)
                command = [str((args.bin_dir / f"verify{suffix}").resolve()), "--export", str(folder),
                           "--benchmark-results", str(args.run_dir / "results.json")]
                result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=300)
                if not result.stdout.strip():
                    raise RuntimeError(result.stderr)
                report = json.loads(result.stdout)
                report["command"] = command
                reports.append(report)
                atomic_json(args.output, {"environment": platform.platform(), "reports": reports,
                                         "complete": False})
                print(json.dumps(report), flush=True)
                if result.returncode:
                    raise RuntimeError(result.stderr)
            finally:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
    atomic_json(args.output, {"environment": platform.platform(), "reports": reports,
                             "complete": True,
                             "note": "Correctness only; local Windows teardown uses process termination."})


if __name__ == "__main__":
    main()
