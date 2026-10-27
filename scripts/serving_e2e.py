"""Verify native multi-process or Compose serving against stored float64 oracles."""
import argparse
import contextlib
import json
import os
import platform
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from threadpoolctl import threadpool_limits
from featurebench.core import atomic_json
from featurebench.export import export_run


def ready(url, processes=(), timeout=120):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if any(p.poll() is not None for p in processes):
            raise RuntimeError("serving process exited before readiness; inspect log")
        try:
            with urllib.request.urlopen(url + "/readyz", timeout=2) as result:
                if result.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.2)
    raise TimeoutError(f"readiness timeout: {url}")


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def checked_json(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise RuntimeError(f"{command}: {result.stderr}\n{result.stdout}")
    return json.loads(result.stdout)


def readonly_mount_permissions(folder):
    """Allow the unprivileged serving UID to read this explicitly mounted export."""
    folder.chmod(folder.stat().st_mode | 0o555)
    for path in folder.iterdir():
        if path.is_file():
            path.chmod(path.stat().st_mode | 0o444)


@contextlib.contextmanager
def native(folder, binary):
    manifest = json.loads((folder / "manifest.json").read_text())
    processes = []
    with tempfile.TemporaryDirectory(prefix="serving-", dir=Path("serving_exports")) as temporary:
        runtime = Path(temporary)
        with (runtime / "processes.log").open("w+") as log:
            try:
                targets = []
                for shard in range(manifest["shard_count"]):
                    replicas = []
                    for replica in range(2):
                        grpc, admin = port(), port()
                        replicas.append(f"127.0.0.1:{grpc}")
                        processes.append(subprocess.Popen([str(binary("shard")), "--export", str(folder),
                            "--shard-index", str(shard), "--listen", replicas[-1],
                            "--admin", f"127.0.0.1:{admin}", "--workers", "2",
                            "--max-concurrent-searches", "2"], stdout=log, stderr=log))
                    targets.append(replicas)
                config = runtime / "coordinator.json"
                config.write_text(json.dumps({"dim": manifest["dim"], "shards": targets}))
                url = f"http://127.0.0.1:{port()}"
                processes.append(subprocess.Popen([str(binary("coordinator")), "--config", str(config),
                    "--listen", url.removeprefix("http://"), "--shard-timeout", "1s",
                    "--request-timeout", "3s"], stdout=log, stderr=log))
                ready(url, processes)
                yield url
            except Exception:
                log.flush()
                log.seek(0)
                print(log.read()[-12000:])
                raise
            finally:
                for process in reversed(processes):
                    process.terminate()
                for process in processes:
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--methods", default="full_fp32,full_fp16_storage,pca64")
    parser.add_argument("--shard-counts", default="1,2,4")
    parser.add_argument("--compose", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suffix = ".exe" if platform.system() == "Windows" else ""
    binary = lambda name: (Path("serving/bin") / (name + suffix)).resolve()
    reports = []
    for method in args.methods.split(","):
        for shards in map(int, args.shard_counts.split(",")):
            with threadpool_limits(limits=1):
                folder = export_run(args.run_dir, method, shards, Path("serving_exports")).resolve()
            environment = os.environ.copy()
            compose = ["docker", "compose", "-f", "deploy/compose/docker-compose.yml"]
            if args.compose:
                if shards != 2:
                    raise ValueError("Compose topology requires --shard-counts 2")
                readonly_mount_permissions(folder)
                runtime = Path("serving_exports/runtime")
                runtime.mkdir(exist_ok=True)
                config = runtime / "coordinator.json"
                config.write_text(json.dumps({"dim": json.loads((folder / "manifest.json").read_text())["dim"],
                    "shards": [["shard-0-a:9090", "shard-0-b:9090"], ["shard-1-a:9090", "shard-1-b:9090"]]}))
                environment.update(EXPORT_DIR=str(folder), COORDINATOR_CONFIG=str(config.resolve()))
                manager = contextlib.nullcontext("http://127.0.0.1:18080")
            else:
                manager = native(folder, binary)
            try:
                if args.compose:
                    subprocess.run(compose + ["up", "-d", "--build", "--wait"], env=environment, check=True, timeout=900)
                with manager as url:
                    report = checked_json([str(binary("verify")), "--export", str(folder), "--http", url,
                                           "--benchmark-results", str(args.run_dir / "results.json")])
                    report.update(shard_count=shards, transport="compose" if args.compose else "native_http")
                    load = checked_json([str(binary("loadgen")), "--export", str(folder), "--url", url,
                                         "--duration", "2s", "--warmup", "1s", "--concurrency", "4"])
                    if load["errors"]:
                        raise RuntimeError(f"load errors: {load['errors']}")
                    report["loadgen"] = load
                    reports.append(report)
                    atomic_json(args.output, {"complete": False, "reports": reports})
                    print(json.dumps({k: v for k, v in report.items() if k != "loadgen"}), flush=True)
            except Exception:
                if args.compose:
                    subprocess.run(compose + ["logs", "--no-color", "--tail", "60"], env=environment, check=False, timeout=30)
                raise
            finally:
                if args.compose:
                    subprocess.run(compose + ["down", "--remove-orphans"], env=environment, check=True, timeout=60)
    atomic_json(args.output, {"complete": True, "reports": reports})


if __name__ == "__main__":
    main()
