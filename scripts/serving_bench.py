"""Repeated local-kind measurements. Raw trials are immutable once published."""
import contextlib
import ctypes
from ctypes import wintypes
import json
import hashlib
import os
import platform
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from threadpoolctl import threadpool_limits
from featurebench.core import atomic_json
from featurebench.export import export_run, export_synthetic, read_vectors
import serving_kind as cluster
from serving_e2e import checked_json, ready

ROOT = cluster.ROOT
REAL = ROOT.parents[1] / "featurebench_runs/benchmark_small-f44fa7c6b95a"
EXPORTS = ROOT / "serving_exports"
COUNT = 50000


def environment():
    samples = []
    if platform.system() != "Windows":
        raise RuntimeError("This recorded environment collector targets the local Windows host")
    script = """[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;
    $p=Get-CimInstance Win32_Processor;@{cpu=$p.Name;logical_cpus=$p.NumberOfLogicalProcessors}|ConvertTo-Json"""
    hardware = json.loads(cluster.command("powershell", "-NoProfile", "-Command", script))

    class Memory(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in ["total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended"]]

    def cpu_times():
        idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
        if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            raise ctypes.WinError()
        values = [(value.dwHighDateTime << 32) | value.dwLowDateTime for value in (idle, kernel, user)]
        return values[0], values[1] + values[2]

    for _ in range(3):
        initial = cpu_times()
        time.sleep(2)
        final = cpu_times()
        memory = Memory()
        memory.length = ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise ctypes.WinError()
        samples.append({**hardware, "cpu_percent": 100 * (1 - (final[0] - initial[0]) / (final[1] - initial[1])),
                        "memory_kib": memory.total // 1024, "free_memory_kib": memory.available // 1024,
                        "collector": "GetSystemTimes delta over two seconds; GlobalMemoryStatusEx"})
    docker = json.loads(cluster.command("docker", "info", "--format", '{{json .}}'))
    containers = cluster.command("docker", "ps", "--format", "{{.Names}}").split()
    if any("stepwise" in name.lower() for name in containers):
        raise RuntimeError("Stop StepWise containers before isolated measurements")
    result = {"host_samples": samples, "docker_cpus": docker["NCPU"], "docker_memory_bytes": docker["MemTotal"],
        "docker_server": docker["ServerVersion"], "running_containers": containers,
        "image_id": cluster.command("docker", "image", "inspect", cluster.IMAGE, "--format", "{{.Id}}").strip(),
        "kind": cluster.command(cluster.kind_binary(), "version").strip(),
        "go": cluster.command(str(Path(os.environ["LOCALAPPDATA"]) / "featurebench-tools/go/bin/go.exe"), "version").strip(),
        "source_commit": cluster.command("git", "rev-parse", "HEAD").strip(),
        "client_loadgen_sha256": hashlib.sha256(cluster.binary("loadgen").read_bytes()).hexdigest(),
        "working_tree": cluster.command("git", "status", "--short").splitlines(),
        "scope": "Single Windows host, Docker Desktop WSL2, single-node kind; loopback kubectl port-forward client path"}
    result["idle_check"] = {"median_cpu_percent": statistics.median(s["cpu_percent"] for s in samples),
        "minimum_free_memory_kib": min(s["free_memory_kib"] for s in samples),
        "criteria": "median CPU <=15%; free host memory >=1.5 GiB; no StepWise containers"}
    return result


def summarize(trials):
    result = {}
    for key in ["offered_qps", "achieved_qps", "successful_qps", "p50_ms", "p95_ms", "p99_ms", "max_ms"]:
        values = [t[key] for t in trials]
        result[key] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
    result["errors_per_trial"] = [t["errors"] for t in trials]
    result["partial_per_trial"] = [t["partial"] for t in trials]
    return result


def synthetic(dim=64, dtype="float32", shards=2):
    with threadpool_limits(limits=1):
        return export_synthetic(COUNT, dim, dtype, shards, 42, EXPORTS, query_count=64)


def counters():
    return {p["metadata"]["name"]: cluster.metric(cluster.scrape(p["metadata"]["name"]), "shard_requests_total")
            for p in cluster.pods() if p["metadata"]["labels"]["app"].startswith("shard-")}


def snapshot():
    result = {}
    for p in cluster.pods():
        if p["metadata"]["labels"]["app"].startswith("shard-"):
            name = p["metadata"]["name"]
            text = cluster.scrape(name)
            result[name] = {"numeric_index_bytes": cluster.metric(text, "shard_numeric_index_bytes"),
                            "rss_bytes": cluster.metric(text, "process_resident_memory_bytes")}
    return result


def load(folder, url, duration=10, warmup=2, concurrency=8, qps=20, env=None, partial=False):
    command = [str(cluster.binary("loadgen")), "--export", str(folder), "--url", url,
               "--duration", f"{duration}s", "--warmup", f"{warmup}s", "--concurrency", str(concurrency),
               "--qps", str(qps), "--timeout", "3s"]
    if env:
        command += ["--environment", str(env)]
    if partial:
        command += ["--allow-partial"]
    return checked_json(command)


def get_retries(url):
    text = urllib.request.urlopen(url + "/metrics", timeout=5).read().decode()
    return cluster.metric(text, "coordinator_retries_total")


def run_suite(action, output, reuse_validated=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Preserve existing evidence: choose a new output instead of {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = output.parent / (output.stem + "-trials")
    raw.mkdir(exist_ok=False)
    report = {"complete": False, "action": action, "experiments": [], "notes": [
        "Three independent measured windows per case; warmup excluded. Short local pilot, not production capacity.",
        "Synthetic dimension-64/128 workloads are dimension-matched Gaussian data, not learned PCA features.",
        "50,000 synthetic rows chosen to limit memory pressure on the shared 32 GiB laptop.",
        "Existing thesis and Python benchmark results are not modified."]}
    attempts = []
    for attempt in range(3):
        meta = environment()
        attempts.append(meta["idle_check"])
        idle = meta["idle_check"]
        if idle["median_cpu_percent"] <= 15 and idle["minimum_free_memory_kib"] >= 1572864:
            break
        if attempt < 2:
            time.sleep(10)
    report["precheck_attempts"] = attempts
    report["environment"] = meta
    atomic_json(output, report)
    env = raw / "environment.json"
    atomic_json(env, meta)
    idle = meta["idle_check"]
    if idle["median_cpu_percent"] > 15 or idle["minimum_free_memory_kib"] < 1572864:
        report["blocked"] = "Host idle/memory prerequisite not met; no benchmark was run"
        atomic_json(output, report)
        raise RuntimeError(report["blocked"] + ": " + json.dumps(idle))

    def publish(experiment):
        report["experiments"].append(experiment)
        atomic_json(output, report)
        print(json.dumps({k: v for k, v in experiment.items() if k not in {"trials", "memory_snapshots"}}), flush=True)

    def repeated(name, folder, **kwargs):
        deployed = cluster.deploy(folder, **kwargs.pop("deployment", {}))
        trials, memory, distributions = [], [], []
        with cluster.forward() as url:
            for repetition in range(3):
                before = counters() if name.startswith("E6") else {}
                trial = load(folder, url, env=env, **kwargs)
                if before:
                    after = counters()
                    distributions.append({p: after[p] - before[p] for p in before})
                atomic_json(raw / f"{name}-{repetition + 1}.json", trial)
                trials.append(trial)
                memory.append(snapshot())
        item = {"name": name, "deployment": deployed, "summary": summarize(trials),
                "trials": [f"{raw.name}/{name}-{i + 1}.json" for i in range(3)], "memory_snapshots": memory}
        if distributions:
            item["per_replica_query_deltas"] = distributions
        publish(item)

    try:
        if action in {"bench", "perf"}:
            if action == "perf":
                if not reuse_validated:
                    raise ValueError("perf requires --reuse-validated")
                source = json.loads(Path(reuse_validated).read_text())
                prior = {e["name"]: e for e in source["experiments"]}
                required = [f"E1-{method}-s{shards}" for method in ["full_fp32", "full_fp16_storage", "pca64"] for shards in [1, 2, 4]]
                for name in required:
                    item = prior.get(name, {})
                    if item.get("queries_per_trial") != 2000 or item.get("id_mismatches") != [0, 0, 0] or len(item.get("max_score_errors", [])) != 3 or max(item["max_score_errors"]) > 1e-12:
                        raise ValueError("Incomplete or failed reusable correctness section: " + name)
                if len(prior.get("pprof", {}).get("profiles", [])) != 3:
                    raise ValueError("Missing reusable profiles")
                report["reused_validation_report"] = str(reuse_validated)
                for name in ["pprof", *required]:
                    publish({**prior[name], "reused_from": str(reuse_validated)})
            # Profiles precede any decision to change scoring implementation.
            if action == "bench":
                folder = synthetic()
                deployed = cluster.deploy(folder)
                pod = cluster.pods("shard-0")[0]["metadata"]["name"]
                profiles = []
                with cluster.forward() as url, cluster.forward("pod/" + pod, 9091) as admin:
                    for repetition in range(3):
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            running = executor.submit(load, folder, url, duration=7, warmup=1, concurrency=8, qps=0)
                            time.sleep(2)
                            for kind, endpoint in [("cpu", "profile?seconds=3"), ("heap", "heap")]:
                                path = raw / f"profile-{kind}-{repetition + 1}.pb.gz"
                                path.write_bytes(urllib.request.urlopen(admin + "/debug/pprof/" + endpoint, timeout=15).read())
                                go = str(Path(os.environ["LOCALAPPDATA"]) / "featurebench-tools/go/bin/go.exe")
                                top = cluster.command(go, "tool", "pprof", "-top", "-nodecount=12", path)
                                (raw / f"profile-{kind}-{repetition + 1}.txt").write_text(top, encoding="utf-8")
                            trial = running.result()
                        atomic_json(raw / f"profile-load-{repetition + 1}.json", trial)
                        profiles.append(f"profile-cpu-{repetition + 1}.txt")
                publish({"name": "pprof", "deployment": deployed, "profiles": profiles,
                         "decision": "Retain the scoring implementation; report observed CPU/RSS costs without premature optimization."})
            for method in (["full_fp32", "full_fp16_storage", "pca64"] if action == "bench" else []):
                for shards in [1, 2, 4]:
                    with threadpool_limits(limits=1):
                        folder = export_run(REAL, method, shards, EXPORTS)
                    deployed = cluster.deploy(folder)
                    checks = []
                    with cluster.forward() as url:
                        for repetition in range(3):
                            check = checked_json([str(cluster.binary("verify")), "--export", str(folder), "--http", url,
                                                   "--benchmark-results", str(REAL / "results.json")])
                            atomic_json(raw / f"E1-{method}-s{shards}-{repetition + 1}.json", check)
                            checks.append(check)
                    publish({"name": f"E1-{method}-s{shards}", "deployment": deployed,
                             "queries_per_trial": checks[0]["query_count"], "id_mismatches": [c["id_mismatches"] for c in checks],
                             "max_score_errors": [c["max_abs_score_difference"] for c in checks]})
            for shards in [1, 2, 4]:
                repeated(f"E2-s{shards}", synthetic(shards=shards))
            for dim, dtype, name in [(768, "float32", "fp32"), (768, "float16", "fp16"), (128, "float32", "dim128"), (64, "float32", "dim64")]:
                repeated("E3-synthetic-" + name, synthetic(dim, dtype), qps=4)
            # Keep actual PCA features separate from dimension-matched synthetic workloads.
            for method in ["full_fp32", "full_fp16_storage", "pca128", "pca64"]:
                with threadpool_limits(limits=1):
                    folder = export_run(REAL, method, 2, EXPORTS)
                repeated("E3-real-" + method, folder, qps=20)
            for workers, concurrent in [(4, 1), (1, 4)]:
                for concurrency in [1, 8]:
                    repeated(f"concurrency-w{workers}-c{concurrent}-load{concurrency}", synthetic(),
                             qps=0, duration=6, concurrency=concurrency,
                             deployment={"workers": workers, "concurrent": concurrent})
            for policy, cluster_ip in [("pick_first", True), ("round_robin", False)]:
                repeated("E6-" + policy, synthetic(), qps=20, duration=6,
                         deployment={"balance": policy, "cluster_ip": cluster_ip})
        else:
            folder = synthetic()
            cluster.deploy(folder)
            for event in ["replica_kill", "rollout", "whole_shard"]:
                trials = []
                for repetition in range(3):
                    with cluster.forward() as url, ThreadPoolExecutor(max_workers=1) as executor:
                        running = executor.submit(load, folder, url, duration=18, warmup=2, qps=10, env=env)
                        time.sleep(6)
                        before = get_retries(url)
                        at = time.monotonic()
                        event_info = {"event_after_loadgen_start_seconds": 6, "measurement_offset_approx_seconds": 4}
                        if event == "replica_kill":
                            victim = cluster.pods("shard-0")[0]["metadata"]["name"]
                            event_info["victim"] = victim
                            cluster.kube("delete", "pod", victim, "--grace-period=0", "--force", "--wait=false")
                            until = time.monotonic() + 90
                            while time.monotonic() < until:
                                current = cluster.pods("shard-0")
                                if len(current) == 2 and all(any(c["type"] == "Ready" and c["status"] == "True" for c in p["status"].get("conditions", [])) for p in current):
                                    break
                                time.sleep(.2)
                            else:
                                raise TimeoutError("replacement readiness")
                            event_info["replacement_ready_seconds"] = time.monotonic() - at
                        elif event == "rollout":
                            cluster.kube("rollout", "restart", "deployment/shard-0")
                            cluster.kube("rollout", "status", "deployment/shard-0", "--timeout=120s")
                            event_info["rollout_seconds"] = time.monotonic() - at
                        else:
                            cluster.kube("scale", "deployment/shard-0", "--replicas=0")
                            cluster.kube("wait", "--for=delete", "pod", "-l", "app=shard-0", "--timeout=60s")
                            # Reuse the established port-forward; coordinator is intentionally unready.
                            request = {"queries": [{"id": "policy-test", "vector": [1.0] * 64}], "k": 10}
                            policy_results = []
                            for partial in [False, True]:
                                request["allow_partial"] = partial
                                req = urllib.request.Request(url + "/v1/search", data=json.dumps(request).encode(), headers={"Content-Type": "application/json"})
                                try:
                                    response = urllib.request.urlopen(req, timeout=5)
                                    policy_results.append({"allow_partial": partial, "status": response.status, "response": json.loads(response.read())})
                                except urllib.error.HTTPError as err:
                                    policy_results.append({"allow_partial": partial, "status": err.code})
                            event_info["policies"] = policy_results
                            if policy_results[0]["status"] != 503 or policy_results[1]["status"] != 200 or not policy_results[1]["response"]["partial"] or policy_results[1]["response"]["missing_shards"] != [0]:
                                raise AssertionError("whole-shard policy mismatch")
                            time.sleep(2)
                            cluster.kube("scale", "deployment/shard-0", "--replicas=2")
                            cluster.kube("rollout", "status", "deployment/shard-0", "--timeout=120s")
                            ready(url)
                        event_info["observed_event_seconds"] = time.monotonic() - at
                        trial = running.result()
                        trial["event"] = event_info
                        trial["event"]["retries_during_run"] = get_retries(url) - before
                        # Preserve raw offsets for inspecting the event window, not just whole-run p99.
                        window_end = min(18, max(10, 6 + event_info["observed_event_seconds"]))
                        event_info["approx_event_window_offsets_seconds"] = [2, window_end]
                        window = [s["latency_ms"] for s in trial["samples"] if 2 <= s["scheduled_seconds"] <= window_end]
                        trial["event"]["event_window_p99_ms"] = sorted(window)[int((len(window) - 1) * .99)] if window else None
                    atomic_json(raw / f"E4E5-{event}-{repetition + 1}.json", trial)
                    trials.append(trial)
                    print(json.dumps({"event": event, "repetition": repetition + 1, "errors": trial["errors"], "details": event_info}), flush=True)
                publish({"name": "E5-rollout" if event == "rollout" else "E4-" + event,
                         "summary": summarize(trials), "events": [t["event"] for t in trials]})
                if event != "whole_shard" and any(t["errors"] for t in trials):
                    report["acceptance_review_required"] = "Nonzero failures during replica/rollout event; investigate before proceeding"
                    for pod in cluster.pods("coordinator"):
                        (raw / (pod["metadata"]["name"] + ".txt")).write_text(cluster.kube("logs", pod["metadata"]["name"], "--tail=1000"), encoding="utf-8")
                    atomic_json(output, report)
                    raise RuntimeError(report["acceptance_review_required"])
        report["complete"] = True
        atomic_json(output, report)
    except Exception as error:
        report["error"] = str(error)
        if action == "failures":
            try:
                cluster.kube("scale", "deployment/shard-0", "--replicas=2")
                cluster.kube("rollout", "status", "deployment/shard-0", "--timeout=120s")
            except Exception as cleanup_error:
                report["cleanup_error"] = str(cleanup_error)
        atomic_json(output, report)
        raise
