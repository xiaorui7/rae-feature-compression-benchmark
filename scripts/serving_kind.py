"""Local kind lifecycle. Every kubectl operation selects this project's context."""
import argparse
import contextlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from serving_e2e import checked_json, port, ready, readonly_mount_permissions

ROOT = Path(__file__).resolve().parents[1]
CLUSTER = "rae-serving"
CONTEXT = "kind-" + CLUSTER
NAMESPACE = "rae-serving"
STATE = ROOT / "serving_exports/kind-state.json"
IMAGE = "rae-serving:local"


def command(*args, timeout=300):
    result = subprocess.run([str(a) for a in args], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{args}: {result.stderr}\n{result.stdout}")
    return result.stdout


def kube(*args, **kwargs):
    return command("kubectl", "--context", CONTEXT, "-n", NAMESPACE, *args, **kwargs)


def kind_binary():
    return shutil.which("kind") or str(Path(os.environ.get("LOCALAPPDATA", "")) / "featurebench-tools/kind.exe")


def binary(name):
    return ROOT / "serving/bin" / (name + (".exe" if platform.system() == "Windows" else ""))


def state():
    return json.loads(STATE.read_text())


def pods(app=None):
    args = ["get", "pods", "-o", "json"]
    if app:
        args += ["-l", "app=" + app]
    return [p for p in json.loads(kube(*args))["items"] if not p["metadata"].get("deletionTimestamp")]


@contextlib.contextmanager
def forward(target="service/coordinator", remote=8080):
    if target == "service/coordinator":
        candidates = [p for p in pods("coordinator") if any(c["type"] == "Ready" and c["status"] == "True" for c in p["status"].get("conditions", []))]
        if not candidates:
            raise RuntimeError("No non-terminating ready coordinator pod")
        target = "pod/" + candidates[0]["metadata"]["name"]
    local = port()
    with tempfile.TemporaryFile(mode="w+", dir=ROOT / "serving_exports") as log:
        process = subprocess.Popen(["kubectl", "--context", CONTEXT, "-n", NAMESPACE, "port-forward", target,
                                    f"{local}:{remote}", "--address", "127.0.0.1"], stdout=log, stderr=log)
        url = f"http://127.0.0.1:{local}"
        try:
            ready(url, [process], timeout=90)
            yield url
        finally:
            process.terminate()
            process.wait(timeout=10)


def scrape(pod):
    with forward("pod/" + pod, 9091) as url:
        return urllib.request.urlopen(url + "/metrics", timeout=5).read().decode()


def metric(text, name):
    return sum(float(line.split()[-1]) for line in text.splitlines()
               if line.startswith(name + " ") or line.startswith(name + "{"))


def resources(folder, workers=2, concurrent=2, total_cpu=4, balance="round_robin", cluster_ip=False, image_revision=""):
    manifest = json.loads((folder / "manifest.json").read_text())
    count = manifest["shard_count"]
    version = manifest["export_id"][:20]
    config_name = "coordinator-config-" + version + ("-cluster" if cluster_ip else "-headless")
    config = {"dim": manifest["dim"], "shards": [[f"shard-{i}-v{version}{'-cluster' if cluster_ip else ''}:9090"] for i in range(count)]}
    objects = [{"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": NAMESPACE}},
               {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": config_name, "namespace": NAMESPACE},
                "immutable": True, "data": {"coordinator.json": json.dumps(config)}}]
    cpu = str(int(total_cpu * 1000 / count / 2)) + "m"
    for i in range(count):
        app = f"shard-{i}"
        for suffix, cluster in [("", "None"), ("-cluster", None)]:
            spec = {"selector": {"app": app, "export-version": version}, "ports": [{"name": "grpc", "port": 9090}]}
            if cluster:
                spec["clusterIP"] = cluster
            objects.append({"apiVersion": "v1", "kind": "Service", "metadata": {"name": app + "-v" + version + suffix, "namespace": NAMESPACE,
                           "labels": {"managed-by": "rae-serving", "export-version": version}}, "spec": spec})
        container = {"name": "shard", "image": IMAGE, "imagePullPolicy": "Never",
            "command": ["/bin/shard"], "args": ["--export=/exports/" + folder.name, f"--shard-index={i}",
                "--listen=0.0.0.0:9090", "--admin=0.0.0.0:9091", f"--workers={workers}",
                f"--max-concurrent-searches={concurrent}", "--drain-timeout=10s", "--pprof"],
            "env": [{"name": "SHARD_INDEX", "value": str(i)}],
            "ports": [{"containerPort": 9090}, {"containerPort": 9091}],
            "resources": {"requests": {"cpu": cpu, "memory": "64Mi"}, "limits": {"cpu": cpu, "memory": "1Gi"}},
            "volumeMounts": [{"name": "exports", "mountPath": "/exports", "readOnly": True}],
            "lifecycle": {"preStop": {"exec": {"command": ["/bin/wait", "3s"]}}}}
        probes(container, 9091)
        deployment = {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": app, "namespace": NAMESPACE},
            "spec": {"replicas": 2, "strategy": {"type": "RollingUpdate", "rollingUpdate": {"maxUnavailable": 0, "maxSurge": 1}},
                     "selector": {"matchLabels": {"app": app}}, "template": {"metadata": {"labels": {"app": app, "export-version": version}, "annotations": {"image-id": image_revision}},
                     "spec": {"terminationGracePeriodSeconds": 20, "securityContext": {"runAsNonRoot": True, "runAsUser": 65532},
                              "containers": [container], "volumes": [{"name": "exports", "hostPath": {"path": "/exports", "type": "Directory"}}]}}}}
        objects += [deployment, {"apiVersion": "policy/v1", "kind": "PodDisruptionBudget",
            "metadata": {"name": app, "namespace": NAMESPACE}, "spec": {"minAvailable": 1, "selector": {"matchLabels": {"app": app}}}}]
    coordinator = {"name": "coordinator", "image": IMAGE, "imagePullPolicy": "Never", "command": ["/bin/coordinator"],
        "args": ["--config=/config/coordinator.json", "--listen=0.0.0.0:8080", "--shard-timeout=500ms", "--request-timeout=2s", "--balance=" + balance],
        "resources": {"requests": {"cpu": "250m", "memory": "32Mi"}, "limits": {"cpu": "1", "memory": "256Mi"}},
        "volumeMounts": [{"name": "config", "mountPath": "/config", "readOnly": True}],
        "lifecycle": {"preStop": {"exec": {"command": ["/bin/wait", "3s"]}}}}
    probes(coordinator, 8080)
    objects += [{"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "coordinator", "namespace": NAMESPACE},
        "spec": {"replicas": 2, "selector": {"matchLabels": {"app": "coordinator"}},
                 "template": {"metadata": {"labels": {"app": "coordinator"}, "annotations": {"config": json.dumps(config), "image-id": image_revision}},
                 "spec": {"terminationGracePeriodSeconds": 20, "containers": [coordinator],
                          "volumes": [{"name": "config", "configMap": {"name": config_name}}]}}}},
        {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "coordinator", "namespace": NAMESPACE},
         "spec": {"selector": {"app": "coordinator"}, "ports": [{"port": 8080, "targetPort": 8080}]}}]
    return {"apiVersion": "v1", "kind": "List", "items": objects}


def probes(container, port_number):
    for key, path in [("startupProbe", "/healthz"), ("readinessProbe", "/readyz"), ("livenessProbe", "/healthz")]:
        container[key] = {"httpGet": {"path": path, "port": port_number}, "periodSeconds": 1 if key == "readinessProbe" else 3,
                          "timeoutSeconds": 1, "failureThreshold": 60 if key == "startupProbe" else 3}


def deploy(folder, workers=2, concurrent=2, total_cpu=4, balance="round_robin", cluster_ip=False):
    folder = folder.resolve()
    readonly_mount_permissions(folder)
    image_revision = command("docker", "image", "inspect", IMAGE, "--format", "{{.Id}}").strip()
    spec = resources(folder, workers, concurrent, total_cpu, balance, cluster_ip, image_revision)
    output = ROOT / "serving_exports/kind-resources.json"
    output.write_text(json.dumps(spec, indent=2))
    kube("apply", "-f", output)
    manifest = json.loads((folder / "manifest.json").read_text())
    count = manifest["shard_count"]
    # Remove previous topology's excess shards only within this dedicated namespace.
    for deployment in json.loads(kube("get", "deploy", "-o", "json"))["items"]:
        name = deployment["metadata"]["name"]
        if name.startswith("shard-") and int(name.split("-")[1]) >= count:
            kube("delete", "deployment,service,pdb", name, "--ignore-not-found")
            kube("delete", "service", name + "-cluster", "--ignore-not-found")
    for i in range(count):
        kube("rollout", "status", f"deployment/shard-{i}", "--timeout=180s")
    kube("rollout", "status", "deployment/coordinator", "--timeout=180s")
    wanted = {f"shard-{i}-v{manifest['export_id'][:20]}{suffix}" for i in range(count) for suffix in ["", "-cluster"]}
    for service in json.loads(kube("get", "service", "-l", "managed-by=rae-serving", "-o", "json"))["items"]:
        if service["metadata"]["name"] not in wanted:
            kube("delete", "service", service["metadata"]["name"])
    info = {"export": str(folder), "workers": workers, "max_concurrent_searches": concurrent,
            "total_shard_cpu_limit": total_cpu, "replicas_per_shard": 2, "coordinator_replicas": 2,
            "balance": balance, "cluster_ip": cluster_ip, "image": IMAGE, "image_id": image_revision, "context": CONTEXT,
            "namespace": NAMESPACE, "node_image": "kindest/node:v1.35.0"}
    STATE.write_text(json.dumps(info, indent=2))
    return info


def provision(folder):
    exports = ROOT / "serving_exports"
    exports.mkdir(exist_ok=True)
    tool = kind_binary()
    if CLUSTER not in command(tool, "get", "clusters").split():
        config = {"kind": "Cluster", "apiVersion": "kind.x-k8s.io/v1alpha4", "nodes": [{"role": "control-plane",
                  "extraMounts": [{"hostPath": str(exports.resolve()), "containerPath": "/exports", "readOnly": True}]}]}
        path = exports / "kind-config.json"
        path.write_text(json.dumps(config))
        command(tool, "create", "cluster", "--name", CLUSTER, "--image", "kindest/node:v1.35.0", "--config", path, "--wait", "180s", timeout=300)
    command("docker", "build", "-f", "serving/Dockerfile", "-t", IMAGE, ".", timeout=600)
    archive = exports / "serving-image.tar"
    architecture = command("docker", "image", "inspect", IMAGE, "--format", "{{.Architecture}}").strip()
    command("docker", "image", "save", "--platform", "linux/" + architecture, "-o", archive, IMAGE)
    command(tool, "load", "image-archive", archive, "--name", CLUSTER, timeout=300)
    return deploy(folder)


def verify(output):
    if output.exists() and output.resolve().is_relative_to((ROOT / "benchmarks").resolve()):
        raise FileExistsError("Published evidence is immutable; choose a fresh output")
    info = state()
    folder = Path(info["export"])
    before = {p["metadata"]["name"]: metric(scrape(p["metadata"]["name"]), "shard_requests_total")
              for p in pods() if p["metadata"]["labels"]["app"].startswith("shard-")}
    with forward() as url:
        result = checked_json([str(binary("verify")), "--export", str(folder), "--http", url])
    after = {name: metric(scrape(name), "shard_requests_total") for name in before}
    deltas = {name: after[name] - before[name] for name in before}
    if not all(value > 0 for value in deltas.values()):
        raise AssertionError(f"replica unused: {deltas}")
    result.update(deployment=info, per_pod_query_delta=deltas)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["up", "verify", "bench", "perf", "failures", "down"])
    parser.add_argument("--export", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reuse-validated", type=Path, help="perf only: existing report with all three-run E1 checks and profiles")
    args = parser.parse_args()
    if args.output is None:
        name = "phase4-kind" if args.action == "verify" else "phase5-" + args.action
        args.output = ROOT / "serving_exports/kind-verification.json" if args.action == "verify" else ROOT / "benchmarks/serving" / (name + ".json")
    if args.action == "up":
        if not args.export:
            parser.error("up requires --export with a completed export under serving_exports")
        folder = args.export.resolve()
        if not folder.is_relative_to((ROOT / "serving_exports").resolve()):
            parser.error("kind demo exports must be inside serving_exports")
        print(json.dumps(provision(folder), indent=2))
    elif args.action == "verify":
        print(json.dumps(verify(args.output), indent=2))
    elif args.action == "down":
        command(kind_binary(), "delete", "cluster", "--name", CLUSTER)
    else:
        from serving_bench import run_suite
        run_suite(args.action, args.output, reuse_validated=args.reuse_validated)


if __name__ == "__main__":
    main()
