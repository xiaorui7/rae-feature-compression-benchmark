"""Deployment contracts that must hold before invoking Kubernetes."""
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from serving_kind import resources
import serving_kind


def manifest(tmp_path, export_id, shards):
    folder = tmp_path / ("export-" + export_id)
    folder.mkdir()
    (folder / "manifest.json").write_text(json.dumps({"dim": 64, "shard_count": shards, "export_id": export_id}))
    return folder


def test_distinct_exports_cannot_share_shard_endpoints(tmp_path):
    snapshots = [resources(manifest(tmp_path, letter * 64, 2))["items"] for letter in ["a", "b"]]
    services = []
    for objects in snapshots:
        config = next(o for o in objects if o["kind"] == "ConfigMap")
        assert config["immutable"] is True
        targets = json.loads(config["data"]["coordinator.json"])["shards"]
        names = {o["metadata"]["name"] for o in objects if o["kind"] == "Service" and o["metadata"]["name"] != "coordinator"}
        assert all(target[0].split(":")[0] in names for target in targets)
        services.append(names)
        for service in [o for o in objects if o["kind"] == "Service" and o["metadata"]["name"] in names]:
            assert "export-version" in service["spec"]["selector"]
            app = service["spec"]["selector"]["app"]
            deployment = next(o for o in objects if o["kind"] == "Deployment" and o["metadata"]["name"] == app)
            assert all(deployment["spec"]["template"]["metadata"]["labels"][k] == v for k, v in service["spec"]["selector"].items())
    assert services[0].isdisjoint(services[1])


def test_shard_topologies_preserve_total_cpu_budget(tmp_path):
    for count in [1, 2, 4]:
        objects = resources(manifest(tmp_path, str(count) * 64, count))["items"]
        shards = [o for o in objects if o["kind"] == "Deployment" and o["metadata"]["name"].startswith("shard-")]
        assert len(shards) == count
        total = sum(int(o["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["cpu"].removesuffix("m")) * o["spec"]["replicas"] for o in shards)
        assert total == 4000
        assert len([o for o in objects if o["kind"] == "PodDisruptionBudget"]) == count
        for deployment in shards:
            spec = deployment["spec"]["template"]["spec"]
            assert spec["terminationGracePeriodSeconds"] > 10
            assert spec["volumes"][0]["hostPath"]["path"] == "/exports"
            assert spec["containers"][0]["volumeMounts"][0]["readOnly"]


def test_verify_refuses_to_overwrite_published_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(serving_kind, "ROOT", tmp_path)
    output = tmp_path / "benchmarks/serving/accepted.json"
    output.parent.mkdir(parents=True)
    output.write_text("published")
    def forbidden_state():
        raise AssertionError("must reject overwrite before contacting a cluster")
    monkeypatch.setattr(serving_kind, "state", forbidden_state)
    with pytest.raises(FileExistsError, match="Published evidence"):
        serving_kind.verify(output)
    assert output.read_text() == "published"


def test_new_image_changes_every_pod_template_even_with_same_export(tmp_path):
    folder = manifest(tmp_path, "f" * 64, 2)
    first = resources(folder, image_revision="sha256:old")["items"]
    second = resources(folder, image_revision="sha256:new")["items"]
    templates = lambda objects: [o["spec"]["template"] for o in objects if o["kind"] == "Deployment"]
    assert len(templates(first)) == len(templates(second)) == 3
    for old, new in zip(templates(first), templates(second)):
        assert old != new
        assert new["metadata"]["annotations"]["image-id"] == "sha256:new"
