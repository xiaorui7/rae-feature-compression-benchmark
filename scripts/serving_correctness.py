"""Repeat real-data oracle checks on kind independently of timed load tests."""
import argparse
from pathlib import Path
from threadpoolctl import threadpool_limits
from featurebench.core import atomic_json
from featurebench.export import export_run
import serving_kind as cluster
from serving_e2e import checked_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--methods", default="full_fp32,full_fp16_storage,pca64,pca128")
    parser.add_argument("--shard-counts", default="1,2,4")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        parser.error("use a fresh output and positive repeat count")
    result = {"complete": False, "scope": "Actual kind oracle checks; not a timing benchmark", "cases": []}
    for method in args.methods.split(","):
        for shards in map(int, args.shard_counts.split(",")):
            with threadpool_limits(limits=1):
                folder = export_run(args.run_dir, method, shards, cluster.ROOT / "serving_exports")
            deployment = cluster.deploy(folder)
            checks = []
            with cluster.forward() as url:
                for _ in range(args.repeats):
                    checks.append(checked_json([str(cluster.binary("verify")), "--export", str(folder), "--http", url,
                                                "--benchmark-results", str(args.run_dir / "results.json")]))
            result["cases"].append({"method": method, "shards": shards, "deployment": deployment, "trials": checks})
            atomic_json(args.output, result)
            print(f"{method} / {shards} shards: {len(checks)} oracle checks passed", flush=True)
    result["complete"] = True
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
