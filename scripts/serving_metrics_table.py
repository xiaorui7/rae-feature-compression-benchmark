"""Flatten accepted serving measurements without changing the raw reports."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output; existing measurements are preserved")
    report = json.loads(args.report.read_text())
    if not report["complete"]:
        parser.error("flatten only a completed accepted report")
    if report.get("action") not in {"bench", "perf"}:
        parser.error("expected a completed performance report, not a failure-event report")
    rows = []
    for experiment in report["experiments"]:
        if "summary" not in experiment:
            continue
        first = json.loads((args.report.parent / experiment["trials"][0]).read_text())
        environment = first["environment"]
        row = {"case": experiment["name"], "input_data_status": environment["data_status"],
               "execution": "actual_timed_measurement", "gallery_rows": environment["count"],
               "dimension": environment["dimension"], "dtype": environment["dtype"], "shards": environment["shards"],
               "client_concurrency": environment["concurrency"], "trial_count": 3,
               "errors_by_trial": json.dumps(experiment["summary"]["errors_per_trial"])}
        for metric in ["achieved_qps", "successful_qps", "p50_ms", "p95_ms", "p99_ms", "max_ms"]:
            for statistic, value in experiment["summary"][metric].items():
                row[metric + "_" + statistic] = value
        for metric in ["numeric_index_bytes", "rss_bytes"]:
            samples = [sum(pod[metric] for pod in snapshot.values()) for snapshot in experiment["memory_snapshots"]]
            for name, value in [("median", statistics.median(samples)), ("min", min(samples)), ("max", max(samples))]:
                row[metric + "_all_shard_replicas_" + name] = value
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} actual measurement cases to {args.output}")


if __name__ == "__main__":
    main()
