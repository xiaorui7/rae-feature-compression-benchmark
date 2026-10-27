import argparse
import json
import time
from pathlib import Path

from .pipeline import ROOT, evaluate_stage, extract_stage, fit_stage, load_config, recommend_stage, run_dir_for
from .reporting import build_report
from .core import atomic_json


def main():
    parser = argparse.ArgumentParser(prog="python -m featurebench")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--queries", default="validation,test")
    p = sub.add_parser("export-synthetic")
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--dtype", choices=["float32", "float16"], required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    p.add_argument("--query-count", type=int, default=100)
    for name in ("smoke", "extract", "fit", "evaluate", "run"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/featurebench_smoke.json" if name == "smoke" else None)
        p.add_argument("--max-runtime-minutes", type=float, default=120)
    p = sub.add_parser("report")
    p.add_argument("--run-dir", required=True)
    p = sub.add_parser("recommend")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--metric", choices=["precision_at_10"], default="precision_at_10")
    p.add_argument("--max-drop-pp", type=float, default=2)
    p.add_argument("--require-smaller-than-full", action="store_true")
    args = parser.parse_args()
    if args.command in ("export", "export-synthetic"):
        from .export import export_run, export_synthetic
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            if args.command == "export":
                result = export_run(args.run_dir, args.method, args.shards, args.out,
                                    tuple(args.queries.split(",")))
            else:
                result = export_synthetic(args.count, args.dim, args.dtype, args.shards,
                                          args.seed, args.out, args.query_count)
        print(result)
        return
    if args.command == "report":
        print(build_report(Path(args.run_dir).resolve()))
        return
    if args.command == "recommend":
        print(json.dumps(recommend_stage(Path(args.run_dir).resolve(), args.max_drop_pp, args.require_smaller_than_full), indent=2))
        return
    if not args.config:
        parser.error("--config is required")
    config = load_config(ROOT / args.config)
    from threadpoolctl import threadpool_limits
    import torch
    threadpool_limits(limits=1)
    torch.set_num_threads(1)
    if args.command == "smoke" and config["profile"] != "smoke":
        parser.error("smoke requires smoke config")
    if args.command != "smoke" and config["profile"] == "smoke" and args.command != "run":
        parser.error("Use the smoke command for synthetic profile")
    t0 = time.monotonic()
    run_dir = run_dir_for(config)
    if args.command in ("smoke", "extract", "run"):
        run_dir = extract_stage(config, args.max_runtime_minutes)
        extraction = json.loads((run_dir / "extract.json").read_text(encoding="utf-8"))
        if extraction["status"] == "partial" and args.command == "run":
            info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
            info["status"] = "partial_not_run_budget"
            atomic_json(run_dir / "run_info.json", info)
            print(f"Partial extraction; later stages not_run. Resume with the same config: {run_dir}")
            return
    if args.command in ("smoke", "fit", "run"):
        fit_stage(run_dir, max(0, args.max_runtime_minutes - (time.monotonic() - t0) / 60))
        if args.command == "run" and not (run_dir / "models" / "full_fp32.npz").exists():
            info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
            info["status"] = "partial_not_run_budget"
            atomic_json(run_dir / "run_info.json", info)
            print(f"Model fit incomplete; evaluation not_run. Resume with the same config: {run_dir}")
            return
    if args.command in ("smoke", "evaluate", "run"):
        evaluate_stage(run_dir, max(0, args.max_runtime_minutes - (time.monotonic() - t0) / 60))
    if args.command in ("smoke", "run"):
        recommend_stage(run_dir)
        build_report(run_dir)
    print(run_dir)


if __name__ == "__main__":
    main()
