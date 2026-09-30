import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core import atomic_json
from .evaluation import break_even


def build_report(run_dir):
    run_dir = Path(run_dir)
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    extraction = json.loads((run_dir / "extract.json").read_text(encoding="utf-8"))
    fitted = json.loads((run_dir / "fit.json").read_text(encoding="utf-8"))["models"]
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    recommendation = json.loads((run_dir / "recommendation.json").read_text(encoding="utf-8")) if (run_dir / "recommendation.json").exists() else None
    rows = results["rows"]
    ready = [r for r in rows if "test_precision_at_10" in r]
    if not ready:
        raise ValueError("No evaluated methods")
    baseline = next(r for r in ready if r["method_id"] == "full_fp32")
    status = "synthetic" if info["profile"] == "smoke" else "pilot" if info["profile"] == "pilot" else "actual_measurement"
    lines = [f"# Feature compression report ({info['profile']})", "",
             f"**Status:** {status}. **Protocol:** `{info['protocol_id']}`.", "",
             "Main representation: spatial mean of official RAE `encode` output, then train-only centering. ",
             "The comparison baseline is full pooled FP32, not an unpooled spatial latent.", "",
             "## Quality and deployment storage", "",
             "| Method | k | Test probe top-1 | Test 5-NN | Test class P@10 | Test overlap@10 | Test pooled MSE | Deployment bytes |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in ready:
        lines.append(f"| {r['method_id']} | {r['dim']} | {r['test_probe_accuracy']:.3f} | {r['test_knn_accuracy']:.3f} | "
                     f"{r['test_precision_at_10']:.3f} | {r['test_neighbor_overlap_at_10']:.3f} | "
                     f"{r['test_pooled_feature_mse']:.5f} | {r['deployment_bytes']:,} |")
    lines += ["", "Test query count: " + str(len(results["query_metrics"]["full_fp32:test"]["precision_at_10"])),
              "Paired 95% bootstrap intervals for test query metric differences versus full FP32 are in `results.json`; "
              "they condition on the fixed train gallery and do not include training uncertainty.",
              "Learned adapter seeds are individual rows; do not interpret a bootstrap interval as seed variation.", "",
              "### Adapter seed variation (sample standard deviation)", "",
              "| Dimension | Seeds run | Test class P@10 mean | Seed SD | Test pooled MSE mean | Seed SD |",
              "|---:|---:|---:|---:|---:|---:|"]
    groups = defaultdict(list)
    for r in ready:
        if r["method"] == "linear_adapter":
            groups[r["dim"]].append(r)
    for dim, group in sorted(groups.items()):
        quality = [r["test_precision_at_10"] for r in group]
        mse = [r["test_pooled_feature_mse"] for r in group]
        lines.append(f"| {dim} | {len(group)} | {np.mean(quality):.4f} | {np.std(quality, ddof=1) if len(group)>1 else float('nan'):.4f} | "
                     f"{np.mean(mse):.5f} | {np.std(mse, ddof=1) if len(group)>1 else float('nan'):.5f} |")
    lines += ["",
              "## Storage and timing boundary", "",
              "Deployment bytes = serialized gallery cache + encoder-side compressor state + train mean. "
              "Inverse state, full checkpoint, decoder, classifier weights, and original training cache are separate. "
              "The gallery file includes IDs/labels/array metadata.",
              "FP16 storage is converted to FP32 for probe and exact cosine search. Its search RAM is not halved.",
              "Search timing excludes encoder extraction and query projection; transform, probe fit, and extraction times are separate. "
              "Search uses a warm-up and five repetitions; the median is reported. Gallery loading/conversion is also separate.",
              "A break-even count uses actual additional state bytes divided by the positive per-vector byte saving. "
              "This is an analytical estimate, not measured large-gallery throughput.", "",
              "| Method | Gallery raw bytes | Gallery file bytes | Encoder state bytes | Inverse state bytes (excluded) | Probe head raw bytes (excluded) | Gallery load/convert s | Transform median s | Search test median s | Break-even count (estimate) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    full_per = baseline["dim"] * 4
    for r in ready:
        item_per = r["dim"] * (2 if r["dtype"] == "float16" else 4)
        be = break_even(max(0, r["encoder_state_bytes"] - baseline["encoder_state_bytes"]), full_per, item_per)
        lines.append(f"| {r['method_id']} | {r['gallery_raw_bytes']:,} | {r['gallery_file_bytes']:,} | "
                     f"{r['encoder_state_bytes']:,} | {r['inverse_state_bytes']:,} | {r.get('probe_head_raw_bytes', 0):,} | "
                     f"{r.get('gallery_load_convert_seconds_median', float('nan')):.5f} | {r['transform_seconds_median']:.5f} | "
                     f"{r['test_search_seconds']:.5f} | {be if be is not None else 'n/a'} |")
    lines += ["", "## Recommendation", "",
              f"Selection uses validation class Precision@10 with allowed drop {recommendation['max_drop_pp'] if recommendation else 'n/a'} percentage points; "
              "test is displayed only after selection."]
    if recommendation:
        lines.append("```json")
        lines.append(json.dumps(recommendation, indent=2))
        lines.append("```")
    lines += ["", "## Provenance and limitations", "",
              f"Extraction status: `{extraction['status']}`. "
              + ("Real extraction uses the official frozen RAE and loads its decoder." if info["profile"] != "smoke" else "Synthetic smoke uses no official weights."),
              "", "| Split | Images | Extract seconds | Peak allocated CUDA MiB | Status |",
              "|---|---:|---:|---:|---|"]
    for split, detail in extraction["splits"].items():
        lines.append(f"| {split} | {detail['count']} | {detail.get('seconds', 0):.2f} | "
                     f"{detail.get('peak_cuda_allocated_mb', float('nan')):.1f} | {detail.get('status', extraction['status'])} |")
    lines += ["",
              "Fit-stage timings are single training-pass measurements, not five-repeat latency medians. Reused models keep their original recorded fit time when available:",
              "", "| Method | Fit status | Timing origin | Fit seconds | Full checkpoint bytes |", "|---|---|---|---:|---:|"]
    for item in fitted:
        fit_seconds = item.get("fit_seconds")
        fit_label = f"{fit_seconds:.3f}" if fit_seconds is not None else "n/a (reused)"
        lines.append(f"| {item['method_id']} | {item['status']} | {item.get('fit_timing_type') or 'not_run'} | {fit_label} | {item.get('state_bytes', 0):,} |")
    lines += ["",
              "Spatial pooling removes location information before channel compression. No pooled embedding is decoded into an image.",
              "CIFAR-10 class Precision@10 is a coarse label retrieval proxy, not fine-grained similarity quality.",
              "This tool does not train a DiT or measure generation FID/KID/IS. Pilot results are exploratory.",
              "Single-run time and accuracy numbers are machine and split specific. See `docs/protocol.md`.", ""]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    handles = []
    palette = plt.get_cmap("tab20")
    for i, row in enumerate(ready):
        storage_kib = row["deployment_bytes"] / 1024
        color = palette(i)
        handles.append(axes[0].scatter(storage_kib, row["test_precision_at_10"], color=color, s=42))
        axes[1].scatter(storage_kib, row["test_probe_accuracy"], color=color, s=42)
        axes[2].scatter(row["test_pooled_feature_mse"], row["test_precision_at_10"], color=color, s=42)
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].set(xlabel="Measured deployment storage (KiB)", ylabel="Class Precision@10", title="Quality vs storage")
    axes[1].set(xlabel="Measured deployment storage (KiB)", ylabel="Probe top-1 accuracy", title="Accuracy vs storage")
    axes[2].set(xlabel="Pooled feature MSE", ylabel="Class Precision@10", title="Reconstruction vs downstream")
    fig.legend(handles, [r["method_id"] for r in ready], loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.23, 1, 1))
    fig.savefig(run_dir / "tradeoffs.png", dpi=160)
    plt.close(fig)
    info["status"] = "complete" if extraction["status"] in ("actual_measurement", "synthetic") and len(ready) == len(rows) else "partial"
    info["evaluation_cpu_threads"] = ready[0].get("cpu_threads")
    info["reported_at"] = time.time()
    atomic_json(run_dir / "run_info.json", info)
    return run_dir / "report.md"
