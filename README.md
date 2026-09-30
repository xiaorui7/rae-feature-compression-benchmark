# RAE Feature Compression Benchmark

A config-driven Python benchmark for reducing the storage and exact-search cost of high-dimensional Representation Autoencoder (RAE) features while measuring retrieval and classification quality.

The project extracts one pooled feature vector per image from the official frozen RAE Stage-1 model, fits several compression methods on the training split, evaluates them on fixed validation/test splits, and writes machine-readable results plus a Markdown report. It is a feature-system benchmark, not a full Diffusion Transformer trainer.

## Why this exists

The DINOv2-B RAE emits a 768-channel spatial representation. Persisting and searching many full-precision vectors has a measurable storage and compute cost. This benchmark answers a practical question: which representation is the smallest one that remains within a configured validation-quality tolerance?

## Pipeline

```text
CIFAR-10 images
    -> official frozen RAE encoder
    -> spatial mean pooling [B, 768]
    -> train-only centering
    -> FP32 / FP16 storage / PCA / linear adapter
    -> exact block cosine search + logistic probe + reconstruction metrics
    -> validation-only recommendation + test audit
```

Each run is content-addressed from its configuration. Atomic JSON/NPZ writes, manifest checks, sample IDs, dataset/checkpoint hashes, and explicit result statuses prevent partial or stale artifacts from being presented as completed measurements.

## Quick start

Python 3.10 or newer is required.

```bash
git clone https://github.com/xiaorui7/rae-feature-compression-benchmark.git
cd rae-feature-compression-benchmark
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
featurebench smoke
pytest -q
```

On macOS or Linux, activate with `source .venv/bin/activate`. The smoke profile uses synthetic data and does not download model weights; its results are labeled `synthetic` and are only an end-to-end health check.

## Running a real benchmark

Real extraction requires the upstream [bytetriper/RAE](https://github.com/bytetriper/RAE) repository, its Stage-1 DINOv2-B assets, and the CIFAR-10 Python archive. These large third-party files are intentionally not committed here.

1. Clone the official repository into `official_RAE/` and install its requirements.
2. Place the CIFAR-10 archive under `data/` so `torchvision.datasets.CIFAR10(..., download=False)` can load it.
3. Confirm the paths in `configs/featurebench_pilot.json`.
4. Run the staged pipeline:

```bash
featurebench extract --config configs/featurebench_pilot.json
featurebench fit --config configs/featurebench_pilot.json
featurebench evaluate --config configs/featurebench_pilot.json
featurebench recommend --run-dir featurebench_runs/<run-id>
featurebench report --run-dir featurebench_runs/<run-id>
```

Use `featurebench run --config ...` to execute the stages in one command. Re-running the same configuration resumes verified artifacts instead of silently changing the protocol.

## Verified example

The checked-in `benchmark_small` report is an actual local measurement on 5,000 train, 1,000 validation, and 2,000 test images. Under its validation-only 2 percentage-point tolerance, PCA-64 was selected:

| Representation | Deployment bytes | Test class P@10 | Exact search median |
|---|---:|---:|---:|
| Full FP32 (768-D) | 15,784,386 | 0.950 | 1.775 s |
| PCA (64-D) | 1,904,296 | 0.954 | 1.343 s |

This run reduced measured serialized deployment storage by 87.9%. Timing is a single-machine benchmark, and class Precision@10 is a coarse CIFAR-10 retrieval proxy. PCA-64 retained only 0.644 neighbor overlap@10 against the full representation, so the recommendation should not be generalized to fine-grained visual search without a task-specific evaluation. See [the complete report](examples/benchmark-small-report.md) and [protocol](docs/protocol.md).

## Engineering features

- Staged CLI (`extract`, `fit`, `evaluate`, `recommend`, `report`) with resumable, content-addressed run directories.
- Verified manifests and caches with atomic writes, unique sample IDs, shape checks, and provenance hashes.
- Common compressor interface for identity FP32, FP16 storage, PCA, and trainable linear adapters.
- Exact block-wise cosine search with deterministic ID tie-breaking and self-match exclusion.
- Validation-only model selection, paired bootstrap intervals, and a held-out test audit.
- Unit and CLI integration tests plus a CPU GitHub Actions workflow.

## Project structure

```text
featurebench/          CLI, pipeline, compression, evaluation, reporting
configs/               smoke, pilot, and benchmark-small protocols
tests/                 focused unit and end-to-end smoke tests
docs/                  protocol, design decisions, and research boundaries
examples/              checked-in measured report and recommendation
.github/workflows/     CPU test automation
```

## Boundaries

- The real benchmark depends on official RAE code and weights; this repository does not redistribute them.
- The main representation is a spatially pooled feature vector, not a decodable spatial latent.
- No DiT is trained, and no image-generation FID/KID/IS claim is made.
- Local runtime and memory values are hardware-specific; synthetic smoke results are never treated as research measurements.

## Documentation

- [Project guide](PROJECT_GUIDE.md): architecture, data flow, and interview walkthrough.
- [Evaluation protocol](docs/protocol.md): exact definitions and measurement boundaries.
- [Design decisions](docs/design.md): trade-offs behind caching, search, and recommendation.
- [Related work and attribution](docs/related_work.md): upstream ownership and research scope.

## License

Project code is released under the MIT License. Official RAE code, pretrained weights, and datasets remain subject to their own licenses.

