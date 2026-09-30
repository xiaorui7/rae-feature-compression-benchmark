# Project Guide

This guide explains the repository from an engineering perspective. Start here if you need to understand the project well enough to modify it or discuss it in an interview.

## 1. The problem

The official DINOv2-B RAE produces a spatial tensor with 768 channels. This project mean-pools the spatial axes into one 768-dimensional vector per image and asks whether a smaller stored representation can preserve enough downstream behavior for exact similarity search and linear classification.

The user is an engineer comparing representation formats before choosing what to persist in a feature store. The input is a benchmark configuration plus images and official model assets. The output is a run directory containing a manifest, cached features, fitted compressors, metric records, a recommendation, and a readable report.

## 2. End-to-end data flow

1. `featurebench.__main__` parses a staged CLI command and caps CPU thread pools for comparable local timing.
2. `source.build_manifest` creates deterministic, class-balanced train/validation/test splits with stable sample IDs.
3. `source.extract` loads the official frozen RAE, encodes images, spatially pools its output, and writes verified feature caches.
4. `pipeline.fit_stage` computes a train-only mean and fits each configured compressor.
5. `pipeline.evaluate_stage` transforms all splits, runs exact retrieval, trains a logistic probe, measures reconstruction/storage/timing, and records each metric.
6. `pipeline.recommend_stage` chooses the smallest candidate meeting a validation Precision@10 tolerance. Test metrics are attached only after selection.
7. `reporting.build_report` converts the structured artifacts into Markdown and a trade-off plot.

The stages share a content-addressed run directory. That makes interrupted runs resumable while ensuring that a changed protocol creates a different destination.

## 3. Main modules

### `core.py`

Contains deterministic hashing, atomic JSON/NPZ writes, cache validation, split construction, tensor reshaping, and environment/Git provenance. Atomic writes create a temporary file in the destination directory and replace the target only after serialization succeeds.

### `source.py`

Owns dataset and official RAE integration. It checks dataset and model provenance, maps manifest rows back to CIFAR-10 samples, applies deterministic preprocessing, extracts RAE tensors, and mean-pools `[B, C, H, W]` into `[B, C]`.

### `methods.py`

Implements a small compressor interface:

- `Full`: FP32 identity baseline.
- `FP16`: storage-only cast; evaluation converts back to FP32.
- `PCAMethod`: train-only full-SVD PCA with explicit transform/inverse transform state.
- `Adapter`: two biased linear layers trained with MSE and validation early stopping.

Every method can fit, transform, invert, save, load, and describe its metadata. The pipeline can therefore benchmark methods without method-specific evaluation branches.

### `evaluation.py`

Implements safe L2 normalization, exact block cosine search, retrieval metrics, reconstruction metrics, paired bootstrap intervals, logistic-probe tuning, storage break-even estimates, and validation-only recommendation.

Exact search is deliberately used instead of an approximate-nearest-neighbor service because the benchmark galleries are small. Blocking limits temporary score-matrix memory while preserving exact results.

### `pipeline.py` and `reporting.py`

`pipeline.py` orchestrates stages and persists intermediate state. `reporting.py` consumes only saved artifacts, so report generation is repeatable and does not rerun expensive extraction or training.

## 4. What makes the result reproducible

- Fixed split seed and stable source IDs.
- Dataset archive hash, upstream Git commit, checkpoint hash, preprocessing config, and schema version in provenance.
- Cache keys derived from configuration, manifest, split, and provenance.
- Validation checks for key, count, dimension, finite values, duplicate IDs, and manifest order.
- Atomic writes for JSON and NumPy archives.
- Explicit `synthetic`, `pilot`, `actual_measurement`, `partial`, and `not_run_*` statuses.
- Validation-only selection followed by a held-out test audit.
- CPU thread limits and repeated/warmed timing for measured evaluation paths.

## 5. Important trade-offs

### Pooling before compression

Pooling makes the benchmark useful for image-level storage and retrieval, but it discards spatial information. The pooled vectors cannot be decoded back into images and are not equivalent to the spatial latents studied in the original thesis.

### Exact search instead of a vector database

The implementation uses NumPy block matrix multiplication. This keeps the algorithm and tie behavior auditable for a few thousand gallery vectors. A production-scale system would need a vector index and separate recall/latency measurements.

### PCA versus a learned adapter

PCA is deterministic and efficient. The adapter tests whether a fitted linear bottleneck helps under the same reconstruction objective. It is not trained end-to-end with a product task, so the project does not claim that it is superior to PCA.

### Serialized deployment bytes

The reported quantity includes the gallery file, encoder-side compressor state, and train mean. It excludes the upstream encoder, inverse state, classifier, decoder, and source cache. This boundary is documented so the number is interpretable rather than a vague storage claim.

## 6. How to trace one command

For `featurebench smoke`:

1. `__main__.py` loads `configs/featurebench_smoke.json`.
2. `extract_stage` creates deterministic synthetic vectors and labels them `synthetic`.
3. `fit_stage` fits all configured method objects and serializes checkpoints.
4. `evaluate_stage` writes representations and metrics.
5. `recommend_stage` selects from validation data.
6. `build_report` creates `report.md` and `tradeoffs.png`.

Search for `atomic_json` calls to see every stage checkpoint. Search for `status` to see how incomplete or synthetic work is prevented from appearing as a real benchmark.

## 7. Testing strategy

The unit tests cover tensor transforms, cache invalidation, compressor serialization, exact-search determinism, self-exclusion, retrieval metrics, and validation-only recommendation. The CLI integration test runs the complete synthetic pipeline in a temporary directory and verifies the generated report and status. GitHub Actions repeats the CPU test suite on pushes and pull requests.

The tests intentionally do not download official weights. A real extraction run is an environment-dependent benchmark and must be executed separately with the upstream assets.

## 8. How to explain it in an interview

Use this short structure:

1. **Problem:** high-dimensional embeddings cost more to store and search.
2. **System:** a staged, resumable benchmark compares storage formats and compressors on fixed data splits.
3. **Hard part:** preserving experimental validity across caching, selection, exact search, and measurements.
4. **Decision:** use exact blocked search and validation-only selection for an auditable small-scale benchmark.
5. **Result:** the checked-in run selected PCA-64 under the configured tolerance and reduced serialized deployment storage from 15.78 MB to 1.90 MB, with documented task and hardware caveats.
6. **Next step:** evaluate an approximate index and a task-specific dataset before making production recommendations.

Do not describe this as a deployed service, distributed system, database, or full RAE-DiT trainer. The strongest honest story is a carefully designed data and benchmarking pipeline with reproducibility and validation safeguards.

## 9. Good first changes

- Add a compressor by implementing the same fit/transform/inverse/save/load/metadata interface and registering it in `load_method` and `methods_for`.
- Add a metric in `evaluation.py`, then persist it as a scalar record in `evaluate_stage`.
- Add a larger-gallery backend behind a search interface and benchmark exact agreement, latency, and memory against the current implementation.

