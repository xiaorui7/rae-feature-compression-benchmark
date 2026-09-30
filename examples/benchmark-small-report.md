# Feature compression report (benchmark_small)

**Status:** actual_measurement. **Protocol:** `9a1b847a6d9ac483`.

Main representation: spatial mean of official RAE `encode` output, then train-only centering. 
The comparison baseline is full pooled FP32, not an unpooled spatial latent.

## Quality and deployment storage

| Method | k | Test probe top-1 | Test 5-NN | Test class P@10 | Test overlap@10 | Test pooled MSE | Deployment bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_fp32 | 768 | 0.983 | 0.970 | 0.950 | 1.000 | 0.00000 | 15,784,386 |
| full_fp16_storage | 768 | 0.983 | 0.970 | 0.950 | 1.000 | 0.00000 | 8,104,418 |
| pca256 | 256 | 0.982 | 0.969 | 0.952 | 0.890 | 0.06148 | 6,334,120 |
| pca128 | 128 | 0.979 | 0.971 | 0.953 | 0.782 | 0.10901 | 3,380,904 |
| pca64 | 64 | 0.980 | 0.974 | 0.954 | 0.644 | 0.15501 | 1,904,296 |
| linear_adapter128_s42 | 128 | 0.979 | 0.971 | 0.946 | 0.761 | 0.10948 | 3,378,322 |
| linear_adapter128_s43 | 128 | 0.980 | 0.969 | 0.945 | 0.759 | 0.10951 | 3,378,322 |
| linear_adapter128_s44 | 128 | 0.980 | 0.967 | 0.945 | 0.761 | 0.10943 | 3,378,322 |
| linear_adapter256_s42 | 256 | 0.981 | 0.970 | 0.938 | 0.831 | 0.06195 | 6,332,050 |
| linear_adapter256_s43 | 256 | 0.981 | 0.965 | 0.936 | 0.830 | 0.06182 | 6,332,050 |
| linear_adapter256_s44 | 256 | 0.981 | 0.965 | 0.936 | 0.830 | 0.06180 | 6,332,050 |

Test query count: 2000
Paired 95% bootstrap intervals for test query metric differences versus full FP32 are in `results.json`; they condition on the fixed train gallery and do not include training uncertainty.
Learned adapter seeds are individual rows; do not interpret a bootstrap interval as seed variation.

### Adapter seed variation (sample standard deviation)

| Dimension | Seeds run | Test class P@10 mean | Seed SD | Test pooled MSE mean | Seed SD |
|---:|---:|---:|---:|---:|---:|
| 128 | 3 | 0.9449 | 0.0007 | 0.10947 | 0.00004 |
| 256 | 3 | 0.9364 | 0.0010 | 0.06186 | 0.00008 |

## Storage and timing boundary

Deployment bytes = serialized gallery cache + encoder-side compressor state + train mean. Inverse state, full checkpoint, decoder, classifier weights, and original training cache are separate. The gallery file includes IDs/labels/array metadata.
FP16 storage is converted to FP32 for probe and exact cosine search. Its search RAM is not halved.
Search timing excludes encoder extraction and query projection; transform, probe fit, and extraction times are separate. Search uses a warm-up and five repetitions; the median is reported. Gallery loading/conversion is also separate.
A break-even count uses actual additional state bytes divided by the positive per-vector byte saving. This is an analytical estimate, not measured large-gallery throughput.

| Method | Gallery raw bytes | Gallery file bytes | Encoder state bytes | Inverse state bytes (excluded) | Probe head raw bytes (excluded) | Gallery load/convert s | Transform median s | Search test median s | Break-even count (estimate) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_fp32 | 15,360,000 | 15,780,750 | 302 | 298 | 61,520 | 0.03419 | 0.00000 | 1.77538 | n/a |
| full_fp16_storage | 7,680,000 | 8,100,750 | 334 | 298 | 61,520 | 0.02501 | 0.00384 | 1.89822 | 1 |
| pca256 | 5,120,000 | 5,540,750 | 790,036 | 790,036 | 20,560 | 0.00890 | 0.02219 | 1.58817 | 386 |
| pca128 | 2,560,000 | 2,980,750 | 396,820 | 396,820 | 10,320 | 0.00821 | 0.01778 | 1.40066 | 155 |
| pca64 | 1,280,000 | 1,700,750 | 200,212 | 200,212 | 5,200 | 0.00469 | 0.01251 | 1.34253 | 71 |
| linear_adapter128_s42 | 2,560,000 | 2,980,750 | 394,238 | 396,790 | 10,320 | 0.00633 | 0.01333 | 1.46515 | 154 |
| linear_adapter128_s43 | 2,560,000 | 2,980,750 | 394,238 | 396,790 | 10,320 | 0.00730 | 0.01382 | 1.48863 | 154 |
| linear_adapter128_s44 | 2,560,000 | 2,980,750 | 394,238 | 396,790 | 10,320 | 0.00842 | 0.01424 | 1.34181 | 154 |
| linear_adapter256_s42 | 5,120,000 | 5,540,750 | 787,966 | 790,006 | 20,560 | 0.01357 | 0.02875 | 1.57816 | 385 |
| linear_adapter256_s43 | 5,120,000 | 5,540,750 | 787,966 | 790,006 | 20,560 | 0.01438 | 0.03067 | 1.53954 | 385 |
| linear_adapter256_s44 | 5,120,000 | 5,540,750 | 787,966 | 790,006 | 20,560 | 0.01287 | 0.02760 | 1.41959 | 385 |

## Recommendation

Selection uses validation class Precision@10 with allowed drop 2 percentage points; test is displayed only after selection.
```json
{
  "max_drop_pp": 2,
  "selected": "pca64",
  "selection_split": "validation_only",
  "status": "selected",
  "test_audit": {
    "knn_accuracy": 0.974,
    "met_validation_threshold": true,
    "precision_at_10": 0.9539
  },
  "threshold": 0.9319999999999999,
  "validation_deployment_bytes": 1904296,
  "validation_precision_at_10": 0.9558
}
```

## Provenance and limitations

Extraction status: `actual_measurement`. Real extraction uses the official frozen RAE and loads its decoder.

| Split | Images | Extract seconds | Peak allocated CUDA MiB | Status |
|---|---:|---:|---:|---|
| test | 2000 | 22.54 | 2006.6 | actual_measurement |
| train | 5000 | 56.37 | 2006.6 | actual_measurement |
| validation | 1000 | 11.40 | 2006.6 | actual_measurement |

Fit-stage timings are single training-pass measurements, not five-repeat latency medians. Re-measured timings use the same data/config but do not replace the saved checkpoint:

| Method | Fit status | Timing origin | Fit seconds | Full checkpoint bytes |
|---|---|---|---:|---:|
| full_fp32 | reused_verified_model | remeasured_without_checkpoint_replacement | 0.014 | 544 |
| full_fp16_storage | reused_verified_model | remeasured_without_checkpoint_replacement | 0.009 | 576 |
| pca256 | reused_verified_model | remeasured_without_checkpoint_replacement | 0.676 | 790,760 |
| pca128 | reused_verified_model | remeasured_without_checkpoint_replacement | 0.000 | 397,544 |
| pca64 | reused_verified_model | remeasured_without_checkpoint_replacement | 0.000 | 200,936 |
| linear_adapter128_s42 | reused_verified_model | remeasured_without_checkpoint_replacement | 12.448 | 792,322 |
| linear_adapter128_s43 | reused_verified_model | remeasured_without_checkpoint_replacement | 9.869 | 792,322 |
| linear_adapter128_s44 | reused_verified_model | remeasured_without_checkpoint_replacement | 9.690 | 792,322 |
| linear_adapter256_s42 | reused_verified_model | remeasured_without_checkpoint_replacement | 16.066 | 1,579,266 |
| linear_adapter256_s43 | reused_verified_model | remeasured_without_checkpoint_replacement | 15.843 | 1,579,266 |
| linear_adapter256_s44 | reused_verified_model | remeasured_without_checkpoint_replacement | 14.905 | 1,579,266 |

Spatial pooling removes location information before channel compression. No pooled embedding is decoded into an image.
CIFAR-10 class Precision@10 is a coarse label retrieval proxy, not fine-grained similarity quality.
This tool does not train a DiT or measure generation FID/KID/IS. Pilot results are exploratory.
Single-run time and accuracy numbers are machine and split specific. See `docs/protocol.md`.
