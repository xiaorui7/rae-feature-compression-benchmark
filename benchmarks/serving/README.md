# Serving evidence index

Input status and execution status are different: synthetic vectors below were
searched by actual Go processes/pods with actual clocks. Unit-test fake clients
are not sources of QPS, RSS, latency or availability measurements.

| Files | Inputs / execution | Use |
|---|---|---|
| `phase2-real.json`, `phase2-smoke.json` | Real / synthetic; actual single-shard gRPC checks | Historical correctness |
| `phase3-real.json`, `phase3-compose-real.json`, `phase3-compose-final-smoke.json` | Real / synthetic; actual HTTP/Compose checks | Functional correctness; their short overlapping load windows are not performance comparisons |
| `phase4-kind.json` | Real; actual initial kind verification | Oracle and per-pod query distribution |
| `phase5-performance.json`, `phase5-performance-trials/` | Real and explicitly synthetic; actual timed requests | Accepted E2/E3/concurrency/E6 results, three trials per case |
| `phase5-measurements.csv` | Derived from accepted performance JSON | Median/min/max for 17 timed cases; summed shard numeric buffers and RSS |
| `phase5-failures.json`, `phase5-failures-trials/` | Synthetic gallery; actual pod deletion, rolling restart and whole-shard scale events | Accepted E4/E5 results and explicit partial policy checks |
| `archive/phase5-bench-isolated-retry-trials/E1-*.json` | Real; actual oracle checks | Accepted 54,000 query checks for FP32, FP16 and PCA64 |
| `phase5-pca128-correctness.json` | Real; actual oracle checks | Additional accepted 18,000 PCA128 query checks |
| `archive/phase5-bench-isolated-retry-trials/profile-*` | Synthetic; actual CPU/heap sampling | Three profiles per type, before retaining the baseline scorer |
| `phase5-version-isolation-verification.json` | Real; actual PCA64 check | Verification of export-version deployment correction |
| `phase6-kind-smoke.json` | Synthetic; actual final-image kind check | 64 oracle queries, zero mismatches, every shard replica received traffic; functional only |

## Retained non-final attempts

Historical reports and their trial directories live under `archive/`. File
contents are unchanged, including paths recorded at measurement time. When a
report references an old `benchmarks/serving/phase5-bench*` location, resolve it
under `benchmarks/serving/archive/` instead. The accepted performance and fault
reports remain at the top level. Archiving a mixed report does not invalidate
its explicitly accepted E1 checks or profiles listed above.

- `archive/phase5-bench.json` and `archive/phase5-bench-trials/`: initial WMI idle precheck
  blocked; no load experiment ran.
- `archive/phase5-bench-isolated.json` and `archive/phase5-bench-isolated-trials/`: CPU idle precheck failed;
  no load experiment ran.
- `archive/phase5-bench-idlechecked.json` and `archive/phase5-bench-idlechecked-trials/`: partial run stopped on an
  export-transition HTTP 503. Preserved for diagnosis, not the accepted suite.
- `archive/phase5-bench-isolated-retry.json` and `archive/phase5-bench-isolated-retry-trials/`: full E1 correctness and profiles completed,
  then performance was interrupted after a timing-denominator defect was found.
  Only its explicitly identified validation/profile sections are reused. Its
  open-loop throughput is superseded; it must not be quoted as accepted QPS.

No original Python benchmark/example output is replaced by these artifacts.
Detailed commands, limits, adverse results and interpretation are in
[`docs/SERVING_EVIDENCE.md`](../../docs/SERVING_EVIDENCE.md).

These are single-host kind pilots, not production traffic, multi-node scaling,
cloud deployments or uptime/SLA claims. ANN and hedging were not run.
