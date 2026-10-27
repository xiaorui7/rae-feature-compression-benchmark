# Serving evidence

## Current acceptance status

Phases 1-6 are implemented and validated.
The sections below preserve the chronological audit, including older not-yet-run
statements at their original phase boundaries. Accepted versus superseded raw
files are indexed in `benchmarks/serving/README.md`.

| Phase | Accepted deliverable | Hosted CPU CI | Hosted serving CI |
|---|---|---|---|
| 3 | Coordinator, bounded replica failover, native/Compose oracle checks | [passed](https://github.com/xiaorui7/rae-feature-compression-benchmark/actions/runs/40461235435) | [passed](https://github.com/xiaorui7/rae-feature-compression-benchmark/actions/runs/40461235333) |
| 4 | One-command kind, probes/PDB/drain, per-pod traffic proof | [passed](https://github.com/xiaorui7/rae-feature-compression-benchmark/actions/runs/40576356936) | [passed](https://github.com/xiaorui7/rae-feature-compression-benchmark/actions/runs/40576356775) |
| 5 | Repeated correctness, performance, profiles, balancing and actual pod events | [passed](https://github.com/xiaorui7/rae-feature-compression-benchmark/actions/runs/40585547637) | [passed](https://github.com/xiaorui7/rae-feature-compression-benchmark/actions/runs/40585547241) |
| 6 | Documentation audit, evidence protection and final-image verification | [passed](https://github.com/xiaorui7/rae-feature-compression-benchmark/actions/runs/40588481769) | [passed](https://github.com/xiaorui7/rae-feature-compression-benchmark/actions/runs/40588481711) |

Phase 6 code was validated at `b34bdb9`; this handoff update changes documentation
only. [Main branch checks](https://github.com/xiaorui7/rae-feature-compression-benchmark/actions?query=branch%3Amain)
also cover subsequent documentation commits.

Key scope: 72,000 real-data query checks (2,000 unique queries), three windows
per timed case, synthetic 50k-row workload shapes clearly separated from actual
5k-row features, no production SLA or cross-host scaling claim. ANN, hedging and
optional research extensions were not executed.

## Phase 1: export and oracle

Implemented on branch `feature/go-retrieval-service`. This phase measures no
serving performance. Go verification is recorded in Phase 2 below; container
deployment and failover are not yet measured.

### Source audit

The real data exists outside the public checkout, at
`../../featurebench_runs/benchmark_small-f44fa7c6b95a` relative to the repository.
The source run reports `complete`, profile `benchmark_small`, protocol
`9a1b847a6d9ac483`. Its original code commit was not recorded (`null`); the
upstream RAE commit is `a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041` with a dirty
working-tree flag. Exporting preserves this run and does not improve its
historical provenance retroactively.

### Commands executed

The local interpreter was `C:\Miniconda\envs\rae-thesis\python.exe`.
Commands below use `python` as shorthand for that interpreter:

```bash
python -m pytest -q
python -m pytest -q tests/test_export.py
python -m featurebench export --run-dir ../../featurebench_runs/benchmark_small-f44fa7c6b95a --method full_fp32 --shards 2 --out serving_exports
python -m featurebench export --run-dir ../../featurebench_runs/benchmark_small-f44fa7c6b95a --method full_fp16_storage --shards 2 --out serving_exports
python -m featurebench export --run-dir ../../featurebench_runs/benchmark_small-f44fa7c6b95a --method pca64 --shards 2 --out serving_exports
```

Environment: Windows `Windows-10-10.0.26200-SP0`, Python 3.10.20 (Conda),
NumPy 1.26.0. CLI oracle generation caps BLAS at one thread.

### Actual-data exports

Each export was independently read with `verify_export`, checking payloads,
checksums, IDs, counts, and dtype. Each has two shards of 2,500 vectors and
1,000 validation plus 2,000 test queries, all with generated top-10 reference
answers. These are successful export checks, not Go search correctness results.

| Method | Stored dtype | Dimension | Export ID | Verification |
|---|---|---:|---|---|
| full_fp32 | float32 | 768 | `5b87268df01b11017c70ea414a57269de3f4a33ea42d84b03e0043d65083e62d` | passed |
| full_fp16_storage | float16 | 768 | `7bd0cdb0f5677083dbe092d54a37c6c02868288d64dcac894252a33141948c7f` | passed |
| pca64 | float32 | 64 | `c5c724cc9e399443a84c60daa63da779c31088c8d7dc9ee0d4b6b52450f43390` | passed |

Exports are local ignored artifacts under `serving_exports/`. The IDs include
the source run path and therefore may differ on another machine.

### Synthetic tests

Tests cover both binary dtypes, signed zero/subnormal preservation, corrupted
checksums, invalid/truncated payloads, balanced 1/2/4-shard coverage, disjoint
query IDs, deterministic ties including zero vectors, multi-block oracle merge,
immutable reuse, and a simulated mid-export failure without a completion marker.
CLI integration runs an actual smoke pipeline, exports FP32/FP16/PCA-4, and
exercises `export-synthetic`. Existing CI runs these tests with `pytest -q`.

Final local suite: **34 passed in 74.21 seconds**. The focused export suite
also passed independently: **22 passed in 7.30 seconds**. These are test-run
durations, not serving performance measurements. The original 11 tests remain
present. Benchmark computation and published example results are unchanged.

### Plan clarifications

- Optional label sidecars were added so Phase 2 can compute class Precision@10
  directly from exported IDs and labels; synthetic Gaussian vectors have none.
- `--query-count` defaults to 100 for synthetic exports, allowing explicit
  control of reference-generation work on later large galleries.
- Oracle scratch arrays are blocked. The complete input arrays still reside
  in memory; no out-of-core capability is claimed.
- Float64 accumulation is not a promise of bitwise equality across languages.
  Phase 2 must measure ID equality and the specified score tolerance.

## Phase 2: single-shard gRPC correctness

Local environment: Windows `Windows-10-10.0.26200-SP0`, amd64, Go 1.26.2,
Python 3.10.20, NumPy 1.26.0. Go is installed in a separate user-local tools
directory; the existing Python environment is reused. Binaries were built with:

```bash
cd serving
go test ./...
go vet ./...
go build -o bin/shard.exe ./cmd/shard
go build -o bin/verify.exe ./cmd/verify
cd ..
python scripts/serving_verify.py --run-dir ../../featurebench_runs/benchmark_small-f44fa7c6b95a --output benchmarks/serving/phase2-real.json
python scripts/serving_verify.py --run-dir ../../featurebench_runs/smoke-c7760544a48d --methods full_fp32,full_fp16_storage,pca4 --output benchmarks/serving/phase2-smoke.json
```

All four Go unit-test packages passed and `go vet ./...` passed. Generated
Protobuf code is committed. `Serving correctness` CI runs formatting, vet,
`go test -race ./...`, and synthetic end-to-end gRPC verification on Ubuntu.
Windows has no C compiler configured, so no local race-detector result is claimed.

Each actual-data run used a 5,000-vector gallery, one shard with four scoring
workers, and all 2,000 test queries sent through gRPC in batches of 16.

| Method | Data status | Queries | ID mismatches | Max absolute score error | Service P@10 | Original Python P@10 |
|---|---|---:|---:|---:|---:|---:|
| full_fp32 | actual_measurement | 2,000 | 0 | 4.551914400963142e-15 | 0.95035 | 0.95035 |
| full_fp16_storage | actual_measurement | 2,000 | 0 | 2.220446049250313e-16 | 0.95035 | 0.95035 |
| pca64 | actual_measurement | 2,000 | 0 | 1.2212453270876722e-15 | 0.95390 | 0.95390 |

Raw reports: `benchmarks/serving/phase2-real.json`. Precision differences are
between 9.99e-16 and 1.44e-15, consistent with different mean/summation reduction
order. They are not evidence of changed classification accuracy or near-tie
ranking changes. ID checks are against the float64 oracle, not directly against
every original float32 neighbor list.

Synthetic smoke: FP32, FP16, and PCA-4 each passed all 30 test queries with zero
ID mismatches and maximum error <= 2.22e-16. The raw output is explicitly labeled
synthetic in `benchmarks/serving/phase2-smoke.json`.

### Environment issue and scope

Docker Desktop is installed but failed to start. Its backend log reported a
Secrets Engine socket rename/access error under `AppData/Local/docker-secrets-engine`.
No factory reset or deletion was performed. Phase 2 ran directly on Windows;
the Docker issue remains to be resolved before Compose/kind acceptance.

The shard has basic health/readiness and numeric metrics, bounded scoring/RPC
admission, and a graceful-shutdown implementation. A rolling-update or drain
under load experiment has not been performed. The local Windows verification
script terminates processes rather than exercising POSIX SIGTERM handling.

## Later-phase measurements

## Phase 3 prerequisite changes

The first Phase 3 commit adds explicit scoring concurrency, caller-authoritative
deadlines with a configurable fallback, explicit query exclusion IDs, the shared
10% overload retry budget, and deterministic randomized heap/uneven-worker tests.
Incomplete export reuse now reports the exact directory and recovery options.
Original Python tests and published benchmark outputs are unchanged.

Validation: `go test ./...` passed after regenerating protobuf and formatting.
Container commands will bind 0.0.0.0 internally; host publication remains local.
No performance optimization has been made; Phase 5 profiling remains required.

The prerequisite commit is `e0aeb35`. Hosted CPU tests and Serving correctness
passed: [CPU](https://github.com/xiaorui7/rae-feature-compression-benchmark/actions/runs/40459507511),
[Serving](https://github.com/xiaorui7/rae-feature-compression-benchmark/actions/runs/40459507632).

## Phase 3 acceptance audit

Docker recovered by fully exiting Docker Desktop, stopping its remaining backend
processes, running `wsl --shutdown`, and restarting Desktop. Server version:
29.8.1. No AppData files were removed; update/Ubuntu fallback was unnecessary.

| Criterion | Result and evidence |
|---|---|
| Coordinator fan-out and merge | HTTP API, gRPC round_robin, explicit alternate-replica retry and bounded heap merge implemented |
| Real oracle, 1/2/4 shards | FP32, FP16 and PCA64 each: 2,000 queries per topology, 0 ID mismatches; 18,000 total query checks |
| Score tolerance | Max FP32 4.5519144e-15, FP16 2.2204461e-16, PCA64 1.2212453e-15, all below 1e-12 |
| Compose E2E | Two shards x two replicas; real FP32 2,000 queries, 0 ID mismatches, short load completed without errors |
| Errors / deadlines / retry | Fake replica tests cover UNAVAILABLE, DEADLINE_EXCEEDED, INVALID_ARGUMENT, RESOURCE_EXHAUSTED; at most one alternate retry, overload retries 10 of 100 external requests |
| Strict/partial policy | HTTP tests require 503 by default and explicit partial:true plus missing_shards when allowed |
| Graceful drain / metadata | Real gRPC wire test verifies request ID propagation and GracefulStop waits for successful in-flight completion |
| Randomized correctness | Heap and k-way merge versus full sort with ties; uneven 103-row worker partitions versus brute force |
| Export residual | New test confirms incomplete directory reuse returns clear recovery instructions |
| CI Compose | Workflow includes synthetic export, compose up, verify, short load, teardown; hosted result recorded after push |

Raw files: `benchmarks/serving/phase3-real.json` and
`benchmarks/serving/phase3-compose-real.json`. The short load runs are functional
checks, not isolated performance benchmarks: local tests overlapped, warmup was
one second and measurement two seconds. No throughput comparison is inferred.

Implementation decisions: DNS is refreshed every second; one primary round-robin
channel plus direct replica channels makes retry destination explicit. A private
Prometheus registry avoids global collector collisions in tests. Configured
gRPC status-code retries are disabled; grpc-go's transparent retry of unsent or
unprocessed calls is separate from the application policy/counter. The merge
is a k-way cursor heap over sorted shard results.
These implement the planned behavior; no scoring optimization was attempted.

Kubernetes, repeated isolated throughput/RSS, failure-under-load and rollout
measurements remain **not run** at this phase boundary.

Hosted CI found a Linux-only deployment issue: atomic export files use private
0600 permissions by default, preventing UID 65532 from reading a bind mount.
The serving deployment helper now explicitly grants read/traverse permissions
on the selected export before mounting it read-only. Original benchmark outputs
and atomic-write behavior are unchanged. Compose errors now print container logs
before teardown. An intermediate formatting CI failure was corrected with gofmt;
neither failure was bypassed to enter Phase 4.

Phase 3 final hosted CI passed at `fef6a64`:
[CPU](https://github.com/xiaorui7/rae-feature-compression-benchmark/actions/runs/40461235435),
[Serving including Compose](https://github.com/xiaorui7/rae-feature-compression-benchmark/actions/runs/40461235333).
The Linux read-permission correction resolved Compose startup. Final local
synthetic smoke evidence is `phase3-compose-final-smoke.json`; 30 queries,
zero ID mismatches and no short-load errors. Linux `go test -race ./...` also
passed locally in the Go builder container.

## Phase 4 acceptance audit

`scripts/serving_kind.py up --export <real-FP32-two-shard-export>` provisioned a
new `rae-serving` cluster, imported the static image and waited for all six pods
(four shard replicas and two coordinators). kind v0.33.0 was downloaded from
the official release and SHA-256 verified; the node image is v1.35.0, matching
the StepWise demo pattern. Windows subprocess logs explicitly decode UTF-8.

`verify` checked 2,000 actual test queries through coordinator port-forward:
0 ID mismatches, max absolute error 4.5519144e-15, Precision@10 0.95035.
Per-pod query deltas were 992/1008 for shard 0 and 992/1008 for shard 1.
All replicas received traffic; all six pods were Ready with zero restarts.
Both shard PDBs reported minAvailable=1 and one allowed disruption.

Raw evidence: `benchmarks/serving/phase4-kind.json`. Captured structured manifest:
`deploy/kind/resources.example.json`; generation logic and reproduction commands
are in `scripts/serving_kind.py` and `deploy/kind/README.md`.

Acceptance: one-command provisioning from scratch passed; oracle verification
passed; both replicas per shard demonstrated by metric deltas. Configured probes,
read-only mount, resource limits, 20-second termination grace, three-second
preStop and ten-second gRPC drain are present in the applied manifest. PDBs do
not guarantee survival of crashes/direct pod deletion. StepWise remained running
during correctness verification; no Phase 4 performance claims are made.

Phase 4 hosted CI passed at `990a2b8`:
[CPU](https://github.com/xiaorui7/rae-feature-compression-benchmark/actions/runs/40576356936),
[Serving/Compose](https://github.com/xiaorui7/rae-feature-compression-benchmark/actions/runs/40576356775).

## Phase 5 measurement protocol

StepWise's kind node was stopped with `docker stop stepwise-control-plane`;
there were no running StepWise Compose containers. Its container/state was
retained. The only running container before measurement was this project's kind
node. Host: Intel Core Ultra 9 285H, 16 logical CPUs, approximately 31.4 GiB RAM.
Docker Desktop 29.8.1 reports 16 CPUs and 16,470,925,312 bytes memory limit.

The initial WMI LoadPercentage precheck reported median 28% and correctly
prevented starting that run (`archive/phase5-bench.json`, no measurements). A subsequent
independent GetSystemTimes CPU-time delta check reported 7.29/9.18/7.86%; the
actual suite's precheck median was 14.50%. The collector now uses three two-second
GetSystemTimes deltas and GlobalMemoryStatusEx, with median CPU <=15%, at least
1.5 GiB available host RAM, and no StepWise containers. Both prechecks are
retained; the metric/method change is explicit, not a replaced result.

Performance data uses 50,000 synthetic gallery rows, seed 42 and 64 queries,
rather than the plan's illustrative million-row example, to fit current laptop
memory headroom. Synthetic dim64/dim128 are Gaussian dimension-matched workloads,
not actual PCA outputs. Separate real 5,000-row PCA exports measure actual methods.
Each case has three measured windows; report median and min/max. Open-loop
latency starts at scheduled arrival. Host UI/background load, short windows,
CPU quotas and loopback kubectl port-forward limit generalization.

Three CPU and heap profiles precede any scoring optimization decision. On the
FP32 50k x64 workload, scan-loop cumulative CPU was 87.27%-88.85%, with inline
scalar access flat CPU 29.09%-32.73%. The heap profile identifies export loading,
IDs and runtime allocations as contributors beyond numeric index bytes. Decision:
retain the readable exact scorer for this baseline; no scoring optimization is
claimed. Profiles and individual runs are stored in the Phase 5 trial directory.

### Export-transition correction

The first active suite (`archive/phase5-bench-idlechecked.json`) stopped with HTTP 503
when switching from four-shard FP16 to one-shard PCA64. Its completed FP32/FP16
correctness trials remain recorded, but it is not a completed benchmark suite.
The failed request was logged at 20:28:05.601 UTC; the stable shard-0 EndpointSlice
changed again at 20:28:07 UTC. The harness reused stable discovery names while
rolling between immutable export versions, allowing old endpoint/config state
to overlap a new format. Readiness alone did not establish export identity.

The harness now uses per-export Service names/selectors and immutable per-export
coordinator ConfigMaps. Port-forward explicitly selects a non-terminating Ready
coordinator pod. Two new regression tests assert that exports have disjoint
discovery names, selectors match the correct version, and 1/2/4-shard topologies
preserve the same total CPU quota. Both tests passed. A fresh isolated PCA64
verification passed 2,000 queries with zero mismatches; raw evidence is
`phase5-version-isolation-verification.json`. The complete suite is rerun from
scratch under the corrected deployment protocol; partial/failed runs are retained.

The version-isolated run completed all nine E1 cases with three trials each:
54,000 query checks, zero ID mismatches. Its subsequent open-loop runs exposed
a load-generator timing defect: when the last response finished before the end
of the configured window, achieved QPS used the shorter elapsed interval (e.g.
20.07 at an offered 20 QPS). That performance run was stopped and preserved as
`archive/phase5-bench-isolated-retry.json`; its open-loop throughput is not accepted.

The generator now waits until the full observation window ends, then drains
remaining requests. A subprocess/HTTP regression test requires five requests in
a 250 ms / 20 QPS run, duration >=250 ms, achieved QPS <=20 and zero errors.
It passed locally. `perf --reuse-validated <report>` reuses only the already
completed three-run E1 checks and profiles, validating all nine zero-mismatch
sections before reuse. All performance cases are rerun with the corrected
generator. It never reuses prior performance numbers.

## Phase 5 accepted results

Accepted timed results: `benchmarks/serving/phase5-performance.json` and
`phase5-failures.json`, each complete. Raw individual windows are in their
corresponding `*-trials/` directories. `phase5-measurements.csv` flattens all
17 timed performance cases with median/min/max throughput, latency and memory;
regenerate with `scripts/serving_metrics_table.py` using a fresh output path.
These are actual timed service requests. Synthetic refers to the input vectors,
not simulated latency numbers. No scoring optimization was introduced.

Accepted performance precheck: median 14.376% CPU, minimum 4,759,348 KiB available
host RAM. Failure precheck: median 9.844%, minimum 4,591,108 KiB. The timed image
ID was `sha256:71c3bdfa0251813ddb3e02bfaec1d46cb2929a409ca77f820a874a37ef3b2906`;
the Windows loadgen SHA-256 is recorded in each accepted environment JSON.
Metadata records the source commit and dirty working-tree files rather than
pretending the measurement started from a clean checkout. The timing fix is
committed as `84aaa0e`; the shard scoring implementation was unchanged.

### E1: real feature correctness

Every row covers 2,000 unique test queries, 1/2/4 shards and three trials per
topology. Together: **72,000 query checks**, not 72,000 distinct queries/users.
Each representation is compared with its own stored-value float64 oracle.

| Representation | Query checks | ID mismatches | Maximum absolute score error |
|---|---:|---:|---:|
| FP32 | 18,000 | 0 | 4.551914400963142e-15 |
| FP16 | 18,000 | 0 | 2.220446049250313e-16 |
| PCA64 | 18,000 | 0 | 1.2212453270876722e-15 |
| PCA128 | 18,000 | 0 | 1.5543122344752192e-15 |

FP32/FP16/PCA64 raw validation is retained in
`archive/phase5-bench-isolated-retry-trials/E1-*.json` (the later open-loop portion of
that report is superseded). PCA128 was additionally checked using
`scripts/serving_correctness.py`; see `phase5-pca128-correctness.json`.
The largest error is below the 1e-12 acceptance tolerance.

### E2: fixed CPU shard comparison

Synthetic 50,000 x64 FP32, offered 20 QPS, three ten-second windows with two
seconds warmup each. Total shard CPU limit is four CPUs across all replicas:
two replicas per shard receive 2/1/0.5 CPU each at 1/2/4 shards. Coordinator
limits remain two replicas x1 CPU, with one reached by port-forward.
Ranges below are min/max across the three measured windows.

| Shards | Achieved QPS median [range] | p50 ms median | p99 ms median [range] | Failed requests |
|---:|---:|---:|---:|---:|
| 1 | 19.9991 [19.9979,19.9992] | 12.010 | 19.016 [18.826,23.526] | 0 |
| 2 | 19.9986 [19.9950,19.9988] | 10.177 | 19.869 [18.154,26.641] | 0 |
| 4 | 19.9990 [19.9985,19.9990] | 9.521 | 19.233 [17.411,20.152] | 0 |

At this low offered load, all topologies kept up. This is not a maximum-capacity
test and gives no clear monotonic p99 improvement. It measures one-host fan-out
and CPU sharing, not cross-machine horizontal scaling.

### E3: representation and memory cost

Two shards x two replicas, four total shard CPUs. Memory columns sum all four
shard processes. Numeric buffers include raw values and float64 inverse norms,
excluding IDs/runtime/allocations. RSS is a post-window snapshot, not peak RSS
or unique physical memory (shared pages can be counted in multiple processes).
Numeric byte gauges were constant across all three snapshots. RSS ranges and
latency ranges are min/max across trials. MiB means 1,048,576 bytes.

Synthetic 50,000-row inputs at four offered QPS (all runs achieved approximately
4 QPS with zero errors; exact ranges are in the CSV/raw JSON):

| Input | Numeric MiB | Sum RSS MiB median [range] | p50 ms median | p99 ms median [range] |
|---|---:|---:|---:|---:|
| FP32 768-D | 293.732 | 658.625 [658.500,659.125] | 26.467 | 32.970 [32.584,35.560] |
| FP16 768-D | 147.247 | 365.062 [365.062,365.312] | 46.343 | 55.575 [54.438,55.987] |
| Gaussian 128-D | 49.591 | 169.355 [169.230,169.355] | 13.083 | 19.302 [19.277,21.812] |
| Gaussian 64-D | 25.177 | 120.250 [119.875,120.375] | 11.074 | 15.467 [14.678,16.525] |

FP16 reduced numeric buffers by **49.87%** and summed RSS by approximately
**44.57%**, but increased median trial p99 from 32.970 to 55.575 ms. The
22.605 ms p99 difference is a measured cost, not an optimization win. The lower
Gaussian dimensions are workload shapes, not evidence of PCA semantic quality.

Actual 5,000-row feature exports at 20 offered QPS (all three trials per method
had zero errors and approximately 20 achieved QPS):

| Representation | Numeric MiB | Sum RSS MiB median [range] | p50 ms median | p99 ms median [range] |
|---|---:|---:|---:|---:|
| FP32 | 29.373 | 117.500 [117.375,117.500] | 10.677 | 20.906 [19.556,21.280] |
| FP16 | 14.725 | 88.250 [87.625,88.625] | 14.386 | 22.838 [16.812,23.254] |
| PCA128 | 4.959 | 72.586 [70.812,75.352] | 7.733 | 14.685 [13.733,22.132] |
| PCA64 | 2.518 | 67.250 [63.750,69.105] | 7.819 | 18.005 [12.452,21.194] |

Small-gallery tail ranges overlap; PCA64 was not consistently lower-latency than
PCA128. These request latencies cannot be compared directly with the original
Python benchmark's whole-batch search timing.

### Explicit concurrency comparison

Synthetic 50,000 x64 FP32, two shards x two replicas, same four-CPU shard quota.
Each cell is the median and min/max of three six-second closed-loop windows,
with two seconds warmup. There were zero errors in every window.

| Workers / concurrent searches | HTTP clients | Achieved QPS median [range] | p50 ms median | p99 ms median [range] |
|---|---:|---:|---:|---:|
| 4 / 1 | 1 | 243.14 [234.64,253.61] | 3.932 | 7.303 [6.895,7.413] |
| 1 / 4 | 1 | 165.83 [158.11,169.04] | 5.714 | 12.449 [11.967,13.109] |
| 4 / 1 | 8 | 1001.46 [982.60,1008.47] | 5.587 | 30.563 [24.868,36.982] |
| 1 / 4 | 8 | 1025.77 [1022.99,1027.53] | 5.768 | 31.125 [28.288,31.514] |

More within-query workers helped the single-client case. More concurrent queries
gave slightly higher saturated throughput here, with overlapping p99 ranges.
Neither configuration dominates every workload. Short closed-loop measurements
are not a production capacity or latency guarantee.

### E4/E5: actual pod events under open-loop load

Synthetic 50,000 x64, 10 offered QPS, three eighteen-second measurement windows
per event, two seconds warmup. Each window contains 180 scheduled requests.
The event starts approximately four seconds into measurement. Event-window
offsets are saved per trial and include buffers around the observed action;
they are approximate because loadgen is a separate process.

| Event | Errors per trial | Whole-run p99 ms median [range] | Event-window p99 ms median [range] | Observed action seconds median [range] |
|---|---|---:|---:|---:|
| Forced single replica deletion | 0 / 0 / 0 | 21.905 [18.088,25.508] | 26.016 [17.861,26.639] | replacement Ready 4.703 [4.015,4.985] |
| Shard rolling restart | 0 / 0 / 0 | 21.325 [21.136,23.027] | 21.325 [21.136,23.027] | rollout 8.500 [8.312,8.719] |
| Whole shard scaled to zero, held, restored | 91 / 91 / 90 expected HTTP 503 | 20.368 [18.416,21.207] | 18.416 [17.042,20.368] | full scale/hold/restore cycle 10.766 [10.594,10.797] |

There were **0 failures out of 540** requests in replica deletion tests and
**0 out of 540** in rolling restarts. All whole-shard trials separately confirmed
strict 503 and explicit partial mode returning `partial:true, missing_shards:[0]`.
Their expected 503s are preserved, not discarded. Fast 503 responses are included
in the whole-run latency distribution; that p99 is not successful-search latency.

Observed application retry-counter deltas were **0 in every event trial**.
This counter does not observe grpc-go transparent transport retries. These tests therefore
demonstrate survival/routing at this workload, not that retries were required.
The alternate-replica retry and 10% overload-budget properties are covered by the
Phase 3 injected-client tests. No availability percentage or cluster-wide SLA is
claimed. The host, node and coordinator failure cases were not measured.

### E6: actual per-replica distribution

20 QPS, three six-second windows, two seconds warmup. Counter deltas include
warmup, giving 160 completed query operations per shard per trial; latency
statistics exclude warmup. `shard_requests_total` counts completed individual
queries, so a batch of 16 increments it 16 times.

| Client/discovery | Each shard's two replica deltas, all three trials | Whole-run p99 ms median [range] |
|---|---|---:|
| ClusterIP + pick_first | 160 / 0 (or 0 / 160) | 21.518 [16.484,23.208] |
| Headless + round_robin | 80 / 80 | 20.646 [18.030,21.461] |

This confirms connection affinity versus request distribution, rather than
inferring balancing from replica count. Tail ranges overlap; the main result is
distribution, not a claimed latency improvement.

### Acceptance and limits

- E1: four actual methods, 1/2/4 shards, three checks each, zero mismatches passed.
- E2/E3: fixed shard CPU, real and explicitly synthetic inputs, three trials,
  raw samples, numeric buffers and RSS passed; negative results included.
- Both requested worker/concurrent-search configurations passed low/high-load trials.
- Three CPU/heap profiles collected before the decision to retain the scorer.
- E4/E5: three real deletion/rollout/whole-shard trials passed the stated policies.
- E6: both replicas demonstrated with actual counters, three trials each.
- Hosted CI for the timing regression passed at `84aaa0e`:
  [CPU](https://github.com/xiaorui7/rae-feature-compression-benchmark/actions/runs/40581165541),
  [Serving](https://github.com/xiaorui7/rae-feature-compression-benchmark/actions/runs/40581165549).

All reported performance is a short local pilot through loopback kubectl
port-forward with host background activity recorded by precheck. Quantiles use
the lower order statistic `sorted[floor((n-1)*p)]`; small four-QPS windows have
only 40 samples, so p99 is particularly uncertain. Min/max across three trials
are variation ranges, not confidence intervals. No cross-host scaling, ANN,
hedging, public serving, or production SLA was tested. Existing Python benchmark
outputs and tests were not edited.

Phase 5 final hosted CI passed at `8d1d9f3` before starting Phase 6:
[CPU](https://github.com/xiaorui7/rae-feature-compression-benchmark/actions/runs/40585547637),
[Serving/Compose](https://github.com/xiaorui7/rae-feature-compression-benchmark/actions/runs/40585547241).

## Phase 6 acceptance audit

- README now has the online request architecture, Compose/kind quick starts,
  workload-qualified measured results and links to evidence/design/walkthrough.
- SERVING_DESIGN covers protocol, concurrency, deadline/retry bounds, explicit
  exclusions, export-version isolation, lifecycle and measured trade-offs.
- SERVING_WALKTHROUGH covers every component, request/failure paths, reproduction,
  result interpretation, twelve design questions and unimplemented next steps.
- Numerical results are linked to raw artifacts in the measurement index;
  limitations and unsuccessful attempts remain documented here.
- Default kind verification writes ignored local output; overwriting a published
  benchmark path is rejected before cluster access, with a regression test.
- Rebuilt image IDs annotate Pod templates so updated code rolls even when a
  tag/export is unchanged. A regression test covers all shard/coordinator templates.
- Final shard-failure logs include correlation ID, shard/code/internal error;
  the HTTP response remains generic. This is a diagnostic change, not a new
  performance result. Phase 5 image-specific results remain untouched.
- Temporary serving logs/configs are kept in the workspace export parent.
- `git diff e82da4d -- tests examples configs` shows only two added Python test
  files, no changes to existing tests, published examples or original configs.
  Original pipeline/evaluation/methods/core files are unchanged in these phases.

Final code/manifest smoke verification and hosted CI passed, as recorded below
and in the acceptance table at the top of this document.

Final-image local kind verification passed on 64 synthetic queries with zero ID
mismatches, maximum score error 4.440892098500626e-16 and positive query deltas
for every shard replica (16/48 per shard in this short startup check). Raw output
is `phase6-kind-smoke.json`; the final captured structured manifest is
`deploy/kind/resources.versioned.example.json`. It is a functional check, not a
replacement performance measurement. Local Go tests/vet/format and four focused
deployment contract tests passed.

The final local Python suite passed **39 tests**. Added tests cover incomplete
exports, versioned endpoint isolation, fixed CPU budgets, published-evidence
overwrite protection and image-identity Pod template changes. Existing Python
test files were not changed. Final Go tests, vet and formatting checks passed.

Hosted Phase 6 CPU and Serving/Compose checks both passed at `b34bdb9`, including
39 Python tests and Linux Go race/vet/format checks. The 17-case CSV was regenerated
from accepted JSON to an ignored local file; SHA-256 matched the committed CSV
exactly (`173a90a1b6cabf3c818efe593bca2c6b7ae96a73a4641ce640c6c1928b9e4905`).

After all timed measurements and final smoke checks, the previously running
StepWise node was restarted and observed Ready; the original `kind-stepwise`
kubectl context was restored. The RAE demo remains running with all six pods
Ready and zero restarts. Its scripts always select `kind-rae-serving` explicitly.

### Departures from the initial plan

- Docker recovery step (a) succeeded; updates and Ubuntu Engine fallback were
  unnecessary. No Docker factory reset was performed.
- Synthetic performance galleries use 50,000 rows rather than the illustrative
  million/200k sizes, matching observed laptop memory headroom.
- Export cutover required versioned discovery/configuration; the failed attempt
  was preserved, the contract regression-tested, and accepted checks rerun.
- An early-window QPS denominator bug was fixed with an actual subprocess/HTTP
  regression test. All performance windows were rerun; only completed correct
  E1/profile sections were reused, with explicit source-report references.
- Profiling led to retaining the baseline scorer; no kernel/ANN/hedging
  optimization or research extension is presented as completed work.
- Phase 6 added diagnostic error context, overwrite protection and image-ID
  rollout annotations. It did not replace or relabel Phase 5 performance data.
