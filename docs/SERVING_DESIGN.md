# Serving design

Status: Phases 1-6 implemented and validated. Hosted code CI and image-specific
measurements are recorded in SERVING_EVIDENCE.md.

The intended system serves immutable exact cosine retrieval over exported
gallery vectors. It will not implement ANN, online updates, authentication,
multi-tenant isolation, public exposure, or multi-node availability guarantees.

## Commands

Run from the repository root after editable installation:

```bash
featurebench export --run-dir <completed-run> --method pca64 --shards 2 --out serving_exports
featurebench export --run-dir <completed-run> --method full_fp16_storage --shards 2 --queries test --out serving_exports
featurebench export-synthetic --count 1000 --dim 64 --dtype float16 --shards 2 --seed 42 --query-count 100 --out serving_exports
```

The command prints the full path to `export-<SHA256>` under `--out`. Source
metadata, array contents, IDs, labels, shard count, and format version determine
that hash. Existing destinations are verified and reused, never overwritten.
An interrupted export retains its files but has no manifest and is rejected.
Use a fresh output parent after a failed export, or inspect and remove only the
incomplete directory before retrying. Concurrent writers to the same export
are not supported; exclusive directory creation prevents overwriting.

The source must be a completed benchmark with a completed method result.
IDs and labels must agree with its split manifest. Queries are validation/test
only, and their IDs must not appear in the gallery. Raw representation hashes
identify the exact current inputs; they do not retrospectively prove that an
old NPZ file was never modified before export.

## Binary layout

All fields are little-endian. The header is exactly 28 bytes without padding.

| Offset | Bytes | Field |
|---:|---:|---|
| 0 | 8 | ASCII `RAEVEC01` |
| 8 | 4 | uint32 version, currently 1 |
| 12 | 4 | uint32 dtype: 1 = IEEE float32; 2 = IEEE float16 |
| 16 | 4 | uint32 dimension |
| 20 | 8 | uint64 row count |
| 28 | count * dim * scalar bytes | Row-major vector payload |

Vectors retain their stored dtype and values. No normalization or additional
compression is performed during export. Non-finite values are rejected; zero
vectors and signed zeros are valid.

## Manifest and sidecars

`manifest.json` is written atomically **last** using the existing JSON helper.
Binary files follow the same temporary-file/replace pattern as existing NPZ
writes. Manifest publication guards against application interruption; this is
not a guarantee of filesystem durability through power failure.

The manifest records schema/version, export identity, source run/protocol/method,
data status, dtype, dimension, gallery/shard counts, and sharding rule. Each
shard/query record includes filenames, row count, and SHA-256 checksums.

- `.ids.json`: ordered JSON array of IDs corresponding to the vector rows.
- `.labels.json`: optional ordered labels, preserving real benchmark labels for
  the later service Precision@10 comparison. Synthetic load data has no labels.
- `oracle-<split>.json`: ordered query results, each with ID and top-k hits
  containing ID and float64 score. The default k is 10, capped by gallery size.

`verify_export` rejects missing manifests, checksum mismatches, invalid binary
headers/payload lengths, non-finite data, duplicate gallery IDs, and overlapping
query/gallery IDs. Consumers must verify files before declaring readiness.

## Partitioning and reference arithmetic

Sort gallery IDs ascending, then assign sorted row i to shard `i % shard_count`.
This yields balanced immutable shards and deterministic tie order.

The oracle widens the **stored** float16/float32 values to float64. It computes
`dot(query, gallery) / (norm(query) * norm(gallery))`; a zero norm produces score
zero. Results sort by descending score, then ascending ID. It batches at most
32 queries against 8,192 gallery rows at a time and merges top-k candidates,
avoiding a full query-by-gallery score matrix. Input arrays themselves still
reside in memory, including synthetic galleries.

The existing benchmark uses float32 arithmetic and remains unchanged. Float64
does not guarantee bitwise identical reductions across NumPy and Go. Phase 2
must measure zero ID mismatches and maximum absolute score difference <= 1e-12,
investigating all ranking changes rather than assuming equivalence. No serving
correctness or latency result exists at Phase 1. Phase 2 results are recorded
separately in `SERVING_EVIDENCE.md`.

## Phase 2: single-shard service

Build with Go 1.26.2 from `serving/`:

```bash
go test ./...
go vet ./...
go build -o bin/shard ./cmd/shard
go build -o bin/verify ./cmd/verify
```

Windows binary names end in `.exe`. Linux CI also executes `go test -race ./...`.
The service depends directly on grpc-go v1.79.3 and protobuf v1.36.11; their
transitive dependencies are pinned by `go.mod`/`go.sum`. Generated files are
committed. Regeneration uses protoc 33.0, protoc-gen-go v1.36.11 and
protoc-gen-go-grpc v1.5.1:

```bash
protoc --go_out=. --go_opt=paths=source_relative --go-grpc_out=. --go-grpc_opt=paths=source_relative proto/search.proto
```

Run a single-shard export (use `--shards 1` when exporting):

```bash
serving/bin/shard --export <export-dir> --shard-index 0 --workers 4
serving/bin/verify --export <export-dir> --benchmark-results <run-dir>/results.json
```

The server defaults to loopback gRPC port 9090 and HTTP admin port 9091. It
verifies only its shard payload/IDs before becoming ready. It stores contiguous
`[]float32` or `[]uint16` values plus a float64 inverse norm per vector. FP16
values are decoded during scoring; there is no persistent FP32 copy. The
numeric-index byte gauge excludes IDs, slice headers, RPC buffers, and runtime
overhead and must not be called RSS.

Worker heaps are request-local. `--max-concurrent-searches` (default 1, range
1-16) controls scoring admission. `--workers` is parallelism within each query;
at most their product scoring goroutines are active across RPCs.
Up to 16 admitted RPCs can wait for that token; additional requests receive
`ResourceExhausted`. Each batch has at most 32 queries, k is in [1,100], query
dimension/finite values are validated, and RPC messages are capped at 4 MiB.
Caller deadlines are authoritative. `--search-timeout` (default five seconds)
is applied only when the caller has not supplied a deadline. Native defaults
bind loopback; container commands must explicitly bind `0.0.0.0` on the internal
network and publish host ports on loopback only.

More workers within one query can reduce low-load latency. More concurrent
queries with fewer workers each can improve throughput under load. Phase 5
must measure both (workers=4, concurrent=1) and (workers=1, concurrent=4)
under the same CPU limits. Neither is presumed faster. No optimization will
precede the Phase 5 CPU/heap pprof measurements.

`Query.id` is an opaque correlation ID and no longer excludes gallery rows.
Only explicit `Query.exclude_ids` excludes matches (at most 100 IDs).

Coordinator retry policy permits at most one retry per shard on a different
replica and only within the overall deadline. UNAVAILABLE and
DEADLINE_EXCEEDED are retryable; INVALID_ARGUMENT is never retryable.
RESOURCE_EXHAUSTED additionally requires a shared budget token: cumulative
overload retries cannot exceed floor(external requests / 10) over the process
lifetime. The first nine requests receive no overload retry credit. This is
a conservative global bound, including fan-outs to multiple shards, not a
per-shard allowance. It is not a rolling-window rate limiter.

`/healthz` indicates a live HTTP handler; `/readyz` and gRPC health indicate a
loaded, non-draining index. Prometheus metrics expose completed queries,
numeric index bytes, vectors scanned, latency, readiness and in-flight work.
Logs are JSON and carry a bounded `x-request-id` from gRPC metadata, without
query vectors. SIGINT/SIGTERM disables readiness and invokes gRPC graceful stop,
bounded by `--drain-timeout` (10 seconds by default), then forced stop if needed.
Windows process termination does not exercise POSIX graceful shutdown.

`scripts/serving_verify.py` exports each selected method to one shard, starts a
real local gRPC process, waits for readiness, verifies all test queries, and
writes JSON evidence. It currently uses ports 9090/9091, which must be free.
Its verifier exits nonzero on any ID mismatch or score error above 1e-12.

## Coordinator, discovery and deployment

`cmd/coordinator` reads JSON with `dim` and `shards`, an array of replica-target
arrays. Each target is host:port. `fanout.Pool.Refresh` resolves names every
second, updates a gRPC manual resolver and probes each direct channel using
the gRPC health protocol. Primary requests use grpc-go `round_robin`; the
`pick_first` flag exists for the balancing comparison. DNS membership changes
close removed direct channels. Health probes run outside membership locks so
deadline-bound retries do not wait for probe network timeouts. Health snapshots
can be up to one refresh old. Readiness requires one healthy
replica per shard, but search attempts remain possible during a stale snapshot.

The first attempt captures gRPC peer address. A retry uses a direct channel to
a different healthy address, preserving the overall request context and a fresh
per-attempt deadline. If no peer was selected, the failure happened before a
replica handled the RPC. Configured gRPC status-code retries are disabled.
grpc-go can still transparently retry an unsent or remotely unprocessed call;
that transport behavior is not an application retry counter. The explicit policy
allows at most two logical calls per shard, both idempotent reads under the same
overall deadline. No hedging is implemented.

`POST /v1/search` accepts `queries:[{id,vector,exclude_ids}]`, `k`, and optional
`allow_partial`. It validates a bounded body, shape and values, then admits at
most `--max-inflight` requests. Excess requests return 429. Each admitted request
fans out concurrently; unavailable shards cause 503 unless partial was explicitly
allowed. Partial responses always include `partial:true` and `missing_shards`.
An HTTP request ID is propagated through outgoing gRPC metadata and JSON logs.
The API is local/demo only and has no authentication.

`internal/obs` uses a private Prometheus registry per component. Histogram labels
are bounded outcomes/shard indices, never query IDs. CPU/heap profiling is
opt-in on the shard admin port. Numeric-index bytes exclude IDs and runtime
overhead; process RSS is measured separately.

`serving/Dockerfile` builds static binaries with Go 1.26.2 and runs as UID/GID
65532 in scratch. Compose mounts exports/config read-only, starts two replicas
of each of two shards, checks readiness, and publishes only the coordinator
on host loopback. Each shard has 1 CPU/512 MiB, coordinator 1 CPU/256 MiB.

`scripts/serving_e2e.py` checks oracle queries through HTTP for native 1/2/4
shards or the fixed two-shard Compose topology. Teardown runs in finally blocks.
Hosted CI uses synthetic smoke data; real-data evidence is collected locally.

`cmd/loadgen` excludes warmup and supports closed-loop concurrency or an
open-loop schedule. Open-loop latency starts at scheduled time, including
client delay; a bounded full queue reports an error. At most one million timing
samples are retained per run; overflow is reported. HTTP workers, connections,
timeouts and queue capacity are bounded. Raw samples retain event offsets.
Achieved QPS includes final drain time; offered QPS uses the configured window.

## Immutable export versions on kind

The generated manifests use a stable Deployment per shard, with version-labelled
pods and a separate Service name for each export hash. Coordinator configuration
is an immutable versioned ConfigMap. This prevents a previous export's cached
endpoints or configuration from receiving a new representation's vectors during
benchmark cutover. Unused generated Services are removed only in this project's
namespace; historical raw measurements remain unchanged.

The local image ID is placed in every Pod template annotation. Rebuilding the
same tag triggers rollout when its content changes, even if the export is
unchanged. Single-platform image import uses the image's actual architecture;
only the recorded amd64 host/CI environments were verified.

Readiness probes use /readyz; liveness and startup probes use /healthz. Each
shard has two replicas, PDB minAvailable=1, maxUnavailable=0, maxSurge=1, a
three-second preStop wait and 20-second termination grace around a ten-second
drain. The scratch image includes Go healthcheck/wait binaries instead of a shell.
The extraMounts/hostPath export is read-only and demo-only. A normal ClusterIP
Service per export is retained for comparison with headless round_robin.

Readiness is strict across shards. During whole-shard loss, new ClusterIP
connections may have no Ready coordinator endpoint. The measured 503/partial
policies use an already established pod port-forward, which remains reachable
when readiness is false. This is not a guarantee of partial-mode availability
through a production load balancer.

Final failed shard attempts are logged with request ID, shard index, gRPC code
and internal error; callers receive generic 503. This Phase 6 diagnostic change
does not replace the Phase 5 image-specific measurements or claim a speedup.

## Measurements and limits

See SERVING_EVIDENCE.md and benchmarks/serving/README.md for accepted versus
superseded attempts. Four actual feature methods passed 72,000 query checks
against their own float64 oracles. Timed cases used 5,000 actual rows or 50,000
synthetic rows, three trials, recorded host/Docker limits and fixed shard CPU.
The two requested parallelism policies were tested at one and eight clients.

The 50k x768 synthetic case halved scalar storage but reduced numeric buffers
by 49.87% (norms remain float64), and summed post-window RSS by about 44.57%.
FP16 p99 was slower, 55.575 versus 32.970 ms. This is a memory/CPU trade-off,
not a universal efficiency improvement. No throughput gain is claimed from
fixed-rate cases merely keeping up with offered load.

The system implements exact O(gallery rows x dimension) scans. Hash sharding,
ANN, online updates, image encoding in the request path, authentication/TLS,
multi-host operation and hedging are not implemented. The measured endpoint
accepts precomputed vectors in the deployed representation; encoding and PCA
fitting are outside serving latency. The original Python evaluation and its
implicit self-match behavior remain unchanged; online exclusions are explicit.
