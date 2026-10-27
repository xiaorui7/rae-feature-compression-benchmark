# Serving walkthrough

This guide covers the complete serving extension. Start with the request path,
then inspect the component map and experiments. The original feature extraction
and quality benchmark are explained separately in PROJECT_GUIDE.md.

An image becomes a numeric feature vector offline. A gallery is the collection
of vectors to search; a query is a vector to compare with that gallery. A shard
holds a different portion of the gallery. A replica holds another copy of the
same shard. The online service takes vectors, not raw images or PCA models.

## Phase 1: transporting benchmark data into a service

The original benchmark writes NumPy archives. A Go service needs a simple
language-independent format and an independent expected result against which
to check its search implementation.

1. `featurebench.__main__.main` dispatches the new export commands without
   rerunning extraction, fitting, or evaluation.
2. `export.export_run` reads a completed run, checks its method result, and
   compares stored IDs/labels against the source manifest.
3. `export.export_arrays` validates data and separates gallery from queries.
   It sorts gallery IDs and distributes rows round-robin across shards.
4. `export.write_vectors` writes a 28-byte header and raw little-endian values.
   IDs and labels travel in separate JSON files with the same row order.
5. `export.oracle` computes reference cosine results using float64 from the
   exact exported values, with ID-based tie-breaking.
6. The exporter writes checksums into `manifest.json` only when every file is
   ready. `export.verify_export` checks those files before reuse.

### Why not export normalized float32 vectors for every method?

That would erase the FP16 storage distinction and prevent a later experiment
on retaining half-precision values in server memory. The service will calculate
norms and widen values while scoring.

### What does a checksum prove?

It detects changed bytes relative to the manifest. It is not authentication
against someone who can replace both the data and manifest. These exports are
local trusted artifacts, with no public-write interface planned.

### Why write the manifest last?

A crash after writing one shard must not make an incomplete index look ready.
The manifest is a completion marker. Missing or mismatched files cause loading
to fail instead of serving a smaller gallery silently.

### Why does sharding not change the correct global top-k?

Any hit outside its shard's local top-k already has at least k hits ordered
ahead of it. It therefore cannot be in the global top-k. Merging each shard's
top-k is sufficient when all shards use the same total ordering, including ID
tie-breaking. The Go implementation of this rule belongs to Phase 2/3.

### Is the synthetic command evidence of real retrieval quality?

No. It creates seeded Gaussian vectors for functional tests and later load
experiments. Its manifest explicitly says `synthetic`. Real run exports retain
their dataset/protocol/method provenance and labels.

## Phase 2: a request through one shard

1. `cmd/shard.run` reads the manifest, calls `format.Load` to check SHA-256,
   validates the header and IDs, and constructs `index.New`.
2. `index.New` computes one inverse norm per gallery row and keeps the original
   FP16 or FP32 buffer. The server then opens gRPC and admin listeners.
3. `shard.Server.Search` validates the entire batch before scoring. It applies
   bounded admission and the caller's deadline, or a configurable fallback.
4. `index.Search` waits for the scoring admission token, divides contiguous rows
   among workers, and accumulates dot products in float64. Each worker uses
   `topk.Heap`, whose root is the worst retained hit.
5. The query results are merged through another bounded heap, sorted, and
   returned through the generated gRPC interface.
6. `cmd/verify.run` sends batches of 16 queries, checks each returned ID/score
   against the exported oracle, and computes class Precision@10 from labels.

### Why keep a worst-first heap?

When retaining k best items, the root must be the easiest item to replace.
Each new score costs O(log k) after the heap fills instead of sorting every
gallery score. For equal scores, the larger ID is worse.

### Why is FP16-in-memory not automatically faster?

It stores two bytes per scalar, but every score decodes half values. The
float64 norm array, IDs, and runtime allocations also remain. Numeric payload
savings and whole-process RSS are different measurements. Latency must be
measured before claiming an improvement.

### Why limit workers across requests?

Creating a full worker pool for every concurrent request multiplies CPU work
and goroutines. Explicit search admission and workers per query bound scoring
concurrency; the RPC admission limit also bounds waiting work.
Phase 5 should measure whether this is an appropriate throughput choice.

### What happens on a bad checksum or invalid query?

A bad index prevents startup, so the shard never becomes ready. Invalid query
dimensions, non-finite values, excessive batch size, or invalid k return
InvalidArgument. Saturated admission returns ResourceExhausted. Cancellation
and deadlines propagate into the scan.

## Phase 3: the complete request path

1. `cmd/loadgen.run` loads verified query vectors and sends HTTP JSON from a
   bounded worker pool. Open-loop mode schedules arrivals independently of replies.
2. `fanout.HTTP.Search` validates shape, finite values, batch/body limits and k,
   bounds concurrent requests and assigns a request ID and overall deadline.
3. `fanout.Engine.Search` starts one goroutine per shard. `Engine.attempt` caps
   each attempt and consults the shared retry budget for retryable failures.
4. `fanout.Pool.Search` calls the round-robin gRPC channel and captures the peer.
   `Pool.Alternative` excludes that address before selecting a direct retry.
5. Shards validate and score, then Engine merges their candidates with the same
   score/ID ordering. A missing shard produces 503 or an explicitly marked partial.
6. HTTP JSON returns IDs and scores; metrics record outcome, latency, retries and
   in-flight work. Logs carry the same request ID without storing query vectors.

To reproduce native correctness, build all `serving/cmd/...` binaries into
`serving/bin` and run `scripts/serving_e2e.py --run-dir <run> --output <json>`.
For Compose add `--compose --shard-counts 2`. That path builds the image, mounts
immutable exports read-only, verifies all test queries and tears down containers.

### Why both round-robin and direct connections?

The primary channel balances requests across replica addresses. A direct channel
makes retry destination explicit: the failed peer is excluded. A second call on
the same round-robin channel alone cannot guarantee a different peer, especially
when other requests run concurrently. Discovery refreshes endpoints and health.

### What happens when a replica or shard fails?

A replica error may use one retry on another healthy address within the original
deadline. INVALID_ARGUMENT cannot retry; overload retries consume a global 10%
budget. If the whole shard fails, complete global top-k is unavailable. The caller
gets 503 unless it explicitly accepts partial results and missing shard indices.

### What does the drain test establish?

The gRPC wire test blocks an in-flight request, starts GracefulStop, verifies that
shutdown waits, releases the request and verifies a successful reply. It also
checks metadata on the actual wire. It does not substitute for the later
Kubernetes rollout-under-load measurement.

### Why does open-loop load matter?

A closed-loop client waits for responses before sending more, so a stalled server
reduces arrivals and can hide overload. Open-loop keeps scheduled arrivals and
measures from scheduled time. Queue drops and errors are reported explicitly.

## Phase 4: local Kubernetes

`serving_kind.provision` creates only the named rae-serving cluster and mounts
the repository's export parent read-only into its single node. It builds a static
image, saves a single-platform archive and imports it into kind's containerd.
`resources` generates native Kubernetes objects; `deploy` applies them and waits
for every rollout. All commands specify context/namespace explicitly.

Each shard has two pods holding the same shard data. Replicas increase serving
capacity and allow failover; they do not increase unique dataset capacity.
Different shard indices hold different gallery partitions. The headless Service
exposes ready pod IPs to the gRPC resolver; the separate ClusterIP Service exists
for the later pick_first comparison. Coordinator itself also has two replicas.

`verify` measures per-pod query counters before/after oracle verification. Positive
deltas for both pods of each shard demonstrate distribution instead of assuming
that two Running pods imply balancing. Port-forward reaches one coordinator pod;
this is local demo access, not a measurement of coordinator load balancing.

Readiness removes an unready pod from normal discovery. Liveness detects a stuck
process; startup probes allow index loading first. PreStop waits three seconds
while endpoints propagate, then SIGTERM turns off readiness and drains gRPC.
The PDB minAvailable=1 constrains voluntary eviction; it does not prevent a crash,
an explicit deletion or loss of the single host.

## Complete component map

Paths are relative to the repository root unless shown under `serving/`.

| Component | Entry points | Responsibility |
|---|---|---|
| Offline export | `featurebench/export.py`: export_run, export_arrays, write_vectors, oracle | Preserve stored values/IDs; write portable files, checksums and reference results |
| Artifact reader | `serving/internal/format/format.go`: ReadManifest, ReadChecked, Load, HalfToFloat | Reject corrupt/inconsistent files before readiness; keep FP16 as uint16 |
| Scorer | `serving/internal/index/index.go`: New, Validate, Search, NumericBytes | Precompute norms; bound concurrent scans; partition rows among workers |
| Selection/merge | `serving/internal/topk`: Heap.Add, Heap.Sorted, Merge | Retain local candidates and merge sorted shard lists with one total order |
| Internal API | `serving/proto/search.proto`, generated Go files | Typed Search RPC, queries, explicit exclude_ids and double-precision scores |
| Shard process | `serving/cmd/shard/main.go`, `internal/shard/server.go` | Load its partition, validate RPCs, expose health/metrics, handle SIGTERM/drain |
| Replica transport | `serving/internal/fanout/pool.go`: NewPool, Refresh, Search, Alternative | DNS membership, gRPC round_robin, peer capture, direct alternate channel |
| Fan-out policy | `serving/internal/fanout/search.go`, `policy.go` | Concurrent shard attempts, deadlines, one retry, global overload budget and merge |
| External API | `serving/internal/fanout/http.go`: HTTP.Handler, HTTP.Search | Bounded JSON validation, request admission/ID, strict or explicit partial response |
| Coordinator process | `serving/cmd/coordinator/main.go` | Read shard config, start pools/probes and HTTP server, drain on shutdown |
| Observability | `serving/internal/obs/metrics.go`: New, Handler | Per-process Prometheus registry, latency/outcomes/retries/inflight/process metrics |
| Correctness client | `serving/cmd/verify/main.go` | Read oracle/query artifacts, compare each ID/score, optionally calculate class P@10 |
| Load client | `serving/cmd/loadgen/main.go` | Closed-loop or scheduled open-loop requests, warmup exclusion, raw latency samples |
| Container utilities | `serving/cmd/healthcheck`, `cmd/wait`, `Dockerfile` | Scratch image health command and preStop wait without requiring a shell |
| Native/Compose automation | `scripts/serving_verify.py`, `serving_e2e.py` | Export, start processes/containers, verify, short load, teardown |
| Kubernetes automation | `scripts/serving_kind.py`: provision, resources, deploy, forward, verify | Explicit context/namespace, versioned discovery/config and per-pod verification |
| Repeated experiments | `scripts/serving_bench.py`, `serving_correctness.py`, `serving_metrics_table.py` | Profiles, repeated windows, real pod events, immutable evidence and flattened CSV |
| Hosted checks | `.github/workflows/tests.yml`, `serving.yml` | Python tests; Go format/vet/race; synthetic Compose end-to-end |

## Data and control flow

```text
Offline images -> frozen RAE -> pooled vectors -> trained compressor
       -> train gallery / held-out query vectors
       -> .vec + .ids.json + manifest + float64 oracle

Online query in the SAME deployed representation
       -> HTTP validation and bounded admission
       -> overall context deadline + request ID
       -> concurrent gRPC call to each shard
       -> one selected replica per shard, optional alternate retry
       -> immutable raw values + cached inverse norms
       -> worker-local top-k -> shard top-k
       -> coordinator k-way merge -> JSON IDs/scores
```

For a 768-D gallery, the query must have 768 values. For a PCA64 gallery, the
query must already be transformed with the same fitted centering/PCA pipeline.
The server does not convert an arbitrary 768-D request into PCA64 automatically.
This separation keeps model loading/training outside the serving path.

### Storage versus RAM

The binary vector file contains a 28-byte header and two or four bytes per
scalar. At startup, the server verifies the file and loads only its shard into
RAM. Inverse norms add eight bytes per row. IDs, Go runtime state, transient
loading buffers, RPC messages and allocation retention cost additional memory.

Consequently, serialized bytes, numeric index buffers, heap live bytes, RSS and
peak memory are different quantities. The experiment measured numeric gauges
and post-window RSS; it did not measure a new process's peak RSS. Summing process
RSS can count shared pages more than once. FP16 does not promise halved RSS.

### Concurrency and cancellation

`--workers` divides a single scan into contiguous row ranges. Each worker keeps
its own bounded heap, so scoring does not need a shared heap lock. The row
boundaries use integer fractions of the total count, covering uneven partitions.

`--max-concurrent-searches` bounds admitted scans. A batch performs its queries
sequentially, while different RPCs may score concurrently. RPC admission and
HTTP admission also have limits. More workers within one query can reduce
single-client latency; more independent scans can improve loaded throughput.
The fixed-CPU measurements showed that neither mode won every metric.

Contexts carry cancellation and deadlines through HTTP, fan-out, gRPC and the
scan. Workers check cancellation periodically; the index returns an error after
waiting for workers rather than silently returning a partially scanned top-k.
A caller deadline overrides the shard's fallback timeout. A retry does not
receive a fresh overall deadline.

## Failure walkthroughs

| Trigger | What the system does | What was verified |
|---|---|---|
| One replica disappears | gRPC routing can continue through another connection; a retryable failed attempt may use one different healthy address | Three real forced deletions at 10 QPS, 0/540 failed requests; observed application retry-counter deltas were zero |
| One replica is slow | Per-attempt deadline expires; one alternate retry is allowed if the overall context still permits | Injected-client deadline/retry tests; a separate slow-replica load benchmark was not run |
| Whole shard disappears | Strict mode returns 503; explicitly allowed partial mode lists missing shards | Three real scale-to-zero/restore events, 91/91/90 expected 503s; explicit partial checks passed |
| Export checksum is wrong | Startup exits nonzero before serving/readiness | Format/export tests; no corrupted index is accepted silently |
| Shard rolling restart | New Ready pod joins; preStop allows endpoint propagation, then gRPC drains | Three actual rolling restarts at 10 QPS, 0/540 failed requests |
| A different export is deployed | Version-specific Service selectors and immutable ConfigMap isolate the new representation | Export-transition bug retained; isolation regression and oracle checks passed |

The strict readiness policy requires a healthy replica for every shard. During
whole-shard loss, the benchmark's existing pod port-forward remains connected
even when /readyz returns 503. New connections through a normal ClusterIP may
have no Ready coordinator endpoint. Do not claim production partial-mode
availability from this local access pattern.

Final shard failure logs include request ID, shard index, gRPC code and internal
error; external callers get generic messages. This diagnostic detail was added
after Phase 5 measurements, not presented as a measured performance improvement.

## Reproducing and reading experiments

Build the binaries first and provision kind using an export under
`serving_exports`. The README has a no-model-download synthetic demo. For actual
features, run the original Python benchmark or use a verified completed run.

```bash
# Full repeated protocol: profiles, E1, E2, E3, concurrency and E6.
python scripts/serving_kind.py bench --output serving_exports/my-bench.json

# Actual pod deletion, rollout and whole-shard events, three trials each.
python scripts/serving_kind.py failures --output serving_exports/my-failures.json

# Real-data correctness independently of timed experiments.
python scripts/serving_correctness.py --run-dir <completed-run> --output serving_exports/my-oracle.json

# Flatten a completed performance report without changing it.
python scripts/serving_metrics_table.py --report serving_exports/my-bench.json --output serving_exports/my-summary.csv
```

The recorded measurement harness's real run path and 50,000-row synthetic size
are constants at the top of `serving_bench.py`; adjust the real path to an
equivalent completed benchmark on another machine. Its environment collector
targets the recorded Windows host; the Go service and CI also run on Linux.
Stop unrelated workloads first. In the recorded run StepWise's node was stopped
without deleting its state. Host CPU-time/memory prechecks and Docker limits are
saved before any timed window. A failed precheck produces no performance claim.

Use a fresh report path: accepted or failed evidence is never overwritten by
the benchmark harness. `perf --reuse-validated <report>` is a recovery path that
validates and reuses completed E1/profile sections only; it does not reuse old
performance results. The timing-defect attempt remains separately archived.

Read a report in this order:

1. Check `complete`, `environment`, `input_data_status`/`data_status`, image ID,
   client hash and dirty source state. Synthetic input is distinct from execution
   status: the requests and pod failures were real, not simulated timings.
2. Find the named case, configured shard/CPU/worker settings and client mode.
   Offered fixed QPS is not a measurement of maximum server capacity.
3. Inspect all three error maps and partial counts before latency. Fast 503s can
   lower a latency percentile while successful throughput collapses.
4. Compare median/min/max and individual `samples`, not just the fastest run.
   Quantiles use a lower order statistic; four-QPS windows have only 40 samples.
5. For fault cases, inspect victim/action duration, approximate event-window
   offsets, policy responses and retry deltas. Replacement Ready time starts at
   the deletion command; it is not a cluster-wide recovery-time guarantee.
6. For E6, compare each pod's counter delta. Those counts include warmup, while
   measured latency samples do not. A batch increments the shard query counter
   once per vector, not once per RPC.

The measurements show about 1,026 QPS at 31.1 ms p99 for a synthetic 50k x64
closed-loop case. Fault tests separately used ten offered QPS. FP16 reduced
numeric buffer bytes 49.87% in the 50k x768 case but increased p99 by 22.605 ms.
These statements must retain their workload qualifiers.

## Design questions

### 1. What did you build, and who uses it?

A Python exporter and a Go exact vector-search service. A coordinator accepts
HTTP query vectors, searches replicated partitions over gRPC and merges IDs and
scores. Benchmark/verification clients are its measured consumers; no real-user
adoption is claimed. It extends an offline feature benchmark into a deployable
serving and reliability experiment.

### 2. Why shard, and why replicate?

Shards divide the unique gallery and scan work. Replicas duplicate a shard for
request distribution and survival of one process loss. Replicas do not increase
unique dataset capacity and multiply stored index memory. On one host, extra
shards also add fan-out, process and scheduling overhead.

### 3. Why is merging only each shard's top-k correct?

An item outside its local top-k already has k items ahead of it in that shard.
It cannot enter the global top-k. This proof needs the same total ordering in
every place: score descending, then ID ascending, including exact ties.

### 4. Why exact search instead of ANN?

Exact scans make the result checkable against an independent oracle and suit
this small reproducible benchmark. Their O(rows x dimension) work becomes costly
at larger scale. ANN would require measuring recall/latency and operational
complexity; it is a future option, not an implemented feature.

### 5. Why float64 arithmetic and a tolerance?

Both oracle and service score the actual stored values after widening them to
float64. This limits arithmetic disagreement. NumPy and Go can still reduce sums
in different orders, so the criterion is zero ID mismatches and score error
<=1e-12, not bitwise equality. The largest measured error was about 4.55e-15.

### 6. Why did FP16 save memory but run slower?

Values occupy two bytes instead of four, while norms and IDs remain. Each
score decodes FP16 before float64 accumulation. In the synthetic 50k x768 case,
numeric buffers fell 49.87%, summed RSS fell about 44.57%, and p99 rose from
32.970 to 55.575 ms. That is a measured storage/CPU trade-off.

### 7. What is the difference between workers and concurrent searches?

Workers split one query; concurrent searches admit multiple independent queries.
Their product bounds scoring goroutines. Four workers/one search helped low-load
latency; one worker/four searches had slightly higher eight-client throughput
here. The CPU budget, workload and latency distribution determine the choice.

### 8. Why does gRPC need client-side balancing here?

A ClusterIP normally balances connections. A long-lived HTTP/2 connection can
keep requests on one pod. Headless discovery exposes replica addresses to the
gRPC round_robin picker. Actual counters showed 160/0 versus 80/80 per shard in
each of three distribution trials, including warmup.

### 9. When is retry safe, and how do you prevent a retry storm?

Search reads an immutable index, so repeating it has no write side effect. One
retry uses a different healthy address under the original overall deadline.
INVALID_ARGUMENT never retries; overload retries additionally consume a global
budget of at most floor(external requests/10). Writes would need separate
idempotency semantics. Configured gRPC status retries are disabled, but grpc-go
can transparently retry an unsent/unprocessed call. Fault trials observed zero
application retry-counter deltas; those counters do not measure transparent
transport retries. Injected tests prove the explicit branches and budget, not
a production success rate.

### 10. What do liveness, readiness and draining each do?

Liveness checks a responsive process. Readiness says its index/membership can
serve the required work. On shutdown, readiness turns off, new work stops and
in-flight gRPC requests drain within a bound. PreStop allows endpoint propagation.
The PDB constrains voluntary eviction, not crashes or direct deletions.

### 11. Why open-loop load, warmup and repeated windows?

A closed-loop client sends less when a server stalls, potentially hiding overload
(coordinated omission). Open-loop keeps scheduled arrivals and times from that
schedule, reporting queue drops/errors. Warmup is excluded; each case has three
windows and reports median/min/max. A regression test prevents early completion
from shrinking the denominator and reporting achieved QPS above offered QPS.

### 12. How do you know the results are reproducible and limited honestly?

Exports are content-addressed and checksummed, queries are held out, tie order
is deterministic, and raw trials record settings, binary/image identity and source
state. Hosted CI checks Go race/vet/format, Python tests and Compose end-to-end.
Failed/superseded attempts remain archived. The results are short single-host
pilots; they establish neither production SLA nor multi-node scaling.

## What I would do next

First extend duration and arrival patterns while keeping raw errors and tail
samples, then repeat on a separate load host. CPU profiles motivate evaluating
specialized scoring kernels only with oracle agreement and before/after controls.
At a larger gallery size, compare ANN recall/latency. For production, replace the
demo hostPath, define a safe artifact rollout contract, and add access control,
TLS and service-level monitoring. These are proposals, not completed work.
