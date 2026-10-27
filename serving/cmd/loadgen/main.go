package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"github.com/xiaorui7/rae-feature-compression-benchmark/serving/internal/fanout"
	"github.com/xiaorui7/rae-feature-compression-benchmark/serving/internal/format"
	"net/http"
	"os"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

type sample struct {
	Latency float64 `json:"latency_ms"`
	Offset  float64 `json:"scheduled_seconds"`
	Status  string  `json:"status"`
	Partial bool    `json:"partial"`
}
type report struct {
	Mode        string         `json:"mode"`
	Offered     float64        `json:"offered_qps"`
	Achieved    float64        `json:"achieved_qps"`
	Success     float64        `json:"successful_qps"`
	Duration    float64        `json:"duration_seconds"`
	Count       int            `json:"completed"`
	Errors      map[string]int `json:"errors"`
	Partial     int            `json:"partial"`
	P50         float64        `json:"p50_ms"`
	P95         float64        `json:"p95_ms"`
	P99         float64        `json:"p99_ms"`
	Max         float64        `json:"max_ms"`
	Samples     []sample       `json:"samples"`
	Environment map[string]any `json:"environment"`
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	dir := flag.String("export", "", "query export")
	url := flag.String("url", "http://127.0.0.1:8080", "coordinator URL")
	split := flag.String("split", "test", "query split")
	duration := flag.Duration("duration", 10*time.Second, "measurement duration")
	warmup := flag.Duration("warmup", 2*time.Second, "excluded warmup")
	timeout := flag.Duration("timeout", 2*time.Second, "per-request deadline")
	concurrency := flag.Int("concurrency", 4, "bounded HTTP workers")
	qps := flag.Float64("qps", 0, "open-loop offered QPS; zero selects closed-loop")
	partial := flag.Bool("allow-partial", false, "explicit partial policy")
	envFile := flag.String("environment", "", "JSON environment metadata to include")
	flag.Parse()
	if *duration <= 0 || *duration > time.Hour || *warmup < 0 || *timeout <= 0 || *concurrency < 1 || *concurrency > 1024 || *qps < 0 || *qps > 100000 {
		return fmt.Errorf("invalid load limits")
	}
	m, err := format.ReadManifest(*dir)
	if err != nil {
		return err
	}
	record, ok := m.Queries[*split]
	if !ok {
		return fmt.Errorf("unknown split")
	}
	q, err := format.Load(*dir, m, record)
	if err != nil {
		return err
	}
	payloads := make([][]byte, len(q.IDs))
	for row, id := range q.IDs {
		values := make([]float32, q.Dim)
		for col := range values {
			values[col] = float32(q.Value(row*q.Dim + col))
		}
		payloads[row], err = json.Marshal(fanout.Request{K: 10, AllowPartial: *partial, Queries: []fanout.Query{{ID: id, Vector: values}}})
		if err != nil {
			return err
		}
	}
	transport := &http.Transport{MaxIdleConns: *concurrency, MaxIdleConnsPerHost: *concurrency, MaxConnsPerHost: *concurrency, IdleConnTimeout: 30 * time.Second}
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: transport, Timeout: *timeout}
	phase := func(d time.Duration) report {
		start := time.Now()
		end := start.Add(d)
		var sequence atomic.Uint64
		var mu sync.Mutex
		samples := make([]sample, 0)
		errors := map[string]int{}
		partials := 0
		offered := 0
		dropped := 0
		execute := func(scheduled time.Time) {
			row := int(sequence.Add(1)-1) % len(payloads)
			ctx, cancel := context.WithDeadline(context.Background(), scheduled.Add(*timeout))
			defer cancel()
			req, _ := http.NewRequestWithContext(ctx, "POST", *url+"/v1/search", bytes.NewReader(payloads[row]))
			req.Header.Set("Content-Type", "application/json")
			response, err := client.Do(req)
			status := "ok"
			isPartial := false
			if err != nil {
				status = "transport"
			} else {
				var result fanout.Response
				decodeErr := json.NewDecoder(response.Body).Decode(&result)
				response.Body.Close()
				if response.StatusCode != 200 {
					status = fmt.Sprint(response.StatusCode)
				} else if decodeErr != nil {
					status = "decode"
				}
				isPartial = result.Partial
			}
			s := sample{Latency: float64(time.Since(scheduled).Microseconds()) / 1000, Offset: scheduled.Sub(start).Seconds(), Status: status, Partial: isPartial}
			mu.Lock()
			if len(samples) < 1000000 {
				samples = append(samples, s)
			} else {
				errors["sample_limit"]++
			}
			if status != "ok" {
				errors[status]++
			}
			if isPartial {
				partials++
			}
			mu.Unlock()
		}
		var wg sync.WaitGroup
		if *qps == 0 {
			for worker := 0; worker < *concurrency; worker++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for time.Now().Before(end) {
						execute(time.Now())
					}
				}()
			}
			wg.Wait()
			offered = int(sequence.Load())
		} else {
			jobs := make(chan time.Time, *concurrency)
			for worker := 0; worker < *concurrency; worker++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for scheduled := range jobs {
						execute(scheduled)
					}
				}()
			}
			for n := 0; ; n++ {
				scheduled := start.Add(time.Duration(float64(n) / *qps * float64(time.Second)))
				if !scheduled.Before(end) {
					break
				}
				if delay := time.Until(scheduled); delay > 0 {
					time.Sleep(delay)
				}
				offered++
				select {
				case jobs <- scheduled:
				default:
					dropped++
				}
			}
			close(jobs)
			// Preserve the full observation window even when the final response is early.
			if remaining := time.Until(end); remaining > 0 {
				time.Sleep(remaining)
			}
			wg.Wait()
		}
		elapsed := time.Since(start).Seconds()
		if dropped > 0 {
			errors["loadgen_queue_full"] = dropped
		}
		latencies := make([]float64, len(samples))
		success := 0
		for i, s := range samples {
			latencies[i] = s.Latency
			if s.Status == "ok" {
				success++
			}
		}
		sort.Float64s(latencies)
		percentile := func(p float64) float64 {
			if len(latencies) == 0 {
				return 0
			}
			return latencies[int(float64(len(latencies)-1)*p)]
		}
		mode := "closed_loop"
		if *qps > 0 {
			mode = "open_loop"
		}
		return report{Mode: mode, Offered: float64(offered) / d.Seconds(), Achieved: float64(len(samples)) / elapsed, Success: float64(success) / elapsed, Duration: elapsed, Count: len(samples), Errors: errors, Partial: partials, P50: percentile(.5), P95: percentile(.95), P99: percentile(.99), Max: percentile(1), Samples: samples}
	}
	if *warmup > 0 {
		_ = phase(*warmup)
	}
	r := phase(*duration)
	r.Environment = map[string]any{"go": runtime.Version(), "os": runtime.GOOS, "arch": runtime.GOARCH, "logical_cpus": runtime.NumCPU(), "concurrency": *concurrency, "configured_qps": *qps, "warmup_seconds": warmup.Seconds(), "export_id": m.ExportID, "data_status": m.Status, "count": m.TotalCount, "dimension": m.Dim, "shards": m.ShardCount, "dtype": m.Dtype}
	if *envFile != "" {
		raw, err := os.ReadFile(*envFile)
		if err != nil {
			return err
		}
		var environment any
		if err = json.Unmarshal(raw, &environment); err != nil {
			return err
		}
		r.Environment["deployment"] = environment
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(r)
}
