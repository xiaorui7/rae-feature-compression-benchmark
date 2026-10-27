package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"github.com/xiaorui7/rae-feature-compression-benchmark/serving/internal/fanout"
	"github.com/xiaorui7/rae-feature-compression-benchmark/serving/internal/obs"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stderr, nil)))
	if err := run(); err != nil {
		slog.Error("coordinator failed", "error", err)
		os.Exit(1)
	}
}
func run() error {
	config := flag.String("config", "", "JSON {dim, shards:[[host:port,...],...]}")
	listen := flag.String("listen", "127.0.0.1:8080", "HTTP address")
	timeout := flag.Duration("request-timeout", time.Second, "overall request deadline")
	perShard := flag.Duration("shard-timeout", 200*time.Millisecond, "per-attempt deadline")
	limit := flag.Int("max-inflight", 64, "maximum admitted HTTP searches")
	policy := flag.String("balance", "round_robin", "gRPC policy: round_robin or pick_first")
	drain := flag.Duration("drain-timeout", 10*time.Second, "shutdown bound")
	flag.Parse()
	if *timeout <= 0 || *perShard <= 0 || *limit < 1 || *limit > 1024 || *drain <= 0 {
		return fmt.Errorf("invalid limits")
	}
	raw, err := os.ReadFile(*config)
	if err != nil {
		return err
	}
	var cfg struct {
		Dim    int        `json:"dim"`
		Shards [][]string `json:"shards"`
	}
	if err = json.Unmarshal(raw, &cfg); err != nil {
		return err
	}
	if cfg.Dim < 1 || cfg.Dim > 65536 || len(cfg.Shards) < 1 || len(cfg.Shards) > 64 {
		return fmt.Errorf("invalid dimensions or shard count")
	}
	engine := &fanout.Engine{Timeout: *perShard, Metrics: obs.New("coordinator")}
	pools := []*fanout.Pool{}
	for _, targets := range cfg.Shards {
		p, err := fanout.NewPool(targets, *policy)
		if err != nil {
			return err
		}
		defer p.Close()
		pools = append(pools, p)
		engine.Pools = append(engine.Pools, p)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	refresh := func() {
		for _, p := range pools {
			probe, cancel := context.WithTimeout(ctx, time.Second)
			p.Refresh(probe)
			cancel()
		}
		if engine.Ready() {
			engine.Metrics.Ready.Set(1)
		} else {
			engine.Metrics.Ready.Set(0)
		}
	}
	refresh()
	go func() {
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				refresh()
			}
		}
	}()
	handler := &fanout.HTTP{Engine: engine, Dim: cfg.Dim, Timeout: *timeout, Slots: make(chan struct{}, *limit)}
	server := &http.Server{Addr: *listen, Handler: handler.Handler(), ReadHeaderTimeout: 2 * time.Second, ReadTimeout: 5 * time.Second, WriteTimeout: *timeout + time.Second, IdleTimeout: 30 * time.Second}
	failure := make(chan error, 1)
	go func() { failure <- server.ListenAndServe() }()
	select {
	case <-ctx.Done():
	case err = <-failure:
		if err != http.ErrServerClosed {
			return err
		}
	}
	handler.Draining.Store(true)
	engine.Metrics.Ready.Set(0)
	shutdown, cancel := context.WithTimeout(context.Background(), *drain)
	defer cancel()
	if err := server.Shutdown(shutdown); err != nil {
		_ = server.Close()
		return err
	}
	return nil
}
