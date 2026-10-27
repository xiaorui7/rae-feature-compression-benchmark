package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"net/http/pprof"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"github.com/xiaorui7/rae-feature-compression-benchmark/serving/internal/format"
	"github.com/xiaorui7/rae-feature-compression-benchmark/serving/internal/index"
	"github.com/xiaorui7/rae-feature-compression-benchmark/serving/internal/shard"
	pb "github.com/xiaorui7/rae-feature-compression-benchmark/serving/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
)

func main() {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stderr, nil)))
	if err := run(); err != nil {
		slog.Error("shard failed", "error", err)
		os.Exit(1)
	}
}
func run() error {
	dir := flag.String("export", "", "export directory")
	number := flag.Int("shard-index", 0, "shard index")
	addr := flag.String("listen", "127.0.0.1:9090", "gRPC address")
	admin := flag.String("admin", "127.0.0.1:9091", "admin HTTP address")
	workers := flag.Int("workers", runtime.GOMAXPROCS(0), "scoring workers per search")
	concurrent := flag.Int("max-concurrent-searches", 1, "concurrent scoring searches (1-16)")
	searchTimeout := flag.Duration("search-timeout", 5*time.Second, "fallback deadline when caller supplies none")
	drain := flag.Duration("drain-timeout", 10*time.Second, "graceful shutdown bound")
	profile := flag.Bool("pprof", false, "enable CPU/heap profiling on private admin port")
	flag.Parse()
	if *dir == "" || *drain <= 0 || *searchTimeout <= 0 {
		return fmt.Errorf("export and positive drain/search timeouts required")
	}
	m, err := format.ReadManifest(*dir)
	if err != nil {
		return err
	}
	if *number < 0 || *number >= m.ShardCount {
		return fmt.Errorf("shard index out of range")
	}
	vectors, err := format.Load(*dir, m, m.Shards[*number])
	if err != nil {
		return err
	}
	idx, err := index.New(vectors, *workers, *concurrent)
	if err != nil {
		return err
	}
	service := shard.New(idx)
	service.DefaultTimeout = *searchTimeout
	server := grpc.NewServer(grpc.MaxRecvMsgSize(shard.MaxMessageBytes), grpc.MaxSendMsgSize(shard.MaxMessageBytes), grpc.MaxConcurrentStreams(16))
	pb.RegisterSearchServiceServer(server, service)
	healthServer := health.NewServer()
	healthpb.RegisterHealthServer(server, healthServer)
	healthServer.SetServingStatus("", healthpb.HealthCheckResponse_SERVING)
	listener, err := net.Listen("tcp", *addr)
	if err != nil {
		return err
	}
	defer listener.Close()
	adminListener, err := net.Listen("tcp", *admin)
	if err != nil {
		return err
	}
	defer adminListener.Close()
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, r *http.Request) {
		if !service.Ready.Load() {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	})
	mux.Handle("/metrics", service.Metrics.Handler())
	if *profile {
		mux.HandleFunc("/debug/pprof/", pprof.Index)
		mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
		mux.HandleFunc("/debug/pprof/heap", pprof.Handler("heap").ServeHTTP)
	}
	httpServer := &http.Server{Handler: mux, ReadHeaderTimeout: 2 * time.Second, ReadTimeout: 5 * time.Second, WriteTimeout: 5 * time.Second, IdleTimeout: 30 * time.Second}
	failures := make(chan error, 2)
	go func() { failures <- server.Serve(listener) }()
	go func() { failures <- httpServer.Serve(adminListener) }()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	slog.Info("ready", "export_id", m.ExportID, "shard", *number, "count", idx.Count(), "dim", idx.Dim(), "dtype", m.Dtype, "numeric_bytes", idx.NumericBytes())
	var serveErr error
	select {
	case <-ctx.Done():
	case serveErr = <-failures:
	}
	service.Ready.Store(false)
	service.Metrics.Ready.Set(0)
	healthServer.Shutdown()
	done := make(chan struct{})
	go func() { server.GracefulStop(); close(done) }()
	select {
	case <-done:
	case <-time.After(*drain):
		server.Stop()
		<-done
	}
	shutdown, cancel := context.WithTimeout(context.Background(), *drain)
	defer cancel()
	_ = httpServer.Shutdown(shutdown)
	return serveErr
}
