package fanout

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"github.com/xiaorui7/rae-feature-compression-benchmark/serving/internal/shard"
	pb "github.com/xiaorui7/rae-feature-compression-benchmark/serving/proto"
	"google.golang.org/grpc/metadata"
	"io"
	"log/slog"
	"math"
	"net/http"
	"sync/atomic"
	"time"
)

type Query struct {
	ID         string    `json:"id"`
	Vector     []float32 `json:"vector"`
	ExcludeIDs []string  `json:"exclude_ids,omitempty"`
}
type Request struct {
	Queries      []Query `json:"queries"`
	K            int     `json:"k"`
	AllowPartial bool    `json:"allow_partial"`
}
type HTTP struct {
	Engine   *Engine
	Dim      int
	Timeout  time.Duration
	Slots    chan struct{}
	Draining atomic.Bool
}

func (h *HTTP) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200) })
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, r *http.Request) {
		if h.Draining.Load() || !h.Engine.Ready() {
			http.Error(w, "unavailable", 503)
			return
		}
		w.WriteHeader(200)
	})
	mux.Handle("/metrics", h.Engine.Metrics.Handler())
	mux.HandleFunc("POST /v1/search", h.Search)
	return mux
}
func (h *HTTP) Search(w http.ResponseWriter, r *http.Request) {
	started := time.Now()
	outcome := "bad_request"
	defer func() {
		h.Engine.Metrics.Requests.WithLabelValues(outcome).Inc()
		h.Engine.Metrics.Latency.WithLabelValues(outcome).Observe(time.Since(started).Seconds())
	}()
	var body Request
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, shard.MaxMessageBytes))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&body); err != nil {
		http.Error(w, "invalid JSON body", 400)
		return
	}
	var extra any
	if decoder.Decode(&extra) != io.EOF {
		http.Error(w, "only one JSON object allowed", 400)
		return
	}
	if len(body.Queries) < 1 || len(body.Queries) > shard.MaxBatch || body.K < 1 || body.K > 100 {
		http.Error(w, "invalid batch or k", 400)
		return
	}
	req := &pb.SearchRequest{K: uint32(body.K)}
	for _, q := range body.Queries {
		if len(q.Vector) != h.Dim || len(q.ID) > 1024 || len(q.ExcludeIDs) > 100 {
			http.Error(w, "invalid query", 400)
			return
		}
		for _, v := range q.Vector {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				http.Error(w, "nonfinite query", 400)
				return
			}
		}
		for _, id := range q.ExcludeIDs {
			if len(id) > 1024 {
				http.Error(w, "invalid excluded ID", 400)
				return
			}
		}
		req.Queries = append(req.Queries, &pb.Query{Id: q.ID, Values: q.Vector, ExcludeIds: q.ExcludeIDs})
	}
	if h.Draining.Load() {
		outcome = "unavailable"
		http.Error(w, "draining", 503)
		return
	}
	select {
	case h.Slots <- struct{}{}:
		defer func() { <-h.Slots }()
	default:
		outcome = "overloaded"
		http.Error(w, "busy", 429)
		return
	}
	h.Engine.Metrics.Inflight.Inc()
	defer h.Engine.Metrics.Inflight.Dec()
	id := r.Header.Get("X-Request-ID")
	if id == "" || len(id) > 128 {
		var b [16]byte
		_, _ = rand.Read(b[:])
		id = hex.EncodeToString(b[:])
	}
	w.Header().Set("X-Request-ID", id)
	ctx, cancel := context.WithTimeout(r.Context(), h.Timeout)
	defer cancel()
	ctx = metadata.NewOutgoingContext(ctx, metadata.Pairs("x-request-id", id))
	response, err := h.Engine.Search(ctx, req, body.AllowPartial)
	if err != nil {
		outcome = "unavailable"
		slog.Warn("search failed", "request_id", id, "error", err)
		http.Error(w, "one or more shards unavailable", 503)
		return
	}
	outcome = "ok"
	if response.Partial {
		outcome = "partial"
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(response)
	slog.Info("search", "request_id", id, "outcome", outcome, "duration_ms", time.Since(started).Milliseconds())
}
