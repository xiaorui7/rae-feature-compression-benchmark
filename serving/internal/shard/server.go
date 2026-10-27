package shard

import (
	"context"
	"log/slog"
	"sync/atomic"
	"time"

	"github.com/xiaorui7/rae-feature-compression-benchmark/serving/internal/index"
	"github.com/xiaorui7/rae-feature-compression-benchmark/serving/internal/obs"
	pb "github.com/xiaorui7/rae-feature-compression-benchmark/serving/proto"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

const MaxBatch = 32
const MaxMessageBytes = 4 << 20

type Server struct {
	pb.UnimplementedSearchServiceServer
	Index          *index.Index
	Ready          atomic.Bool
	Queries        atomic.Uint64
	slots          chan struct{}
	DefaultTimeout time.Duration
	Metrics        *obs.Metrics
}

func New(i *index.Index) *Server {
	s := &Server{Index: i, slots: make(chan struct{}, 16), DefaultTimeout: 5 * time.Second, Metrics: obs.New("shard")}
	s.Metrics.Bytes.Set(float64(i.NumericBytes()))
	s.Metrics.Ready.Set(1)
	s.Ready.Store(true)
	return s
}

func (s *Server) Search(ctx context.Context, req *pb.SearchRequest) (*pb.SearchResponse, error) {
	if !s.Ready.Load() {
		return nil, status.Error(codes.Unavailable, "shard not ready")
	}
	if req == nil || len(req.Queries) < 1 || len(req.Queries) > MaxBatch || req.K < 1 || req.K > 100 {
		return nil, status.Error(codes.InvalidArgument, "invalid batch or k")
	}
	for _, q := range req.Queries {
		if q == nil || len(q.Id) > 1024 {
			return nil, status.Error(codes.InvalidArgument, "invalid query")
		}
		if len(q.ExcludeIds) > 100 {
			return nil, status.Error(codes.InvalidArgument, "too many excluded IDs")
		}
		for _, id := range q.ExcludeIds {
			if len(id) > 1024 {
				return nil, status.Error(codes.InvalidArgument, "excluded ID too long")
			}
		}
		if err := s.Index.Validate(q.Values, int(req.K)); err != nil {
			return nil, status.Error(codes.InvalidArgument, "invalid query dimension or values")
		}
	}
	select {
	case s.slots <- struct{}{}:
		defer func() { <-s.slots }()
	default:
		return nil, status.Error(codes.ResourceExhausted, "shard busy")
	}
	if _, present := ctx.Deadline(); !present {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, s.DefaultTimeout)
		defer cancel()
	}
	started := time.Now()
	s.Metrics.Inflight.Inc()
	defer s.Metrics.Inflight.Dec()
	requestID := ""
	if md, ok := metadata.FromIncomingContext(ctx); ok {
		values := md.Get("x-request-id")
		if len(values) > 0 && len(values[0]) <= 128 {
			requestID = values[0]
		}
	}
	defer func() {
		s.Metrics.Latency.WithLabelValues("admitted").Observe(time.Since(started).Seconds())
		slog.Info("search", "request_id", requestID, "queries", len(req.Queries), "duration_ms", time.Since(started).Milliseconds())
	}()
	response := &pb.SearchResponse{}
	for _, q := range req.Queries {
		hits, err := s.Index.Search(ctx, q.Id, q.Values, int(req.K), q.ExcludeIds...)
		if err != nil {
			return nil, status.FromContextError(err).Err()
		}
		result := &pb.QueryResult{Id: q.Id}
		for _, h := range hits {
			result.Hits = append(result.Hits, &pb.Hit{Id: h.ID, Score: h.Score})
		}
		response.Results = append(response.Results, result)
		s.Queries.Add(1)
		s.Metrics.Requests.WithLabelValues("ok").Inc()
		s.Metrics.Scanned.Add(float64(s.Index.Count()))
	}
	return response, nil
}
