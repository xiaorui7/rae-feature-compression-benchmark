package fanout

import (
	"context"
	"fmt"
	"github.com/xiaorui7/rae-feature-compression-benchmark/serving/internal/obs"
	"github.com/xiaorui7/rae-feature-compression-benchmark/serving/internal/topk"
	pb "github.com/xiaorui7/rae-feature-compression-benchmark/serving/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"log/slog"
	"sync"
	"time"
)

type ReplicaPool interface {
	Search(context.Context, *pb.SearchRequest) (*pb.SearchResponse, string, error)
	Alternative(string) (pb.SearchServiceClient, bool)
	Ready() bool
}
type Engine struct {
	Pools   []ReplicaPool
	Timeout time.Duration
	Budget  Budget
	Metrics *obs.Metrics
}
type Result struct {
	ID   string     `json:"id"`
	Hits []topk.Hit `json:"hits"`
}
type Response struct {
	Results []Result `json:"results"`
	Partial bool     `json:"partial"`
	Missing []int    `json:"missing_shards"`
}

func (e *Engine) Ready() bool {
	for _, p := range e.Pools {
		if !p.Ready() {
			return false
		}
	}
	return len(e.Pools) > 0
}
func (e *Engine) attempt(ctx context.Context, p ReplicaPool, req *pb.SearchRequest) (*pb.SearchResponse, error) {
	attempt, cancel := context.WithTimeout(ctx, e.Timeout)
	result, failed, err := p.Search(attempt, req)
	cancel()
	if err == nil || ctx.Err() != nil {
		return result, err
	}
	alt, ok := p.Alternative(failed)
	if !ok || !e.Budget.Allow(status.Code(err)) {
		return nil, err
	}
	if e.Metrics != nil {
		e.Metrics.Retries.WithLabelValues(status.Code(err).String()).Inc()
	}
	retry, cancel := context.WithTimeout(ctx, e.Timeout)
	defer cancel()
	return alt.Search(retry, req, grpc.WaitForReady(false))
}
func (e *Engine) Search(ctx context.Context, req *pb.SearchRequest, partial bool) (Response, error) {
	e.Budget.BeginRequest()
	replies := make([]*pb.SearchResponse, len(e.Pools))
	errs := make([]error, len(e.Pools))
	var wg sync.WaitGroup
	for i, p := range e.Pools {
		wg.Add(1)
		go func(i int, p ReplicaPool) {
			defer wg.Done()
			started := time.Now()
			replies[i], errs[i] = e.attempt(ctx, p, req)
			if e.Metrics != nil {
				e.Metrics.Fanout.WithLabelValues(fmt.Sprint(i)).Observe(time.Since(started).Seconds())
			}
		}(i, p)
	}
	wg.Wait()
	requestID := ""
	if md, ok := metadata.FromOutgoingContext(ctx); ok {
		if ids := md.Get("x-request-id"); len(ids) > 0 {
			requestID = ids[0]
		}
	}
	response := Response{Results: make([]Result, len(req.Queries)), Missing: []int{}}
	for i, err := range errs {
		if err == nil && (replies[i] == nil || len(replies[i].Results) != len(req.Queries)) {
			err = status.Error(codes.Internal, "invalid shard response")
			errs[i] = err
		}
		if err != nil {
			slog.Warn("shard_search_failed", "request_id", requestID, "shard", i, "code", status.Code(err).String(), "error", err)
			response.Missing = append(response.Missing, i)
		}
	}
	if len(response.Missing) > 0 && !partial {
		return response, status.Error(codes.Unavailable, "one or more shards unavailable")
	}
	response.Partial = len(response.Missing) > 0
	for q, query := range req.Queries {
		lists := make([][]topk.Hit, 0, len(replies))
		for i, r := range replies {
			if errs[i] != nil {
				continue
			}
			if r.Results[q].Id != query.Id {
				return response, status.Error(codes.Internal, "invalid query correlation")
			}
			list := make([]topk.Hit, 0, len(r.Results[q].Hits))
			for _, hit := range r.Results[q].Hits {
				list = append(list, topk.Hit{ID: hit.Id, Score: hit.Score})
			}
			lists = append(lists, list)
		}
		response.Results[q] = Result{ID: query.Id, Hits: topk.Merge(lists, int(req.K))}
	}
	return response, nil
}
