package shard

import (
	"context"
	"github.com/xiaorui7/rae-feature-compression-benchmark/serving/internal/format"
	"github.com/xiaorui7/rae-feature-compression-benchmark/serving/internal/index"
	pb "github.com/xiaorui7/rae-feature-compression-benchmark/serving/proto"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"testing"
)

func TestBatchValidationAndReadiness(t *testing.T) {
	i, err := index.New(&format.Vectors{Dim: 1, IDs: []string{"a"}, Full: []float32{1}}, 1)
	if err != nil {
		t.Fatal(err)
	}
	s := New(i)
	request := &pb.SearchRequest{K: 1, Queries: []*pb.Query{{Id: "q", Values: []float32{1}}}}
	got, err := s.Search(context.Background(), request)
	if err != nil || got.Results[0].Hits[0].Id != "a" {
		t.Fatal(got, err)
	}
	for _, bad := range []*pb.SearchRequest{nil, {}, {K: 1, Queries: make([]*pb.Query, MaxBatch+1)}, {K: 1, Queries: []*pb.Query{nil}}} {
		if _, err := s.Search(context.Background(), bad); status.Code(err) != codes.InvalidArgument {
			t.Fatal(err)
		}
	}
	s.Ready.Store(false)
	if _, err := s.Search(context.Background(), request); status.Code(err) != codes.Unavailable {
		t.Fatal(err)
	}
}
