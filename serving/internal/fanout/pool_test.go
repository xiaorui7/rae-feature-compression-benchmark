package fanout

import (
	"context"
	pb "github.com/xiaorui7/rae-feature-compression-benchmark/serving/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/metadata"
	"net"
	"sync/atomic"
	"testing"
	"time"
)

type blockingServer struct {
	pb.UnimplementedSearchServiceServer
	entered chan struct{}
	release chan struct{}
	id      atomic.Bool
}

func (s *blockingServer) Search(c context.Context, r *pb.SearchRequest) (*pb.SearchResponse, error) {
	md, _ := metadata.FromIncomingContext(c)
	ids := md.Get("x-request-id")
	s.id.Store(len(ids) == 1 && ids[0] == "wire-id")
	close(s.entered)
	select {
	case <-s.release:
		return reply("a", 1), nil
	case <-c.Done():
		return nil, c.Err()
	}
}
func TestWirePropagationAndGracefulDrain(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server := grpc.NewServer()
	s := &blockingServer{entered: make(chan struct{}), release: make(chan struct{})}
	pb.RegisterSearchServiceServer(server, s)
	healthpb.RegisterHealthServer(server, health.NewServer())
	go server.Serve(listener)
	defer server.Stop()
	p, err := NewPool([]string{listener.Addr().String()}, "round_robin")
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	p.Refresh(ctx)
	if !p.Ready() {
		t.Fatal("pool not ready")
	}
	result := make(chan error, 1)
	go func() {
		c := metadata.NewOutgoingContext(ctx, metadata.Pairs("x-request-id", "wire-id"))
		_, _, err := p.Search(c, request())
		result <- err
	}()
	select {
	case <-s.entered:
	case <-ctx.Done():
		t.Fatal("request not started")
	}
	drained := make(chan struct{})
	go func() { server.GracefulStop(); close(drained) }()
	select {
	case <-drained:
		t.Fatal("drain ended with in-flight request")
	case <-time.After(20 * time.Millisecond):
	}
	close(s.release)
	if err := <-result; err != nil {
		t.Fatal(err)
	}
	select {
	case <-drained:
	case <-ctx.Done():
		t.Fatal("drain did not complete")
	}
	if !s.id.Load() {
		t.Fatal("metadata missing on wire")
	}
}
