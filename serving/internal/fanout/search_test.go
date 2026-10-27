package fanout

import (
	"context"
	"fmt"
	"github.com/xiaorui7/rae-feature-compression-benchmark/serving/internal/obs"
	pb "github.com/xiaorui7/rae-feature-compression-benchmark/serving/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

type clientFunc func(context.Context, *pb.SearchRequest) (*pb.SearchResponse, error)

func (f clientFunc) Search(c context.Context, r *pb.SearchRequest, _ ...grpc.CallOption) (*pb.SearchResponse, error) {
	return f(c, r)
}

type fake struct {
	first  clientFunc
	second clientFunc
	calls  atomic.Int32
	retry  atomic.Int32
	t      *testing.T
}

func (f *fake) Ready() bool { return true }
func (f *fake) Search(c context.Context, r *pb.SearchRequest) (*pb.SearchResponse, string, error) {
	f.calls.Add(1)
	result, err := f.first(c, r)
	return result, "first", err
}
func (f *fake) Alternative(exclude string) (pb.SearchServiceClient, bool) {
	if exclude != "first" {
		f.t.Errorf("retry did not exclude first replica: %s", exclude)
	}
	if f.second == nil {
		return nil, false
	}
	return clientFunc(func(c context.Context, r *pb.SearchRequest) (*pb.SearchResponse, error) {
		f.retry.Add(1)
		return f.second(c, r)
	}), true
}
func reply(id string, score float64) *pb.SearchResponse {
	return &pb.SearchResponse{Results: []*pb.QueryResult{{Id: "q", Hits: []*pb.Hit{{Id: id, Score: score}}}}}
}
func request() *pb.SearchRequest {
	return &pb.SearchRequest{K: 2, Queries: []*pb.Query{{Id: "q", Values: []float32{1}}}}
}
func TestMergeShardCounts(t *testing.T) {
	for _, n := range []int{1, 2, 4} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			e := &Engine{Timeout: time.Second}
			for s := 0; s < n; s++ {
				s := s
				e.Pools = append(e.Pools, &fake{t: t, first: func(context.Context, *pb.SearchRequest) (*pb.SearchResponse, error) {
					r := reply("a", 1)
					r.Results[0].Hits = nil
					for j := s; j < 8; j += n {
						r.Results[0].Hits = append(r.Results[0].Hits, &pb.Hit{Id: fmt.Sprint(j), Score: float64(8 - j)})
					}
					return r, nil
				}})
			}
			r, err := e.Search(context.Background(), request(), false)
			if err != nil || len(r.Results[0].Hits) != 2 || r.Results[0].Hits[0].ID != "0" || r.Results[0].Hits[1].ID != "1" {
				t.Fatalf("merge: %+v %v", r, err)
			}
		})
	}
}
func TestRetryCodesAndBudget(t *testing.T) {
	for _, code := range []codes.Code{codes.Unavailable, codes.DeadlineExceeded, codes.InvalidArgument, codes.ResourceExhausted} {
		t.Run(code.String(), func(t *testing.T) {
			p := &fake{t: t, first: func(context.Context, *pb.SearchRequest) (*pb.SearchResponse, error) {
				return nil, status.Error(code, "injected")
			}, second: func(context.Context, *pb.SearchRequest) (*pb.SearchResponse, error) { return reply("a", 1), nil }}
			e := &Engine{Timeout: time.Second, Pools: []ReplicaPool{p}}
			for i := 0; i < 100; i++ {
				_, _ = e.Search(context.Background(), request(), false)
			}
			want := int32(100)
			if code == codes.InvalidArgument {
				want = 0
			}
			if code == codes.ResourceExhausted {
				want = 10
			}
			if p.retry.Load() != want {
				t.Fatalf("retry=%d want=%d", p.retry.Load(), want)
			}
		})
	}
}
func TestDeadlineAndPartial(t *testing.T) {
	p := &fake{t: t, first: func(c context.Context, _ *pb.SearchRequest) (*pb.SearchResponse, error) {
		<-c.Done()
		return nil, status.FromContextError(c.Err()).Err()
	}, second: func(c context.Context, _ *pb.SearchRequest) (*pb.SearchResponse, error) {
		<-c.Done()
		return nil, status.FromContextError(c.Err()).Err()
	}}
	e := &Engine{Timeout: 10 * time.Millisecond, Pools: []ReplicaPool{p}}
	c, cancel := context.WithTimeout(context.Background(), 15*time.Millisecond)
	defer cancel()
	start := time.Now()
	if _, err := e.Search(c, request(), false); err == nil {
		t.Fatal("missing strict failure")
	}
	if time.Since(start) > 100*time.Millisecond || p.retry.Load() != 1 {
		t.Fatal("deadline or attempt bound")
	}
	r, err := e.Search(context.Background(), request(), true)
	if err != nil || !r.Partial || len(r.Missing) != 1 {
		t.Fatalf("partial %+v %v", r, err)
	}
}
func TestHTTPPoliciesAndRequestID(t *testing.T) {
	p := &fake{t: t, first: func(c context.Context, _ *pb.SearchRequest) (*pb.SearchResponse, error) {
		md, _ := metadata.FromOutgoingContext(c)
		if md.Get("x-request-id")[0] != "test-id" {
			t.Error("request ID lost")
		}
		return nil, status.Error(codes.Unavailable, "injected")
	}}
	e := &Engine{Timeout: time.Second, Pools: []ReplicaPool{p}, Metrics: obs.New("test")}
	h := &HTTP{Engine: e, Dim: 1, Timeout: time.Second, Slots: make(chan struct{}, 1)}
	for _, tc := range []struct {
		body string
		code int
	}{{`{"queries":[{"id":"q","vector":[1]}],"k":1}`, 503}, {`{"queries":[{"id":"q","vector":[1]}],"k":1,"allow_partial":true}`, 200}, {`{"queries":[{"vector":[1,2]}],"k":1}`, 400}} {
		w := httptest.NewRecorder()
		r := httptest.NewRequest("POST", "/v1/search", strings.NewReader(tc.body))
		r.Header.Set("X-Request-ID", "test-id")
		h.Handler().ServeHTTP(w, r)
		if w.Code != tc.code {
			t.Fatalf("%d %s", w.Code, w.Body.String())
		}
		if tc.code == 200 && !strings.Contains(w.Body.String(), `"partial":true`) {
			t.Fatal("silent partial")
		}
	}
	h.Draining.Store(true)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/v1/search", strings.NewReader(`{"queries":[{"vector":[1]}],"k":1}`))
	h.Handler().ServeHTTP(w, r)
	if w.Code != 503 {
		t.Fatal("drain accepts requests")
	}
}
