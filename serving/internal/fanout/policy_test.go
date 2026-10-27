package fanout

import (
	"google.golang.org/grpc/codes"
	"testing"
)

func TestRetryBudget(t *testing.T) {
	b := &Budget{}
	for n := 0; n < 100; n++ {
		b.BeginRequest()
	}
	for n := 0; n < 20; n++ {
		if got := b.Allow(codes.ResourceExhausted); got != (n < 10) {
			t.Fatal(n, got)
		}
	}
	if b.Allow(codes.InvalidArgument) || !b.Allow(codes.Unavailable) || !b.Allow(codes.DeadlineExceeded) {
		t.Fatal("policy")
	}
}
