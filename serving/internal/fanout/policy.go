package fanout

import (
	"google.golang.org/grpc/codes"
	"sync/atomic"
)

// Budget counts external requests, not shard attempts, across the coordinator lifetime.
type Budget struct {
	Requests        atomic.Uint64
	OverloadRetries atomic.Uint64
}

func (b *Budget) BeginRequest() { b.Requests.Add(1) }
func (b *Budget) Allow(code codes.Code) bool {
	switch code {
	case codes.Unavailable, codes.DeadlineExceeded:
		return true
	case codes.ResourceExhausted:
		for {
			used := b.OverloadRetries.Load()
			if used >= b.Requests.Load()/10 {
				return false
			}
			if b.OverloadRetries.CompareAndSwap(used, used+1) {
				return true
			}
		}
	default:
		return false
	}
}
