package fanout

import (
	"context"
	"fmt"
	pb "github.com/xiaorui7/rae-feature-compression-benchmark/serving/proto"
	"google.golang.org/grpc"
	_ "google.golang.org/grpc/balancer/roundrobin"
	"google.golang.org/grpc/credentials/insecure"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/resolver/manual"
	"net"
	"sort"
	"sync"
	"time"
)

// Pool maintains one round-robin channel and direct channels for explicit failover.
// Discovery removes dead endpoints; retained connections are bounded by current membership.
type Pool struct {
	Targets  []string
	mu       sync.RWMutex
	direct   map[string]*grpc.ClientConn
	healthy  map[string]bool
	rr       *grpc.ClientConn
	resolver *manual.Resolver
	closed   bool
}

func NewPool(targets []string, policy string) (*Pool, error) {
	if len(targets) == 0 || (policy != "round_robin" && policy != "pick_first") {
		return nil, fmt.Errorf("invalid replica targets or policy")
	}
	r := manual.NewBuilderWithScheme("replicas")
	r.InitialState(resolver.State{})
	conn, err := grpc.NewClient(r.Scheme()+":///shard", grpc.WithResolvers(r), grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithDefaultServiceConfig(`{"loadBalancingConfig":[{"`+policy+`":{}}]}`), grpc.WithDisableRetry())
	if err != nil {
		return nil, err
	}
	return &Pool{Targets: targets, direct: map[string]*grpc.ClientConn{}, healthy: map[string]bool{}, rr: conn, resolver: r}, nil
}
func (p *Pool) Refresh(ctx context.Context) {
	addresses := map[string]bool{}
	for _, target := range p.Targets {
		host, port, err := net.SplitHostPort(target)
		if err != nil {
			continue
		}
		if net.ParseIP(host) != nil {
			addresses[target] = true
			continue
		}
		ips, err := net.DefaultResolver.LookupHost(ctx, host)
		if err != nil {
			continue
		}
		for _, ip := range ips {
			addresses[net.JoinHostPort(ip, port)] = true
		}
	}
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return
	}
	for addr, c := range p.direct {
		if !addresses[addr] {
			c.Close()
			delete(p.direct, addr)
			delete(p.healthy, addr)
		}
	}
	sorted := make([]string, 0, len(addresses))
	for addr := range addresses {
		sorted = append(sorted, addr)
	}
	sort.Strings(sorted)
	resolved := make([]resolver.Address, 0, len(sorted))
	clients := make(map[string]*grpc.ClientConn, len(sorted))
	for _, addr := range sorted {
		c := p.direct[addr]
		if c == nil {
			var err error
			c, err = grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithDisableRetry())
			if err != nil {
				continue
			}
			p.direct[addr] = c
		}
		clients[addr] = c
		// Keep unhealthy endpoints in gRPC resolver: connection recovery is automatic.
		resolved = append(resolved, resolver.Address{Addr: addr})
	}
	p.mu.Unlock()
	// A failed health probe must not hold the lock needed by deadline-bound retries.
	for addr, c := range clients {
		probe, cancel := context.WithTimeout(ctx, 150*time.Millisecond)
		health, err := healthpb.NewHealthClient(c).Check(probe, &healthpb.HealthCheckRequest{})
		cancel()
		p.mu.Lock()
		if p.direct[addr] == c && !p.closed {
			p.healthy[addr] = err == nil && health.Status == healthpb.HealthCheckResponse_SERVING
		}
		p.mu.Unlock()
	}
	p.resolver.UpdateState(resolver.State{Addresses: resolved})
}
func (p *Pool) Ready() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	for _, h := range p.healthy {
		if h {
			return true
		}
	}
	return false
}
func (p *Pool) Search(ctx context.Context, req *pb.SearchRequest) (*pb.SearchResponse, string, error) {
	var remote peer.Peer
	result, err := pb.NewSearchServiceClient(p.rr).Search(ctx, req, grpc.Peer(&remote))
	addr := ""
	if remote.Addr != nil {
		addr = remote.Addr.String()
	}
	return result, addr, err
}
func (p *Pool) Alternative(exclude string) (pb.SearchServiceClient, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	keys := make([]string, 0, len(p.direct))
	for addr := range p.direct {
		if addr != exclude && p.healthy[addr] {
			keys = append(keys, addr)
		}
	}
	sort.Strings(keys)
	if len(keys) == 0 {
		return nil, false
	}
	// Without a selected peer the first attempt failed before reaching any replica.
	return pb.NewSearchServiceClient(p.direct[keys[0]]), true
}
func (p *Pool) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.closed = true
	p.rr.Close()
	for _, c := range p.direct {
		c.Close()
	}
}
