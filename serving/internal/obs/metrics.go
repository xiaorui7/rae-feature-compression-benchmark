package obs

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"net/http"
)

type Metrics struct {
	Registry *prometheus.Registry
	Requests *prometheus.CounterVec
	Latency  *prometheus.HistogramVec
	Fanout   *prometheus.HistogramVec
	Retries  *prometheus.CounterVec
	Inflight prometheus.Gauge
	Scanned  prometheus.Counter
	Bytes    prometheus.Gauge
	Ready    prometheus.Gauge
}

func New(prefix string) *Metrics {
	r := prometheus.NewRegistry()
	m := &Metrics{Registry: r,
		Requests: prometheus.NewCounterVec(prometheus.CounterOpts{Name: prefix + "_requests_total", Help: "Completed requests by outcome."}, []string{"outcome"}),
		Latency:  prometheus.NewHistogramVec(prometheus.HistogramOpts{Name: prefix + "_request_duration_seconds", Help: "Request latency.", Buckets: prometheus.ExponentialBuckets(.0001, 2, 18)}, []string{"outcome"}),
		Fanout:   prometheus.NewHistogramVec(prometheus.HistogramOpts{Name: prefix + "_fanout_duration_seconds", Help: "Per-shard latency including retry.", Buckets: prometheus.ExponentialBuckets(.0001, 2, 18)}, []string{"shard"}),
		Retries:  prometheus.NewCounterVec(prometheus.CounterOpts{Name: prefix + "_retries_total", Help: "Retried shard attempts by status."}, []string{"code"}),
		Inflight: prometheus.NewGauge(prometheus.GaugeOpts{Name: prefix + "_inflight", Help: "Admitted requests."}),
		Scanned:  prometheus.NewCounter(prometheus.CounterOpts{Name: prefix + "_vectors_scanned_total", Help: "Vectors considered in completed searches."}),
		Bytes:    prometheus.NewGauge(prometheus.GaugeOpts{Name: prefix + "_numeric_index_bytes", Help: "Raw values and inverse norms, excluding IDs and overhead."}),
		Ready:    prometheus.NewGauge(prometheus.GaugeOpts{Name: prefix + "_ready", Help: "Ready for requests."}),
	}
	r.MustRegister(m.Requests, m.Latency, m.Fanout, m.Retries, m.Inflight, m.Scanned, m.Bytes, m.Ready, prometheus.NewGoCollector(), prometheus.NewProcessCollector(prometheus.ProcessCollectorOpts{}))
	return m
}
func (m *Metrics) Handler() http.Handler {
	return promhttp.HandlerFor(m.Registry, promhttp.HandlerOpts{})
}
