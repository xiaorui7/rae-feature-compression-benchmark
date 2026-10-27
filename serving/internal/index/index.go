package index

import (
	"context"
	"fmt"
	"math"
	"sync"

	"github.com/xiaorui7/rae-feature-compression-benchmark/serving/internal/format"
	"github.com/xiaorui7/rae-feature-compression-benchmark/serving/internal/topk"
)

type Index struct {
	vectors     *format.Vectors
	inverseNorm []float64
	workers     int
	admission   chan struct{}
}

func New(v *format.Vectors, workers int, concurrent ...int) (*Index, error) {
	maxSearches := 1
	if len(concurrent) > 0 {
		maxSearches = concurrent[0]
	}
	if workers < 1 || maxSearches < 1 || maxSearches > 16 || len(v.IDs) == 0 || v.Dim < 1 {
		return nil, fmt.Errorf("invalid index configuration")
	}
	if len(v.Full)+len(v.Half) != len(v.IDs)*v.Dim || (len(v.Full) > 0 && len(v.Half) > 0) {
		return nil, fmt.Errorf("invalid vector payload")
	}
	if workers > len(v.IDs) {
		workers = len(v.IDs)
	}
	index := &Index{vectors: v, workers: workers, inverseNorm: make([]float64, len(v.IDs)), admission: make(chan struct{}, maxSearches)}
	for row := range v.IDs {
		var norm float64
		for col := 0; col < v.Dim; col++ {
			value := v.Value(row*v.Dim + col)
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, fmt.Errorf("non-finite index value")
			}
			norm += value * value
		}
		if norm > 0 {
			index.inverseNorm[row] = 1 / math.Sqrt(norm)
		}
	}
	return index, nil
}
func (i *Index) Dim() int   { return i.vectors.Dim }
func (i *Index) Count() int { return len(i.vectors.IDs) }
func (i *Index) NumericBytes() int {
	return 4*len(i.vectors.Full) + 2*len(i.vectors.Half) + 8*len(i.inverseNorm)
}
func (i *Index) Validate(q []float32, k int) error {
	if len(q) != i.Dim() || k < 1 || k > 100 {
		return fmt.Errorf("invalid dimension or k")
	}
	for _, v := range q {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("non-finite query")
		}
	}
	return nil
}
func (i *Index) Search(ctx context.Context, id string, q []float32, k int, excludeIDs ...string) ([]topk.Hit, error) {
	if err := i.Validate(q, k); err != nil {
		return nil, err
	}
	excluded := make(map[string]bool, len(excludeIDs))
	for _, excludedID := range excludeIDs {
		excluded[excludedID] = true
	}
	// Admission bounds total scoring goroutines to searches times workers.
	select {
	case i.admission <- struct{}{}:
		defer func() { <-i.admission }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	var norm float64
	for _, v := range q {
		norm += float64(v) * float64(v)
	}
	inv := 0.0
	if norm > 0 {
		inv = 1 / math.Sqrt(norm)
	}
	results := make([][]topk.Hit, i.workers)
	var wg sync.WaitGroup
	for worker := 0; worker < i.workers; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			best := topk.New(k)
			start, end := i.Count()*worker/i.workers, i.Count()*(worker+1)/i.workers
			for row := start; row < end; row++ {
				if row%64 == 0 && ctx.Err() != nil {
					return
				}
				if excluded[i.vectors.IDs[row]] {
					continue
				}
				var dot float64
				for col, v := range q {
					dot += float64(v) * i.vectors.Value(row*i.Dim()+col)
				}
				best.Add(topk.Hit{ID: i.vectors.IDs[row], Score: dot * inv * i.inverseNorm[row]})
			}
			results[worker] = best.Sorted()
		}(worker)
	}
	wg.Wait()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	best := topk.New(k)
	for _, hits := range results {
		for _, hit := range hits {
			best.Add(hit)
		}
	}
	return best.Sorted(), nil
}
