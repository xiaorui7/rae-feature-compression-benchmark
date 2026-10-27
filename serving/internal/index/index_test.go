package index

import (
	"context"
	"github.com/xiaorui7/rae-feature-compression-benchmark/serving/internal/format"
	"math"
	"reflect"
	"sync"
	"testing"
)

func fixture(t *testing.T, workers int) *Index {
	t.Helper()
	i, err := New(&format.Vectors{Dim: 2, IDs: []string{"b", "a", "c", "z"}, Full: []float32{1, 0, 1, 0, 0, 1, 0, 0}}, workers)
	if err != nil {
		t.Fatal(err)
	}
	return i
}
func TestScoringAndWorkers(t *testing.T) {
	one := fixture(t, 1)
	many := fixture(t, 3)
	want, err := one.Search(context.Background(), "q", []float32{1, 0}, 3)
	if err != nil {
		t.Fatal(err)
	}
	got, err := many.Search(context.Background(), "q", []float32{1, 0}, 3)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) || got[0].ID != "a" || got[1].ID != "b" || got[0].Score != 1 || got[2].Score != 0 {
		t.Fatalf("%v %v", got, want)
	}
	got, err = many.Search(context.Background(), "a", []float32{0, 0}, 10, "a")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || got[0].ID != "b" {
		t.Fatal(got)
	}
}
func TestValidationAndCancellation(t *testing.T) {
	i := fixture(t, 2)
	for _, q := range [][]float32{{1}, {1, float32(math.NaN())}, {1, float32(math.Inf(1))}} {
		if _, err := i.Search(context.Background(), "q", q, 1); err == nil {
			t.Fatal("accepted invalid query")
		}
	}
	for _, k := range []int{0, 101} {
		if _, err := i.Search(context.Background(), "q", []float32{1, 0}, k); err == nil {
			t.Fatal("accepted k")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := i.Search(ctx, "q", []float32{1, 0}, 1); err == nil {
		t.Fatal("ignored cancellation")
	}
}
func TestConcurrentReaders(t *testing.T) {
	i := fixture(t, 3)
	var wg sync.WaitGroup
	for n := 0; n < 20; n++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			hits, err := i.Search(context.Background(), "q", []float32{1, 0}, 2)
			if err != nil || len(hits) != 2 {
				t.Errorf("search: %v", err)
			}
		}()
	}
	wg.Wait()
}
