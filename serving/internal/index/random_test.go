package index

import (
	"context"
	"fmt"
	"github.com/xiaorui7/rae-feature-compression-benchmark/serving/internal/format"
	"github.com/xiaorui7/rae-feature-compression-benchmark/serving/internal/topk"
	"math"
	"math/rand"
	"sort"
	"testing"
)

func TestUnevenWorkersAgainstBruteForce(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	for trial := 0; trial < 20; trial++ {
		v := &format.Vectors{Dim: 7, IDs: make([]string, 103), Full: make([]float32, 103*7)}
		for i := range v.IDs {
			v.IDs[i] = fmt.Sprintf("g%04d", i)
		}
		for i := range v.Full {
			v.Full[i] = float32(rng.Intn(5) - 2)
		}
		q := []float32{1, 2, 1, 2, 1, 2, 1}
		var qn float64
		for _, x := range q {
			qn += float64(x * x)
		}
		want := make([]topk.Hit, len(v.IDs))
		for row, id := range v.IDs {
			var dot, norm float64
			for col, x := range q {
				g := v.Value(row*7 + col)
				dot += float64(x) * g
				norm += g * g
			}
			score := 0.0
			if norm > 0 {
				score = dot / math.Sqrt(qn) / math.Sqrt(norm)
			}
			want[row] = topk.Hit{ID: id, Score: score}
		}
		sort.Slice(want, func(i, j int) bool { return topk.Better(want[i], want[j]) })
		for _, workers := range []int{1, 2, 4, 8} {
			idx, err := New(v, workers, 4)
			if err != nil {
				t.Fatal(err)
			}
			got, err := idx.Search(context.Background(), "g0000", q, 10)
			if err != nil {
				t.Fatal(err)
			}
			for j := range got {
				if got[j].ID != want[j].ID || math.Abs(got[j].Score-want[j].Score) > 1e-12 {
					t.Fatal(trial, workers, j, got[j], want[j])
				}
			}
		}
	}
}
