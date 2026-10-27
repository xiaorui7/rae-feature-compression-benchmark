package topk

import (
	"fmt"
	"math/rand"
	"sort"
	"testing"
)

func TestMergeAgainstFullSort(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	for trial := 0; trial < 50; trial++ {
		for _, n := range []int{1, 2, 4} {
			lists := make([][]Hit, n)
			all := []Hit{}
			for i := 0; i < 137; i++ {
				h := Hit{ID: fmt.Sprintf("%04d", i), Score: float64(r.Intn(5))}
				lists[i%n] = append(lists[i%n], h)
				all = append(all, h)
			}
			sort.Slice(all, func(i, j int) bool { return Better(all[i], all[j]) })
			for _, list := range lists {
				sort.Slice(list, func(i, j int) bool { return Better(list[i], list[j]) })
			}
			for _, k := range []int{1, 10, 100, 200} {
				got := Merge(lists, k)
				if len(got) != min(k, len(all)) {
					t.Fatal("length")
				}
				for i, h := range got {
					if h != all[i] {
						t.Fatalf("trial %d shards %d k %d", trial, n, k)
					}
				}
			}
		}
	}
}
