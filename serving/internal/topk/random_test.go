package topk

import (
	"fmt"
	"math/rand"
	"reflect"
	"sort"
	"testing"
)

func TestRandomAgainstSort(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for trial := 0; trial < 100; trial++ {
		all := make([]Hit, 137)
		for i := range all {
			all[i] = Hit{fmt.Sprintf("id%04d", i), float64(rng.Intn(7) - 3)}
		}
		for _, k := range []int{1, 3, 10, 100, 200} {
			h := New(k)
			for _, i := range rng.Perm(len(all)) {
				h.Add(all[i])
			}
			sort.Slice(all, func(i, j int) bool { return Better(all[i], all[j]) })
			if !reflect.DeepEqual(h.Sorted(), all[:min(k, len(all))]) {
				t.Fatal(trial, k)
			}
		}
	}
}
