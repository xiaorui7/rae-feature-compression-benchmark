package topk

import (
	"container/heap"
	"sort"
)

type Hit struct {
	ID    string  `json:"id"`
	Score float64 `json:"score"`
}

func Better(a, b Hit) bool {
	return a.Score > b.Score || (a.Score == b.Score && a.ID < b.ID)
}

type worstFirst []Hit

func (h worstFirst) Len() int           { return len(h) }
func (h worstFirst) Less(i, j int) bool { return Better(h[j], h[i]) }
func (h worstFirst) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *worstFirst) Push(x any)        { *h = append(*h, x.(Hit)) }
func (h *worstFirst) Pop() any          { a := *h; v := a[len(a)-1]; *h = a[:len(a)-1]; return v }

type Heap struct {
	k    int
	hits worstFirst
}

func New(k int) *Heap { return &Heap{k: k} }
func (h *Heap) Add(hit Hit) {
	if h.k <= 0 {
		return
	}
	if len(h.hits) < h.k {
		heap.Push(&h.hits, hit)
		return
	}
	if Better(hit, h.hits[0]) {
		h.hits[0] = hit
		heap.Fix(&h.hits, 0)
	}
}
func (h *Heap) Sorted() []Hit {
	hits := append([]Hit{}, h.hits...)
	sort.Slice(hits, func(i, j int) bool { return Better(hits[i], hits[j]) })
	return hits
}
