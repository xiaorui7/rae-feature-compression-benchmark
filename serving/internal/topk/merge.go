package topk

import "container/heap"

type cursor struct {
	hit          Hit
	list, offset int
}
type bestFirst []cursor

func (h bestFirst) Len() int           { return len(h) }
func (h bestFirst) Less(i, j int) bool { return Better(h[i].hit, h[j].hit) }
func (h bestFirst) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *bestFirst) Push(x any)        { *h = append(*h, x.(cursor)) }
func (h *bestFirst) Pop() any          { old := *h; x := old[len(old)-1]; *h = old[:len(old)-1]; return x }

// Merge consumes sorted, disjoint shard candidates under the shared total order.
func Merge(lists [][]Hit, k int) []Hit {
	pending := &bestFirst{}
	for i, list := range lists {
		if len(list) > 0 {
			heap.Push(pending, cursor{hit: list[0], list: i})
		}
	}
	result := make([]Hit, 0, k)
	for pending.Len() > 0 && len(result) < k {
		c := heap.Pop(pending).(cursor)
		result = append(result, c.hit)
		c.offset++
		if c.offset < len(lists[c.list]) {
			c.hit = lists[c.list][c.offset]
			heap.Push(pending, c)
		}
	}
	return result
}
