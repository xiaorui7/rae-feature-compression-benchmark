package topk

import "testing"

func TestOrdering(t *testing.T) {
	h := New(3)
	for _, v := range []Hit{{"z", 0}, {"b", 1}, {"a", 1}, {"c", 2}, {"x", -1}} {
		h.Add(v)
	}
	hits := h.Sorted()
	for i, want := range []string{"c", "a", "b"} {
		if hits[i].ID != want {
			t.Fatalf("got %v", hits)
		}
	}
}
