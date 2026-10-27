package main

import (
	"net/http"
	"os"
	"time"
)

func main() {
	if len(os.Args) != 2 {
		os.Exit(2)
	}
	c := http.Client{Timeout: time.Second}
	r, err := c.Get(os.Args[1])
	if err != nil {
		os.Exit(1)
	}
	defer r.Body.Close()
	if r.StatusCode != 200 {
		os.Exit(1)
	}
}
