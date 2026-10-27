package main

import (
	"os"
	"time"
)

func main() {
	if len(os.Args) != 2 {
		os.Exit(2)
	}
	d, err := time.ParseDuration(os.Args[1])
	if err != nil || d < 0 || d > 30*time.Second {
		os.Exit(2)
	}
	time.Sleep(d)
}
