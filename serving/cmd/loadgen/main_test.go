package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"github.com/xiaorui7/rae-feature-compression-benchmark/serving/internal/format"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestOpenLoopUsesFullWindow(t *testing.T) {
	folder := t.TempDir()
	write := func(name string, b []byte) string {
		t.Helper()
		if err := os.WriteFile(filepath.Join(folder, name), b, 0600); err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(b)
		return hex.EncodeToString(sum[:])
	}
	payload := make([]byte, 32)
	copy(payload, []byte("RAEVEC01"))
	binary.LittleEndian.PutUint32(payload[8:], 1)
	binary.LittleEndian.PutUint32(payload[12:], 1)
	binary.LittleEndian.PutUint32(payload[16:], 1)
	binary.LittleEndian.PutUint64(payload[20:], 1)
	binary.LittleEndian.PutUint32(payload[28:], 0x3f800000)
	record := format.Record{Count: 1, VecFile: "queries.vec", IDsFile: "queries.ids.json", SHA256Vec: write("queries.vec", payload), SHA256IDs: write("queries.ids.json", []byte(`["q"]`))}
	m := format.Manifest{SchemaVersion: 1, Status: "synthetic", Dtype: "float32", Dim: 1, TotalCount: 1, ShardCount: 1, Shards: []format.Record{{Index: 0, Count: 1}}, Queries: map[string]format.Record{"test": record}}
	raw, _ := json.Marshal(m)
	write("manifest.json", raw)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"results":[],"partial":false,"missing_shards":[]}`))
	}))
	defer server.Close()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	child := exec.Command(executable, "-test.run=TestLoadgenWindowProcess")
	child.Env = append(os.Environ(), "LOADGEN_HELPER=1", "LOADGEN_EXPORT="+folder, "LOADGEN_URL="+server.URL)
	data, err := child.Output()
	if err != nil {
		t.Fatal(err)
	}
	var result report
	if err = json.Unmarshal(data, &result); err != nil {
		t.Fatalf("%v %s", err, data)
	}
	if result.Count != 5 || result.Duration < .25 || result.Achieved > 20 || len(result.Errors) != 0 {
		t.Fatalf("invalid observation window: %+v", result)
	}
}
func TestLoadgenWindowProcess(t *testing.T) {
	if os.Getenv("LOADGEN_HELPER") != "1" {
		return
	}
	flag.CommandLine = flag.NewFlagSet("loadgen", flag.ExitOnError)
	os.Args = []string{"loadgen", "--export", os.Getenv("LOADGEN_EXPORT"), "--url", os.Getenv("LOADGEN_URL"), "--duration", "250ms", "--warmup", "0s", "--qps", "20", "--concurrency", "1"}
	if err := run(); err != nil {
		os.Exit(1)
	}
	os.Exit(0)
}
