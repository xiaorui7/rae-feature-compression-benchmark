package format

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestHalfDecode(t *testing.T) {
	cases := []struct {
		bits uint16
		want float32
	}{{0, 0}, {0x8000, math.Float32frombits(0x80000000)}, {1, float32(math.Ldexp(1, -24))}, {0x03ff, float32(1023 * math.Ldexp(1, -24))}, {0x0400, float32(math.Ldexp(1, -14))}, {0x3c00, 1}, {0xc000, -2}, {0x7bff, 65504}}
	for _, c := range cases {
		if got := HalfToFloat(c.bits); math.Float32bits(got) != math.Float32bits(c.want) {
			t.Fatalf("%x: %g != %g", c.bits, got, c.want)
		}
	}
	if !math.IsInf(float64(HalfToFloat(0x7c00)), 1) || !math.IsNaN(float64(HalfToFloat(0x7e00))) {
		t.Fatal("special values")
	}
}

func TestLoadAndCorruption(t *testing.T) {
	for _, bits := range []uint16{0x3c00, 0x7c00, 0x7e00} {
		dir := t.TempDir()
		payload := make([]byte, 30)
		copy(payload, "RAEVEC01")
		binary.LittleEndian.PutUint32(payload[8:], 1)
		binary.LittleEndian.PutUint32(payload[12:], 2)
		binary.LittleEndian.PutUint32(payload[16:], 1)
		binary.LittleEndian.PutUint64(payload[20:], 1)
		binary.LittleEndian.PutUint16(payload[28:], bits)
		ids := []byte(`["a"]`)
		write := func(name string, b []byte) string {
			t.Helper()
			if err := os.WriteFile(filepath.Join(dir, name), b, 0600); err != nil {
				t.Fatal(err)
			}
			h := sha256.Sum256(b)
			return hex.EncodeToString(h[:])
		}
		r := Record{Count: 1, VecFile: "s.vec", IDsFile: "s.ids.json", SHA256Vec: write("s.vec", payload), SHA256IDs: write("s.ids.json", ids)}
		m := &Manifest{Dim: 1, Dtype: "float16"}
		v, err := Load(dir, m, r)
		if bits != 0x3c00 {
			if err == nil {
				t.Fatal("accepted nonfinite")
			}
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		if v.Full != nil || len(v.Half) != 1 || v.Value(0) != 1 {
			t.Fatal("half storage lost")
		}
		payload[28] = 7
		write("s.vec", payload)
		if _, err = Load(dir, m, r); err == nil {
			t.Fatal("accepted corrupt checksum")
		}
		r.SHA256Vec = write("s.vec", payload[:20])
		if _, err = Load(dir, m, r); err == nil {
			t.Fatal("accepted truncated header")
		}
	}
}
