package format

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
)

type Record struct {
	Index        int    `json:"index"`
	VecFile      string `json:"vec_file"`
	IDsFile      string `json:"ids_file"`
	LabelsFile   string `json:"labels_file"`
	OracleFile   string `json:"oracle_file"`
	Count        uint64 `json:"count"`
	SHA256Vec    string `json:"sha256_vec"`
	SHA256IDs    string `json:"sha256_ids"`
	SHA256Labels string `json:"sha256_labels"`
	SHA256Oracle string `json:"sha256_oracle"`
}
type Manifest struct {
	SchemaVersion int               `json:"schema_version"`
	ExportID      string            `json:"export_id"`
	Status        string            `json:"status"`
	Dtype         string            `json:"dtype"`
	Dim           uint32            `json:"dim"`
	TotalCount    uint64            `json:"total_count"`
	ShardCount    int               `json:"shard_count"`
	Shards        []Record          `json:"shards"`
	Queries       map[string]Record `json:"queries"`
	Source        map[string]any    `json:"source"`
}
type Vectors struct {
	IDs  []string
	Dim  int
	Full []float32
	Half []uint16
}

func ReadManifest(dir string) (*Manifest, error) {
	data, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return nil, err
	}
	var m Manifest
	if err = json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	if m.SchemaVersion != 1 || m.Dim == 0 || m.TotalCount == 0 || m.ShardCount < 1 || len(m.Shards) != m.ShardCount || (m.Dtype != "float16" && m.Dtype != "float32") {
		return nil, fmt.Errorf("invalid manifest metadata")
	}
	if m.Status != "actual_measurement" && m.Status != "synthetic" {
		return nil, fmt.Errorf("invalid data status")
	}
	var count uint64
	for i, s := range m.Shards {
		if s.Index != i || s.Count == 0 {
			return nil, fmt.Errorf("invalid shard metadata")
		}
		count += s.Count
	}
	if count != m.TotalCount {
		return nil, fmt.Errorf("shard counts do not match total")
	}
	return &m, nil
}

func checkedPath(dir, name string) (string, error) {
	if name == "" || name == "." || name == ".." || strings.ContainsAny(name, "/\\:") {
		return "", fmt.Errorf("invalid export filename")
	}
	return filepath.Join(dir, name), nil
}

func ReadChecked(dir, name, expected string) ([]byte, error) {
	path, err := checkedPath(dir, name)
	if err != nil {
		return nil, err
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	sum := sha256.Sum256(b)
	if hex.EncodeToString(sum[:]) != expected {
		return nil, fmt.Errorf("checksum mismatch: %s", name)
	}
	return b, nil
}

func Load(dir string, m *Manifest, r Record) (*Vectors, error) {
	path, err := checkedPath(dir, r.VecFile)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	hash := sha256.New()
	if _, err = io.Copy(hash, f); err != nil {
		return nil, err
	}
	if hex.EncodeToString(hash.Sum(nil)) != r.SHA256Vec {
		return nil, fmt.Errorf("checksum mismatch: %s", r.VecFile)
	}
	if _, err = f.Seek(0, 0); err != nil {
		return nil, err
	}
	header := make([]byte, 28)
	if _, err = io.ReadFull(f, header); err != nil {
		return nil, err
	}
	version, dtype, dim, count := binary.LittleEndian.Uint32(header[8:]), binary.LittleEndian.Uint32(header[12:]), binary.LittleEndian.Uint32(header[16:]), binary.LittleEndian.Uint64(header[20:])
	if string(header[:8]) != "RAEVEC01" || version != 1 || dim != m.Dim || count != r.Count || count == 0 || (dtype != 1 && dtype != 2) {
		return nil, fmt.Errorf("invalid vector header")
	}
	if (dtype == 1) != (m.Dtype == "float32") {
		return nil, fmt.Errorf("dtype mismatch")
	}
	bytesPer := uint64(4)
	if dtype == 2 {
		bytesPer = 2
	}
	// Check lengths before allocating so corrupt headers cannot trigger huge allocations.
	maxInt := uint64(^uint(0) >> 1)
	if count > (maxInt-28)/uint64(dim)/bytesPer {
		return nil, fmt.Errorf("vector size overflow")
	}
	size, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if uint64(size.Size()) != 28+count*uint64(dim)*bytesPer {
		return nil, fmt.Errorf("vector length mismatch")
	}
	b, err := ReadChecked(dir, r.IDsFile, r.SHA256IDs)
	if err != nil {
		return nil, err
	}
	v := &Vectors{Dim: int(dim)}
	if err = json.Unmarshal(b, &v.IDs); err != nil {
		return nil, err
	}
	if uint64(len(v.IDs)) != count {
		return nil, fmt.Errorf("ID count mismatch")
	}
	seen := make(map[string]bool, len(v.IDs))
	for _, id := range v.IDs {
		if id == "" || seen[id] {
			return nil, fmt.Errorf("invalid or duplicate ID")
		}
		seen[id] = true
	}
	n := int(count * uint64(dim))
	if dtype == 1 {
		v.Full = make([]float32, n)
		err = binary.Read(f, binary.LittleEndian, v.Full)
	} else {
		v.Half = make([]uint16, n)
		err = binary.Read(f, binary.LittleEndian, v.Half)
	}
	if err != nil {
		return nil, err
	}
	for i := 0; i < n; i++ {
		x := v.Value(i)
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return nil, fmt.Errorf("non-finite index value")
		}
	}
	return v, nil
}

func HalfToFloat(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exponent := uint32(h>>10) & 31
	mantissa := uint32(h & 1023)
	if exponent == 0 {
		if mantissa == 0 {
			return math.Float32frombits(sign)
		}
		e := int32(-14)
		for mantissa&1024 == 0 {
			mantissa <<= 1
			e--
		}
		return math.Float32frombits(sign | uint32(e+127)<<23 | (mantissa&1023)<<13)
	}
	if exponent == 31 {
		return math.Float32frombits(sign | 0x7f800000 | mantissa<<13)
	}
	return math.Float32frombits(sign | (exponent+112)<<23 | mantissa<<13)
}
func (v *Vectors) Value(i int) float64 {
	if v.Half != nil {
		return float64(HalfToFloat(v.Half[i]))
	}
	return float64(v.Full[i])
}
