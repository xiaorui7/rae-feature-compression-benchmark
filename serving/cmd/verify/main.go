package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"net/http"
	"os"
	"runtime"
	"time"

	"github.com/xiaorui7/rae-feature-compression-benchmark/serving/internal/fanout"
	"github.com/xiaorui7/rae-feature-compression-benchmark/serving/internal/format"
	"github.com/xiaorui7/rae-feature-compression-benchmark/serving/internal/topk"
	pb "github.com/xiaorui7/rae-feature-compression-benchmark/serving/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	dir := flag.String("export", "", "export directory")
	address := flag.String("address", "127.0.0.1:9090", "shard gRPC address")
	split := flag.String("split", "test", "query split")
	results := flag.String("benchmark-results", "", "optional original results.json")
	timeout := flag.Duration("timeout", 5*time.Second, "deadline per RPC")
	httpURL := flag.String("http", "", "coordinator base URL instead of direct gRPC")
	flag.Parse()
	m, err := format.ReadManifest(*dir)
	if err != nil {
		return err
	}
	if m.ShardCount != 1 && *httpURL == "" {
		return fmt.Errorf("Phase 2 verifier requires a single-shard export")
	}
	record, ok := m.Queries[*split]
	if !ok {
		return fmt.Errorf("query split not found")
	}
	queries, err := format.Load(*dir, m, record)
	if err != nil {
		return err
	}
	data, err := format.ReadChecked(*dir, record.OracleFile, record.SHA256Oracle)
	if err != nil {
		return err
	}
	var expected struct {
		K       int `json:"k"`
		Results []struct {
			ID   string     `json:"id"`
			Hits []topk.Hit `json:"hits"`
		} `json:"results"`
	}
	if err = json.Unmarshal(data, &expected); err != nil {
		return err
	}
	if len(expected.Results) != len(queries.IDs) {
		return fmt.Errorf("oracle query count mismatch")
	}
	var queryLabels []int
	labels := map[string]int{}
	if record.LabelsFile != "" {
		data, err = format.ReadChecked(*dir, record.LabelsFile, record.SHA256Labels)
		if err != nil {
			return err
		}
		if err = json.Unmarshal(data, &queryLabels); err != nil {
			return err
		}
		if len(queryLabels) != len(queries.IDs) {
			return fmt.Errorf("query label count mismatch")
		}
		for _, shardRecord := range m.Shards {
			gallery, err := format.Load(*dir, m, shardRecord)
			if err != nil {
				return err
			}
			data, err = format.ReadChecked(*dir, shardRecord.LabelsFile, shardRecord.SHA256Labels)
			if err != nil {
				return err
			}
			var values []int
			if err = json.Unmarshal(data, &values); err != nil {
				return err
			}
			if len(values) != len(gallery.IDs) {
				return fmt.Errorf("gallery label count mismatch")
			}
			for i, id := range gallery.IDs {
				labels[id] = values[i]
			}
		}
	}
	conn, err := grpc.NewClient(*address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return err
	}
	defer conn.Close()
	client := pb.NewSearchServiceClient(conn)
	mismatches := 0
	maxDifference := 0.0
	precision := 0.0
	for start := 0; start < len(queries.IDs); start += 16 {
		end := min(start+16, len(queries.IDs))
		request := &pb.SearchRequest{K: uint32(expected.K)}
		for row := start; row < end; row++ {
			q := &pb.Query{Id: queries.IDs[row], Values: make([]float32, queries.Dim)}
			for col := range q.Values {
				q.Values[col] = float32(queries.Value(row*queries.Dim + col))
			}
			request.Queries = append(request.Queries, q)
		}
		ctx, cancel := context.WithTimeout(context.Background(), *timeout)
		var response *pb.SearchResponse
		if *httpURL == "" {
			response, err = client.Search(ctx, request)
		} else {
			body := fanout.Request{K: int(request.K)}
			for _, q := range request.Queries {
				body.Queries = append(body.Queries, fanout.Query{ID: q.Id, Vector: q.Values})
			}
			raw, _ := json.Marshal(body)
			httpReq, makeErr := http.NewRequestWithContext(ctx, "POST", *httpURL+"/v1/search", bytes.NewReader(raw))
			if makeErr != nil {
				cancel()
				return makeErr
			}
			result, callErr := http.DefaultClient.Do(httpReq)
			if callErr != nil {
				cancel()
				return callErr
			}
			var decoded fanout.Response
			err = json.NewDecoder(result.Body).Decode(&decoded)
			result.Body.Close()
			if result.StatusCode != 200 || decoded.Partial {
				cancel()
				return fmt.Errorf("coordinator status %d or partial", result.StatusCode)
			}
			response = &pb.SearchResponse{}
			for _, r := range decoded.Results {
				q := &pb.QueryResult{Id: r.ID}
				for _, hit := range r.Hits {
					q.Hits = append(q.Hits, &pb.Hit{Id: hit.ID, Score: hit.Score})
				}
				response.Results = append(response.Results, q)
			}
		}
		cancel()
		if err != nil {
			return err
		}
		if len(response.Results) != end-start {
			return fmt.Errorf("response query count mismatch")
		}
		for i, result := range response.Results {
			row := start + i
			want := expected.Results[row]
			if result.Id != queries.IDs[row] || want.ID != queries.IDs[row] || len(result.Hits) != len(want.Hits) {
				return fmt.Errorf("invalid response or oracle structure")
			}
			correct := 0
			for j, hit := range result.Hits {
				if hit.Id != want.Hits[j].ID {
					mismatches++
				}
				if math.IsNaN(hit.Score) || math.IsInf(hit.Score, 0) {
					return fmt.Errorf("non-finite returned score")
				}
				maxDifference = math.Max(maxDifference, math.Abs(hit.Score-want.Hits[j].Score))
				if queryLabels != nil {
					label, ok := labels[hit.Id]
					if !ok {
						return fmt.Errorf("unknown returned ID")
					}
					if label == queryLabels[row] {
						correct++
					}
				}
			}
			if queryLabels != nil {
				precision += float64(correct) / float64(len(result.Hits))
			}
		}
	}
	report := map[string]any{"export_id": m.ExportID, "data_status": m.Status, "method": m.Source["method_id"], "split": *split, "query_count": len(queries.IDs), "id_mismatches": mismatches, "max_abs_score_difference": maxDifference, "score_tolerance": 1e-12, "go_version": runtime.Version(), "os": runtime.GOOS, "arch": runtime.GOARCH, "status": "passed"}
	if queryLabels != nil {
		report["service_precision_at_10"] = precision / float64(len(queries.IDs))
	}
	if *results != "" {
		raw, err := os.ReadFile(*results)
		if err != nil {
			return err
		}
		var original struct {
			Rows []map[string]any `json:"rows"`
		}
		if err = json.Unmarshal(raw, &original); err != nil {
			return err
		}
		found := false
		for _, row := range original.Rows {
			if row["method_id"] == m.Source["method_id"] {
				v, ok := row[*split+"_precision_at_10"].(float64)
				if !ok {
					return fmt.Errorf("original precision missing")
				}
				report["python_float32_precision_at_10"] = v
				report["precision_delta"] = precision/float64(len(queries.IDs)) - v
				found = true
			}
		}
		if !found {
			return fmt.Errorf("original method not found")
		}
	}
	if mismatches != 0 || maxDifference > 1e-12 {
		report["status"] = "failed"
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err = enc.Encode(report); err != nil {
		return err
	}
	if report["status"] == "failed" {
		return fmt.Errorf("oracle verification failed")
	}
	return nil
}
