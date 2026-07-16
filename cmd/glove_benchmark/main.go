package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"vector-engine/index"
)

type bfResult struct {
	ID    string
	Score float32
}

func cosineDistance(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 1.0
	}
	var dotProduct, normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 1.0
	}
	sim := dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
	return 1.0 - sim
}

func bruteForceTopK(query []float32, data map[string][]float32, k int) []string {
	var res []bfResult
	for id, vec := range data {
		dist := cosineDistance(query, vec)
		res = append(res, bfResult{ID: id, Score: 1.0 - dist})
	}
	sort.Slice(res, func(i, j int) bool {
		return res[i].Score > res[j].Score // Higher score is better
	})
	var out []string
	for i := 0; i < k && i < len(res); i++ {
		out = append(out, res[i].ID)
	}
	return out
}

func loadGloVe(filepath string, N int, Q int, dim int) (map[string][]float32, [][]float32, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	dataset := make(map[string][]float32)
	var queries [][]float32

	scanner := bufio.NewScanner(file)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	count := 0
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, " ")
		if len(parts) != dim+1 {
			continue // Skip malformed lines
		}
		
		word := parts[0]
		vec := make([]float32, dim)
		for i := 1; i <= dim; i++ {
			val, _ := strconv.ParseFloat(parts[i], 32)
			vec[i-1] = float32(val)
		}

		if count < N {
			dataset[word] = vec
		} else if count < N+Q {
			queries = append(queries, vec)
		} else {
			break
		}
		count++
	}
	
	if err := scanner.Err(); err != nil {
		return nil, nil, err
	}
	
	return dataset, queries, nil
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	
	N := 20000
	Q := 500
	k := 10
	dim := 50 // Use 50d glove
	
	fmt.Printf("Loading %d GloVe 50d vectors + %d queries...\n", N, Q)
	dataset, queries, err := loadGloVe("glove/glove.6B.50d.txt", N, Q, dim)
	if err != nil {
		fmt.Printf("Error loading GloVe: %v\n", err)
		return
	}
	fmt.Printf("Successfully loaded %d vectors and %d queries.\n", len(dataset), len(queries))

	M := 16
	efConstruction := 100
	
	config := index.HNSWConfig{
		M:              M,
		MaxM0:          2 * M,
		EfConstruction: efConstruction,
		EfSearch:       50,
		ML:             1.0 / math.Log(float64(M)),
	}

	hIndex := index.NewHNSWIndex(config)

	fmt.Print("Building HNSW Index... ")
	buildStart := time.Now()
	for id, vec := range dataset {
		hIndex.Insert(index.NodeID(id), vec)
	}
	buildTime := time.Since(buildStart)
	fmt.Printf("Done in %v.\n", buildTime)

	fmt.Print("Running Brute-Force baseline... ")
	groundTruths := make([][]string, Q)
	bfStart := time.Now()
	for i := 0; i < Q; i++ {
		groundTruths[i] = bruteForceTopK(queries[i], dataset, k)
	}
	fmt.Printf("Done in %v.\n", time.Since(bfStart))

	efValues := []int{50, 100, 200, 400}
	
	fmt.Printf("\n### RESULTS: Dataset Size N = %d (GloVe 50d)\n", N)
	fmt.Printf("| efSearch | Recall@10 | Avg Latency (µs) | QPS       |\n")
	fmt.Printf("|----------|-----------|------------------|-----------|\n")

	for _, ef := range efValues {
		hIndex.Config.EfSearch = ef
		
		hStart := time.Now()
		var totalMatches int
		
		for i := 0; i < Q; i++ {
			matches, _ := hIndex.Query(queries[i], k)
			
			// Compute recall
			matchCount := 0
			gtMap := make(map[string]bool)
			for _, gt := range groundTruths[i] {
				gtMap[gt] = true
			}
			for _, m := range matches {
				if gtMap[m.ID] {
					matchCount++
				}
			}
			totalMatches += matchCount
		}
		
		hDuration := time.Since(hStart)
		avgLatencyUs := float64(hDuration.Microseconds()) / float64(Q)
		qps := float64(Q) / hDuration.Seconds()
		recall := float64(totalMatches) / float64(Q*k) * 100.0

		fmt.Printf("| %-8d | %-9.1f | %-16.1f | %-9.0f |\n", ef, recall, avgLatencyUs, qps)
	}
}
