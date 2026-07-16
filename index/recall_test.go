package index

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"testing"
	"time"
)

func genRandomVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := 0; i < dim; i++ {
		vec[i] = rand.Float32()
	}
	return vec
}

type bfResult struct {
	id   string
	dist float32
}

// bruteForceTopK calculates exact ground truth nearest neighbors using O(N) linear scan
func bruteForceTopK(query []float32, data map[string][]float32, k int) []string {
	var res []bfResult
	for id, vec := range data {
		dist := cosineDistance(query, vec) // distance = 1 - similarity
		res = append(res, bfResult{id: id, dist: dist})
	}
	
	sort.Slice(res, func(i, j int) bool {
		return res[i].dist < res[j].dist // lowest distance = highest similarity
	})
	
	if len(res) > k {
		res = res[:k]
	}
	
	out := make([]string, len(res))
	for i, r := range res {
		out[i] = r.id
	}
	return out
}

func TestHNSW_RecallAtK(t *testing.T) {
	rand.Seed(time.Now().UnixNano())

	N := 2000
	Q := 100
	k := 10
	dim := 128

	dataset := make(map[string][]float32)
	
	M := 16
	config := HNSWConfig{
		M:              M,
		MaxM0:          2 * M,
		EfConstruction: 100,
		EfSearch:       50,
		ML:             1.0 / math.Log(float64(M)),
	}
	hIndex := NewHNSWIndex(config)

	// Populate Both Indexes
	for i := 0; i < N; i++ {
		id := fmt.Sprintf("doc_%d", i)
		vec := genRandomVector(dim)
		dataset[id] = vec
		err := hIndex.Insert(NodeID(id), vec)
		if err != nil {
			t.Fatalf("Insert failed: %v", err)
		}
	}

	// Generate Queries
	queries := make([][]float32, Q)
	for i := 0; i < Q; i++ {
		queries[i] = genRandomVector(dim)
	}

	var totalRecall float64
	var totalBFTime time.Duration
	var totalHNSWTime time.Duration

	// Execute Comparison & Calculate Recall@k
	for i := 0; i < Q; i++ {
		qVec := queries[i]

		// 1. Brute force ground truth
		bfStart := time.Now()
		bfResults := bruteForceTopK(qVec, dataset, k)
		totalBFTime += time.Since(bfStart)

		// 2. HNSW ANN search
		hnswStart := time.Now()
		hnswMatches, _ := hIndex.Query(qVec, k)
		totalHNSWTime += time.Since(hnswStart)

		// 3. Intersection
		groundTruth := make(map[string]bool)
		for _, id := range bfResults {
			groundTruth[id] = true
		}

		intersection := 0
		for _, match := range hnswMatches {
			if groundTruth[match.ID] {
				intersection++
			}
		}

		// 4. Compute Recall
		recall := float64(intersection) / float64(k)
		totalRecall += recall
	}

	avgRecall := totalRecall / float64(Q)
	avgBF := float64(totalBFTime.Microseconds()) / 1000.0 / float64(Q)
	avgHNSW := float64(totalHNSWTime.Microseconds()) / 1000.0 / float64(Q)
	
	var speedup float64
	if avgHNSW > 0 {
		speedup = avgBF / avgHNSW
	}

	// Console Report
	fmt.Println("\n=======================================================")
	fmt.Printf("               HNSW RECALL@%d TEST REPORT               \n", k)
	fmt.Println("=======================================================")
	fmt.Printf("Dataset Size (N)   : %d vectors\n", N)
	fmt.Printf("Query Count (Q)    : %d\n", Q)
	fmt.Printf("HNSW Config        : M=%d, EfConst=%d, EfSearch=%d\n", config.M, config.EfConstruction, config.EfSearch)
	fmt.Println("-------------------------------------------------------")
	fmt.Printf("Avg Brute-Force    : %.4f ms / query\n", avgBF)
	fmt.Printf("Avg HNSW Latency   : %.4f ms / query\n", avgHNSW)
	fmt.Printf("Speedup Multiplier : %.2fx faster\n", speedup)
	fmt.Println("-------------------------------------------------------")
	fmt.Printf("Average Recall@%d   : %.2f%%\n", k, avgRecall*100.0)
	fmt.Println("=======================================================")

	if avgRecall < 0.85 {
		t.Fatalf("Recall@%d is too low! Expected >= 85%%, got %.2f%%", k, avgRecall*100.0)
	}
}
