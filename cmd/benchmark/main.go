package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"time"

	"vector-engine/index"
)

func genRandomVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := 0; i < dim; i++ {
		vec[i] = rand.Float32()
	}
	return vec
}

func cosineDistance(a, b []float32) float32 {
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 1.0
	}
	return 1.0 - (dot / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB)))))
}

type bfResult struct {
	id   string
	dist float32
}

func bruteForceTopK(query []float32, data map[string][]float32, k int) []string {
	var res []bfResult
	for id, vec := range data {
		dist := cosineDistance(query, vec)
		res = append(res, bfResult{id: id, dist: dist})
	}
	
	sort.Slice(res, func(i, j int) bool {
		return res[i].dist < res[j].dist
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

func main() {
	rand.Seed(time.Now().UnixNano())

	datasetSizes := []int{1000, 2000, 4000, 8000, 16000}
	efSearchValues := []int{50}
	Q := 500
	k := 10
	dim := 128

	M := 16
	efConstruction := 100

	fmt.Println("=========================================================================================")
	fmt.Println("                  HNSW vs. BRUTE-FORCE COMPREHENSIVE BENCHMARK SWEEP                     ")
	fmt.Println("=========================================================================================")
	fmt.Printf("Fixed Parameters: M=%d, efConstruction=%d, Queries (Q)=%d, Top-K (k)=%d, Dim=%d\n", M, efConstruction, Q, k, dim)

	for _, N := range datasetSizes {
		fmt.Printf("\n>>> INITIALIZING DATASET N = %d\n", N)
		
		dataset := make(map[string][]float32, N)
		
		config := index.HNSWConfig{
			M:              M,
			MaxM0:          2 * M,
			EfConstruction: efConstruction,
			EfSearch:       50,
			ML:             1.0 / math.Log(float64(M)),
		}
		hIndex := index.NewHNSWIndex(config)

		// 1. Generate & Populate
		fmt.Print("    Generating vectors and building Brute-Force baseline... ")
		for i := 0; i < N; i++ {
			id := fmt.Sprintf("doc_%d", i)
			dataset[id] = genRandomVector(dim)
		}
		fmt.Println("Done.")

		fmt.Print("    Building HNSW Index... ")
		buildStart := time.Now()
		for id, vec := range dataset {
			hIndex.Insert(index.NodeID(id), vec)
		}
		buildTime := time.Since(buildStart)
		fmt.Printf("Done in %v.\n", buildTime)

		// Generate Queries
		queries := make([][]float32, Q)
		for i := 0; i < Q; i++ {
			queries[i] = genRandomVector(dim)
		}

		// 2. Execute Brute Force Baseline
		fmt.Print("    Running Brute-Force exact queries for ground truth... ")
		
		bfDurations := make([]time.Duration, Q)
		groundTruths := make([][]string, Q)
		
		bfStart := time.Now()
		for i := 0; i < Q; i++ {
			qStart := time.Now()
			groundTruths[i] = bruteForceTopK(queries[i], dataset, k)
			bfDurations[i] = time.Since(qStart)
		}
		totalBFTime := time.Since(bfStart)
		fmt.Println("Done.")

		avgBFTime := totalBFTime / time.Duration(Q)
		bfQPS := float64(Q) / totalBFTime.Seconds()

		fmt.Printf("\n### RESULTS: Dataset Size N = %d\n", N)
		fmt.Printf("**Brute-Force Baseline** -> Avg Latency: %.2f ms | QPS: %.0f\n\n", float64(avgBFTime.Microseconds())/1000.0, bfQPS)
		
		// Markdown Table Header
		fmt.Println("| efSearch | Recall@10 | Avg Latency (µs) | p99 Latency (µs) | QPS       | Speedup vs BF |")
		fmt.Println("|----------|-----------|------------------|------------------|-----------|---------------|")

		// 3. Execute HNSW Sweep
		for _, efSearch := range efSearchValues {
			hIndex.Config.EfSearch = efSearch

			var totalRecall float64
			hnswDurations := make([]time.Duration, Q)

			sweepStart := time.Now()
			for i := 0; i < Q; i++ {
				qVec := queries[i]
				
				qStart := time.Now()
				hnswMatches, _ := hIndex.Query(qVec, k)
				hnswDurations[i] = time.Since(qStart)

				// Calculate Recall
				gtMap := make(map[string]bool)
				for _, id := range groundTruths[i] {
					gtMap[id] = true
				}

				intersection := 0
				for _, match := range hnswMatches {
					if gtMap[match.ID] {
						intersection++
					}
				}
				totalRecall += float64(intersection) / float64(k)
			}
			totalSweepTime := time.Since(sweepStart)

			// Metrics
			avgRecall := totalRecall / float64(Q)
			avgSweepTime := totalSweepTime / time.Duration(Q)
			qps := float64(Q) / totalSweepTime.Seconds()
			speedup := float64(avgBFTime.Nanoseconds()) / float64(avgSweepTime.Nanoseconds())

			// p99 Calculation
			sort.Slice(hnswDurations, func(i, j int) bool {
				return hnswDurations[i] < hnswDurations[j]
			})
			p99Idx := int(float64(Q) * 0.99)
			if p99Idx >= len(hnswDurations) {
				p99Idx = len(hnswDurations) - 1
			}
			p99Lat := hnswDurations[p99Idx]

			avgUs := float64(avgSweepTime.Microseconds())
			p99Us := float64(p99Lat.Microseconds())
			
			// Format Row
			fmt.Printf("| %-8d | %5.1f%%    | %-16.1f | %-16.1f | %-9.0f | %-13.1fx |\n", 
				efSearch, avgRecall*100.0, avgUs, p99Us, qps, speedup)
		}
		
		// Prevent Memory Spikes
		runtime.GC()
	}
	fmt.Println("\n=========================================================================================")
}
