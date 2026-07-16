package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"vector-engine/index"
)

func generateRandomVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := 0; i < dim; i++ {
		vec[i] = rand.Float32()
	}
	return vec
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// Standard HNSW configuration
	M := 16
	config := index.HNSWConfig{
		M:              M,
		MaxM0:          2 * M,
		EfConstruction: 200,
		EfSearch:       50,
		ML:             1.0 / math.Log(float64(M)),
	}

	idx := index.NewHNSWIndex(config)
	numVectors := 500
	dim := 128

	fmt.Printf("Inserting %d random vectors into HNSW Index...\n", numVectors)
	var exactMatchTarget []float32
	var exactMatchID string

	for i := 0; i < numVectors; i++ {
		id := index.NodeID(fmt.Sprintf("doc_%d", i))
		vec := generateRandomVector(dim)
		err := idx.Insert(id, vec)
		if err != nil {
			fmt.Printf("Error inserting %s: %v\n", id, err)
		}
		
		// Save one of the vectors to ensure our query can find a perfect match
		if i == 250 {
			exactMatchTarget = vec
			exactMatchID = string(id)
		}
	}

	fmt.Printf("\n--- HNSW Status ---\n")
	fmt.Printf("Total Nodes mapped:   %d\n", len(idx.Nodes))
	fmt.Printf("Global MaxLayer:      %d\n", idx.MaxLayer)

	fmt.Printf("\n--- HNSW Query Search (k=5) ---\n")
	fmt.Printf("Searching for target that perfectly matches '%s'...\n", exactMatchID)
	
	matches, execTime := idx.Query(exactMatchTarget, 5)
	
	fmt.Printf("Query Execution Time: %d ns (%.4f ms)\n", execTime, float64(execTime)/1000000.0)
	fmt.Println("Top 5 Matches:")
	for i, match := range matches {
		fmt.Printf("  %d. ID: %-8s | Score (Cosine Sim): %.6f\n", i+1, match.ID, match.Score)
	}
}
