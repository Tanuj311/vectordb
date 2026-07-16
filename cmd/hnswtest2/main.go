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
	numVectors := 100
	dim := 128

	fmt.Printf("Inserting %d random vectors into HNSW Index...\n", numVectors)
	for i := 0; i < numVectors; i++ {
		id := index.NodeID(fmt.Sprintf("doc_%d", i))
		vec := generateRandomVector(dim)
		err := idx.Insert(id, vec)
		if err != nil {
			fmt.Printf("Error inserting %s: %v\n", id, err)
		}
	}

	fmt.Printf("\n--- HNSW Status (Verification) ---\n")
	fmt.Printf("Total Nodes mapped:   %d\n", len(idx.Nodes))
	fmt.Printf("Global MaxLayer:      %d\n", idx.MaxLayer)
	fmt.Printf("Current EntryPoint:   %s\n", idx.EntryPoint)
	
	entryNode := idx.Nodes[idx.EntryPoint]
	layer0Neighbors := 0
	if len(entryNode.Neighbors) > 0 {
		layer0Neighbors = len(entryNode.Neighbors[0])
	}
	fmt.Printf("EntryPoint Layer 0 Neighbors: %d (Max allowed: %d)\n", layer0Neighbors, config.MaxM0)
}
