package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"vector-engine/index"
)

func main() {
	// Seed the global random number generator
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

	fmt.Printf("Initializing HNSW Index with M=%d, mL=%.4f\n", config.M, config.ML)
	idx := index.NewHNSWIndex(config)
	fmt.Printf("Initial MaxLayer: %d\n", idx.MaxLayer)
	fmt.Printf("Total Nodes: %d\n\n", len(idx.Nodes))

	fmt.Println("Generating 20 random layers to verify exponential decay distribution...")
	fmt.Println("---------------------------------------------------------------------")

	// Print exactly 20 nodes
	for i := 0; i < 20; i++ {
		layer := index.GenerateRandomLayer(idx.Config)
		fmt.Printf("Random Node %2d -> Assigned Layer: %d\n", i+1, layer)
	}

	fmt.Println("\n--- Extended Distribution Test (100,000 nodes) ---")
	extendedCounts := make(map[int]int)
	for i := 0; i < 100000; i++ {
		layer := index.GenerateRandomLayer(idx.Config)
		extendedCounts[layer]++
	}

	for l := 0; l < 10; l++ {
		if count, ok := extendedCounts[l]; ok {
			fmt.Printf("Layer %d: %8d nodes (%.2f%%)\n", l, count, float64(count)/1000.0)
		}
	}
}
