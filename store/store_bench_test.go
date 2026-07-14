package store

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"vector-engine/index"
)

func generateRandomVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := 0; i < dim; i++ {
		vec[i] = rand.Float32()
	}
	return vec
}

func BenchmarkInsert(b *testing.B) {
	config := index.HNSWConfig{M: 16, MaxM0: 32, EfConstruction: 100, EfSearch: 50, ML: 1.0 / math.Log(16)}
	store := NewVectorStore("bruteforce", config)
	dim := 128

	vectors := make([][]float32, b.N)
	ids := make([]string, b.N)
	for i := 0; i < b.N; i++ {
		vectors[i] = generateRandomVector(dim)
		ids[i] = fmt.Sprintf("vec_%d", i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store.Insert(ids[i], vectors[i])
	}
}

func BenchmarkQuery(b *testing.B) {
	config := index.HNSWConfig{M: 16, MaxM0: 32, EfConstruction: 100, EfSearch: 50, ML: 1.0 / math.Log(16)}
	store := NewVectorStore("bruteforce", config)
	dim := 128
	dbSize := 1000

	for i := 0; i < dbSize; i++ {
		store.Insert(fmt.Sprintf("vec_%d", i), generateRandomVector(dim))
	}

	target := generateRandomVector(dim)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		store.Query(target, 10)
	}
}

func setupBenchmarkStore(b *testing.B, numVectors, dim int) (*VectorStore, []float32) {
	store := NewVectorStore("bruteforce", index.HNSWConfig{})
	for i := 0; i < numVectors; i++ {
		store.Insert(fmt.Sprintf("doc_%d", i), generateRandomVector(dim))
	}
	target := generateRandomVector(dim)
	b.ResetTimer()
	return store, target
}

func BenchmarkVectorQuery(b *testing.B) {
	numVectors := 10000
	dim := 128
	k := 10

	store, target := setupBenchmarkStore(b, numVectors, dim)

	b.Run("SingleThreaded", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			store.Query(target, k)
		}
	})

	b.Run("Parallelized", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			store.Query(target, k)
		}
	})
}
