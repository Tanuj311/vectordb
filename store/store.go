package store

import (
	"math"
	"runtime"
	"sort"
	"sync"
	"time"

	"vector-engine/index"
)

type VectorMatch struct {
	ID    string
	Score float32
}

type VectorStore struct {
	mu        sync.RWMutex
	data      map[string][]float32
	IndexType string
	HNSW      *index.HNSWIndex
}

func NewVectorStore(indexType string, config index.HNSWConfig) *VectorStore {
	store := &VectorStore{
		data:      make(map[string][]float32),
		IndexType: indexType,
	}
	if indexType == "hnsw" {
		store.HNSW = index.NewHNSWIndex(config)
	}
	return store
}

func (s *VectorStore) Insert(id string, vec []float32) {
	if s.IndexType == "hnsw" {
		s.HNSW.Insert(index.NodeID(id), vec)
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.data[id] = vec
}

// Query routes the execution path dynamically
func (s *VectorStore) Query(target []float32, k int) ([]VectorMatch, int64) {
	if s.IndexType == "hnsw" {
		idxMatches, execTime := s.HNSW.Query(target, k)
		// Convert from index.VectorMatch to store.VectorMatch
		matches := make([]VectorMatch, len(idxMatches))
		for i, m := range idxMatches {
			matches[i] = VectorMatch{ID: m.ID, Score: m.Score}
		}
		return matches, execTime
	}
	return s.queryBruteForce(target, k)
}

// queryBruteForce is the parallelized Stage 3 baseline using a goroutine worker pool
func (s *VectorStore) queryBruteForce(target []float32, k int) ([]VectorMatch, int64) {
	start := time.Now()
	
	s.mu.RLock()
	defer s.mu.RUnlock()

	numWorkers := runtime.NumCPU()
	if numWorkers < 1 {
		numWorkers = 1
	}

	keys := make([]string, 0, len(s.data))
	for id := range s.data {
		keys = append(keys, id)
	}

	total := len(keys)
	if total == 0 {
		return []VectorMatch{}, time.Since(start).Nanoseconds()
	}

	chunkSize := total / numWorkers
	if chunkSize == 0 {
		chunkSize = 1
		numWorkers = total
	}

	resultsCh := make(chan []VectorMatch, numWorkers)
	var wg sync.WaitGroup

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			startIdx := workerID * chunkSize
			endIdx := startIdx + chunkSize
			if workerID == numWorkers-1 {
				endIdx = total
			}

			if startIdx >= total {
				return
			}

			var localMatches []VectorMatch
			for _, id := range keys[startIdx:endIdx] {
				vec := s.data[id]
				score := cosineSimilarity(target, vec)
				localMatches = append(localMatches, VectorMatch{ID: id, Score: score})
			}

			sort.Slice(localMatches, func(i, j int) bool {
				return localMatches[i].Score > localMatches[j].Score
			})

			localK := k
			if localK > len(localMatches) {
				localK = len(localMatches)
			}
			
			resultsCh <- localMatches[:localK]
		}(i)
	}

	wg.Wait()
	close(resultsCh)

	var globalMatches []VectorMatch
	for res := range resultsCh {
		globalMatches = append(globalMatches, res...)
	}

	sort.Slice(globalMatches, func(i, j int) bool {
		return globalMatches[i].Score > globalMatches[j].Score
	})

	if k > len(globalMatches) {
		k = len(globalMatches)
	}

	execTime := time.Since(start).Nanoseconds()
	return globalMatches[:k], execTime
}

func (s *VectorStore) TotalVectors() int64 {
	if s.IndexType == "hnsw" {
		return s.HNSW.TotalNodes()
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return int64(len(s.data))
}

func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dotProduct, normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}
