package index

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"testing"
)

// readFvecs reads an fvecs file into a slice of float32 slices
func readFvecs(filename string) ([][]float32, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var vectors [][]float32
	var d int32
	for {
		err := binary.Read(file, binary.LittleEndian, &d)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		vec := make([]float32, d)
		err = binary.Read(file, binary.LittleEndian, &vec)
		if err != nil {
			return nil, err
		}
		vectors = append(vectors, vec)
	}
	return vectors, nil
}

// readIvecs reads an ivecs file into a slice of int32 slices
func readIvecs(filename string) ([][]int32, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var vectors [][]int32
	var d int32
	for {
		err := binary.Read(file, binary.LittleEndian, &d)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		vec := make([]int32, d)
		err = binary.Read(file, binary.LittleEndian, &vec)
		if err != nil {
			return nil, err
		}
		vectors = append(vectors, vec)
	}
	return vectors, nil
}

// TestHNSW_SIFT10K_Regression runs a regression test against the SIFT10K dataset.
// It asserts that the index correctly builds and achieves > 95% Recall@10 at efSearch=50.
func TestHNSW_SIFT10K_Regression(t *testing.T) {
	baseFile := "../siftsmall/siftsmall_base.fvecs"
	queryFile := "../siftsmall/siftsmall_query.fvecs"
	gtFile := "../siftsmall/siftsmall_groundtruth.ivecs"

	if _, err := os.Stat(baseFile); os.IsNotExist(err) {
		t.Skip("SIFT10K dataset not found. Skipping regression test.")
	}

	baseVecs, err := readFvecs(baseFile)
	if err != nil {
		t.Fatalf("Failed to read base vectors: %v", err)
	}
	queryVecs, err := readFvecs(queryFile)
	if err != nil {
		t.Fatalf("Failed to read query vectors: %v", err)
	}
	gtVecs, err := readIvecs(gtFile)
	if err != nil {
		t.Fatalf("Failed to read ground truth: %v", err)
	}

	t.Logf("Loaded %d base vectors, %d query vectors.", len(baseVecs), len(queryVecs))

	M := 16
	config := HNSWConfig{
		M:              M,
		MaxM0:          2 * M,
		EfConstruction: 100,
		EfSearch:       50,
		ML:             1.0 / math.Log(float64(M)),
	}

	hIndex := NewHNSWIndex(config)

	// Build Index
	for i, vec := range baseVecs {
		hIndex.Insert(NodeID(fmt.Sprintf("%d", i)), vec)
	}
	t.Log("HNSW Index built successfully.")

	// Test Recall@10
	k := 10
	var totalMatches int

	for i := 0; i < len(queryVecs); i++ {
		matches, _ := hIndex.Query(queryVecs[i], k)

		matchCount := 0
		gtMap := make(map[string]bool)
		for j := 0; j < k && j < len(gtVecs[i]); j++ {
			gtMap[fmt.Sprintf("%d", gtVecs[i][j])] = true
		}

		for _, m := range matches {
			if gtMap[m.ID] {
				matchCount++
			}
		}
		totalMatches += matchCount
	}

	recall := float64(totalMatches) / float64(len(queryVecs)*k) * 100.0
	t.Logf("Recall@10 with efSearch=%d: %.2f%%", config.EfSearch, recall)

	if recall < 95.0 {
		t.Errorf("Regression failed! Expected Recall@10 >= 95.0%%, got %.2f%%", recall)
	}
}
