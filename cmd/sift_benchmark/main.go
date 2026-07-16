package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"time"

	"vector-engine/index"
)

// ReadFvecs reads an fvecs file into a slice of float32 slices
func ReadFvecs(filename string) ([][]float32, error) {
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

// ReadIvecs reads an ivecs file into a slice of int32 slices
func ReadIvecs(filename string) ([][]int32, error) {
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

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	
	baseFile := "siftsmall/siftsmall_base.fvecs"
	queryFile := "siftsmall/siftsmall_query.fvecs"
	gtFile := "siftsmall/siftsmall_groundtruth.ivecs"

	fmt.Printf("Loading SIFT10K dataset...\n")
	baseVecs, err := ReadFvecs(baseFile)
	if err != nil {
		fmt.Printf("Error loading base: %v\n", err)
		return
	}
	queryVecs, err := ReadFvecs(queryFile)
	if err != nil {
		fmt.Printf("Error loading query: %v\n", err)
		return
	}
	gtVecs, err := ReadIvecs(gtFile)
	if err != nil {
		fmt.Printf("Error loading ground truth: %v\n", err)
		return
	}
	
	fmt.Printf("Loaded %d base vectors, %d query vectors, %d ground truth vectors.\n", len(baseVecs), len(queryVecs), len(gtVecs))

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
	for i, vec := range baseVecs {
		hIndex.Insert(index.NodeID(fmt.Sprintf("%d", i)), vec)
	}
	buildTime := time.Since(buildStart)
	fmt.Printf("Done in %v.\n", buildTime)

	k := 10
	efValues := []int{10, 50, 100, 200, 400}
	
	fmt.Printf("\n### RESULTS: Dataset Size N = %d (SIFT10K)\n", len(baseVecs))
	fmt.Printf("| efSearch | Recall@10 | Avg Latency (µs) | QPS       |\n")
	fmt.Printf("|----------|-----------|------------------|-----------|\n")

	for _, ef := range efValues {
		hIndex.Config.EfSearch = ef
		
		hStart := time.Now()
		var totalMatches int
		
		for i := 0; i < len(queryVecs); i++ {
			matches, _ := hIndex.Query(queryVecs[i], k)
			
			// Compute recall against exact ground truth provided by SIFT
			matchCount := 0
			gtMap := make(map[string]bool)
			for j := 0; j < k && j < len(gtVecs[i]); j++ { // top-k ground truth
				gtMap[fmt.Sprintf("%d", gtVecs[i][j])] = true
			}
			
			for _, m := range matches {
				if gtMap[m.ID] {
					matchCount++
				}
			}
			totalMatches += matchCount
		}
		
		hDuration := time.Since(hStart)
		Q := len(queryVecs)
		avgLatencyUs := float64(hDuration.Microseconds()) / float64(Q)
		qps := float64(Q) / hDuration.Seconds()
		recall := float64(totalMatches) / float64(Q*k) * 100.0

		fmt.Printf("| %-8d | %-9.1f | %-16.1f | %-9.0f |\n", ef, recall, avgLatencyUs, qps)
	}
}
