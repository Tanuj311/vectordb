package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "vector-engine/proto"
)

func generateRandomVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := 0; i < dim; i++ {
		vec[i] = rand.Float32()
	}
	return vec
}

func main() {
	target := flag.String("target", "localhost:50051", "Target gRPC address")
	numVectors := flag.Int("vectors", 10000, "Number of vectors to insert")
	concurrency := flag.Int("concurrency", 20, "Number of concurrent workers")
	numQueries := flag.Int("queries", 5000, "Number of queries to execute")
	flag.Parse()

	fmt.Printf("Connecting to %s...\n", *target)
	conn, err := grpc.Dial(*target, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()
	
	dim := 128
	client := pb.NewVectorServiceClient(conn)

	// --- Phase 1: Ingestion Stress Test ---
	fmt.Printf("\n--- Phase 1: Ingestion Stress Test ---\n")
	fmt.Printf("Generating and inserting %d vectors using %d concurrent workers...\n", *numVectors, *concurrency)

	insertJobs := make(chan int, *numVectors)
	for i := 0; i < *numVectors; i++ {
		insertJobs <- i
	}
	close(insertJobs)

	var insertWg sync.WaitGroup
	startInsert := time.Now()

	for i := 0; i < *concurrency; i++ {
		insertWg.Add(1)
		go func() {
			defer insertWg.Done()
			for idx := range insertJobs {
				vec := generateRandomVector(dim)
				id := fmt.Sprintf("doc_%d", idx)
				
				_, err := client.Insert(context.Background(), &pb.InsertRequest{
					Id:     id,
					Vector: vec,
				})
				if err != nil {
					log.Printf("Insert failed for %s: %v", id, err)
				}
			}
		}()
	}

	insertWg.Wait()
	insertDuration := time.Since(startInsert)
	insertThroughput := float64(*numVectors) / insertDuration.Seconds()
	
	fmt.Printf("Total Ingestion Time: %v\n", insertDuration)
	fmt.Printf("Ingestion Throughput: %.2f Inserts/sec\n", insertThroughput)

	// --- Phase 2: Concurrent Query Stress Test ---
	fmt.Printf("\n--- Phase 2: Concurrent Query Stress Test ---\n")
	fmt.Printf("Executing %d queries using %d concurrent workers...\n", *numQueries, *concurrency)

	queryJobs := make(chan struct{}, *numQueries)
	for i := 0; i < *numQueries; i++ {
		queryJobs <- struct{}{}
	}
	close(queryJobs)

	// Pre-allocate slice per worker to avoid lock contention
	workerLatencies := make([][]time.Duration, *concurrency)
	for i := 0; i < *concurrency; i++ {
		workerLatencies[i] = make([]time.Duration, 0, (*numQueries / *concurrency) + 1)
	}

	var queryWg sync.WaitGroup
	var completedQueries uint64
	startQuery := time.Now()

	for i := 0; i < *concurrency; i++ {
		queryWg.Add(1)
		go func(workerID int) {
			defer queryWg.Done()
			
			for range queryJobs {
				vec := generateRandomVector(dim)
				
				qStart := time.Now()
				_, err := client.Query(context.Background(), &pb.QueryRequest{
					Vector: vec,
					K:      10,
				})
				qDuration := time.Since(qStart)
				
				if err != nil {
					log.Printf("Query failed: %v", err)
					continue
				}

				workerLatencies[workerID] = append(workerLatencies[workerID], qDuration)
				atomic.AddUint64(&completedQueries, 1)
			}
		}(i)
	}

	queryWg.Wait()
	totalQueryDuration := time.Since(startQuery)

	// --- Phase 3: Metrics Calculation & Report ---
	fmt.Printf("\n--- Phase 3: Metrics Calculation & Report ---\n")

	var allLatencies []time.Duration
	for _, latencies := range workerLatencies {
		allLatencies = append(allLatencies, latencies...)
	}

	if len(allLatencies) == 0 {
		log.Fatalf("No successful queries recorded.")
	}

	sort.Slice(allLatencies, func(i, j int) bool {
		return allLatencies[i] < allLatencies[j]
	})

	totalActualQueries := len(allLatencies)
	qps := float64(totalActualQueries) / totalQueryDuration.Seconds()
	
	var totalLatency time.Duration
	for _, l := range allLatencies {
		totalLatency += l
	}
	avgLatency := totalLatency / time.Duration(totalActualQueries)
	
	p50 := allLatencies[int(float64(totalActualQueries)*0.50)]
	p95 := allLatencies[int(float64(totalActualQueries)*0.95)]
	p99 := allLatencies[int(float64(totalActualQueries)*0.99)]

	fmt.Printf("Total Queries Executed: %d\n", totalActualQueries)
	fmt.Printf("Total Elapsed Time:     %v\n", totalQueryDuration)
	fmt.Printf("Throughput:             %.2f QPS\n", qps)
	fmt.Printf("Average Latency:        %.2f ms\n", float64(avgLatency.Microseconds())/1000.0)
	fmt.Printf("p50 Latency (Median):   %.2f ms\n", float64(p50.Microseconds())/1000.0)
	fmt.Printf("p95 Latency:            %.2f ms\n", float64(p95.Microseconds())/1000.0)
	fmt.Printf("p99 Latency:            %.2f ms\n", float64(p99.Microseconds())/1000.0)
}
