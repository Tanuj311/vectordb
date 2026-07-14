package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "vector-engine/proto"
)

func main() {
	// grpc.NewClient is available in grpc v1.62+ 
	conn, err := grpc.Dial("localhost:50051", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()
	c := pb.NewVectorServiceClient(conn)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	fmt.Println("Connecting to gRPC server and inserting mock vectors...")
	_, err = c.Insert(ctx, &pb.InsertRequest{Id: "doc1", Vector: []float32{1.0, 0.0, 0.0}})
	if err != nil { log.Fatalf("insert failed: %v", err) }
	
	_, err = c.Insert(ctx, &pb.InsertRequest{Id: "doc2", Vector: []float32{0.0, 1.0, 0.0}})
	if err != nil { log.Fatalf("insert failed: %v", err) }

	fmt.Println("Successfully inserted vectors. Executing query...")
	qRes, err := c.Query(ctx, &pb.QueryRequest{
		Vector: []float32{1.0, 0.1, 0.0},
		K:      3,
	})
	if err != nil {
		log.Fatalf("could not query: %v", err)
	}
	
	fmt.Printf("Query Response (Execution Time: %d ns):\n", qRes.ExecutionTimeNs)
	for i, match := range qRes.Matches {
		fmt.Printf("  %d. ID: %s, Score: %f\n", i+1, match.Id, match.Score)
	}
}
