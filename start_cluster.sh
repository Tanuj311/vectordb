#!/bin/bash

# Start Shard 1
go run main.go -type=shard -port=50051 -index-type=hnsw &
SHARD1_PID=$!

# Start Shard 2
go run main.go -type=shard -port=50052 -index-type=hnsw &
SHARD2_PID=$!

# Start Shard 3
go run main.go -type=shard -port=50053 -index-type=hnsw &
SHARD3_PID=$!

echo "Waiting for shards to spin up..."
sleep 2

# Start the Coordinator & HTTP Gateway
go run main.go -type=coordinator -port=50050 -http-port=8080 -shards="localhost:50051,localhost:50052,localhost:50053" &
COORD_PID=$!

echo "Cluster is running."
echo "Press Ctrl+C to shut down."

# Trap SIGINT (Ctrl+C) to gracefully shut down the cluster
trap "echo 'Shutting down cluster...'; kill $SHARD1_PID $SHARD2_PID $SHARD3_PID $COORD_PID; exit" SIGINT SIGTERM

# Wait indefinitely so the script doesn't exit
wait
