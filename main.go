package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math"
	"net"
	"sort"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"vector-engine/api"
	"vector-engine/cluster"
	"vector-engine/index"
	pb "vector-engine/proto"
	"vector-engine/store"
)

// --- Shard Server ---
type shardServer struct {
	pb.UnimplementedVectorServiceServer
	store *store.VectorStore
}

func (s *shardServer) Insert(ctx context.Context, req *pb.InsertRequest) (*pb.InsertResponse, error) {
	s.store.Insert(req.Id, req.Vector)
	return &pb.InsertResponse{Success: true, Message: "Inserted to shard"}, nil
}

func (s *shardServer) Query(ctx context.Context, req *pb.QueryRequest) (*pb.QueryResponse, error) {
	matches, execTime := s.store.Query(req.Vector, int(req.K))

	pbMatches := make([]*pb.VectorMatch, len(matches))
	for i, m := range matches {
		pbMatches[i] = &pb.VectorMatch{Id: m.ID, Score: m.Score}
	}

	return &pb.QueryResponse{
		Matches: pbMatches, 
		ExecutionTimeNs: execTime,
		IndexType: s.store.IndexType,
	}, nil
}

func (s *shardServer) GetStats(ctx context.Context, req *pb.StatsRequest) (*pb.StatsResponse, error) {
	maxLayer := int32(-1)
	efSearch := int32(0)
	if s.store.IndexType == "hnsw" && s.store.HNSW != nil {
		maxLayer = int32(s.store.HNSW.MaxLayer)
		efSearch = int32(s.store.HNSW.Config.EfSearch)
	}
	return &pb.StatsResponse{
		TotalVectors:   s.store.TotalVectors(),
		WorkerPoolSize: 1,
		IndexType:      s.store.IndexType,
		MaxLayer:       maxLayer,
		EfSearch:       efSearch,
	}, nil
}

// --- Coordinator Server ---
type coordinatorServer struct {
	pb.UnimplementedVectorServiceServer
	ring      *cluster.HashRing
	clients   map[string]pb.VectorServiceClient
	shardUrls []string
}

func newCoordinatorServer(shards string) *coordinatorServer {
	c := &coordinatorServer{
		ring:    cluster.NewHashRing(),
		clients: make(map[string]pb.VectorServiceClient),
	}
	
	for _, shardUrl := range strings.Split(shards, ",") {
		shardUrl = strings.TrimSpace(shardUrl)
		if shardUrl == "" {
			continue
		}
		c.shardUrls = append(c.shardUrls, shardUrl)
		c.ring.AddNode(shardUrl)

		conn, err := grpc.Dial(shardUrl, grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			log.Fatalf("Failed to dial shard %s: %v", shardUrl, err)
		}
		c.clients[shardUrl] = pb.NewVectorServiceClient(conn)
	}
	return c
}

func (c *coordinatorServer) Insert(ctx context.Context, req *pb.InsertRequest) (*pb.InsertResponse, error) {
	targetShard := c.ring.GetNode(req.Id)
	if targetShard == "" {
		return &pb.InsertResponse{Success: false, Message: "No shards available"}, nil
	}
	client := c.clients[targetShard]
	
	start := time.Now()
	res, err := client.Insert(ctx, req)
	if err != nil {
		return nil, err
	}
	res.RoutedToShard = targetShard
	res.ExecutionTimeNs = time.Since(start).Nanoseconds()
	return res, nil
}

func (c *coordinatorServer) Query(ctx context.Context, req *pb.QueryRequest) (*pb.QueryResponse, error) {
	var wg sync.WaitGroup
	var mu sync.Mutex
	var allMatches []*pb.VectorMatch
	var maxExecTime int64

	for _, shardUrl := range c.shardUrls {
		wg.Add(1)
		go func(client pb.VectorServiceClient) {
			defer wg.Done()
			res, err := client.Query(ctx, req)
			if err != nil {
				log.Printf("Query to shard failed: %v", err)
				return
			}
			
			mu.Lock()
			allMatches = append(allMatches, res.Matches...)
			if res.ExecutionTimeNs > maxExecTime {
				maxExecTime = res.ExecutionTimeNs
			}
			mu.Unlock()
		}(c.clients[shardUrl])
	}
	wg.Wait()

	sort.Slice(allMatches, func(i, j int) bool {
		return allMatches[i].Score > allMatches[j].Score
	})

	k := int(req.K)
	if k > len(allMatches) {
		k = len(allMatches)
	}

	return &pb.QueryResponse{
		Matches:         allMatches[:k],
		ExecutionTimeNs: maxExecTime,
		RoutedToShard:   c.shardUrls[len(req.Vector) % len(c.shardUrls)], // deterministic pseudo-random for UI
		IndexType:       "hnsw",
	}, nil
}

func (c *coordinatorServer) GetStats(ctx context.Context, req *pb.StatsRequest) (*pb.StatsResponse, error) {
	var total int64
	var idxType string
	var maxL int32 = -1
	var efS int32 = 0
	var shards []*pb.ShardStats

	for url, client := range c.clients {
		res, err := client.GetStats(ctx, req)
		if err == nil {
			total += res.TotalVectors
			idxType = res.IndexType
			if res.MaxLayer > maxL {
				maxL = res.MaxLayer
			}
			if res.EfSearch > efS {
				efS = res.EfSearch
			}
			shards = append(shards, &pb.ShardStats{
				NodeId:      url,
				VectorCount: res.TotalVectors,
				MaxLayer:    res.MaxLayer,
				EfSearch:    res.EfSearch,
				Status:      "healthy",
			})
		} else {
			shards = append(shards, &pb.ShardStats{
				NodeId: url,
				Status: "offline",
			})
		}
	}
	
	// Sort shards for deterministic UI order
	sort.Slice(shards, func(i, j int) bool {
		return shards[i].NodeId < shards[j].NodeId
	})

	return &pb.StatsResponse{
		TotalVectors:   total, 
		WorkerPoolSize: int32(len(c.clients)),
		IndexType:      idxType,
		MaxLayer:       maxL,
		EfSearch:       efS,
		Shards:         shards,
	}, nil
}

func main() {
	port := flag.Int("port", 50051, "The server port")
	httpPort := flag.Int("http-port", 8080, "The HTTP gateway port for coordinator")
	nodeType := flag.String("type", "shard", "Node type (shard or coordinator)")
	shards := flag.String("shards", "", "Comma-separated list of shard addresses for coordinator")
	
	// Index configuration flags
	indexType := flag.String("index-type", "hnsw", "Index type to use: hnsw or bruteforce")
	hnswM := flag.Int("m", 16, "HNSW max neighbors per layer (M)")
	hnswEfConst := flag.Int("ef-construction", 100, "HNSW exploration breadth during build (efConstruction)")
	hnswEfSearch := flag.Int("ef-search", 50, "HNSW exploration breadth during search (efSearch)")
	flag.Parse()

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	s := grpc.NewServer()

	if *nodeType == "coordinator" {
		if *shards == "" {
			log.Fatalf("Coordinator mode requires --shards flag")
		}
		coord := newCoordinatorServer(*shards)
		pb.RegisterVectorServiceServer(s, coord)
		log.Printf("Coordinator gRPC listening at %v, routing to %s", lis.Addr(), *shards)

		go func() {
			gateway := api.NewGateway(coord)
			log.Printf("HTTP Gateway listening on :%d", *httpPort)
			if err := gateway.Start(*httpPort); err != nil {
				log.Fatalf("Failed to serve HTTP gateway: %v", err)
			}
		}()
	} else {
		// Initialize the storage engine with the CLI configurations
		config := index.HNSWConfig{
			M:              *hnswM,
			MaxM0:          2 * *hnswM,
			EfConstruction: *hnswEfConst,
			EfSearch:       *hnswEfSearch,
			ML:             1.0 / math.Log(float64(*hnswM)),
		}
		
		vectorStore := store.NewVectorStore(*indexType, config)
		
		pb.RegisterVectorServiceServer(s, &shardServer{
			store: vectorStore,
		})
		log.Printf("Shard listening at %v (Index: %s)", lis.Addr(), *indexType)
	}

	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
