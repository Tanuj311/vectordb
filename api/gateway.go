package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	pb "vector-engine/proto"
)

type Gateway struct {
	coordinator pb.VectorServiceServer
}

func NewGateway(coordinator pb.VectorServiceServer) *Gateway {
	return &Gateway{coordinator: coordinator}
}

type InsertReq struct {
	ID     string    `json:"id"`
	Vector []float32 `json:"vector"`
}

func (g *Gateway) handleInsert(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req InsertReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	res, err := g.coordinator.Insert(ctx, &pb.InsertRequest{
		Id:     req.ID,
		Vector: req.Vector,
	})
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(res)
}

type QueryReq struct {
	Vector []float32 `json:"vector"`
	K      int32     `json:"k"`
}

func (g *Gateway) handleQuery(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req QueryReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	res, err := g.coordinator.Query(ctx, &pb.QueryRequest{
		Vector: req.Vector,
		K:      req.K,
	})
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(res)
}

func (g *Gateway) handleStats(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	res, err := g.coordinator.GetStats(ctx, &pb.StatsRequest{})
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(res)
}

func (g *Gateway) Start(port int) error {
	mux := http.NewServeMux()
	mux.HandleFunc("/insert", g.handleInsert)
	mux.HandleFunc("/query", g.handleQuery)
	mux.HandleFunc("/stats", g.handleStats)

	addr := fmt.Sprintf(":%d", port)
	return http.ListenAndServe(addr, mux)
}
