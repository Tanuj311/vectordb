package index

import (
	"container/heap"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"
)

type NodeID string

type HNSWNode struct {
	ID        NodeID
	Vector    []float32
	Neighbors [][]NodeID // Neighbors[l] = neighbor IDs at layer l
}

type HNSWConfig struct {
	M              int     // Max neighbors per node per layer
	MaxM0          int     // Max neighbors for Layer 0
	EfConstruction int     // Exploration breadth during build
	EfSearch       int     // Exploration breadth during query
	ML             float64 // Normalization factor for level generation (1 / ln(M))
}

type HNSWIndex struct {
	mu         sync.RWMutex
	Nodes      map[NodeID]*HNSWNode
	EntryPoint NodeID
	MaxLayer   int
	Config     HNSWConfig
}

// NewHNSWIndex initializes a brand new, empty HNSW graph
func NewHNSWIndex(config HNSWConfig) *HNSWIndex {
	return &HNSWIndex{
		Nodes:    make(map[NodeID]*HNSWNode),
		MaxLayer: -1, // Indicates an empty graph
		Config:   config,
	}
}

// GenerateRandomLayer calculates the exponential decay layer assignment
func GenerateRandomLayer(config HNSWConfig) int {
	// Prevent math.Log(0) by ensuring rand.Float64() is strictly > 0
	r := rand.Float64()
	for r == 0.0 {
		r = rand.Float64()
	}
	return int(-math.Log(r) * config.ML)
}

func cosineDistance(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 1.0 // Max distance
	}
	var dotProduct, normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 1.0
	}
	sim := dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
	// distance = 1 - similarity (0 is identical, 2 is opposite)
	return 1.0 - sim
}

// --- Priority Queues for Greedy Search ---

type Item struct {
	ID   NodeID
	Dist float32
}

// MinQueue (closest first)
type MinQueue []Item
func (pq MinQueue) Len() int { return len(pq) }
func (pq MinQueue) Less(i, j int) bool { return pq[i].Dist < pq[j].Dist }
func (pq MinQueue) Swap(i, j int)      { pq[i], pq[j] = pq[j], pq[i] }
func (pq *MinQueue) Push(x interface{}) {
	*pq = append(*pq, x.(Item))
}
func (pq *MinQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	*pq = old[0 : n-1]
	return item
}

// MaxQueue (furthest first, for capacity clipping)
type MaxQueue []Item
func (pq MaxQueue) Len() int { return len(pq) }
func (pq MaxQueue) Less(i, j int) bool { return pq[i].Dist > pq[j].Dist }
func (pq MaxQueue) Swap(i, j int)      { pq[i], pq[j] = pq[j], pq[i] }
func (pq *MaxQueue) Push(x interface{}) {
	*pq = append(*pq, x.(Item))
}
func (pq *MaxQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	*pq = old[0 : n-1]
	return item
}

// searchLayer performs a greedy graph traversal on a specific layer
// Note: It assumes the caller holds the appropriate lock (Write lock during Insert)
func (idx *HNSWIndex) searchLayer(target []float32, entryPoints []NodeID, ef int, layer int) []NodeID {
	visited := make(map[NodeID]bool)

	candidates := &MinQueue{}
	heap.Init(candidates)
	
	results := &MaxQueue{}
	heap.Init(results)

	for _, ep := range entryPoints {
		if visited[ep] {
			continue
		}
		visited[ep] = true
		entryDist := cosineDistance(target, idx.Nodes[ep].Vector)
		heap.Push(candidates, Item{ID: ep, Dist: entryDist})
		heap.Push(results, Item{ID: ep, Dist: entryDist})
		if results.Len() > ef {
			heap.Pop(results)
		}
	}

	for candidates.Len() > 0 {
		c := heap.Pop(candidates).(Item)
		f := (*results)[0] // furthest node in our current results

		if c.Dist > f.Dist {
			// Early exit condition: the closest candidate is further than our worst result
			break
		}

		cNode := idx.Nodes[c.ID]
		if layer < len(cNode.Neighbors) {
			for _, neighborID := range cNode.Neighbors[layer] {
				if !visited[neighborID] {
					visited[neighborID] = true
					nNode := idx.Nodes[neighborID]
					nDist := cosineDistance(target, nNode.Vector)
					
					f = (*results)[0]
					if nDist < f.Dist || results.Len() < ef {
						heap.Push(candidates, Item{ID: neighborID, Dist: nDist})
						heap.Push(results, Item{ID: neighborID, Dist: nDist})
						if results.Len() > ef {
							heap.Pop(results)
						}
					}
				}
			}
		}
	}

	// Extract from max queue and sort ascending by distance
	out := make([]Item, 0, results.Len())
	for results.Len() > 0 {
		out = append(out, heap.Pop(results).(Item))
	}
	
	// reverse to make it closest-first
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}

	resIDs := make([]NodeID, len(out))
	for i, item := range out {
		resIDs[i] = item.ID
	}
	return resIDs
}

// selectNeighborsHeuristic implements the standard HNSW diversity heuristic (Relative Neighborhood Graph).
// It ensures that we only connect to a candidate if it's closer to the target than it is to any already-selected neighbor.
// This prevents densely clustered cliques and creates long-range 'bridge' edges across the graph.
func (idx *HNSWIndex) selectNeighborsHeuristic(target []float32, candidates []NodeID, M int) []NodeID {
	var selected []NodeID
	var discarded []NodeID

	for _, cID := range candidates {
		if len(selected) >= M {
			break
		}

		cNode := idx.Nodes[cID]
		cDist := cosineDistance(target, cNode.Vector)

		keep := true
		for _, sID := range selected {
			sNode := idx.Nodes[sID]
			distToNeighbor := cosineDistance(cNode.Vector, sNode.Vector)
			if distToNeighbor < cDist {
				keep = false
				break
			}
		}

		if keep {
			selected = append(selected, cID)
		} else {
			discarded = append(discarded, cID)
		}
	}

	// keepPrunedConnections: pad with discarded nodes to ensure graph connectivity in dense regions
	for _, dID := range discarded {
		if len(selected) >= M {
			break
		}
		selected = append(selected, dID)
	}

	return selected
}

// Insert adds a new vector into the HNSW Index
func (idx *HNSWIndex) Insert(id NodeID, vector []float32) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	targetLayer := GenerateRandomLayer(idx.Config)

	newNode := &HNSWNode{
		ID:        id,
		Vector:    vector,
		Neighbors: make([][]NodeID, targetLayer+1),
	}

	// Handle Empty Graph Case
	if idx.MaxLayer == -1 {
		idx.EntryPoint = id
		idx.MaxLayer = targetLayer
		idx.Nodes[id] = newNode
		return nil
	}

	// Register node early so distance calculations during pruning can find its vector
	idx.Nodes[id] = newNode

	currEntryPoints := []NodeID{idx.EntryPoint}
	
	// Phase 1: Top-Down Routing (Layers > targetLayer)
	for l := idx.MaxLayer; l > targetLayer; l-- {
		// greedily traverse to the single closest neighbor (ef = 1)
		res := idx.searchLayer(vector, currEntryPoints, 1, l)
		if len(res) > 0 {
			currEntryPoints = []NodeID{res[0]}
		}
	}

	// Phase 2: Bidirectional Insertion & Pruning (Layers <= targetLayer)
	maxL := idx.MaxLayer
	if targetLayer < maxL {
		maxL = targetLayer
	}

	for l := maxL; l >= 0; l-- {
		// Find best candidate neighbors using EfConstruction
		candidates := idx.searchLayer(vector, currEntryPoints, idx.Config.EfConstruction, l)

		maxM := idx.Config.M
		if l == 0 {
			maxM = idx.Config.MaxM0
		}

		// Apply the standard HNSW diversity heuristic for edge selection
		selected := idx.selectNeighborsHeuristic(vector, candidates, maxM)

		// Add connections bi-directionally
		newNode.Neighbors[l] = selected
		for _, neighborID := range selected {
			neighborNode := idx.Nodes[neighborID]
			
			// Ensure neighbor has neighbor slices up to layer l initialized
			for len(neighborNode.Neighbors) <= l {
				neighborNode.Neighbors = append(neighborNode.Neighbors, nil)
			}
			
			neighborNode.Neighbors[l] = append(neighborNode.Neighbors[l], id)

			// Pruning logic for neighbors that exceed capacity
			if len(neighborNode.Neighbors[l]) > maxM {
				type nItem struct {
					id   NodeID
					dist float32
				}
				var nList []nItem
				for _, nnID := range neighborNode.Neighbors[l] {
					d := cosineDistance(neighborNode.Vector, idx.Nodes[nnID].Vector)
					nList = append(nList, nItem{id: nnID, dist: d})
				}
				
				sort.Slice(nList, func(i, j int) bool {
					return nList[i].dist < nList[j].dist
				})
				
				var sortedCands []NodeID
				for _, item := range nList {
					sortedCands = append(sortedCands, item.id)
				}
				
				neighborNode.Neighbors[l] = idx.selectNeighborsHeuristic(neighborNode.Vector, sortedCands, maxM)
			}
		}

		if len(candidates) > 0 {
			currEntryPoints = candidates
		}
	}

	// Phase 3: Update Global State
	if targetLayer > idx.MaxLayer {
		idx.MaxLayer = targetLayer
		idx.EntryPoint = id
	}

	return nil
}

type VectorMatch struct {
	ID    string
	Score float32
}

// Query performs an HNSW search to find the top-k nearest neighbors
func (idx *HNSWIndex) Query(target []float32, k int) ([]VectorMatch, int64) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	start := time.Now()

	// Edge case: Empty graph
	if idx.MaxLayer < 0 || idx.EntryPoint == "" {
		return nil, 0
	}

	currEntryPoints := []NodeID{idx.EntryPoint}

	// Phase 1: Greedy Descent (Layers > 0)
	for l := idx.MaxLayer; l > 0; l-- {
		res := idx.searchLayer(target, currEntryPoints, 1, l)
		if len(res) > 0 {
			currEntryPoints = []NodeID{res[0]}
		}
	}

	// Phase 2: Beam Search on Layer 0
	efSearch := idx.Config.EfSearch
	if k > efSearch {
		efSearch = k
	}

	candidates := idx.searchLayer(target, currEntryPoints, efSearch, 0)

	// Phase 3: Extract Top-K Results
	var matches []VectorMatch
	for _, candID := range candidates {
		candNode := idx.Nodes[candID]
		dist := cosineDistance(target, candNode.Vector)
		// Our distance is 1 - similarity. So similarity = 1 - distance.
		score := 1.0 - dist
		matches = append(matches, VectorMatch{
			ID:    string(candID),
			Score: score,
		})
	}

	if len(matches) > k {
		matches = matches[:k]
	}

	return matches, time.Since(start).Nanoseconds()
}

func (idx *HNSWIndex) TotalNodes() int64 {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return int64(len(idx.Nodes))
}
