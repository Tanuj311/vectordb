package cluster

import (
	"hash/crc32"
	"sort"
	"sync"
)

type HashRing struct {
	mu    sync.RWMutex
	nodes map[uint32]string
	keys  []uint32
}

func NewHashRing() *HashRing {
	return &HashRing{
		nodes: make(map[uint32]string),
	}
}

func (r *HashRing) AddNode(nodeUrl string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	hash := crc32.ChecksumIEEE([]byte(nodeUrl))
	if _, exists := r.nodes[hash]; !exists {
		r.nodes[hash] = nodeUrl
		r.keys = append(r.keys, hash)
		sort.Slice(r.keys, func(i, j int) bool {
			return r.keys[i] < r.keys[j]
		})
	}
}

func (r *HashRing) RemoveNode(nodeUrl string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	hash := crc32.ChecksumIEEE([]byte(nodeUrl))
	if _, exists := r.nodes[hash]; exists {
		delete(r.nodes, hash)
		
		var newKeys []uint32
		for _, k := range r.keys {
			if k != hash {
				newKeys = append(newKeys, k)
			}
		}
		r.keys = newKeys
	}
}

func (r *HashRing) GetNode(key string) string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if len(r.keys) == 0 {
		return ""
	}

	hash := crc32.ChecksumIEEE([]byte(key))
	
	idx := sort.Search(len(r.keys), func(i int) bool {
		return r.keys[i] >= hash
	})

	if idx == len(r.keys) {
		idx = 0
	}

	return r.nodes[r.keys[idx]]
}
