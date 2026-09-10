#pragma once

#include <cstdint>
#include <vector>

namespace hvax {

// Row-major kNN lists: nbr[i*k + j] is the j-th neighbour of node i or
// UINT32_MAX when missing; sim holds the cosine.
struct KnnGraph {
  uint64_t n = 0;
  int k = 0;
  std::vector<uint32_t> nbr;
  std::vector<float> sim;
};

// Undirected weighted graph in CSR form.
struct CsrGraph {
  std::vector<uint64_t> offsets;  // n + 1
  std::vector<uint32_t> adj;
  std::vector<float> w;
  uint64_t n() const { return offsets.empty() ? 0 : offsets.size() - 1; }
  uint64_t edges() const { return adj.size() / 2; }
};

// Keep edges with sim >= edge_min between eligible nodes, symmetrize
// (max weight when both directions exist), drop self loops.
CsrGraph symmetrize_knn(const KnnGraph& g, float edge_min, const std::vector<uint8_t>& eligible);

// Chinese Whispers with deterministic in-place updates: every node adopts the
// label with the largest summed edge weight among its neighbours, ties broken
// by the smaller label, and only moves when that weight strictly exceeds the
// weight of its current label. Nodes with fixed[i] != 0 never change. Returns
// iterations run. Labels must be initialised by the caller (usually node
// index, or a shared id for pinned members). `threads` is accepted for API
// stability; propagation runs sequentially so results are reproducible.
int chinese_whispers(const CsrGraph& g, std::vector<uint32_t>& labels, const std::vector<uint8_t>& fixed,
                     int max_iters, int threads);

// Renumber labels to 0..C-1 in order of first appearance; returns C.
uint32_t compact_labels(std::vector<uint32_t>& labels);

struct UnionFind {
  std::vector<uint32_t> parent;
  explicit UnionFind(uint32_t n) : parent(n) {
    for (uint32_t i = 0; i < n; ++i) parent[i] = i;
  }
  uint32_t find(uint32_t x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }
  bool unite(uint32_t a, uint32_t b) {
    a = find(a);
    b = find(b);
    if (a == b) return false;
    if (a < b) parent[b] = a;
    else parent[a] = b;
    return true;
  }
};

}  // namespace hvax
