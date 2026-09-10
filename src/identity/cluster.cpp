#include "hvax/identity/cluster.hpp"

#include <algorithm>
#include <numeric>

namespace hvax {

CsrGraph symmetrize_knn(const KnnGraph& g, float edge_min, const std::vector<uint8_t>& eligible) {
  const uint64_t n = g.n;
  const uint64_t k = static_cast<uint64_t>(g.k);
  struct E {
    uint32_t u, v;
    float w;
  };
  std::vector<E> edges;
  edges.reserve(n * k / 4 + 16);
  for (uint64_t i = 0; i < n; ++i) {
    if (!eligible[i]) continue;
    for (uint64_t j = 0; j < k; ++j) {
      const uint32_t v = g.nbr[i * k + j];
      if (v == UINT32_MAX || v == i || v >= n || !eligible[v]) continue;
      const float w = g.sim[i * k + j];
      if (w < edge_min) continue;
      const uint32_t a = std::min<uint32_t>(static_cast<uint32_t>(i), v);
      const uint32_t b = std::max<uint32_t>(static_cast<uint32_t>(i), v);
      edges.push_back(E{a, b, w});
    }
  }
  std::sort(edges.begin(), edges.end(), [](const E& x, const E& y) {
    if (x.u != y.u) return x.u < y.u;
    if (x.v != y.v) return x.v < y.v;
    return x.w > y.w;
  });
  // dedupe (keep max weight), then emit both directions
  std::vector<E> uniq;
  uniq.reserve(edges.size());
  for (const auto& e : edges) {
    if (!uniq.empty() && uniq.back().u == e.u && uniq.back().v == e.v) continue;
    uniq.push_back(e);
  }
  edges.clear();
  edges.shrink_to_fit();

  CsrGraph out;
  out.offsets.assign(n + 1, 0);
  for (const auto& e : uniq) {
    ++out.offsets[e.u + 1];
    ++out.offsets[e.v + 1];
  }
  for (uint64_t i = 0; i < n; ++i) out.offsets[i + 1] += out.offsets[i];
  out.adj.resize(out.offsets[n]);
  out.w.resize(out.offsets[n]);
  std::vector<uint64_t> fill(out.offsets.begin(), out.offsets.end() - 1);
  for (const auto& e : uniq) {
    out.adj[fill[e.u]] = e.v;
    out.w[fill[e.u]++] = e.w;
    out.adj[fill[e.v]] = e.u;
    out.w[fill[e.v]++] = e.w;
  }
  return out;
}

int chinese_whispers(const CsrGraph& g, std::vector<uint32_t>& labels, const std::vector<uint8_t>& fixed,
                     int max_iters, int /*threads*/) {
  // Updates are applied in place in node order (Gauss-Seidel). Synchronous
  // updates oscillate on two-node components (each adopts the other's label
  // forever); in-place updates converge, and a fixed order keeps the result
  // deterministic. One pass over 1M nodes with ~40 edges each is well under
  // a second, so this is not the bottleneck of a recluster run.
  const uint64_t n = g.n();
  if (n == 0) return 0;
  std::vector<std::pair<uint32_t, float>> votes;
  int iters = 0;
  for (; iters < max_iters; ++iters) {
    uint64_t changed = 0;
    for (uint64_t i = 0; i < n; ++i) {
      if (fixed[i]) continue;
      const uint64_t b = g.offsets[i], e = g.offsets[i + 1];
      if (b == e) continue;
      const uint32_t cur = labels[i];
      votes.clear();
      for (uint64_t p = b; p < e; ++p) votes.emplace_back(labels[g.adj[p]], g.w[p]);
      std::sort(votes.begin(), votes.end(), [](auto& x, auto& y) { return x.first < y.first; });
      uint32_t best = cur;
      float best_w = -1.f, cur_w = 0.f;
      for (size_t p = 0; p < votes.size();) {
        const uint32_t lab = votes[p].first;
        float sum = 0.f;
        while (p < votes.size() && votes[p].first == lab) sum += votes[p++].second;
        if (lab == cur) cur_w = sum;
        if (sum > best_w || (sum == best_w && lab < best)) {
          best_w = sum;
          best = lab;
        }
      }
      // hysteresis: only move for a strictly heavier label
      if (best != cur && best_w > cur_w) {
        labels[i] = best;
        ++changed;
      }
    }
    if (changed == 0) {
      ++iters;
      break;
    }
  }
  return iters;
}

uint32_t compact_labels(std::vector<uint32_t>& labels) {
  std::vector<uint32_t> map;
  std::vector<uint32_t> order(labels.size());
  std::iota(order.begin(), order.end(), 0u);
  std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
    return labels[a] != labels[b] ? labels[a] < labels[b] : a < b;
  });
  uint32_t c = 0;
  uint32_t prev = UINT32_MAX;
  std::vector<uint32_t> out(labels.size());
  for (uint32_t idx : order) {
    if (labels[idx] != prev) {
      prev = labels[idx];
      ++c;
    }
    out[idx] = c - 1;
  }
  labels.swap(out);
  return c;
}

}  // namespace hvax
