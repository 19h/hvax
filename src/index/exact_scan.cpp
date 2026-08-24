#include "hvax/index/exact_scan.hpp"

#include "hvax/embed/arcface.hpp"

#include <algorithm>
#include <cmath>
#include <queue>

#include <immintrin.h>

namespace hvax {

void quantize_i8(const float* f32, int8_t* i8) {
  for (int i = 0; i < kDim; ++i) {
    float x = f32[i] * 127.f;
    if (x > 127.f) x = 127.f;
    if (x < -127.f) x = -127.f;
    i8[i] = static_cast<int8_t>(std::lrintf(x));
  }
}

namespace {

struct Item {
  float score;
  uint64_t row;
  bool operator<(const Item& o) const { return score > o.score; }  // min-heap of high scores: wait
};

// Keep the k largest: min-heap by score.
struct MinScore {
  bool operator()(const Item& a, const Item& b) const { return a.score > b.score; }
};

}  // namespace

std::vector<ScanHit> exact_topk_f32(const float* rows, uint64_t n, const float* query, int k,
                                    const uint32_t* face_flags, float min_score) {
  if (k <= 0 || n == 0) return {};
  std::priority_queue<Item, std::vector<Item>, MinScore> heap;
  for (uint64_t i = 0; i < n; ++i) {
    if (face_flags && (face_flags[i] & kTombstone)) continue;
    const float* row = rows + i * static_cast<uint64_t>(kDim);
#if defined(__AVX2__)
    if ((i + 8) < n) {
      _mm_prefetch(reinterpret_cast<const char*>(rows + (i + 8) * kDim), _MM_HINT_T0);
    }
#endif
    const float s = dot512(query, row);
    if (s < min_score) continue;
    if (static_cast<int>(heap.size()) < k) {
      heap.push(Item{s, i});
    } else if (s > heap.top().score) {
      heap.pop();
      heap.push(Item{s, i});
    }
  }
  std::vector<ScanHit> out(heap.size());
  for (int i = static_cast<int>(out.size()) - 1; i >= 0; --i) {
    out[static_cast<size_t>(i)] = ScanHit{heap.top().row, heap.top().score};
    heap.pop();
  }
  std::sort(out.begin(), out.end(), [](auto& a, auto& b) { return a.score > b.score; });
  return out;
}

std::vector<ScanHit> exact_topk_f32_batch(const float* rows, uint64_t n, const float* queries, int nq, int k,
                                          const uint32_t* face_flags, float min_score) {
  // Sequential exact; caller packs per-query. For true GEMM we'd tile; this stays correct.
  std::vector<ScanHit> all;
  all.reserve(static_cast<size_t>(nq * k));
  for (int q = 0; q < nq; ++q) {
    auto part = exact_topk_f32(rows, n, queries + q * kDim, k, face_flags, min_score);
    all.insert(all.end(), part.begin(), part.end());
  }
  return all;
}

}  // namespace hvax
