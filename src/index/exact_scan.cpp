#include "hvax/index/exact_scan.hpp"

#include "hvax/embed/arcface.hpp"

#include <algorithm>
#include <cmath>
#include <queue>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace hvax {

void quantize_i8(const float* f32, int8_t* i8) {
  for (int i = 0; i < kDim; ++i) {
    float x = f32[i] * kI8Scale;
    if (x > 127.f) x = 127.f;
    if (x < -127.f) x = -127.f;
    i8[i] = static_cast<int8_t>(std::lrintf(x));
  }
}

const char* i8_kernel_name() {
#if defined(__AVX2__)
  return "avx2";
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  return "neon-dotprod";
#elif defined(__ARM_NEON)
  return "neon";
#else
  return "scalar";
#endif
}

int32_t dot512_i8(const int8_t* a, const int8_t* b) {
#if defined(__AVX2__)
  __m256i acc = _mm256_setzero_si256();
  for (int i = 0; i < kDim; i += 16) {
    // Sign-extend 16 int8 lanes to int16 and multiply-add pairs into int32.
    const __m256i va = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)));
    const __m256i vb = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i)));
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va, vb));
  }
  __m128i lo = _mm256_castsi256_si128(acc);
  __m128i hi = _mm256_extracti128_si256(acc, 1);
  lo = _mm_add_epi32(lo, hi);
  lo = _mm_hadd_epi32(lo, lo);
  lo = _mm_hadd_epi32(lo, lo);
  return _mm_cvtsi128_si32(lo);
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
  for (int i = 0; i < kDim; i += 32) {
    acc0 = vdotq_s32(acc0, vld1q_s8(a + i), vld1q_s8(b + i));
    acc1 = vdotq_s32(acc1, vld1q_s8(a + i + 16), vld1q_s8(b + i + 16));
  }
  return vaddvq_s32(vaddq_s32(acc0, acc1));
#elif defined(__ARM_NEON)
  int32x4_t acc = vdupq_n_s32(0);
  for (int i = 0; i < kDim; i += 16) {
    const int8x16_t va = vld1q_s8(a + i);
    const int8x16_t vb = vld1q_s8(b + i);
    const int16x8_t lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
    const int16x8_t hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
    acc = vpadalq_s16(acc, lo);
    acc = vpadalq_s16(acc, hi);
  }
  return vaddvq_s32(acc);
#else
  int32_t acc = 0;
  for (int i = 0; i < kDim; ++i) acc += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
  return acc;
#endif
}

namespace {

struct Item {
  float score;
  uint64_t row;
};

// Keep the k largest: min-heap by score.
struct MinScore {
  bool operator()(const Item& a, const Item& b) const { return a.score > b.score; }
};

template <typename Score>
std::vector<ScanHit> topk_generic(uint64_t n, int k, const uint32_t* face_flags, uint32_t skip_mask,
                                  float min_score, Score score_of) {
  if (k <= 0 || n == 0) return {};
  std::priority_queue<Item, std::vector<Item>, MinScore> heap;
  for (uint64_t i = 0; i < n; ++i) {
    if (face_flags && (face_flags[i] & skip_mask)) continue;
    const float s = score_of(i);
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

}  // namespace

std::vector<ScanHit> exact_topk_f32(const float* rows, uint64_t n, const float* query, int k,
                                    const uint32_t* face_flags, float min_score, uint32_t skip_mask) {
  return topk_generic(n, k, face_flags, skip_mask, min_score, [&](uint64_t i) {
    if ((i + 8) < n) __builtin_prefetch(rows + (i + 8) * kDim, 0, 1);
    return dot512(query, rows + i * static_cast<uint64_t>(kDim));
  });
}

std::vector<ScanHit> exact_topk_i8(const int8_t* rows, uint64_t n, const int8_t* query, int k,
                                   const uint32_t* face_flags, float min_score, uint32_t skip_mask) {
  return topk_generic(n, k, face_flags, skip_mask, min_score, [&](uint64_t i) {
    if ((i + 16) < n) __builtin_prefetch(rows + (i + 16) * kDim, 0, 1);
    return i8_dot_to_score(dot512_i8(query, rows + i * static_cast<uint64_t>(kDim)));
  });
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
