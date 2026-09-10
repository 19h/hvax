#pragma once

#include "hvax/types.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace hvax {

struct ScanHit {
  uint64_t row = 0;
  float score = 0;
};

// Exact inner-product top-k over L2-normalized f32 rows. Rows whose flags
// intersect skip_mask are ignored (default: tombstones and low-quality faces).
std::vector<ScanHit> exact_topk_f32(const float* rows, uint64_t n, const float* query, int k,
                                    const uint32_t* face_flags, float min_score,
                                    uint32_t skip_mask = kTombstone | kLowQuality);

std::vector<ScanHit> exact_topk_f32_batch(const float* rows, uint64_t n, const float* queries, int nq, int k,
                                          const uint32_t* face_flags, float min_score);

// int8 candidate scan: same contract as exact_topk_f32 but reads the quantized
// matrix (4x less memory traffic). Scores are approximate (dot / kI8Scale^2);
// callers rerank the candidates against f32 rows.
std::vector<ScanHit> exact_topk_i8(const int8_t* rows, uint64_t n, const int8_t* query, int k,
                                   const uint32_t* face_flags, float min_score,
                                   uint32_t skip_mask = kTombstone | kLowQuality);

// round(f32 * kI8Scale) clamped to [-127, 127].
void quantize_i8(const float* f32, int8_t* i8);
int32_t dot512_i8(const int8_t* a, const int8_t* b);
inline float i8_dot_to_score(int32_t dot) { return static_cast<float>(dot) / (kI8Scale * kI8Scale); }

// Which SIMD path dot512_i8 uses on this build: "neon-dotprod", "neon",
// "avx2", or "scalar". Exposed for tests and /v1/stats.
const char* i8_kernel_name();

}  // namespace hvax
