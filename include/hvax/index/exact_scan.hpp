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

// Exact inner-product top-k over L2-normalized f32 rows. Skips tombstoned rows.
std::vector<ScanHit> exact_topk_f32(const float* rows, uint64_t n, const float* query, int k,
                                    const uint32_t* face_flags, float min_score);

std::vector<ScanHit> exact_topk_f32_batch(const float* rows, uint64_t n, const float* queries, int nq, int k,
                                          const uint32_t* face_flags, float min_score);

void quantize_i8(const float* f32, int8_t* i8);

}  // namespace hvax
