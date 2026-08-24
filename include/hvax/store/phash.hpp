#pragma once

#include <cstdint>
#include <utility>

#include <opencv2/core.hpp>

namespace hvax {

struct PerceptualHash {
  uint64_t phash = 0;
  uint64_t dhash = 0;
};

PerceptualHash hash_image(const cv::Mat& bgr);

inline int hamming64(uint64_t a, uint64_t b) {
  return static_cast<int>(__builtin_popcountll(a ^ b));
}

}  // namespace hvax
