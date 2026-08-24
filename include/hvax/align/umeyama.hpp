#pragma once

#include "hvax/types.hpp"

#include <opencv2/core.hpp>

namespace hvax {

// 2x3 similarity mapping 5 landmarks onto the ArcFace 112 template.
cv::Matx23f estimate_norm(const Landmark5& lmk, int image_size = 112);

cv::Mat norm_crop(const cv::Mat& bgr, const Landmark5& lmk, int image_size = 112);

inline const Landmark5& arcface_dst() {
  static const Landmark5 dst = [] {
    Landmark5 d;
    d.xy[0] = {38.2946f, 51.6963f};
    d.xy[1] = {73.5318f, 51.5014f};
    d.xy[2] = {56.0252f, 71.7366f};
    d.xy[3] = {41.5493f, 92.3655f};
    d.xy[4] = {70.7299f, 92.2041f};
    return d;
  }();
  return dst;
}

}  // namespace hvax
