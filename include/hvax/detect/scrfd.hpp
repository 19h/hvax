#pragma once

#include "hvax/infer/ort.hpp"
#include "hvax/types.hpp"

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace hvax {

class Scrfd {
 public:
  Scrfd(OrtContext& ctx, const std::string& model_path, int det_size, float det_thresh, float nms_thresh);

  std::vector<DetectedFace> detect(const cv::Mat& bgr);

  int det_size() const { return det_size_; }

 private:
  struct ForwardOut {
    std::vector<float> scores;
    std::vector<BBox> boxes;
    std::vector<Landmark5> kps;
  };

  ForwardOut forward(const cv::Mat& padded_bgr);
  std::vector<int> nms(const std::vector<BBox>& boxes, const std::vector<float>& scores) const;
  const std::vector<std::array<float, 2>>& centers(int height, int width, int stride);

  std::unique_ptr<Ort::Session> session_;
  SessionIo io_;
  OrtContext* ctx_ = nullptr;
  int det_size_ = 640;
  float det_thresh_ = 0.5f;
  float nms_thresh_ = 0.4f;
  bool batched_ = false;
  bool use_kps_ = false;
  int fmc_ = 3;
  int num_anchors_ = 2;
  std::vector<int> strides_{8, 16, 32};
  std::mutex center_cache_mu_;
  std::map<std::tuple<int, int, int>, std::vector<std::array<float, 2>>> center_cache_;
};

// Exposed for tests.
std::vector<int> nms_boxes(const std::vector<BBox>& boxes, const std::vector<float>& scores, float thresh);

}  // namespace hvax
