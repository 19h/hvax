#pragma once

#include "hvax/align/umeyama.hpp"
#include "hvax/detect/scrfd.hpp"
#include "hvax/embed/arcface.hpp"
#include "hvax/infer/ort.hpp"
#include "hvax/types.hpp"

#include <memory>
#include <span>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace hvax {

class Pipeline {
 public:
  Pipeline(const std::string& models_dir, int det_size, float det_thresh, float nms_thresh, int ort_threads,
           bool cuda = false, int cuda_device = 0);

  std::vector<DetectedFace> run(const cv::Mat& bgr);

  OrtContext& ort() { return *ort_; }

 private:
  std::unique_ptr<OrtContext> ort_;
  std::unique_ptr<Scrfd> det_;
  std::unique_ptr<ArcFace> rec_;
};

cv::Mat decode_image(std::span<const uint8_t> bytes, int64_t max_pixels);
Mime sniff_mime(std::span<const uint8_t> bytes);

}  // namespace hvax
