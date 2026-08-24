#pragma once

#include "hvax/infer/ort.hpp"
#include "hvax/types.hpp"

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace hvax {

class ArcFace {
 public:
  ArcFace(OrtContext& ctx, const std::string& model_path);

  Embedding embed(const cv::Mat& aligned_bgr_112);
  std::vector<Embedding> embed_batch(const std::vector<cv::Mat>& aligned);

  int input_size() const { return input_size_; }

 private:
  Embedding run_blob(std::vector<float>& blob, int n);

  std::unique_ptr<Ort::Session> session_;
  SessionIo io_;
  OrtContext* ctx_ = nullptr;
  int input_size_ = 112;
  bool dynamic_batch_ = false;
};

void l2_normalize(float* v, int n = kDim);
float dot512(const float* a, const float* b);

}  // namespace hvax
