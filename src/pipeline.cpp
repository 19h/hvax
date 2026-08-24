#include "hvax/pipeline.hpp"

#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

namespace hvax {

Mime sniff_mime(std::span<const uint8_t> b) {
  if (b.size() >= 3 && b[0] == 0xff && b[1] == 0xd8 && b[2] == 0xff) return Mime::jpeg;
  if (b.size() >= 8 && b[0] == 0x89 && b[1] == 0x50 && b[2] == 0x4e && b[3] == 0x47) return Mime::png;
  if (b.size() >= 12 && b[0] == 'R' && b[1] == 'I' && b[2] == 'F' && b[3] == 'F' && b[8] == 'W' && b[9] == 'E' &&
      b[10] == 'B' && b[11] == 'P')
    return Mime::webp;
  return Mime::unknown;
}

cv::Mat decode_image(std::span<const uint8_t> bytes, int64_t max_pixels) {
  if (bytes.empty()) return {};
  cv::Mat buf(1, static_cast<int>(bytes.size()), CV_8UC1, const_cast<uint8_t*>(bytes.data()));
  cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
  if (img.empty()) return {};
  const int64_t px = static_cast<int64_t>(img.rows) * img.cols;
  if (px > max_pixels) {
    spdlog::warn("image {}x{} exceeds max pixels", img.cols, img.rows);
    return {};
  }
  return img;
}

Pipeline::Pipeline(const std::string& models_dir, int det_size, float det_thresh, float nms_thresh, int ort_threads) {
  ort_ = std::make_unique<OrtContext>(ort_threads);
  const auto det_path = models_dir + "/det_10g.onnx";
  const auto rec_path = models_dir + "/w600k_r50.onnx";
  det_ = std::make_unique<Scrfd>(*ort_, det_path, det_size, det_thresh, nms_thresh);
  rec_ = std::make_unique<ArcFace>(*ort_, rec_path);
}

std::vector<DetectedFace> Pipeline::run(const cv::Mat& bgr) {
  auto faces = det_->detect(bgr);
  if (faces.empty()) return faces;
  std::vector<cv::Mat> crops;
  crops.reserve(faces.size());
  for (auto& f : faces) crops.push_back(norm_crop(bgr, f.kps, rec_->input_size()));
  auto embs = rec_->embed_batch(crops);
  for (size_t i = 0; i < faces.size() && i < embs.size(); ++i) faces[i].embedding = embs[i];
  return faces;
}

}  // namespace hvax
