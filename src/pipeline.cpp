#include "hvax/pipeline.hpp"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

namespace hvax {

namespace {

std::string coreml_model_namespace(const std::string& models_dir, int det_size) {
  namespace fs = std::filesystem;
  std::ostringstream identity;
  std::error_code error;
  identity << "static-detector-v1:" << det_size << '\n';
  for (const char* name : {"det_10g.onnx", "w600k_r50.onnx"}) {
    const fs::path path = fs::absolute(fs::path(models_dir) / name, error);
    if (error) throw std::runtime_error("cannot resolve model path: " + error.message());
    const auto size = fs::file_size(path, error);
    if (error) throw std::runtime_error("cannot stat model " + path.string() + ": " + error.message());
    const auto modified = fs::last_write_time(path, error);
    if (error) throw std::runtime_error("cannot stat model " + path.string() + ": " + error.message());
    identity << path.string() << ':' << size << ':'
             << static_cast<int64_t>(modified.time_since_epoch().count()) << '\n';
  }
  uint64_t hash = 14695981039346656037ULL;
  for (const unsigned char byte : identity.str()) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  std::ostringstream encoded;
  encoded << std::hex << std::setfill('0') << std::setw(16) << hash;
  return encoded.str();
}

}  // namespace

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

Pipeline::Pipeline(const std::string& models_dir, int det_size, float det_thresh, float nms_thresh, int ort_threads,
                   bool cuda, int cuda_device)
    : Pipeline(models_dir, det_size, det_thresh, nms_thresh,
               InferenceOptions{.intra_threads = ort_threads,
                                .expected_concurrency = 1,
                                .provider = cuda ? InferenceProvider::cuda : InferenceProvider::cpu,
                                .device_id = cuda_device}) {}

Pipeline::Pipeline(const std::string& models_dir, int det_size, float det_thresh, float nms_thresh,
                   InferenceOptions inference) {
  if (inference.provider == InferenceProvider::coreml && !inference.coreml_cache_dir.empty()) {
    inference.coreml_cache_dir =
        (std::filesystem::path(inference.coreml_cache_dir) / coreml_model_namespace(models_dir, det_size)).string();
  }
  InferenceOptions detector_inference = inference;
  InferenceOptions recognizer_inference = inference;
  const bool coreml = inference.provider == InferenceProvider::coreml;
  const bool hybrid_coreml = coreml &&
                             inference.coreml_model_format == CoreMlModelFormat::automatic &&
                             inference.expected_concurrency <= 1;
  if (hybrid_coreml) {
    detector_inference.coreml_model_format = CoreMlModelFormat::ml_program;
    recognizer_inference.coreml_model_format = CoreMlModelFormat::neural_network;
  } else if (inference.provider == InferenceProvider::coreml &&
             inference.coreml_model_format == CoreMlModelFormat::automatic) {
    detector_inference.coreml_model_format = CoreMlModelFormat::neural_network;
    recognizer_inference.coreml_model_format = CoreMlModelFormat::neural_network;
  }
  if (coreml) {
    // The pinned SCRFD model names both dynamic spatial axes "?". Fixing that
    // symbol specializes them together and lets ORT fold its shape subgraphs
    // before CoreML partitions the graph. ArcFace retains its dynamic batch.
    detector_inference.free_dimension_overrides.emplace_back("?", det_size);
    detector_inference.coreml_require_static_input_shapes = true;
    recognizer_inference.free_dimension_overrides.clear();
    recognizer_inference.coreml_require_static_input_shapes = false;
  }
  ort_ = std::make_unique<OrtContext>(std::move(detector_inference));
  if (coreml) rec_ort_ = std::make_unique<OrtContext>(std::move(recognizer_inference));
  const auto det_path = models_dir + "/det_10g.onnx";
  const auto rec_path = models_dir + "/w600k_r50.onnx";
  det_ = std::make_unique<Scrfd>(*ort_, det_path, det_size, det_thresh, nms_thresh);
  rec_ = std::make_unique<ArcFace>(rec_ort_ ? *rec_ort_ : *ort_, rec_path);
}

std::vector<DetectedFace> Pipeline::run(const cv::Mat& bgr) {
  return run(bgr, nullptr);
}

std::vector<DetectedFace> Pipeline::run(const cv::Mat& bgr, PipelineTimings* timings) {
  const auto detection_start = std::chrono::steady_clock::now();
  auto faces = det_->detect(bgr);
  const auto detection_end = std::chrono::steady_clock::now();
  if (timings) {
    *timings = {};
    timings->detection_ms =
        std::chrono::duration<double, std::milli>(detection_end - detection_start).count();
  }
  if (faces.empty()) return faces;
  const auto alignment_start = detection_end;
  std::vector<cv::Mat> crops;
  crops.reserve(faces.size());
  for (auto& f : faces) crops.push_back(norm_crop(bgr, f.kps, rec_->input_size()));
  const auto alignment_end = std::chrono::steady_clock::now();
  auto embs = rec_->embed_batch(crops);
  const auto embedding_end = std::chrono::steady_clock::now();
  if (timings) {
    timings->alignment_ms =
        std::chrono::duration<double, std::milli>(alignment_end - alignment_start).count();
    timings->embedding_ms =
        std::chrono::duration<double, std::milli>(embedding_end - alignment_end).count();
  }
  for (size_t i = 0; i < faces.size() && i < embs.size(); ++i) faces[i].embedding = embs[i];
  return faces;
}

}  // namespace hvax
