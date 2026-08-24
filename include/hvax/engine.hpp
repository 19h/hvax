#pragma once

#include "hvax/config.hpp"
#include "hvax/pipeline.hpp"
#include "hvax/store/gallery.hpp"

#include <atomic>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace hvax {

struct Metrics {
  std::atomic<uint64_t> ingest_total{0};
  std::atomic<uint64_t> ingest_ignored{0};
  std::atomic<uint64_t> ingest_dup_sha{0};
  std::atomic<uint64_t> ingest_dup_phash{0};
  std::atomic<uint64_t> master_replace{0};
  std::atomic<uint64_t> query_emb{0};
  std::atomic<uint64_t> query_img{0};
  std::atomic<uint64_t> query_us_sum{0};
};

class Engine {
 public:
  explicit Engine(Config cfg);
  ~Engine();

  IngestResult ingest(std::span<const uint8_t> bytes);
  std::vector<Hit> query_embedding(std::span<const float> vec, int k, float min_score);
  std::vector<std::vector<Hit>> query_embedding_batch(std::span<const float> vecs, int nq, int k, float min_score);
  std::vector<std::pair<DetectedFace, std::vector<Hit>>> query_image(std::span<const uint8_t> bytes, int k,
                                                                     float min_score);

  ImageView get_image(int64_t id) const { return gallery_->image(id); }
  FaceView get_face(int64_t id) const { return gallery_->face(id); }
  bool get_face_embedding(int64_t id, Embedding& e) const { return gallery_->face_embedding(id, e); }
  std::filesystem::path image_file(int64_t id) const;
  std::string image_mime(int64_t id) const;
  bool delete_image(int64_t id) { return gallery_->remove_image(id); }

  const Config& config() const { return cfg_; }
  Gallery& gallery() { return *gallery_; }
  const Gallery& gallery() const { return *gallery_; }
  Metrics& metrics() { return metrics_; }
  std::string prometheus() const;

  std::vector<DetectedFace> debug_once(const cv::Mat& bgr) { return pipe_->run(bgr); }

 private:
  bool confirm_soft(int64_t image_id, const std::vector<DetectedFace>& faces) const;

  Config cfg_;
  std::unique_ptr<Pipeline> pipe_;
  std::unique_ptr<Gallery> gallery_;
  Metrics metrics_;
};

}  // namespace hvax
