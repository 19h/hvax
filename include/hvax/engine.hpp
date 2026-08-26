#pragma once

#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "hvax/config.hpp"
#include "hvax/pipeline.hpp"
#include "hvax/store/gallery.hpp"

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
  IngestResult ingest_processed(std::span<const uint8_t> bytes, const cv::Mat& bgr,
                                const std::vector<DetectedFace>& faces);
  IngestCheckResult check_ingest(const std::array<uint8_t, 32>& sha, uint64_t phash, uint64_t dhash, int width,
                                 int height) const;
  std::vector<Hit> query_embedding(std::span<const float> vec, int k, float min_score);
  std::vector<std::vector<Hit>> query_embedding_batch(std::span<const float> vecs, int nq, int k, float min_score);
  std::vector<std::pair<DetectedFace, std::vector<Hit>>> query_image(std::span<const uint8_t> bytes, int k,
                                                                     float min_score);

  ImageView get_image(int64_t id) const { return gallery_->image(id); }
  ImageView get_image(const std::array<uint8_t, 32>& sha) const;
  FaceView get_face(int64_t id) const { return gallery_->face(id); }
  bool get_face_embedding(int64_t id, Embedding& e) const { return gallery_->face_embedding(id, e); }
  std::filesystem::path image_file(int64_t id) const;
  std::filesystem::path image_file(const std::array<uint8_t, 32>& sha) const;
  std::string image_mime(int64_t id) const;
  std::string image_mime(const std::array<uint8_t, 32>& sha) const;
  bool delete_image(int64_t id) { return gallery_->remove_image(id); }
  bool delete_image(const std::array<uint8_t, 32>& sha);

  const Config& config() const { return cfg_; }
  Gallery& gallery() { return *gallery_; }
  const Gallery& gallery() const { return *gallery_; }
  Metrics& metrics() { return metrics_; }
  std::string prometheus() const;

  std::vector<DetectedFace> debug_once(const cv::Mat& bgr);

 private:
  bool confirm_soft(int64_t image_id, const std::vector<DetectedFace>& faces) const;
  Pipeline& pipeline();
  IngestResult persist_ingest(std::span<const uint8_t> bytes, const cv::Mat& img,
                              const std::vector<DetectedFace>& faces);

  Config cfg_;
  std::unique_ptr<Pipeline> pipe_;
  std::unique_ptr<Gallery> gallery_;
  std::mutex pipeline_mu_;
  std::mutex ingest_mu_;
  Metrics metrics_;
};

}  // namespace hvax
