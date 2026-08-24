#pragma once

#include "hvax/config.hpp"
#include "hvax/index/exact_scan.hpp"
#include "hvax/index/hnsw_index.hpp"
#include "hvax/store/phash.hpp"
#include "hvax/types.hpp"
#include "hvax/util/mmap_slots.hpp"

#include <array>
#include <filesystem>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

namespace hvax {

struct ShaKey {
  std::array<uint8_t, 32> v{};
  bool operator==(const ShaKey& o) const { return v == o.v; }
};

struct ShaKeyHash {
  size_t operator()(const ShaKey& k) const {
    uint64_t a = 0, b = 0, c = 0, d = 0;
    std::memcpy(&a, k.v.data(), 8);
    std::memcpy(&b, k.v.data() + 8, 8);
    std::memcpy(&c, k.v.data() + 16, 8);
    std::memcpy(&d, k.v.data() + 24, 8);
    return static_cast<size_t>(a ^ (b * 0x9e3779b97f4a7c15ull) ^ c ^ (d << 1));
  }
};

struct DedupHit {
  int64_t image_id = 0;
  int ph_dist = 0;
  int dh_dist = 0;
  bool hard = false;
};

// All persistent state except raw image bytes. No SQL.
// Embeddings live in a 64-byte-aligned f32 mmap. A search hit is a row index
// into that matrix; FaceSlot[row] is the metadata. One load, no query planner.
class Gallery {
 public:
  explicit Gallery(const Config& cfg);
  ~Gallery();

  Gallery(const Gallery&) = delete;
  Gallery& operator=(const Gallery&) = delete;

  std::optional<ImageView> find_by_sha(const std::array<uint8_t, 32>& sha) const;
  std::optional<DedupHit> find_perceptual(uint64_t phash, uint64_t dhash) const;

  ImageView image(int64_t image_id) const;
  std::vector<FaceView> faces_of(int64_t image_id) const;
  FaceView face(int64_t face_id) const;
  bool face_embedding(int64_t face_id, Embedding& out) const;
  std::filesystem::path image_path(const std::array<uint8_t, 32>& sha) const;
  std::string mime_string(Mime m) const;

  // Insert a brand-new photograph. Returns image_id.
  int64_t insert(std::span<const uint8_t> bytes, const cv::Mat& bgr, const std::array<uint8_t, 32>& sha,
                 Mime mime, PerceptualHash ph, const std::vector<DetectedFace>& faces);

  // Replace master pixels; remap faces. Returns updated view of faces.
  std::vector<FaceView> upgrade(int64_t image_id, std::span<const uint8_t> bytes, const cv::Mat& bgr,
                                const std::array<uint8_t, 32>& sha, Mime mime, PerceptualHash ph,
                                const std::vector<DetectedFace>& new_faces);

  bool remove_image(int64_t image_id);

  std::vector<Hit> search(const float* query, int k, float min_score) const;
  std::vector<std::vector<Hit>> search_batch(const float* queries, int nq, int k, float min_score) const;

  uint64_t live_faces() const;
  uint64_t live_images() const;
  uint64_t embedding_rows() const { return embs_.size(); }
  bool hnsw_active() const;

  void flush();
  const Config& config() const { return cfg_; }

  // For tests / bench: raw matrix.
  const float* emb_data() const;
  std::vector<uint32_t> face_flags_copy() const;

 private:
  void write_master(const std::array<uint8_t, 32>& sha, std::span<const uint8_t> bytes);
  void unlink_master(const std::array<uint8_t, 32>& sha);
  void rebuild_maps();
  std::vector<Hit> hydrate(const std::vector<ScanHit>& rows) const;
  uint64_t pixels_of(int64_t image_id) const;

  Config cfg_;
  std::filesystem::path data_dir_;
  mutable std::shared_mutex mu_;
  SlotFile<ImageSlot> images_;
  SlotFile<FaceSlot> faces_;
  SlotFile<EmbF32> embs_;
  SlotFile<EmbI8> embs_i8_;
  HnswIndex hnsw_;
  std::unordered_map<ShaKey, uint64_t, ShaKeyHash> sha_to_idx_;
  std::vector<uint32_t> flags_cache_;  // per face row, for exact scan
};

}  // namespace hvax
