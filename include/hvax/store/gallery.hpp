#pragma once

#include "hvax/config.hpp"
#include "hvax/index/exact_scan.hpp"
#include "hvax/index/hnsw_index.hpp"
#include "hvax/store/phash.hpp"
#include "hvax/types.hpp"
#include "hvax/util/mmap_slots.hpp"

#include <array>
#include <atomic>
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

struct SearchOptions {
  int k = 10;
  float min_score = 0.f;
  // Also consider faces flagged kLowQuality. They are not in the HNSW, so this
  // forces the exact tier.
  bool include_low_quality = false;
  // Return every hit >= min_score (capped at Config::max_range) instead of k.
  bool range = false;
  // Collapse hits to the best face per identity; k then counts groups.
  bool group_by_identity = false;
};

enum class IdentitySort { size, recent, id };
enum class FaceSort { score, time };

struct ClusterParams {
  int neighbors = kClusterNeighbors;
  float edge = kClusterEdgeCosine;
  float merge = kIdentityMergeCosine;
  int max_iters = 20;
  int threads = 0;  // 0 = hardware concurrency
  // Clusters with fewer faces stay unassigned: an identity is someone seen at
  // least twice. Pinned identities are exempt.
  uint32_t min_size = 2;
};

struct ClusterReport {
  uint64_t faces_eligible = 0;
  uint64_t faces_assigned = 0;
  uint64_t identities = 0;
  uint64_t identities_pinned = 0;
  uint64_t clusters_before_merge = 0;
  uint64_t merged = 0;
  uint64_t singletons_dropped = 0;  // clusters below min_size left unassigned
  uint64_t largest = 0;
  uint64_t edges = 0;
  int iterations = 0;
  double knn_ms = 0, cw_ms = 0, merge_ms = 0, apply_ms = 0, total_ms = 0;
};

struct ReindexReport {
  uint64_t rows = 0;
  uint64_t live = 0;
  uint64_t low_quality = 0;
  uint64_t indexed = 0;
  uint64_t unassigned_by_gate = 0;
  uint64_t requantized = 0;
  double ms = 0;
};

// Label-free evaluation of the embedding model on this gallery: faces that
// share a photo without overlapping boxes are near-certain different people.
struct ImpostorReport {
  uint64_t images_used = 0;
  uint64_t same_image_pairs = 0;
  uint64_t random_pairs = 0;
  double same_image_mean = 0, same_image_sd = 0;
  double random_mean = 0, random_sd = 0, random_mode = 0;
  // threshold -> fraction of pairs above it
  std::vector<std::pair<float, double>> same_image_far;
  std::vector<std::pair<float, double>> random_above;
  // random_above minus the mirrored left tail: same-identity mass in random pairs
  std::vector<std::pair<float, double>> random_excess;
  float impostor_ceiling = 0;  // smallest listed threshold with zero mirrored impostor mass
};

// All persistent state except raw image bytes. No SQL.
// Embeddings live in a 64-byte-aligned f32 mmap. A search hit is a row index
// into that matrix; FaceSlot[row] is the metadata. One load, no query planner.
//
// Identity layer: FaceSlot::identity_ref points at a row of identities.slots;
// centroids.f32 is row-aligned with it. In-memory member lists make
// identity -> faces and image -> faces O(1).
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
  std::vector<int64_t> identities_of(int64_t image_id) const;
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
  std::vector<Hit> search(const float* query, const SearchOptions& opts) const;
  std::vector<std::vector<Hit>> search_batch(const float* queries, int nq, int k, float min_score) const;
  std::vector<std::vector<Hit>> search_batch(const float* queries, int nq, const SearchOptions& opts) const;

  // ---- identities ----
  std::vector<IdentityHit> search_identities(const float* query, int k, float min_score) const;
  std::optional<IdentityView> identity(int64_t identity_id) const;
  std::vector<IdentityView> list_identities(IdentitySort sort, uint64_t offset, uint64_t limit) const;
  std::vector<FaceView> identity_faces(int64_t identity_id, FaceSort sort, uint64_t offset, uint64_t limit,
                                       std::vector<float>* scores = nullptr) const;
  std::vector<CooccurrenceEntry> cooccurring(int64_t identity_id, uint64_t limit) const;
  std::vector<TimelineBucket> timeline(int64_t identity_id, int64_t bucket_ms) const;
  bool identity_centroid(int64_t identity_id, Embedding& out) const;

  // Curation. All of these pin the identities they touch.
  std::optional<int64_t> merge_identities(const std::vector<int64_t>& ids);
  std::optional<int64_t> split_identity(int64_t identity_id, const std::vector<int64_t>& face_ids);
  bool rename_identity(int64_t identity_id, const std::string& name);
  bool dissolve_identity(int64_t identity_id);
  bool assign_face(int64_t face_id, int64_t identity_id);  // -1 unassigns

  // Batch clustering over every indexable face. Safe to call while serving;
  // ingest continues between kNN chunks. Returns nullopt if a run is already
  // in progress.
  std::optional<ClusterReport> recluster(const ClusterParams& params);
  bool clustering() const { return clustering_.load(); }

  // Recompute quality flags from the current gate, requantize int8 from f32,
  // rebuild the HNSW.
  ReindexReport reindex();

  ImpostorReport impostor_eval(uint64_t max_pairs, uint64_t seed) const;
  std::vector<Hit> search_template(std::span<const Embedding> positives, std::span<const Embedding> negatives,
                                   std::span<const int64_t> excluded_image_ids, int k, float min_score) const;

  uint64_t live_faces() const;
  uint64_t live_images() const;
  uint64_t embedding_rows() const { return embs_.size(); }
  uint64_t indexed_faces() const;
  uint64_t low_quality_faces() const;
  uint64_t unassigned_faces() const;
  uint64_t identity_count() const;
  uint64_t identity_rows() const { return identities_.size(); }
  uint64_t largest_identity() const;
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
  void ensure_i8_version();
  std::vector<Hit> hydrate(const std::vector<ScanHit>& rows) const;
  Hit hydrate_one(uint64_t row, float score, bool& ok) const;
  std::vector<ScanHit> candidates_locked(const float* query, int want, float min_score, bool include_low_quality,
                                         bool range) const;
  std::vector<Hit> search_locked(const float* query, const SearchOptions& opts) const;
  std::vector<Hit> group_hits(std::vector<Hit> hits, int k) const;
  uint64_t pixels_of(int64_t image_id) const;
  bool passes_gate(const DetectedFace& f) const;
  bool passes_gate(const FaceSlot& s) const;
  void append_face_locked(uint64_t image_id, const DetectedFace& f, int64_t now);
  void tombstone_face_locked(uint64_t row);
  void track_image_face_locked(uint64_t image_id, uint64_t row);

  // identity internals; caller holds the unique lock
  void load_identity_state_locked();
  void refresh_centroid_norm_locked(int64_t id);
  void assign_locked(uint64_t row, int64_t id, bool touch_stats);
  void unassign_locked(uint64_t row);
  int64_t new_identity_locked(int64_t now);
  void recompute_identity_locked(int64_t id, int64_t now);
  void try_join_identity_locked(uint64_t row);
  int64_t best_identity_locked(const float* query, float* score_out) const;
  bool identity_live(int64_t id) const;
  void apply_labels_locked(const std::vector<int64_t>& labels, uint64_t n, int64_t now);
  void mark_repeat_images_locked();

  Config cfg_;
  std::filesystem::path data_dir_;
  mutable std::shared_mutex mu_;
  SlotFile<ImageSlot> images_;
  SlotFile<FaceSlot> faces_;
  SlotFile<EmbF32> embs_;
  SlotFile<EmbI8> embs_i8_;
  SlotFile<IdentitySlot> identities_;
  SlotFile<EmbF32> centroids_;  // unnormalized member mean
  HnswIndex hnsw_;
  std::unordered_map<ShaKey, uint64_t, ShaKeyHash> sha_to_idx_;
  std::vector<uint32_t> flags_cache_;                 // per face row, for exact scan
  std::vector<float> centroid_norm_;                  // identity_rows * kDim, L2-normalized
  std::vector<std::vector<uint32_t>> members_;        // identity -> live face rows
  std::vector<std::vector<uint32_t>> image_faces_;    // image idx -> live face rows
  std::atomic<bool> clustering_{false};
  std::mutex cluster_mu_;
};

}  // namespace hvax
