#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace hvax {

inline constexpr int kDim = 512;
inline constexpr int kAlign = 112;
inline constexpr uint32_t kChunkRows = 65536;
inline constexpr uint32_t kTombstone = 1u;
// Face is stored and reachable through exact/range search, but is excluded from
// the HNSW index and from identity clustering. Set at ingest from the index
// gate (--index-min-face-px / --index-min-det) and recomputed by --reindex.
inline constexpr uint32_t kLowQuality = 2u;
// Image contains the same identity more than once (collage, contact sheet).
inline constexpr uint16_t kImageRepeatIdentity = 2u;
inline constexpr float kSoftConfirmCosine = 0.97f;
inline constexpr float kRemapCosine = 0.5f;

// Measured on the buffalo_l model over a 1.1M-face gallery: the impostor
// distribution has no mass above cosine 0.30, so 0.35 is a safe search floor.
inline constexpr float kDefaultMinScore = 0.35f;
// Identity layer thresholds, from the same measurement.
inline constexpr float kIdentityJoinCosine = 0.55f;   // face joins nearest centroid
inline constexpr float kClusterEdgeCosine = 0.55f;    // kNN edge kept for label propagation
inline constexpr float kIdentityMergeCosine = 0.70f;  // clusters merged by centroid similarity
inline constexpr int kClusterNeighbors = 50;
inline constexpr int kDefaultIndexMinFacePx = 24;
inline constexpr float kDefaultIndexMinDet = 0.65f;

// int8 embeddings are round(f32 * kI8Scale) clamped to [-127, 127]. The
// largest component seen in 1.1M ArcFace vectors is 0.26, so 480 uses the
// whole int8 range without clipping; the old scale of 127 used only ±33.
inline constexpr float kI8Scale = 480.f;
inline constexpr uint32_t kI8FileVersion = 2;

using Embedding = std::array<float, kDim>;

enum class Mime : uint16_t { unknown = 0, jpeg = 1, png = 2, webp = 3 };

enum class DedupMode { off, sha256, perceptual };

enum class IngestStatus {
  stored,
  duplicate,
  upgraded,
  ignored_no_face,
  bad_image,
};

struct BBox {
  float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
};

struct Landmark5 {
  std::array<std::array<float, 2>, 5> xy{};
};

struct DetectedFace {
  BBox box;
  float det_score = 0;
  Landmark5 kps;
  Embedding embedding{};
};

inline float face_size_px(const BBox& b) {
  const float w = std::max(b.x2 - b.x1, 0.f);
  const float h = std::max(b.y2 - b.y1, 0.f);
  return std::sqrt(w * h);
}

// Quality in [0, 1]: detector confidence scaled by how much the crop had to
// be upsampled to reach the 112 px ArcFace input. 16 px -> 0, >= 112 px -> 1.
inline float face_quality(float det_score, const BBox& b) {
  const float size = face_size_px(b);
  if (size <= 16.f) return 0.f;
  const float upscale = std::log2(size / 16.f) / std::log2(static_cast<float>(kAlign) / 16.f);
  return std::clamp(det_score, 0.f, 1.f) * std::clamp(upscale, 0.f, 1.f);
}

struct Hit {
  int64_t face_id = 0;
  int64_t image_id = 0;
  std::array<uint8_t, 32> sha256{};
  int64_t row = 0;
  float score = 0;
  BBox box;
  float det_score = 0;
  float quality = 0;
  uint32_t flags = 0;
  int64_t identity_id = -1;
  bool hidden = false;  // identity is hidden; only admins receive such hits
  // Only set when hits are grouped by identity: further hits in the same
  // identity that were collapsed into this one.
  int64_t collapsed = 0;
};

struct IdentityHit {
  int64_t identity_id = -1;
  float score = 0;
  uint32_t size = 0;
  uint32_t n_images = 0;
  int64_t rep_face_id = -1;
  std::string name;
  std::optional<Hit> best_face;
};

struct FaceView {
  int64_t face_id = 0;
  int64_t image_id = 0;
  int64_t row = 0;
  BBox box;
  float det_score = 0;
  Landmark5 kps;
  float quality = 0;
  uint32_t flags = 0;
  int64_t identity_id = -1;
  bool hidden = false;
  int64_t created_at = 0;
};

struct ImageView {
  int64_t image_id = 0;
  std::array<uint8_t, 32> sha256{};
  int width = 0;
  int height = 0;
  Mime mime = Mime::unknown;
  uint32_t nbytes = 0;
  uint64_t phash = 0;
  uint64_t dhash = 0;
  int64_t created_at = 0;
  int64_t updated_at = 0;
  uint32_t nfaces = 0;
  uint16_t flags = 0;
  std::vector<int64_t> face_ids;
};

struct IdentityView {
  int64_t identity_id = -1;
  uint32_t size = 0;
  uint32_t n_images = 0;
  int64_t rep_face_id = -1;
  uint32_t flags = 0;
  int64_t created_at = 0;
  int64_t updated_at = 0;
  int64_t first_seen = 0;
  int64_t last_seen = 0;
  std::string name;
  float cohesion = 0;  // mean cosine of members to the centroid
};

struct CooccurrenceEntry {
  int64_t identity_id = -1;
  uint32_t shared_images = 0;
};

struct TimelineBucket {
  int64_t start_ms = 0;
  uint32_t faces = 0;
};

struct PreviousMaster {
  int width = 0;
  int height = 0;
  std::array<uint8_t, 32> sha256{};
};

struct IngestResult {
  IngestStatus status = IngestStatus::bad_image;
  int64_t image_id = 0;
  std::array<uint8_t, 32> sha256{};
  int width = 0;
  int height = 0;
  bool duplicate = false;
  std::string duplicate_kind;
  bool master_replaced = false;
  std::optional<PreviousMaster> previous;
  std::vector<FaceView> faces;
};

struct IngestCheckResult {
  bool duplicate = false;
  bool process_required = true;
  std::string duplicate_kind;
  int64_t image_id = 0;
  int width = 0;
  int height = 0;
};

// On-disk packed records. Little-endian. Never put embeddings in here.
struct alignas(8) ImageSlot {
  uint64_t image_id = 0;
  uint8_t sha256[32]{};
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t nbytes = 0;
  uint16_t mime = 0;
  uint16_t flags = 0;
  uint64_t phash = 0;
  uint64_t dhash = 0;
  uint64_t created_at = 0;
  uint64_t updated_at = 0;
  uint32_t nfaces = 0;
  uint32_t pad = 0;
};
static_assert(sizeof(ImageSlot) == 96);

struct alignas(8) FaceSlot {
  uint64_t face_id = 0;
  uint64_t image_id = 0;
  float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
  float det_score = 0;
  float kps[10]{};
  uint32_t flags = 0;
  uint32_t pad = 0;
  uint64_t created_at = 0;
  // identity_id + 1; 0 means unassigned. Galleries written before the
  // identity layer have 0 here, which reads as "unassigned" without migration.
  uint64_t identity_ref = 0;
};
static_assert(sizeof(FaceSlot) == 104);

inline constexpr size_t kIdentityNameBytes = 64;

// One row per identity; row index == identity_id. centroids.f32 is row-aligned
// and stores the unnormalized mean of member embeddings so running updates are
// exact: mean' = (mean * size + x) / (size + 1).
struct alignas(8) IdentitySlot {
  uint64_t identity_id = 0;
  uint32_t size = 0;
  uint32_t n_images = 0;
  uint64_t rep_face_id = 0;
  uint32_t flags = 0;
  uint32_t pad = 0;
  uint64_t created_at = 0;
  uint64_t updated_at = 0;
  uint64_t first_seen = 0;
  uint64_t last_seen = 0;
  char name[kIdentityNameBytes]{};
};
static_assert(sizeof(IdentitySlot) == 128);
// Identity was curated by hand (merge/split/rename). The clustering job keeps
// its member faces fixed and never dissolves it.
inline constexpr uint32_t kIdentityPinned = 2u;
// Hidden by an operator: the identity's faces never surface in searches,
// lookups or crops without the identity-management key. Hiding also pins.
inline constexpr uint32_t kIdentityHidden = 4u;

struct alignas(64) EmbF32 {
  float v[kDim];
};
static_assert(sizeof(EmbF32) == kDim * 4);

struct alignas(64) EmbI8 {
  int8_t v[kDim];
};
static_assert(sizeof(EmbI8) == kDim);

inline bool slot_live(uint32_t flags) { return (flags & kTombstone) == 0; }
inline bool face_indexable(uint32_t flags) { return (flags & (kTombstone | kLowQuality)) == 0; }
inline int64_t identity_of(const FaceSlot& s) { return s.identity_ref == 0 ? -1 : static_cast<int64_t>(s.identity_ref - 1); }

inline FaceView face_from_slot(const FaceSlot& s) {
  FaceView f;
  f.face_id = static_cast<int64_t>(s.face_id);
  f.image_id = static_cast<int64_t>(s.image_id);
  f.row = static_cast<int64_t>(s.face_id);
  f.box = {s.x1, s.y1, s.x2, s.y2};
  f.det_score = s.det_score;
  for (int i = 0; i < 5; ++i) {
    f.kps.xy[static_cast<size_t>(i)][0] = s.kps[i * 2];
    f.kps.xy[static_cast<size_t>(i)][1] = s.kps[i * 2 + 1];
  }
  f.quality = face_quality(s.det_score, f.box);
  f.flags = s.flags;
  f.identity_id = identity_of(s);
  f.created_at = static_cast<int64_t>(s.created_at);
  return f;
}

inline ImageView image_from_slot(const ImageSlot& s) {
  ImageView v;
  v.image_id = static_cast<int64_t>(s.image_id);
  std::memcpy(v.sha256.data(), s.sha256, 32);
  v.width = static_cast<int>(s.width);
  v.height = static_cast<int>(s.height);
  v.mime = static_cast<Mime>(s.mime);
  v.nbytes = s.nbytes;
  v.phash = s.phash;
  v.dhash = s.dhash;
  v.created_at = static_cast<int64_t>(s.created_at);
  v.updated_at = static_cast<int64_t>(s.updated_at);
  v.nfaces = s.nfaces;
  v.flags = s.flags;
  return v;
}

inline IdentityView identity_from_slot(const IdentitySlot& s) {
  IdentityView v;
  v.identity_id = static_cast<int64_t>(s.identity_id);
  v.size = s.size;
  v.n_images = s.n_images;
  v.rep_face_id = static_cast<int64_t>(s.rep_face_id);
  v.flags = s.flags;
  v.created_at = static_cast<int64_t>(s.created_at);
  v.updated_at = static_cast<int64_t>(s.updated_at);
  v.first_seen = static_cast<int64_t>(s.first_seen);
  v.last_seen = static_cast<int64_t>(s.last_seen);
  v.name.assign(s.name, strnlen(s.name, kIdentityNameBytes));
  return v;
}

}  // namespace hvax
