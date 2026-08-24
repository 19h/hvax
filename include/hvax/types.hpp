#pragma once

#include <array>
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
inline constexpr float kSoftConfirmCosine = 0.97f;
inline constexpr float kRemapCosine = 0.5f;

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

struct Hit {
  int64_t face_id = 0;
  int64_t image_id = 0;
  int64_t row = 0;
  float score = 0;
  BBox box;
  float det_score = 0;
};

struct FaceView {
  int64_t face_id = 0;
  int64_t image_id = 0;
  int64_t row = 0;
  BBox box;
  float det_score = 0;
  Landmark5 kps;
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
  std::vector<int64_t> face_ids;
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
  uint64_t reserved = 0;
};
static_assert(sizeof(FaceSlot) == 104);

struct alignas(64) EmbF32 {
  float v[kDim];
};
static_assert(sizeof(EmbF32) == kDim * 4);

struct alignas(64) EmbI8 {
  int8_t v[kDim];
};
static_assert(sizeof(EmbI8) == kDim);

inline bool slot_live(uint16_t flags) { return (flags & kTombstone) == 0; }

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
  return v;
}

}  // namespace hvax
