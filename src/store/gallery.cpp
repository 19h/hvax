#include "hvax/store/gallery.hpp"

#include "hvax/embed/arcface.hpp"
#include "hvax/util/hex.hpp"
#include "hvax/util/time.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <unordered_map>

#include <spdlog/spdlog.h>

namespace hvax {
namespace {

void fsync_file(const std::filesystem::path& path, std::span<const uint8_t> bytes) {
  std::filesystem::create_directories(path.parent_path());
  const auto tmp = path.string() + ".tmp";
  int fd = ::open(tmp.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
  if (fd < 0) throw std::runtime_error("open image tmp");
  size_t off = 0;
  while (off < bytes.size()) {
    ssize_t n = ::write(fd, bytes.data() + off, bytes.size() - off);
    if (n <= 0) {
      ::close(fd);
      throw std::runtime_error("write image");
    }
    off += static_cast<size_t>(n);
  }
  ::fsync(fd);
  ::close(fd);
  std::filesystem::rename(tmp, path);
}

FaceSlot make_face_slot(uint64_t face_id, uint64_t image_id, const DetectedFace& f, int64_t now) {
  FaceSlot s;
  s.face_id = face_id;
  s.image_id = image_id;
  s.x1 = f.box.x1;
  s.y1 = f.box.y1;
  s.x2 = f.box.x2;
  s.y2 = f.box.y2;
  s.det_score = f.det_score;
  for (int i = 0; i < 5; ++i) {
    s.kps[i * 2] = f.kps.xy[static_cast<size_t>(i)][0];
    s.kps[i * 2 + 1] = f.kps.xy[static_cast<size_t>(i)][1];
  }
  s.flags = 0;
  s.created_at = static_cast<uint64_t>(now);
  s.identity_ref = 0;
  return s;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

Gallery::Gallery(const Config& cfg) : cfg_(cfg), data_dir_(cfg.data_dir) {
  std::filesystem::create_directories(data_dir_ / "images");
  images_.open(data_dir_ / "images.slots", "HVAXIMG1");
  faces_.open(data_dir_ / "faces.slots", "HVAXFCE1");
  embs_.open(data_dir_ / "embeddings.f32", "HVAXEMB1");
  embs_i8_.open(data_dir_ / "embeddings.i8", "HVAXEI81", kI8FileVersion);
  identities_.open(data_dir_ / "identities.slots", "HVAXIDN1");
  centroids_.open(data_dir_ / "centroids.f32", "HVAXCEN1");
  hnsw_.open(data_dir_ / "index.usearch");

  if (embs_.size() != faces_.size()) {
    spdlog::warn("embedding rows {} != face slots {}; truncating to min", embs_.size(), faces_.size());
  }
  if (identities_.size() != centroids_.size()) {
    spdlog::warn("identity rows {} != centroid rows {}; identities will be recomputed", identities_.size(),
                 centroids_.size());
  }
  ensure_i8_version();
  rebuild_maps();
  load_identity_state_locked();

  const uint64_t n = std::min(embs_.size(), faces_.size());
  // --reindex rebuilds the index itself right after construction.
  if (hnsw_.size() == 0 && n > 0 && !cfg_.reindex) {
    hnsw_.rebuild_from(embs_.at(0).v, n, flags_cache_.data());
    hnsw_.save();
  }
  spdlog::info("gallery images={} faces={} emb_rows={} indexed={} identities={} i8_kernel={}", images_.size(),
               faces_.size(), embs_.size(), hnsw_.size(), identity_count(), i8_kernel_name());
}

Gallery::~Gallery() {
  try {
    flush();
  } catch (...) {
  }
}

void Gallery::ensure_i8_version() {
  const uint64_t n = std::min(embs_.size(), faces_.size());
  const bool stale = embs_i8_.version() != kI8FileVersion || embs_i8_.size() < n;
  if (!stale) return;
  spdlog::info("requantizing {} int8 rows (file version {} -> {}, scale {})", n, embs_i8_.version(),
               kI8FileVersion, kI8Scale);
  while (embs_i8_.size() < n) embs_i8_.append(EmbI8{});
  for (uint64_t i = 0; i < n; ++i) quantize_i8(embs_.at(i).v, embs_i8_.at(i).v);
  embs_i8_.set_version(kI8FileVersion);
  embs_i8_.fsync_all();
}

void Gallery::rebuild_maps() {
  sha_to_idx_.clear();
  sha_to_idx_.reserve(static_cast<size_t>(images_.size()));
  flags_cache_.assign(static_cast<size_t>(faces_.size()), kTombstone);
  image_faces_.assign(static_cast<size_t>(images_.size()), {});
  for (uint64_t i = 0; i < images_.size(); ++i) {
    const auto& im = images_.at(i);
    if (!slot_live(im.flags)) continue;
    ShaKey k;
    std::memcpy(k.v.data(), im.sha256, 32);
    sha_to_idx_[k] = i;
  }
  for (uint64_t i = 0; i < faces_.size(); ++i) {
    const auto& f = faces_.at(i);
    flags_cache_[static_cast<size_t>(i)] = f.flags;
    if (slot_live(f.flags)) track_image_face_locked(f.image_id, i);
  }
}

void Gallery::track_image_face_locked(uint64_t image_id, uint64_t row) {
  if (image_id == 0 || image_id > image_faces_.size()) return;
  image_faces_[static_cast<size_t>(image_id - 1)].push_back(static_cast<uint32_t>(row));
}

std::filesystem::path Gallery::image_path(const std::array<uint8_t, 32>& sha) const {
  const std::string hex = to_hex(sha);
  return data_dir_ / "images" / hex.substr(0, 2) / hex;
}

std::string Gallery::mime_string(Mime m) const {
  switch (m) {
    case Mime::jpeg:
      return "image/jpeg";
    case Mime::png:
      return "image/png";
    case Mime::webp:
      return "image/webp";
    default:
      return "application/octet-stream";
  }
}

void Gallery::write_master(const std::array<uint8_t, 32>& sha, std::span<const uint8_t> bytes) {
  fsync_file(image_path(sha), bytes);
}

void Gallery::unlink_master(const std::array<uint8_t, 32>& sha) {
  std::error_code ec;
  std::filesystem::remove(image_path(sha), ec);
}

std::optional<ImageView> Gallery::find_by_sha(const std::array<uint8_t, 32>& sha) const {
  std::shared_lock lock(mu_);
  ShaKey k;
  k.v = sha;
  auto it = sha_to_idx_.find(k);
  if (it == sha_to_idx_.end()) return std::nullopt;
  return image_from_slot(images_.at(it->second));
}

std::optional<DedupHit> Gallery::find_perceptual(uint64_t phash, uint64_t dhash) const {
  std::shared_lock lock(mu_);
  std::optional<DedupHit> hard;
  std::optional<DedupHit> soft;
  int best_soft = 1e9;
  for (uint64_t i = 0; i < images_.size(); ++i) {
    const auto& im = images_.at(i);
    if (!slot_live(im.flags)) continue;
    const int pd = hamming64(phash, im.phash);
    const int dd = hamming64(dhash, im.dhash);
    const bool ph_ok = pd <= cfg_.phash_threshold;
    const bool dh_ok = dd <= cfg_.dhash_threshold;
    if (ph_ok && dh_ok) {
      DedupHit h;
      h.image_id = static_cast<int64_t>(im.image_id);
      h.ph_dist = pd;
      h.dh_dist = dd;
      h.hard = true;
      if (!hard || pd + dd < hard->ph_dist + hard->dh_dist) hard = h;
    } else if (ph_ok || dh_ok) {
      const int s = pd + dd;
      if (s < best_soft) {
        best_soft = s;
        DedupHit h;
        h.image_id = static_cast<int64_t>(im.image_id);
        h.ph_dist = pd;
        h.dh_dist = dd;
        h.hard = false;
        soft = h;
      }
    }
  }
  if (hard) return hard;
  return soft;
}

ImageView Gallery::image(int64_t image_id) const {
  std::shared_lock lock(mu_);
  if (image_id <= 0 || static_cast<uint64_t>(image_id) > images_.size()) throw std::runtime_error("no image");
  const auto& s = images_.at(static_cast<uint64_t>(image_id - 1));
  if (!slot_live(s.flags) || static_cast<int64_t>(s.image_id) != image_id) throw std::runtime_error("no image");
  auto v = image_from_slot(s);
  const auto& rows = image_faces_[static_cast<size_t>(image_id - 1)];
  v.face_ids.reserve(rows.size());
  for (uint32_t r : rows) v.face_ids.push_back(static_cast<int64_t>(faces_.at(r).face_id));
  return v;
}

std::vector<FaceView> Gallery::faces_of(int64_t image_id) const {
  std::shared_lock lock(mu_);
  std::vector<FaceView> out;
  if (image_id <= 0 || static_cast<uint64_t>(image_id) > image_faces_.size()) return out;
  for (uint32_t r : image_faces_[static_cast<size_t>(image_id - 1)]) out.push_back(face_from_slot(faces_.at(r)));
  return out;
}

std::vector<int64_t> Gallery::identities_of(int64_t image_id) const {
  std::shared_lock lock(mu_);
  std::vector<int64_t> out;
  if (image_id <= 0 || static_cast<uint64_t>(image_id) > image_faces_.size()) return out;
  for (uint32_t r : image_faces_[static_cast<size_t>(image_id - 1)]) {
    const int64_t id = identity_of(faces_.at(r));
    if (id >= 0 && std::find(out.begin(), out.end(), id) == out.end()) out.push_back(id);
  }
  return out;
}

FaceView Gallery::face(int64_t face_id) const {
  std::shared_lock lock(mu_);
  if (face_id < 0 || static_cast<uint64_t>(face_id) >= faces_.size()) throw std::runtime_error("no face");
  const auto& s = faces_.at(static_cast<uint64_t>(face_id));
  if (!slot_live(s.flags)) throw std::runtime_error("no face");
  return face_from_slot(s);
}

bool Gallery::face_embedding(int64_t face_id, Embedding& out) const {
  std::shared_lock lock(mu_);
  if (face_id < 0 || static_cast<uint64_t>(face_id) >= embs_.size()) return false;
  if (!slot_live(faces_.at(static_cast<uint64_t>(face_id)).flags)) return false;
  std::memcpy(out.data(), embs_.at(static_cast<uint64_t>(face_id)).v, sizeof(float) * kDim);
  return true;
}

uint64_t Gallery::pixels_of(int64_t image_id) const {
  const auto& s = images_.at(static_cast<uint64_t>(image_id - 1));
  return static_cast<uint64_t>(s.width) * static_cast<uint64_t>(s.height);
}

bool Gallery::passes_gate(const DetectedFace& f) const {
  return face_size_px(f.box) >= static_cast<float>(cfg_.index_min_face_px) && f.det_score >= cfg_.index_min_det;
}

bool Gallery::passes_gate(const FaceSlot& s) const {
  const BBox b{s.x1, s.y1, s.x2, s.y2};
  return face_size_px(b) >= static_cast<float>(cfg_.index_min_face_px) && s.det_score >= cfg_.index_min_det;
}

void Gallery::append_face_locked(uint64_t image_id, const DetectedFace& f, int64_t now) {
  const uint64_t row = faces_.size();
  FaceSlot fs = make_face_slot(row, image_id, f, now);
  if (!passes_gate(f)) fs.flags |= kLowQuality;
  faces_.append(fs);
  EmbF32 e{};
  std::memcpy(e.v, f.embedding.data(), sizeof(float) * kDim);
  embs_.append(e);
  EmbI8 q{};
  quantize_i8(e.v, q.v);
  embs_i8_.append(q);
  flags_cache_.push_back(fs.flags);
  track_image_face_locked(image_id, row);
  if (face_indexable(fs.flags)) {
    hnsw_.add(row, e.v);
    try_join_identity_locked(row);
  }
}

void Gallery::tombstone_face_locked(uint64_t row) {
  FaceSlot& f = faces_.at(row);
  if (!slot_live(f.flags)) return;
  unassign_locked(row);
  f.flags |= kTombstone;
  flags_cache_[static_cast<size_t>(row)] |= kTombstone;
  hnsw_.remove(row);
  if (f.image_id > 0 && f.image_id <= image_faces_.size()) {
    auto& rows = image_faces_[static_cast<size_t>(f.image_id - 1)];
    rows.erase(std::remove(rows.begin(), rows.end(), static_cast<uint32_t>(row)), rows.end());
  }
}

int64_t Gallery::insert(std::span<const uint8_t> bytes, const cv::Mat& bgr, const std::array<uint8_t, 32>& sha,
                        Mime mime, PerceptualHash ph, const std::vector<DetectedFace>& faces) {
  std::unique_lock lock(mu_);
  const int64_t now = unix_ms();
  write_master(sha, bytes);

  ImageSlot im;
  im.image_id = images_.size() + 1;
  std::memcpy(im.sha256, sha.data(), 32);
  im.width = static_cast<uint32_t>(bgr.cols);
  im.height = static_cast<uint32_t>(bgr.rows);
  im.nbytes = static_cast<uint32_t>(bytes.size());
  im.mime = static_cast<uint16_t>(mime);
  im.flags = 0;
  im.phash = ph.phash;
  im.dhash = ph.dhash;
  im.created_at = static_cast<uint64_t>(now);
  im.updated_at = static_cast<uint64_t>(now);
  im.nfaces = static_cast<uint32_t>(faces.size());
  const uint64_t idx = images_.append(im);
  image_faces_.emplace_back();

  for (const auto& f : faces) append_face_locked(im.image_id, f, now);

  ShaKey k;
  k.v = sha;
  sha_to_idx_[k] = idx;
  images_.sync_header();
  faces_.sync_header();
  embs_.sync_header();
  embs_i8_.sync_header();
  identities_.sync_header();
  centroids_.sync_header();
  return static_cast<int64_t>(im.image_id);
}

std::vector<FaceView> Gallery::upgrade(int64_t image_id, std::span<const uint8_t> bytes, const cv::Mat& bgr,
                                       const std::array<uint8_t, 32>& sha, Mime mime, PerceptualHash ph,
                                       const std::vector<DetectedFace>& new_faces) {
  std::unique_lock lock(mu_);
  if (image_id <= 0 || static_cast<uint64_t>(image_id) > images_.size()) throw std::runtime_error("no image");
  ImageSlot& im = images_.at(static_cast<uint64_t>(image_id - 1));
  if (!slot_live(im.flags)) throw std::runtime_error("no image");

  std::array<uint8_t, 32> old_sha{};
  std::memcpy(old_sha.data(), im.sha256, 32);

  write_master(sha, bytes);

  ShaKey oldk;
  oldk.v = old_sha;
  sha_to_idx_.erase(oldk);

  std::memcpy(im.sha256, sha.data(), 32);
  im.width = static_cast<uint32_t>(bgr.cols);
  im.height = static_cast<uint32_t>(bgr.rows);
  im.nbytes = static_cast<uint32_t>(bytes.size());
  im.mime = static_cast<uint16_t>(mime);
  im.phash = ph.phash;
  im.dhash = ph.dhash;
  im.updated_at = static_cast<uint64_t>(unix_ms());

  ShaKey newk;
  newk.v = sha;
  sha_to_idx_[newk] = static_cast<uint64_t>(image_id - 1);

  std::vector<uint64_t> old_rows;
  for (uint32_t r : image_faces_[static_cast<size_t>(image_id - 1)]) old_rows.push_back(r);

  std::vector<char> old_used(old_rows.size(), 0);
  std::vector<char> new_used(new_faces.size(), 0);

  struct Pair {
    int ni, oi;
    float s;
  };
  std::vector<Pair> pairs;
  for (int ni = 0; ni < static_cast<int>(new_faces.size()); ++ni) {
    for (int oi = 0; oi < static_cast<int>(old_rows.size()); ++oi) {
      const float* oldv = embs_.at(old_rows[static_cast<size_t>(oi)]).v;
      const float s = dot512(new_faces[static_cast<size_t>(ni)].embedding.data(), oldv);
      if (s >= kRemapCosine) pairs.push_back(Pair{ni, oi, s});
    }
  }
  std::sort(pairs.begin(), pairs.end(), [](auto& a, auto& b) { return a.s > b.s; });

  const int64_t now = unix_ms();
  std::vector<int64_t> touched_identities;
  for (const auto& p : pairs) {
    if (new_used[static_cast<size_t>(p.ni)] || old_used[static_cast<size_t>(p.oi)]) continue;
    new_used[static_cast<size_t>(p.ni)] = 1;
    old_used[static_cast<size_t>(p.oi)] = 1;
    const uint64_t row = old_rows[static_cast<size_t>(p.oi)];
    FaceSlot& fs = faces_.at(row);
    const auto& nf = new_faces[static_cast<size_t>(p.ni)];
    fs.x1 = nf.box.x1;
    fs.y1 = nf.box.y1;
    fs.x2 = nf.box.x2;
    fs.y2 = nf.box.y2;
    fs.det_score = nf.det_score;
    for (int i = 0; i < 5; ++i) {
      fs.kps[i * 2] = nf.kps.xy[static_cast<size_t>(i)][0];
      fs.kps[i * 2 + 1] = nf.kps.xy[static_cast<size_t>(i)][1];
    }
    EmbF32 e{};
    std::memcpy(e.v, nf.embedding.data(), sizeof(float) * kDim);
    embs_.at(row) = e;
    EmbI8 q{};
    quantize_i8(e.v, q.v);
    embs_i8_.at(row) = q;
    const bool was_indexable = face_indexable(fs.flags);
    fs.flags = (fs.flags & ~kLowQuality) | (passes_gate(nf) ? 0u : kLowQuality);
    flags_cache_[static_cast<size_t>(row)] = fs.flags;
    const bool now_indexable = face_indexable(fs.flags);
    if (now_indexable) {
      if (was_indexable) hnsw_.update(row, e.v);
      else hnsw_.add(row, e.v);
      const int64_t id = identity_of(fs);
      if (id >= 0) touched_identities.push_back(id);
      else try_join_identity_locked(row);
    } else {
      if (was_indexable) hnsw_.remove(row);
      unassign_locked(row);
    }
  }

  for (int ni = 0; ni < static_cast<int>(new_faces.size()); ++ni) {
    if (new_used[static_cast<size_t>(ni)]) continue;
    append_face_locked(static_cast<uint64_t>(image_id), new_faces[static_cast<size_t>(ni)], now);
  }

  for (int oi = 0; oi < static_cast<int>(old_rows.size()); ++oi) {
    if (old_used[static_cast<size_t>(oi)]) continue;
    tombstone_face_locked(old_rows[static_cast<size_t>(oi)]);
  }
  // Remapped faces changed their embeddings: refresh the identities they belong to.
  std::sort(touched_identities.begin(), touched_identities.end());
  touched_identities.erase(std::unique(touched_identities.begin(), touched_identities.end()),
                           touched_identities.end());
  for (int64_t id : touched_identities) recompute_identity_locked(id, now);

  im.nfaces = static_cast<uint32_t>(image_faces_[static_cast<size_t>(image_id - 1)].size());

  unlink_master(old_sha);
  images_.sync_header();
  faces_.sync_header();
  embs_.sync_header();
  embs_i8_.sync_header();
  identities_.sync_header();
  centroids_.sync_header();

  std::vector<FaceView> out;
  for (uint32_t r : image_faces_[static_cast<size_t>(image_id - 1)]) out.push_back(face_from_slot(faces_.at(r)));
  return out;
}

bool Gallery::remove_image(int64_t image_id) {
  std::unique_lock lock(mu_);
  if (image_id <= 0 || static_cast<uint64_t>(image_id) > images_.size()) return false;
  ImageSlot& im = images_.at(static_cast<uint64_t>(image_id - 1));
  if (!slot_live(im.flags)) return false;
  std::array<uint8_t, 32> sha{};
  std::memcpy(sha.data(), im.sha256, 32);
  im.flags |= kTombstone;
  ShaKey k;
  k.v = sha;
  sha_to_idx_.erase(k);
  const std::vector<uint32_t> rows = image_faces_[static_cast<size_t>(image_id - 1)];
  for (uint32_t r : rows) tombstone_face_locked(r);
  unlink_master(sha);
  images_.sync_header();
  faces_.sync_header();
  identities_.sync_header();
  return true;
}

Hit Gallery::hydrate_one(uint64_t row, float score, bool& ok) const {
  Hit h;
  ok = false;
  if (row >= faces_.size()) return h;
  const auto& s = faces_.at(row);
  if (!slot_live(s.flags)) return h;
  if (s.image_id == 0 || s.image_id > images_.size()) return h;
  const auto& image = images_.at(s.image_id - 1);
  if (!slot_live(image.flags)) return h;
  h.face_id = static_cast<int64_t>(s.face_id);
  h.image_id = static_cast<int64_t>(s.image_id);
  std::memcpy(h.sha256.data(), image.sha256, h.sha256.size());
  h.row = static_cast<int64_t>(row);
  h.score = score;
  h.box = {s.x1, s.y1, s.x2, s.y2};
  h.det_score = s.det_score;
  h.quality = face_quality(s.det_score, h.box);
  h.flags = s.flags;
  h.identity_id = identity_of(s);
  ok = true;
  return h;
}

std::vector<Hit> Gallery::hydrate(const std::vector<ScanHit>& rows) const {
  std::vector<Hit> out;
  out.reserve(rows.size());
  for (const auto& r : rows) {
    bool ok = false;
    Hit h = hydrate_one(r.row, r.score, ok);
    if (ok) out.push_back(h);
  }
  return out;
}

std::vector<ScanHit> Gallery::candidates_locked(const float* query, int want, float min_score,
                                                bool include_low_quality, bool range) const {
  const uint64_t n = std::min(embs_.size(), faces_.size());
  if (n == 0 || want <= 0) return {};
  const float* rows = embs_.at(0).v;
  const uint32_t* flags = flags_cache_.empty() ? nullptr : flags_cache_.data();
  const uint32_t skip = include_low_quality ? kTombstone : (kTombstone | kLowQuality);

  const bool exact = n < cfg_.exact_until || include_low_quality || hnsw_.size() == 0;
  if (exact) {
    if (cfg_.i8_scan && embs_i8_.size() >= n) {
      // int8 first pass with a wider pool, then exact f32 rerank. The
      // quantization error on a cosine is ~0.003, so a 0.02 margin on the
      // floor and a 4x pool keep the reranked top-k exact in practice.
      int8_t q8[kDim];
      quantize_i8(query, q8);
      const int pool = std::max(want * 4, want + 32);
      auto cand = exact_topk_i8(embs_i8_.at(0).v, n, q8, pool, flags, min_score - 0.02f, skip);
      std::vector<ScanHit> rerank;
      rerank.reserve(cand.size());
      for (const auto& c : cand) {
        const float s = dot512(query, rows + c.row * kDim);
        if (s >= min_score) rerank.push_back(ScanHit{c.row, s});
      }
      std::sort(rerank.begin(), rerank.end(), [](auto& a, auto& b) { return a.score > b.score; });
      if (static_cast<int>(rerank.size()) > want) rerank.resize(static_cast<size_t>(want));
      return rerank;
    }
    return exact_topk_f32(rows, n, query, want, flags, min_score, skip);
  }

  int pool = range ? std::max(64, want) : std::max(want * 8, 64);
  const int max_pool = std::max(pool, cfg_.max_range);
  for (;;) {
    auto approx = hnsw_.search(query, pool);
    std::vector<ScanHit> rerank;
    rerank.reserve(approx.size());
    for (auto& [row, _] : approx) {
      if (row >= n) continue;
      if (flags && (flags[row] & skip)) continue;
      const float s = dot512(query, rows + row * kDim);
      if (s < min_score) continue;
      rerank.push_back(ScanHit{row, s});
    }
    // Range mode: if everything the pool returned clears the floor, there may
    // be more beyond it — widen and retry until the pool has slack.
    const bool saturated = !approx.empty() && rerank.size() == approx.size() &&
                           static_cast<int>(approx.size()) >= pool;
    if (range && saturated && pool < max_pool) {
      pool = std::min(max_pool, pool * 2);
      continue;
    }
    std::sort(rerank.begin(), rerank.end(), [](auto& a, auto& b) { return a.score > b.score; });
    if (static_cast<int>(rerank.size()) > want) rerank.resize(static_cast<size_t>(want));
    return rerank;
  }
}

std::vector<Hit> Gallery::group_hits(std::vector<Hit> hits, int k) const {
  std::vector<Hit> out;
  std::unordered_map<int64_t, size_t> slot;
  for (auto& h : hits) {
    if (h.identity_id < 0) {
      if (static_cast<int>(out.size()) < k) out.push_back(h);
      continue;
    }
    auto it = slot.find(h.identity_id);
    if (it != slot.end()) {
      ++out[it->second].collapsed;
      continue;
    }
    if (static_cast<int>(out.size()) >= k) continue;
    slot[h.identity_id] = out.size();
    out.push_back(h);
  }
  return out;
}

std::vector<Hit> Gallery::search_locked(const float* query, const SearchOptions& opts) const {
  const int k = std::max(opts.k, 1);
  int want = k;
  if (opts.range) want = std::max(cfg_.max_range, k);
  else if (opts.group_by_identity) want = std::max(k * 8, 64);
  auto hits = hydrate(candidates_locked(query, want, opts.min_score, opts.include_low_quality, opts.range));
  if (opts.group_by_identity) return group_hits(std::move(hits), k);
  if (!opts.range && static_cast<int>(hits.size()) > k) hits.resize(static_cast<size_t>(k));
  return hits;
}

std::vector<Hit> Gallery::search(const float* query, const SearchOptions& opts) const {
  std::shared_lock lock(mu_);
  return search_locked(query, opts);
}

std::vector<Hit> Gallery::search(const float* query, int k, float min_score) const {
  SearchOptions o;
  o.k = k;
  o.min_score = min_score;
  return search(query, o);
}

std::vector<std::vector<Hit>> Gallery::search_batch(const float* queries, int nq, const SearchOptions& opts) const {
  std::shared_lock lock(mu_);
  std::vector<std::vector<Hit>> out;
  out.reserve(static_cast<size_t>(nq));
  for (int i = 0; i < nq; ++i) out.push_back(search_locked(queries + i * kDim, opts));
  return out;
}

std::vector<std::vector<Hit>> Gallery::search_batch(const float* queries, int nq, int k, float min_score) const {
  SearchOptions o;
  o.k = k;
  o.min_score = min_score;
  return search_batch(queries, nq, o);
}

ReindexReport Gallery::reindex() {
  std::unique_lock lock(mu_);
  const auto t0 = std::chrono::steady_clock::now();
  const int64_t now = unix_ms();
  ReindexReport r;
  const uint64_t n = std::min(embs_.size(), faces_.size());
  r.rows = n;
  std::vector<int64_t> touched;
  for (uint64_t i = 0; i < n; ++i) {
    FaceSlot& f = faces_.at(i);
    if (!slot_live(f.flags)) continue;
    ++r.live;
    const uint32_t nf = (f.flags & ~kLowQuality) | (passes_gate(f) ? 0u : kLowQuality);
    if ((nf & kLowQuality) && identity_of(f) >= 0) {
      touched.push_back(identity_of(f));
      f.identity_ref = 0;
      ++r.unassigned_by_gate;
    }
    f.flags = nf;
    flags_cache_[static_cast<size_t>(i)] = nf;
    if (nf & kLowQuality) ++r.low_quality;
  }
  while (embs_i8_.size() < n) embs_i8_.append(EmbI8{});
  for (uint64_t i = 0; i < n; ++i) quantize_i8(embs_.at(i).v, embs_i8_.at(i).v);
  embs_i8_.set_version(kI8FileVersion);
  r.requantized = n;
  hnsw_.rebuild_from(n ? embs_.at(0).v : nullptr, n, flags_cache_.data());
  r.indexed = hnsw_.size();
  if (!touched.empty()) {
    // membership lists must be rebuilt before recomputing the touched identities
    load_identity_state_locked();
    std::sort(touched.begin(), touched.end());
    touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
    for (int64_t id : touched) recompute_identity_locked(id, now);
  }
  faces_.fsync_all();
  embs_i8_.fsync_all();
  identities_.fsync_all();
  centroids_.fsync_all();
  hnsw_.save();
  r.ms = ms_since(t0);
  spdlog::info("reindex rows={} live={} low_quality={} indexed={} unassigned_by_gate={} {:.0f}ms", r.rows, r.live,
               r.low_quality, r.indexed, r.unassigned_by_gate, r.ms);
  return r;
}

uint64_t Gallery::live_faces() const {
  std::shared_lock lock(mu_);
  uint64_t n = 0;
  for (uint32_t f : flags_cache_)
    if (slot_live(f)) ++n;
  return n;
}

uint64_t Gallery::live_images() const {
  std::shared_lock lock(mu_);
  uint64_t n = 0;
  for (uint64_t i = 0; i < images_.size(); ++i)
    if (slot_live(images_.at(i).flags)) ++n;
  return n;
}

uint64_t Gallery::indexed_faces() const { return hnsw_.size(); }

uint64_t Gallery::low_quality_faces() const {
  std::shared_lock lock(mu_);
  uint64_t n = 0;
  for (uint32_t f : flags_cache_)
    if (slot_live(f) && (f & kLowQuality)) ++n;
  return n;
}

uint64_t Gallery::unassigned_faces() const {
  std::shared_lock lock(mu_);
  uint64_t n = 0;
  for (uint64_t i = 0; i < faces_.size(); ++i) {
    const auto& f = faces_.at(i);
    if (face_indexable(f.flags) && f.identity_ref == 0) ++n;
  }
  return n;
}

bool Gallery::hnsw_active() const {
  std::shared_lock lock(mu_);
  return std::min(embs_.size(), faces_.size()) >= cfg_.exact_until;
}

void Gallery::flush() {
  std::unique_lock lock(mu_);
  images_.fsync_all();
  faces_.fsync_all();
  embs_.fsync_all();
  embs_i8_.fsync_all();
  identities_.fsync_all();
  centroids_.fsync_all();
  hnsw_.save();
}

const float* Gallery::emb_data() const {
  if (embs_.size() == 0) return nullptr;
  return embs_.at(0).v;
}

std::vector<uint32_t> Gallery::face_flags_copy() const {
  std::shared_lock lock(mu_);
  return flags_cache_;
}

}  // namespace hvax
