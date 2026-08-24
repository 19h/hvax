#include "hvax/store/gallery.hpp"

#include "hvax/embed/arcface.hpp"
#include "hvax/util/hex.hpp"
#include "hvax/util/time.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <fstream>

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
  return s;
}

}  // namespace

Gallery::Gallery(const Config& cfg) : cfg_(cfg), data_dir_(cfg.data_dir) {
  std::filesystem::create_directories(data_dir_ / "images");
  images_.open(data_dir_ / "images.slots", "HVAXIMG1");
  faces_.open(data_dir_ / "faces.slots", "HVAXFCE1");
  embs_.open(data_dir_ / "embeddings.f32", "HVAXEMB1");
  embs_i8_.open(data_dir_ / "embeddings.i8", "HVAXEI81");
  hnsw_.open(data_dir_ / "index.usearch");

  if (embs_.size() != faces_.size()) {
    spdlog::warn("embedding rows {} != face slots {}; truncating to min", embs_.size(), faces_.size());
  }
  rebuild_maps();

  const uint64_t n = std::min(embs_.size(), faces_.size());
  if (hnsw_.size() == 0 && n > 0) {
    std::vector<uint32_t> flags(static_cast<size_t>(n));
    for (uint64_t i = 0; i < n; ++i) flags[static_cast<size_t>(i)] = faces_.at(i).flags;
    hnsw_.rebuild_from(embs_.at(0).v, n, flags.data());
    hnsw_.save();
  }
  spdlog::info("gallery images={} faces={} emb_rows={}", images_.size(), faces_.size(), embs_.size());
}

Gallery::~Gallery() {
  try {
    flush();
  } catch (...) {
  }
}

void Gallery::rebuild_maps() {
  sha_to_idx_.clear();
  sha_to_idx_.reserve(static_cast<size_t>(images_.size()));
  flags_cache_.assign(static_cast<size_t>(faces_.size()), kTombstone);
  for (uint64_t i = 0; i < images_.size(); ++i) {
    const auto& im = images_.at(i);
    if (!slot_live(im.flags)) continue;
    ShaKey k;
    std::memcpy(k.v.data(), im.sha256, 32);
    sha_to_idx_[k] = i;
  }
  for (uint64_t i = 0; i < faces_.size(); ++i) {
    flags_cache_[static_cast<size_t>(i)] = faces_.at(i).flags;
  }
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
  v.face_ids.reserve(s.nfaces);
  for (uint64_t i = 0; i < faces_.size(); ++i) {
    const auto& f = faces_.at(i);
    if (slot_live(f.flags) && static_cast<int64_t>(f.image_id) == image_id) v.face_ids.push_back(static_cast<int64_t>(f.face_id));
  }
  return v;
}

std::vector<FaceView> Gallery::faces_of(int64_t image_id) const {
  std::shared_lock lock(mu_);
  std::vector<FaceView> out;
  for (uint64_t i = 0; i < faces_.size(); ++i) {
    const auto& f = faces_.at(i);
    if (slot_live(f.flags) && static_cast<int64_t>(f.image_id) == image_id) out.push_back(face_from_slot(f));
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

  for (const auto& f : faces) {
    const uint64_t row = faces_.size();
    FaceSlot fs = make_face_slot(row, im.image_id, f, now);
    faces_.append(fs);
    EmbF32 e{};
    std::memcpy(e.v, f.embedding.data(), sizeof(float) * kDim);
    embs_.append(e);
    EmbI8 q{};
    quantize_i8(e.v, q.v);
    embs_i8_.append(q);
    flags_cache_.push_back(0);
    hnsw_.add(row, e.v);
  }

  ShaKey k;
  k.v = sha;
  sha_to_idx_[k] = idx;
  images_.sync_header();
  faces_.sync_header();
  embs_.sync_header();
  embs_i8_.sync_header();
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
  for (uint64_t i = 0; i < faces_.size(); ++i) {
    if (slot_live(faces_.at(i).flags) && static_cast<int64_t>(faces_.at(i).image_id) == image_id)
      old_rows.push_back(i);
  }

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
    if (row < embs_i8_.size()) embs_i8_.at(row) = q;
    hnsw_.update(row, e.v);
  }

  for (int ni = 0; ni < static_cast<int>(new_faces.size()); ++ni) {
    if (new_used[static_cast<size_t>(ni)]) continue;
    const uint64_t row = faces_.size();
    FaceSlot fs = make_face_slot(row, static_cast<uint64_t>(image_id), new_faces[static_cast<size_t>(ni)], now);
    faces_.append(fs);
    EmbF32 e{};
    std::memcpy(e.v, new_faces[static_cast<size_t>(ni)].embedding.data(), sizeof(float) * kDim);
    embs_.append(e);
    EmbI8 q{};
    quantize_i8(e.v, q.v);
    embs_i8_.append(q);
    flags_cache_.push_back(0);
    hnsw_.add(row, e.v);
  }

  for (int oi = 0; oi < static_cast<int>(old_rows.size()); ++oi) {
    if (old_used[static_cast<size_t>(oi)]) continue;
    const uint64_t row = old_rows[static_cast<size_t>(oi)];
    faces_.at(row).flags |= kTombstone;
    flags_cache_[static_cast<size_t>(row)] |= kTombstone;
    hnsw_.remove(row);
  }

  uint32_t live = 0;
  for (uint64_t i = 0; i < faces_.size(); ++i)
    if (slot_live(faces_.at(i).flags) && static_cast<int64_t>(faces_.at(i).image_id) == image_id) ++live;
  im.nfaces = live;

  unlink_master(old_sha);
  images_.sync_header();
  faces_.sync_header();
  embs_.sync_header();

  std::vector<FaceView> out;
  for (uint64_t i = 0; i < faces_.size(); ++i) {
    const auto& f = faces_.at(i);
    if (slot_live(f.flags) && static_cast<int64_t>(f.image_id) == image_id) out.push_back(face_from_slot(f));
  }
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
  for (uint64_t i = 0; i < faces_.size(); ++i) {
    auto& f = faces_.at(i);
    if (slot_live(f.flags) && static_cast<int64_t>(f.image_id) == image_id) {
      f.flags |= kTombstone;
      flags_cache_[static_cast<size_t>(i)] |= kTombstone;
      hnsw_.remove(i);
    }
  }
  unlink_master(sha);
  images_.sync_header();
  faces_.sync_header();
  return true;
}

std::vector<Hit> Gallery::hydrate(const std::vector<ScanHit>& rows) const {
  std::vector<Hit> out;
  out.reserve(rows.size());
  for (const auto& r : rows) {
    if (r.row >= faces_.size()) continue;
    const auto& s = faces_.at(r.row);
    if (!slot_live(s.flags)) continue;
    Hit h;
    h.face_id = static_cast<int64_t>(s.face_id);
    h.image_id = static_cast<int64_t>(s.image_id);
    h.row = static_cast<int64_t>(r.row);
    h.score = r.score;
    h.box = {s.x1, s.y1, s.x2, s.y2};
    h.det_score = s.det_score;
    out.push_back(h);
  }
  return out;
}

std::vector<Hit> Gallery::search(const float* query, int k, float min_score) const {
  std::shared_lock lock(mu_);
  const uint64_t n = std::min(embs_.size(), faces_.size());
  if (n == 0) return {};
  const float* rows = embs_.at(0).v;
  const uint32_t* flags = flags_cache_.empty() ? nullptr : flags_cache_.data();

  std::vector<ScanHit> cand;
  if (n < cfg_.exact_until) {
    cand = exact_topk_f32(rows, n, query, k, flags, min_score);
  } else {
    auto approx = hnsw_.search(query, std::max(k * 8, 64));
    std::vector<ScanHit> rerank;
    rerank.reserve(approx.size());
    for (auto& [row, _] : approx) {
      if (row >= n) continue;
      if (flags && (flags[row] & kTombstone)) continue;
      const float s = dot512(query, rows + row * kDim);
      if (s < min_score) continue;
      rerank.push_back(ScanHit{row, s});
    }
    std::sort(rerank.begin(), rerank.end(), [](auto& a, auto& b) { return a.score > b.score; });
    if (static_cast<int>(rerank.size()) > k) rerank.resize(static_cast<size_t>(k));
    cand = std::move(rerank);
  }
  return hydrate(cand);
}

std::vector<std::vector<Hit>> Gallery::search_batch(const float* queries, int nq, int k, float min_score) const {
  std::vector<std::vector<Hit>> out;
  out.reserve(static_cast<size_t>(nq));
  for (int i = 0; i < nq; ++i) out.push_back(search(queries + i * kDim, k, min_score));
  return out;
}

uint64_t Gallery::live_faces() const {
  std::shared_lock lock(mu_);
  uint64_t n = 0;
  for (uint64_t i = 0; i < faces_.size(); ++i)
    if (slot_live(faces_.at(i).flags)) ++n;
  return n;
}

uint64_t Gallery::live_images() const {
  std::shared_lock lock(mu_);
  uint64_t n = 0;
  for (uint64_t i = 0; i < images_.size(); ++i)
    if (slot_live(images_.at(i).flags)) ++n;
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
