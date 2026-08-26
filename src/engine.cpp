#include "hvax/engine.hpp"

#include <spdlog/spdlog.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <sstream>

#include "hvax/embed/arcface.hpp"
#include "hvax/util/sha256.hpp"
#include "hvax/util/time.hpp"

namespace hvax {

Engine::Engine(Config cfg) : cfg_(std::move(cfg)) { gallery_ = std::make_unique<Gallery>(cfg_); }

Engine::~Engine() {
  try {
    if (gallery_) gallery_->flush();
  } catch (...) {
  }
}

bool Engine::confirm_soft(int64_t image_id, const std::vector<DetectedFace>& faces) const {
  auto existing = gallery_->faces_of(image_id);
  if (faces.empty() || existing.empty()) return false;
  for (const auto& nf : faces) {
    bool ok = false;
    for (const auto& of : existing) {
      Embedding e{};
      if (!gallery_->face_embedding(of.face_id, e)) continue;
      if (dot512(nf.embedding.data(), e.data()) >= kSoftConfirmCosine) {
        ok = true;
        break;
      }
    }
    if (!ok) return false;
  }
  return true;
}

Pipeline& Engine::pipeline() {
  std::lock_guard lock(pipeline_mu_);
  if (!pipe_) {
    spdlog::warn(
        "InsightFace buffalo_l weights are licensed for non-commercial "
        "research. "
        "See insightface.ai — this binary is MIT, the models are not.");
    pipe_ = std::make_unique<Pipeline>(cfg_.models_dir, cfg_.det_size, cfg_.det_thresh, cfg_.nms_thresh,
                                       cfg_.ort_intra_threads);
  }
  return *pipe_;
}

IngestResult Engine::persist_ingest(std::span<const uint8_t> bytes, const cv::Mat& img,
                                    const std::vector<DetectedFace>& faces) {
  std::lock_guard ingest_lock(ingest_mu_);
  IngestResult r;
  const auto sha = sha256_bytes(bytes);
  r.sha256 = sha;

  if (auto hit = gallery_->find_by_sha(sha)) {
    r.status = IngestStatus::duplicate;
    r.duplicate = true;
    r.duplicate_kind = "sha256";
    r.master_replaced = false;
    r.image_id = hit->image_id;
    r.width = hit->width;
    r.height = hit->height;
    r.faces = gallery_->faces_of(hit->image_id);
    metrics_.ingest_dup_sha.fetch_add(1, std::memory_order_relaxed);
    return r;
  }

  r.width = img.cols;
  r.height = img.rows;
  const Mime mime = sniff_mime(bytes);
  const auto ph = hash_image(img);
  const uint64_t new_px = static_cast<uint64_t>(img.cols) * static_cast<uint64_t>(img.rows);

  std::optional<DedupHit> dup;
  if (cfg_.dedup == DedupMode::perceptual) dup = gallery_->find_perceptual(ph.phash, ph.dhash);

  auto fill_from = [&](int64_t id) {
    auto im = gallery_->image(id);
    r.image_id = id;
    r.width = im.width;
    r.height = im.height;
    r.sha256 = im.sha256;
    r.faces = gallery_->faces_of(id);
  };

  if (dup && dup->hard) {
    auto im = gallery_->image(dup->image_id);
    const uint64_t old_px = static_cast<uint64_t>(im.width) * static_cast<uint64_t>(im.height);
    if (new_px <= old_px) {
      r.status = IngestStatus::duplicate;
      r.duplicate = true;
      r.duplicate_kind = "perceptual";
      r.master_replaced = false;
      fill_from(dup->image_id);
      metrics_.ingest_dup_phash.fetch_add(1, std::memory_order_relaxed);
      return r;
    }
    if (faces.empty()) {
      spdlog::warn("perceptual upgrade of image {} had 0 faces; keeping old master", dup->image_id);
      r.status = IngestStatus::duplicate;
      r.duplicate = true;
      r.duplicate_kind = "perceptual";
      r.master_replaced = false;
      fill_from(dup->image_id);
      return r;
    }
    PreviousMaster prev{im.width, im.height, im.sha256};
    r.faces = gallery_->upgrade(dup->image_id, bytes, img, sha, mime, ph, faces);
    r.status = IngestStatus::upgraded;
    r.duplicate = true;
    r.duplicate_kind = "perceptual";
    r.master_replaced = true;
    r.previous = prev;
    r.image_id = dup->image_id;
    r.width = img.cols;
    r.height = img.rows;
    r.sha256 = sha;
    metrics_.master_replace.fetch_add(1, std::memory_order_relaxed);
    return r;
  }

  if (dup && !dup->hard) {
    if (confirm_soft(dup->image_id, faces)) {
      auto im = gallery_->image(dup->image_id);
      const uint64_t old_px = static_cast<uint64_t>(im.width) * static_cast<uint64_t>(im.height);
      if (new_px <= old_px || faces.empty()) {
        r.status = IngestStatus::duplicate;
        r.duplicate = true;
        r.duplicate_kind = "perceptual";
        fill_from(dup->image_id);
        metrics_.ingest_dup_phash.fetch_add(1, std::memory_order_relaxed);
        return r;
      }
      PreviousMaster prev{im.width, im.height, im.sha256};
      r.faces = gallery_->upgrade(dup->image_id, bytes, img, sha, mime, ph, faces);
      r.status = IngestStatus::upgraded;
      r.duplicate = true;
      r.duplicate_kind = "perceptual";
      r.master_replaced = true;
      r.previous = prev;
      r.image_id = dup->image_id;
      r.width = img.cols;
      r.height = img.rows;
      r.sha256 = sha;
      metrics_.master_replace.fetch_add(1, std::memory_order_relaxed);
      return r;
    }
  }

  if (faces.empty()) {
    r.status = IngestStatus::ignored_no_face;
    metrics_.ingest_ignored.fetch_add(1, std::memory_order_relaxed);
    return r;
  }

  r.image_id = gallery_->insert(bytes, img, sha, mime, ph, faces);
  r.status = IngestStatus::stored;
  r.duplicate = false;
  r.master_replaced = false;
  r.width = img.cols;
  r.height = img.rows;
  r.sha256 = sha;
  r.faces = gallery_->faces_of(r.image_id);
  return r;
}

IngestResult Engine::ingest(std::span<const uint8_t> bytes) {
  metrics_.ingest_total.fetch_add(1, std::memory_order_relaxed);
  IngestResult r;
  if (bytes.empty() || bytes.size() > cfg_.max_upload) {
    r.status = IngestStatus::bad_image;
    return r;
  }
  const auto sha = sha256_bytes(bytes);
  if (auto hit = gallery_->find_by_sha(sha)) {
    r.status = IngestStatus::duplicate;
    r.duplicate = true;
    r.duplicate_kind = "sha256";
    r.image_id = hit->image_id;
    r.width = hit->width;
    r.height = hit->height;
    r.sha256 = hit->sha256;
    r.faces = gallery_->faces_of(hit->image_id);
    metrics_.ingest_dup_sha.fetch_add(1, std::memory_order_relaxed);
    return r;
  }
  cv::Mat img = decode_image(bytes, cfg_.max_pixels);
  if (img.empty()) {
    r.status = IngestStatus::bad_image;
    return r;
  }
  auto faces = pipeline().run(img);
  return persist_ingest(bytes, img, faces);
}

IngestResult Engine::ingest_processed(std::span<const uint8_t> bytes, const cv::Mat& bgr,
                                      const std::vector<DetectedFace>& faces) {
  metrics_.ingest_total.fetch_add(1, std::memory_order_relaxed);
  IngestResult r;
  if (bytes.empty() || bytes.size() > cfg_.max_upload || bgr.empty()) {
    r.status = IngestStatus::bad_image;
    return r;
  }
  return persist_ingest(bytes, bgr, faces);
}

IngestCheckResult Engine::check_ingest(const std::array<uint8_t, 32>& sha, uint64_t phash, uint64_t dhash, int width,
                                       int height) const {
  IngestCheckResult result;
  auto duplicate = gallery_->find_by_sha(sha);
  if (duplicate) {
    result.duplicate = true;
    result.process_required = false;
    result.duplicate_kind = "sha256";
    result.image_id = duplicate->image_id;
    result.width = duplicate->width;
    result.height = duplicate->height;
    return result;
  }

  if (cfg_.dedup != DedupMode::perceptual || width <= 0 || height <= 0) return result;
  const auto hit = gallery_->find_perceptual(phash, dhash);
  if (!hit || !hit->hard) return result;

  const auto existing = gallery_->image(hit->image_id);
  const uint64_t incoming_pixels = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
  const uint64_t existing_pixels = static_cast<uint64_t>(existing.width) * static_cast<uint64_t>(existing.height);
  if (incoming_pixels > existing_pixels) return result;

  result.duplicate = true;
  result.process_required = false;
  result.duplicate_kind = "perceptual";
  result.image_id = existing.image_id;
  result.width = existing.width;
  result.height = existing.height;
  return result;
}

std::vector<DetectedFace> Engine::debug_once(const cv::Mat& bgr) { return pipeline().run(bgr); }

std::vector<Hit> Engine::query_embedding(std::span<const float> vec, int k, float min_score) {
  metrics_.query_emb.fetch_add(1, std::memory_order_relaxed);
  auto t0 = std::chrono::steady_clock::now();
  if (vec.size() != static_cast<size_t>(kDim)) return {};
  Embedding q{};
  std::copy(vec.begin(), vec.end(), q.begin());
  l2_normalize(q.data());
  if (k <= 0) k = cfg_.default_k;
  auto hits = gallery_->search(q.data(), k, min_score);
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
  metrics_.query_us_sum.fetch_add(static_cast<uint64_t>(us), std::memory_order_relaxed);
  return hits;
}

std::vector<std::vector<Hit>> Engine::query_embedding_batch(std::span<const float> vecs, int nq, int k,
                                                            float min_score) {
  if (nq <= 0) return {};
  std::vector<float> q(static_cast<size_t>(nq * kDim));
  std::memcpy(q.data(), vecs.data(), static_cast<size_t>(nq * kDim) * sizeof(float));
  for (int i = 0; i < nq; ++i) l2_normalize(q.data() + i * kDim);
  if (k <= 0) k = cfg_.default_k;
  metrics_.query_emb.fetch_add(static_cast<uint64_t>(nq), std::memory_order_relaxed);
  return gallery_->search_batch(q.data(), nq, k, min_score);
}

std::vector<std::pair<DetectedFace, std::vector<Hit>>> Engine::query_image(std::span<const uint8_t> bytes, int k,
                                                                           float min_score) {
  metrics_.query_img.fetch_add(1, std::memory_order_relaxed);
  cv::Mat img = decode_image(bytes, cfg_.max_pixels);
  if (img.empty()) return {};
  auto faces = pipeline().run(img);
  std::vector<std::pair<DetectedFace, std::vector<Hit>>> out;
  out.reserve(faces.size());
  if (k <= 0) k = cfg_.default_k;
  for (auto& f : faces) {
    auto hits = gallery_->search(f.embedding.data(), k, min_score);
    out.emplace_back(std::move(f), std::move(hits));
  }
  return out;
}

std::filesystem::path Engine::image_file(int64_t id) const {
  auto im = gallery_->image(id);
  return gallery_->image_path(im.sha256);
}

ImageView Engine::get_image(const std::array<uint8_t, 32>& sha) const {
  const auto image = gallery_->find_by_sha(sha);
  if (!image) throw std::runtime_error("no image");
  return gallery_->image(image->image_id);
}

std::filesystem::path Engine::image_file(const std::array<uint8_t, 32>& sha) const {
  const auto image = get_image(sha);
  return gallery_->image_path(image.sha256);
}

std::string Engine::image_mime(int64_t id) const {
  auto im = gallery_->image(id);
  return gallery_->mime_string(im.mime);
}

std::string Engine::image_mime(const std::array<uint8_t, 32>& sha) const {
  return gallery_->mime_string(get_image(sha).mime);
}

bool Engine::delete_image(const std::array<uint8_t, 32>& sha) {
  const auto image = gallery_->find_by_sha(sha);
  return image && gallery_->remove_image(image->image_id);
}

std::string Engine::prometheus() const {
  std::ostringstream o;
  o << "hvax_ingest_total " << metrics_.ingest_total.load() << "\n";
  o << "hvax_ingest_ignored_total " << metrics_.ingest_ignored.load() << "\n";
  o << "hvax_ingest_duplicate_total{kind=\"sha256\"} " << metrics_.ingest_dup_sha.load() << "\n";
  o << "hvax_ingest_duplicate_total{kind=\"perceptual\"} " << metrics_.ingest_dup_phash.load() << "\n";
  o << "hvax_master_replace_total " << metrics_.master_replace.load() << "\n";
  o << "hvax_query_total{type=\"embedding\"} " << metrics_.query_emb.load() << "\n";
  o << "hvax_query_total{type=\"image\"} " << metrics_.query_img.load() << "\n";
  o << "hvax_query_microseconds_sum " << metrics_.query_us_sum.load() << "\n";
  o << "hvax_faces " << gallery_->live_faces() << "\n";
  o << "hvax_images " << gallery_->live_images() << "\n";
  o << "hvax_hnsw " << (gallery_->hnsw_active() ? 1 : 0) << "\n";
  return o.str();
}

}  // namespace hvax
