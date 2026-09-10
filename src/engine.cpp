#include "hvax/engine.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <sstream>

#include "hvax/embed/arcface.hpp"
#include "hvax/util/sha256.hpp"
#include "hvax/util/time.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace hvax {

namespace {

class SemaphorePermit {
 public:
  explicit SemaphorePermit(std::counting_semaphore<256>& semaphore) : semaphore_(semaphore) {
    semaphore_.acquire();
  }
  ~SemaphorePermit() { semaphore_.release(); }

  SemaphorePermit(const SemaphorePermit&) = delete;
  SemaphorePermit& operator=(const SemaphorePermit&) = delete;

 private:
  std::counting_semaphore<256>& semaphore_;
};

}  // namespace

Engine::Engine(Config cfg)
    : cfg_(std::move(cfg)),
      inference_slots_(cfg_.inference.provider == InferenceProvider::coreml
                           ? std::clamp(cfg_.inference.expected_concurrency, 1, 256)
                           : 256) {
  gallery_ = std::make_unique<Gallery>(cfg_);
}

Engine::~Engine() {
  stop_background_cluster();
  try {
    if (gallery_) gallery_->flush();
  } catch (...) {
  }
}

ClusterParams Engine::cluster_params() const {
  ClusterParams p;
  p.neighbors = cfg_.cluster_neighbors;
  p.edge = cfg_.cluster_edge;
  p.merge = cfg_.cluster_merge;
  return p;
}

std::optional<ClusterReport> Engine::recluster() {
  auto t0 = std::chrono::steady_clock::now();
  auto r = gallery_->recluster(cluster_params());
  if (r) {
    metrics_.cluster_runs.fetch_add(1, std::memory_order_relaxed);
    metrics_.cluster_ms_last.store(
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count()),
        std::memory_order_relaxed);
  }
  return r;
}

ReindexReport Engine::reindex() { return gallery_->reindex(); }

ImpostorReport Engine::impostor_eval(uint64_t max_pairs, uint64_t seed) const {
  return gallery_->impostor_eval(max_pairs, seed);
}

void Engine::start_background_cluster() {
  if (cfg_.cluster_interval_s <= 0 || cluster_thread_.joinable()) return;
  cluster_thread_ = std::thread([this] {
    std::unique_lock lock(cluster_cv_mu_);
    while (!cluster_stop_) {
      if (cluster_cv_.wait_for(lock, std::chrono::seconds(cfg_.cluster_interval_s), [this] { return cluster_stop_; }))
        break;
      lock.unlock();
      try {
        recluster();
      } catch (const std::exception& e) {
        spdlog::error("background recluster failed: {}", e.what());
      }
      lock.lock();
    }
  });
}

void Engine::stop_background_cluster() {
  {
    std::lock_guard lock(cluster_cv_mu_);
    cluster_stop_ = true;
  }
  cluster_cv_.notify_all();
  if (cluster_thread_.joinable()) cluster_thread_.join();
}

bool Engine::face_crop(int64_t face_id, const CropOptions& opts, std::vector<uint8_t>& jpeg) const {
  FaceView f;
  std::filesystem::path path;
  try {
    f = gallery_->face(face_id);
    path = image_file(f.image_id);
  } catch (...) {
    return false;
  }
  cv::Mat img = cv::imread(path.string(), cv::IMREAD_COLOR);
  if (img.empty()) return false;
  const float w = f.box.x2 - f.box.x1, h = f.box.y2 - f.box.y1;
  const float side = std::max(w, h) * (1.f + 2.f * std::max(opts.pad, 0.f));
  const float cx = (f.box.x1 + f.box.x2) / 2.f, cy = (f.box.y1 + f.box.y2) / 2.f;
  // Square crop around the box centre; parts outside the image are padded black.
  const int x0 = static_cast<int>(std::lround(cx - side / 2.f)), y0 = static_cast<int>(std::lround(cy - side / 2.f));
  const int s = std::max(1, static_cast<int>(std::lround(side)));
  cv::Mat canvas(s, s, CV_8UC3, cv::Scalar(0, 0, 0));
  const cv::Rect src = cv::Rect(x0, y0, s, s) & cv::Rect(0, 0, img.cols, img.rows);
  if (src.width > 0 && src.height > 0) {
    img(src).copyTo(canvas(cv::Rect(src.x - x0, src.y - y0, src.width, src.height)));
  }
  const int out = std::clamp(opts.size, 16, 1024);
  cv::Mat resized;
  cv::resize(canvas, resized, cv::Size(out, out), 0, 0, s > out ? cv::INTER_AREA : cv::INTER_LINEAR);
  std::vector<uint8_t> buf;
  if (!cv::imencode(".jpg", resized, buf, {cv::IMWRITE_JPEG_QUALITY, std::clamp(opts.jpeg_quality, 30, 100)}))
    return false;
  jpeg.swap(buf);
  metrics_.crops.fetch_add(1, std::memory_order_relaxed);
  return true;
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
                                       cfg_.inference);
  }
  return *pipe_;
}

std::vector<DetectedFace> Engine::run_pipeline(const cv::Mat& bgr) {
  SemaphorePermit permit(inference_slots_);
  return pipeline().run(bgr);
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
  auto faces = run_pipeline(img);
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

std::vector<DetectedFace> Engine::debug_once(const cv::Mat& bgr) { return run_pipeline(bgr); }

std::vector<Hit> Engine::query_embedding(std::span<const float> vec, const SearchOptions& opts_in) {
  metrics_.query_emb.fetch_add(1, std::memory_order_relaxed);
  auto t0 = std::chrono::steady_clock::now();
  if (vec.size() != static_cast<size_t>(kDim)) return {};
  Embedding q{};
  std::copy(vec.begin(), vec.end(), q.begin());
  l2_normalize(q.data());
  SearchOptions opts = opts_in;
  if (opts.k <= 0) opts.k = cfg_.default_k;
  auto hits = gallery_->search(q.data(), opts);
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
  metrics_.query_us_sum.fetch_add(static_cast<uint64_t>(us), std::memory_order_relaxed);
  return hits;
}

std::vector<Hit> Engine::query_embedding(std::span<const float> vec, int k, float min_score) {
  SearchOptions o;
  o.k = k;
  o.min_score = min_score;
  return query_embedding(vec, o);
}

std::vector<std::vector<Hit>> Engine::query_embedding_batch(std::span<const float> vecs, int nq,
                                                            const SearchOptions& opts_in) {
  if (nq <= 0) return {};
  std::vector<float> q(static_cast<size_t>(nq * kDim));
  std::memcpy(q.data(), vecs.data(), static_cast<size_t>(nq * kDim) * sizeof(float));
  for (int i = 0; i < nq; ++i) l2_normalize(q.data() + i * kDim);
  SearchOptions opts = opts_in;
  if (opts.k <= 0) opts.k = cfg_.default_k;
  metrics_.query_emb.fetch_add(static_cast<uint64_t>(nq), std::memory_order_relaxed);
  return gallery_->search_batch(q.data(), nq, opts);
}

std::vector<std::vector<Hit>> Engine::query_embedding_batch(std::span<const float> vecs, int nq, int k,
                                                            float min_score) {
  SearchOptions o;
  o.k = k;
  o.min_score = min_score;
  return query_embedding_batch(vecs, nq, o);
}

std::vector<std::pair<DetectedFace, std::vector<Hit>>> Engine::query_image(std::span<const uint8_t> bytes,
                                                                           const SearchOptions& opts_in,
                                                                           bool detect_only) {
  metrics_.query_img.fetch_add(1, std::memory_order_relaxed);
  cv::Mat img = decode_image(bytes, cfg_.max_pixels);
  if (img.empty()) return {};
  auto faces = run_pipeline(img);
  std::vector<std::pair<DetectedFace, std::vector<Hit>>> out;
  out.reserve(faces.size());
  SearchOptions opts = opts_in;
  if (opts.k <= 0) opts.k = cfg_.default_k;
  for (auto& f : faces) {
    auto hits = detect_only ? std::vector<Hit>{} : gallery_->search(f.embedding.data(), opts);
    out.emplace_back(std::move(f), std::move(hits));
  }
  return out;
}

std::vector<std::pair<DetectedFace, std::vector<Hit>>> Engine::query_image(std::span<const uint8_t> bytes, int k,
                                                                           float min_score, bool detect_only) {
  SearchOptions o;
  o.k = k;
  o.min_score = min_score;
  return query_image(bytes, o, detect_only);
}

std::vector<Hit> Engine::query_template(std::span<const Embedding> positive_embeddings,
                                        std::span<const int64_t> positive_face_ids,
                                        std::span<const Embedding> negative_embeddings,
                                        std::span<const int64_t> negative_face_ids, int k,
                                        float min_score) {
  metrics_.query_template.fetch_add(1, std::memory_order_relaxed);
  const auto started = std::chrono::steady_clock::now();

  std::vector<Embedding> positives(positive_embeddings.begin(), positive_embeddings.end());
  std::vector<Embedding> negatives(negative_embeddings.begin(), negative_embeddings.end());
  std::vector<int64_t> excluded_image_ids;
  excluded_image_ids.reserve(positive_face_ids.size() + negative_face_ids.size());

  auto append_faces = [&](std::span<const int64_t> face_ids, std::vector<Embedding>& embeddings) {
    for (const int64_t face_id : face_ids) {
      const auto face = gallery_->face(face_id);
      Embedding embedding{};
      if (!gallery_->face_embedding(face_id, embedding)) throw std::runtime_error("no face");
      embeddings.push_back(embedding);
      excluded_image_ids.push_back(face.image_id);
    }
  };
  append_faces(positive_face_ids, positives);
  append_faces(negative_face_ids, negatives);

  auto normalize_all = [](std::vector<Embedding>& embeddings) {
    for (auto& embedding : embeddings) {
      double norm2 = 0.0;
      for (const float value : embedding) norm2 += static_cast<double>(value) * value;
      if (!std::isfinite(norm2) || norm2 <= 0.0) throw std::invalid_argument("invalid reference embedding");
      l2_normalize(embedding.data());
    }
  };
  normalize_all(positives);
  normalize_all(negatives);
  if (positives.empty()) throw std::invalid_argument("at least one positive reference is required");
  if (k <= 0) k = cfg_.default_k;

  auto hits = gallery_->search_template(positives, negatives, excluded_image_ids, k, min_score);
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count();
  metrics_.query_us_sum.fetch_add(static_cast<uint64_t>(elapsed), std::memory_order_relaxed);
  return hits;
}

std::vector<IdentityHit> Engine::query_embedding_identities(std::span<const float> vec, int k, float min_score) {
  metrics_.query_identity.fetch_add(1, std::memory_order_relaxed);
  if (vec.size() != static_cast<size_t>(kDim)) return {};
  Embedding q{};
  std::copy(vec.begin(), vec.end(), q.begin());
  l2_normalize(q.data());
  if (k <= 0) k = cfg_.default_k;
  return gallery_->search_identities(q.data(), k, min_score);
}

std::vector<std::pair<DetectedFace, std::vector<IdentityHit>>> Engine::query_image_identities(
    std::span<const uint8_t> bytes, int k, float min_score) {
  metrics_.query_img.fetch_add(1, std::memory_order_relaxed);
  metrics_.query_identity.fetch_add(1, std::memory_order_relaxed);
  cv::Mat img = decode_image(bytes, cfg_.max_pixels);
  if (img.empty()) return {};
  auto faces = run_pipeline(img);
  std::vector<std::pair<DetectedFace, std::vector<IdentityHit>>> out;
  out.reserve(faces.size());
  if (k <= 0) k = cfg_.default_k;
  for (auto& f : faces) {
    auto hits = gallery_->search_identities(f.embedding.data(), k, min_score);
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
  o << "hvax_query_total{type=\"template\"} " << metrics_.query_template.load() << "\n";
  o << "hvax_query_microseconds_sum " << metrics_.query_us_sum.load() << "\n";
  o << "hvax_query_total{type=\"identity\"} " << metrics_.query_identity.load() << "\n";
  o << "hvax_crop_total " << metrics_.crops.load() << "\n";
  o << "hvax_faces " << gallery_->live_faces() << "\n";
  o << "hvax_faces_indexed " << gallery_->indexed_faces() << "\n";
  o << "hvax_faces_low_quality " << gallery_->low_quality_faces() << "\n";
  o << "hvax_faces_unassigned " << gallery_->unassigned_faces() << "\n";
  o << "hvax_images " << gallery_->live_images() << "\n";
  o << "hvax_identities " << gallery_->identity_count() << "\n";
  o << "hvax_identity_largest " << gallery_->largest_identity() << "\n";
  o << "hvax_cluster_runs_total " << metrics_.cluster_runs.load() << "\n";
  o << "hvax_cluster_last_milliseconds " << metrics_.cluster_ms_last.load() << "\n";
  o << "hvax_clustering " << (gallery_->clustering() ? 1 : 0) << "\n";
  o << "hvax_hnsw " << (gallery_->hnsw_active() ? 1 : 0) << "\n";
  return o.str();
}

}  // namespace hvax
