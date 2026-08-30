#include "hvax/index/hnsw_index.hpp"

#include "hvax/index/exact_scan.hpp"

#include <usearch/index.hpp>
#include <usearch/index_dense.hpp>

#include <algorithm>
#include <stdexcept>

#include <spdlog/spdlog.h>

namespace hvax {

using unum::usearch::index_dense_config_t;
using unum::usearch::index_dense_t;
using unum::usearch::metric_kind_t;
using unum::usearch::metric_punned_t;
using unum::usearch::scalar_kind_t;

struct HnswIndex::Impl {
  index_dense_t index;
};

namespace {
index_dense_t make_index() {
  auto metric = metric_punned_t(static_cast<std::size_t>(kDim), metric_kind_t::ip_k, scalar_kind_t::f32_k);
  index_dense_config_t cfg;
  cfg.connectivity = 16;
  cfg.expansion_add = 128;
  cfg.expansion_search = 64;
  auto made = index_dense_t::make(metric, cfg);
  if (!made) throw std::runtime_error("usearch make failed");
  return std::move(made.index);
}
}  // namespace

HnswIndex::HnswIndex() : impl_(std::make_unique<Impl>()) { impl_->index = make_index(); }

HnswIndex::~HnswIndex() {
  try {
    save();
  } catch (...) {
  }
}

void HnswIndex::open(const std::filesystem::path& path) {
  path_ = path;
  if (std::filesystem::exists(path)) {
    auto ok = impl_->index.load(path.string().c_str());
    if (!ok) {
      spdlog::warn("failed to load HNSW {}, starting empty", path.string().c_str());
      impl_->index = make_index();
    } else {
      // USEARCH restores the graph capacity but not the dense wrapper's pool
      // of search contexts. Every operation is serialized by mu_, so one
      // context is sufficient and prevents the first search after a reload
      // from returning its thread ID to an uninitialized ring buffer.
      impl_->index.reserve(unum::usearch::index_limits_t{impl_->index.capacity(), 1});
      spdlog::info("loaded HNSW {} size={}", path.string().c_str(), impl_->index.size());
    }
  }
}

void HnswIndex::add(uint64_t key, const float* vec) {
  std::lock_guard<std::mutex> g(mu_);
  impl_->index.reserve(unum::usearch::index_limits_t{impl_->index.size() + 16});
  impl_->index.add(key, vec);
}

void HnswIndex::update(uint64_t key, const float* vec) {
  std::lock_guard<std::mutex> g(mu_);
  impl_->index.remove(key);
  impl_->index.reserve(unum::usearch::index_limits_t{impl_->index.size() + 16});
  impl_->index.add(key, vec);
}

void HnswIndex::remove(uint64_t key) {
  std::lock_guard<std::mutex> g(mu_);
  impl_->index.remove(key);
}

std::vector<std::pair<uint64_t, float>> HnswIndex::search(const float* query, int k) const {
  std::lock_guard<std::mutex> g(mu_);
  auto results = impl_->index.search(query, static_cast<std::size_t>(std::max(k, 1)));
  std::vector<std::pair<uint64_t, float>> out;
  out.reserve(results.size());
  for (std::size_t i = 0; i < results.size(); ++i) {
    // usearch IP distance is typically 1 - ip (smaller is closer). Convert to cosine.
    const float dist = static_cast<float>(results[i].distance);
    const float score = 1.f - dist;
    out.emplace_back(static_cast<uint64_t>(results[i].member.key), score);
  }
  return out;
}

void HnswIndex::save() {
  if (path_.empty()) return;
  std::lock_guard<std::mutex> g(mu_);
  if (impl_->index.size() == 0) return;
  auto ok = impl_->index.save(path_.string().c_str());
  if (!ok) spdlog::warn("HNSW save failed: {}", path_.string().c_str());
}

void HnswIndex::rebuild_from(const float* rows, uint64_t n, const uint32_t* flags) {
  std::lock_guard<std::mutex> g(mu_);
  impl_->index = make_index();
  impl_->index.reserve(unum::usearch::index_limits_t{static_cast<std::size_t>(n + 16)});
  uint64_t added = 0;
  for (uint64_t i = 0; i < n; ++i) {
    if (flags && (flags[i] & kTombstone)) continue;
    impl_->index.add(i, rows + i * kDim);
    ++added;
  }
  spdlog::info("rebuilt HNSW with {} live vectors", added);
}

uint64_t HnswIndex::size() const {
  std::lock_guard<std::mutex> g(mu_);
  return impl_->index.size();
}

}  // namespace hvax
