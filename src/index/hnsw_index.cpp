#include "hvax/index/hnsw_index.hpp"

#include "hvax/index/exact_scan.hpp"

#include <usearch/index.hpp>
#include <usearch/index_dense.hpp>

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <thread>

#include <spdlog/spdlog.h>

namespace hvax {

using unum::usearch::index_dense_config_t;
using unum::usearch::index_dense_t;
using unum::usearch::index_limits_t;
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

std::size_t default_threads() { return std::max<std::size_t>(1, std::thread::hardware_concurrency()); }
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
      // load() restores the graph but not the per-thread search contexts;
      // without this reserve the first search on a freshly loaded index
      // takes an uninitialised thread slot and crashes.
      impl_->index.reserve(index_limits_t{impl_->index.size() + 16, default_threads()});
      spdlog::info("loaded HNSW {} size={}", path.string().c_str(), impl_->index.size());
    }
  }
}

void HnswIndex::add(uint64_t key, const float* vec) {
  std::lock_guard<std::mutex> g(mu_);
  impl_->index.reserve(index_limits_t{impl_->index.size() + 16, default_threads()});
  impl_->index.add(key, vec);
}

void HnswIndex::update(uint64_t key, const float* vec) {
  std::lock_guard<std::mutex> g(mu_);
  impl_->index.remove(key);
  impl_->index.reserve(index_limits_t{impl_->index.size() + 16, default_threads()});
  impl_->index.add(key, vec);
}

void HnswIndex::remove(uint64_t key) {
  std::lock_guard<std::mutex> g(mu_);
  impl_->index.remove(key);
}

bool HnswIndex::contains(uint64_t key) const {
  std::lock_guard<std::mutex> g(mu_);
  return impl_->index.contains(key);
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

void HnswIndex::search_many(const float* queries, size_t count, int k, int ef, int threads, uint64_t* keys,
                            float* scores) const {
  std::lock_guard<std::mutex> g(mu_);
  const std::size_t kk = static_cast<std::size_t>(std::max(k, 1));
  std::fill(keys, keys + count * kk, UINT64_MAX);
  std::fill(scores, scores + count * kk, -2.f);
  if (count == 0 || impl_->index.size() == 0) return;

  const std::size_t saved_ef = impl_->index.expansion_search();
  if (ef > 0) impl_->index.change_expansion_search(static_cast<std::size_t>(ef));
  const std::size_t nthreads = std::clamp<std::size_t>(static_cast<std::size_t>(std::max(threads, 1)), 1,
                                                       std::max<std::size_t>(1, count));
  impl_->index.reserve(index_limits_t{impl_->index.size() + 16, std::max(nthreads, default_threads())});

  auto worker = [&](std::size_t tid) {
    for (std::size_t q = tid; q < count; q += nthreads) {
      auto results = impl_->index.search(queries + q * kDim, kk, tid);
      for (std::size_t j = 0; j < results.size() && j < kk; ++j) {
        keys[q * kk + j] = static_cast<uint64_t>(results[j].member.key);
        scores[q * kk + j] = 1.f - static_cast<float>(results[j].distance);
      }
    }
  };
  std::vector<std::thread> pool;
  for (std::size_t t = 1; t < nthreads; ++t) pool.emplace_back(worker, t);
  worker(0);
  for (auto& t : pool) t.join();
  if (ef > 0) impl_->index.change_expansion_search(saved_ef);
}

void HnswIndex::save() {
  if (path_.empty()) return;
  std::lock_guard<std::mutex> g(mu_);
  if (impl_->index.size() == 0) return;
  auto ok = impl_->index.save(path_.string().c_str());
  if (!ok) spdlog::warn("HNSW save failed: {}", path_.string().c_str());
}

void HnswIndex::rebuild_from(const float* rows, uint64_t n, const uint32_t* flags, uint32_t skip_mask) {
  std::lock_guard<std::mutex> g(mu_);
  impl_->index = make_index();
  const std::size_t nthreads = std::max<std::size_t>(1, std::min<std::size_t>(default_threads(), n / 4096 + 1));
  impl_->index.reserve(index_limits_t{static_cast<std::size_t>(n + 16), std::max(nthreads, default_threads())});
  std::atomic<uint64_t> added{0};
  // usearch supports concurrent inserts when each worker passes its own
  // thread slot; interleave rows so all workers see the whole distribution.
  auto worker = [&](std::size_t tid) {
    uint64_t local = 0;
    for (uint64_t i = tid; i < n; i += nthreads) {
      if (flags && (flags[i] & skip_mask)) continue;
      impl_->index.add(i, rows + i * kDim, tid);
      ++local;
    }
    added.fetch_add(local, std::memory_order_relaxed);
  };
  std::vector<std::thread> pool;
  for (std::size_t t = 1; t < nthreads; ++t) pool.emplace_back(worker, t);
  worker(0);
  for (auto& t : pool) t.join();
  spdlog::info("rebuilt HNSW with {} indexed vectors of {} rows on {} threads", added.load(), n, nthreads);
}

void HnswIndex::clear() {
  std::lock_guard<std::mutex> g(mu_);
  impl_->index = make_index();
}

uint64_t HnswIndex::size() const {
  std::lock_guard<std::mutex> g(mu_);
  return impl_->index.size();
}

}  // namespace hvax
