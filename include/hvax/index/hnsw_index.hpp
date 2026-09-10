#pragma once

#include "hvax/types.hpp"

#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace hvax {

class HnswIndex {
 public:
  HnswIndex();
  ~HnswIndex();

  HnswIndex(const HnswIndex&) = delete;
  HnswIndex& operator=(const HnswIndex&) = delete;

  void open(const std::filesystem::path& path);
  void add(uint64_t key, const float* vec);
  void update(uint64_t key, const float* vec);
  void remove(uint64_t key);
  bool contains(uint64_t key) const;
  std::vector<std::pair<uint64_t, float>> search(const float* query, int k) const;

  // k nearest neighbours for `count` query vectors, run on `threads` workers
  // with search expansion `ef` (0 keeps the index default). Results are
  // written row-major into keys[count*k] / scores[count*k]; missing slots are
  // filled with UINT64_MAX / -2. Used by identity clustering.
  void search_many(const float* queries, size_t count, int k, int ef, int threads, uint64_t* keys,
                   float* scores) const;

  void save();
  // Rebuild from rows [0, n); rows whose flags intersect skip_mask are omitted.
  void rebuild_from(const float* rows, uint64_t n, const uint32_t* flags,
                    uint32_t skip_mask = kTombstone | kLowQuality);
  void clear();
  uint64_t size() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::filesystem::path path_;
  mutable std::mutex mu_;
};

}  // namespace hvax
