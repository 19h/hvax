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
  std::vector<std::pair<uint64_t, float>> search(const float* query, int k) const;
  void save();
  void rebuild_from(const float* rows, uint64_t n, const uint32_t* flags);
  uint64_t size() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::filesystem::path path_;
  mutable std::mutex mu_;
};

}  // namespace hvax
