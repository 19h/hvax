#pragma once

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace hvax {

struct SlotFileHeader {
  char magic[8];
  uint32_t version;
  uint32_t rec_size;
  uint64_t count;
  uint64_t capacity;
  uint8_t pad[4096 - 32];
};
static_assert(sizeof(SlotFileHeader) == 4096);

// Append-only mmap of POD records. Grow under the caller's exclusive lock.
// Searchers must not hold a raw pointer across a grow; take a shared lock.
template <typename T>
class SlotFile {
 public:
  SlotFile() = default;
  SlotFile(const SlotFile&) = delete;
  SlotFile& operator=(const SlotFile&) = delete;

  ~SlotFile() { close(); }

  void open(const std::filesystem::path& path, const char* magic) {
    close();
    path_ = path;
    std::memcpy(magic_, magic, 8);
    std::filesystem::create_directories(path.parent_path());
    fd_ = ::open(path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd_ < 0) throw std::runtime_error("open slots: " + path.string() + ": " + std::strerror(errno));

    struct stat st {};
    if (fstat(fd_, &st) != 0) throw std::runtime_error("fstat slots");
    if (st.st_size == 0) {
      capacity_ = 1024;
      count_ = 0;
      remap_locked(true);
      header()->count = 0;
      header()->capacity = capacity_;
      std::memcpy(header()->magic, magic_, 8);
      header()->version = 1;
      header()->rec_size = static_cast<uint32_t>(sizeof(T));
      msync(map_, sizeof(SlotFileHeader), MS_SYNC);
    } else {
      if (static_cast<size_t>(st.st_size) < sizeof(SlotFileHeader))
        throw std::runtime_error("slots file too small: " + path.string());
      map_size_ = static_cast<size_t>(st.st_size);
      map_ = mmap(nullptr, map_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
      if (map_ == MAP_FAILED) throw std::runtime_error("mmap slots");
      auto* h = header();
      if (std::memcmp(h->magic, magic_, 8) != 0) throw std::runtime_error("bad slots magic");
      if (h->rec_size != sizeof(T)) throw std::runtime_error("slots record size mismatch");
      count_ = h->count;
      capacity_ = h->capacity;
      if (capacity_ < count_) throw std::runtime_error("corrupt slots header");
    }
  }

  void close() {
    if (map_ && map_ != MAP_FAILED) {
      munmap(map_, map_size_);
      map_ = nullptr;
    }
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  uint64_t size() const { return count_; }
  uint64_t capacity() const { return capacity_; }

  T& at(uint64_t i) { return recs()[i]; }
  const T& at(uint64_t i) const { return recs()[i]; }

  uint64_t append(const T& rec) {
    if (count_ >= capacity_) grow();
    recs()[count_] = rec;
    uint64_t idx = count_;
    ++count_;
    header()->count = count_;
    return idx;
  }

  void sync_header() {
    if (!map_) return;
    header()->count = count_;
    header()->capacity = capacity_;
    msync(map_, sizeof(SlotFileHeader), MS_ASYNC);
  }

  void fsync_all() {
    if (!map_) return;
    header()->count = count_;
    msync(map_, map_size_, MS_SYNC);
    ::fsync(fd_);
  }

 private:
  SlotFileHeader* header() { return reinterpret_cast<SlotFileHeader*>(map_); }
  T* recs() { return reinterpret_cast<T*>(static_cast<char*>(map_) + sizeof(SlotFileHeader)); }
  const T* recs() const {
    return reinterpret_cast<const T*>(static_cast<const char*>(map_) + sizeof(SlotFileHeader));
  }

  void grow() {
    uint64_t cap = capacity_ * 2;
    if (cap < 1024) cap = 1024;
    capacity_ = cap;
    remap_locked(false);
    header()->capacity = capacity_;
  }

  void remap_locked(bool init) {
    size_t need = sizeof(SlotFileHeader) + static_cast<size_t>(capacity_) * sizeof(T);
    if (ftruncate(fd_, static_cast<off_t>(need)) != 0)
      throw std::runtime_error("ftruncate slots: " + std::string(std::strerror(errno)));
    if (map_ && map_ != MAP_FAILED) munmap(map_, map_size_);
    map_size_ = need;
    map_ = mmap(nullptr, map_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (map_ == MAP_FAILED) throw std::runtime_error("mmap grow slots");
    if (init) std::memset(map_, 0, sizeof(SlotFileHeader));
  }

  std::filesystem::path path_;
  char magic_[8]{};
  int fd_ = -1;
  void* map_ = nullptr;
  size_t map_size_ = 0;
  uint64_t count_ = 0;
  uint64_t capacity_ = 0;
};

}  // namespace hvax
