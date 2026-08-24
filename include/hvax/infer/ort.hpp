#pragma once

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace hvax {

class OrtContext {
 public:
  explicit OrtContext(int intra_threads);

  Ort::Env& env() { return env_; }
  Ort::SessionOptions& options() { return opts_; }
  Ort::AllocatorWithDefaultOptions& alloc() { return alloc_; }
  const Ort::MemoryInfo& cpu_mem() const { return cpu_mem_; }

  std::unique_ptr<Ort::Session> load(const std::string& path);

 private:
  Ort::Env env_;
  Ort::SessionOptions opts_;
  Ort::AllocatorWithDefaultOptions alloc_;
  Ort::MemoryInfo cpu_mem_;
};

struct SessionIo {
  std::string input_name;
  std::vector<std::string> output_names;
  std::vector<int64_t> input_shape;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<const char*> input_name_ptrs;
  std::vector<const char*> output_name_ptrs;
};

SessionIo inspect(Ort::Session& session);
void bind_io(SessionIo& io);

}  // namespace hvax
