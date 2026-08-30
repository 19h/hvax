#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace hvax {

enum class InferenceProvider { cpu, cuda, coreml };
enum class CoreMlComputeUnits { all, cpu_and_gpu, cpu_and_neural_engine, cpu_only };
enum class CoreMlModelFormat { automatic, ml_program, neural_network };

struct InferenceOptions {
  int intra_threads = 1;
  int expected_concurrency = 1;
  InferenceProvider provider = InferenceProvider::cpu;
  int device_id = 0;
  CoreMlComputeUnits coreml_compute_units = CoreMlComputeUnits::all;
  CoreMlModelFormat coreml_model_format = CoreMlModelFormat::automatic;
  std::string coreml_cache_dir;
  bool coreml_require_static_input_shapes = false;
  bool coreml_fast_prediction = false;
  bool coreml_allow_low_precision_accumulation = false;
  bool coreml_profile_compute_plan = false;
  std::vector<std::pair<std::string, int64_t>> free_dimension_overrides;
};

const char* to_string(InferenceProvider provider);
const char* to_string(CoreMlComputeUnits units);
const char* to_string(CoreMlModelFormat format);
CoreMlComputeUnits parse_coreml_compute_units(std::string_view value);
CoreMlModelFormat parse_coreml_model_format(std::string_view value);

class OrtContext {
 public:
  explicit OrtContext(InferenceOptions options = {});
  OrtContext(int intra_threads, bool cuda, int cuda_device = 0);

  Ort::Env& env() { return env_; }
  Ort::SessionOptions& options() { return opts_; }
  Ort::AllocatorWithDefaultOptions& alloc() { return alloc_; }
  const Ort::MemoryInfo& cpu_mem() const { return cpu_mem_; }
  InferenceProvider provider() const { return config_.provider; }
  bool cuda_enabled() const { return config_.provider == InferenceProvider::cuda; }
  bool coreml_enabled() const { return config_.provider == InferenceProvider::coreml; }
  const InferenceOptions& config() const { return config_; }

  std::unique_ptr<Ort::Session> load(const std::string& path);

 private:
  InferenceOptions config_;
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
