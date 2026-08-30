#include "hvax/infer/ort.hpp"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>

#include <spdlog/spdlog.h>

namespace hvax {

const char* to_string(InferenceProvider provider) {
  switch (provider) {
    case InferenceProvider::cpu:
      return "CPU";
    case InferenceProvider::cuda:
      return "CUDA";
    case InferenceProvider::coreml:
      return "CoreML";
  }
  return "unknown";
}

const char* to_string(CoreMlComputeUnits units) {
  switch (units) {
    case CoreMlComputeUnits::all:
      return "ALL";
    case CoreMlComputeUnits::cpu_and_gpu:
      return "CPUAndGPU";
    case CoreMlComputeUnits::cpu_and_neural_engine:
      return "CPUAndNeuralEngine";
    case CoreMlComputeUnits::cpu_only:
      return "CPUOnly";
  }
  return "unknown";
}

const char* to_string(CoreMlModelFormat format) {
  switch (format) {
    case CoreMlModelFormat::automatic:
      return "Auto";
    case CoreMlModelFormat::ml_program:
      return "MLProgram";
    case CoreMlModelFormat::neural_network:
      return "NeuralNetwork";
  }
  return "unknown";
}

CoreMlComputeUnits parse_coreml_compute_units(std::string_view value) {
  if (value == "all") return CoreMlComputeUnits::all;
  if (value == "cpu-gpu" || value == "gpu") return CoreMlComputeUnits::cpu_and_gpu;
  if (value == "cpu-ane" || value == "ane") return CoreMlComputeUnits::cpu_and_neural_engine;
  if (value == "cpu") return CoreMlComputeUnits::cpu_only;
  throw std::invalid_argument("CoreML compute units must be all, cpu-gpu, cpu-ane, or cpu");
}

CoreMlModelFormat parse_coreml_model_format(std::string_view value) {
  if (value == "auto") return CoreMlModelFormat::automatic;
  if (value == "mlprogram") return CoreMlModelFormat::ml_program;
  if (value == "neuralnetwork") return CoreMlModelFormat::neural_network;
  throw std::invalid_argument("CoreML model format must be auto, mlprogram, or neuralnetwork");
}

OrtContext::OrtContext(int intra_threads, bool cuda, int cuda_device)
    : OrtContext(InferenceOptions{.intra_threads = intra_threads,
                                  .expected_concurrency = 1,
                                  .provider = cuda ? InferenceProvider::cuda : InferenceProvider::cpu,
                                  .device_id = cuda_device}) {}

OrtContext::OrtContext(InferenceOptions options)
    : config_(std::move(options)),
      env_(config_.coreml_profile_compute_plan ? ORT_LOGGING_LEVEL_VERBOSE : ORT_LOGGING_LEVEL_WARNING, "hvax"),
      cpu_mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
  opts_.SetIntraOpNumThreads(config_.intra_threads > 0 ? config_.intra_threads : 1);
  opts_.SetInterOpNumThreads(1);
  opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  opts_.EnableMemPattern();
  opts_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  for (const auto& [name, value] : config_.free_dimension_overrides) {
    if (name.empty() || value <= 0)
      throw std::invalid_argument("ONNX free-dimension overrides require a name and positive value");
    Ort::ThrowOnError(Ort::GetApi().AddFreeDimensionOverrideByName(opts_, name.c_str(), value));
  }

  const auto providers = Ort::GetAvailableProviders();
  if (config_.provider == InferenceProvider::cuda) {
    if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") == providers.end()) {
      throw std::runtime_error(
          "CUDAExecutionProvider is unavailable; configure with -DHVAX_ORT_ROOT=<onnxruntime-gpu directory>");
    }
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = config_.device_id;
    opts_.AppendExecutionProvider_CUDA(cuda_options);
    spdlog::info("ORT {} execution provider=CUDA device={}", Ort::GetVersionString(), config_.device_id);
    return;
  }

  if (config_.provider == InferenceProvider::coreml) {
    if (std::find(providers.begin(), providers.end(), "CoreMLExecutionProvider") == providers.end()) {
      throw std::runtime_error(
          "CoreMLExecutionProvider is unavailable; use the official macOS arm64 ONNX Runtime package");
    }
    const auto model_format = config_.coreml_model_format == CoreMlModelFormat::automatic
                                  ? CoreMlModelFormat::neural_network
                                  : config_.coreml_model_format;
    std::unordered_map<std::string, std::string> coreml_options = {
        {"ModelFormat", to_string(model_format)},
        {"MLComputeUnits", to_string(config_.coreml_compute_units)},
        {"RequireStaticInputShapes", config_.coreml_require_static_input_shapes ? "1" : "0"},
        {"EnableOnSubgraphs", "0"},
        {"SpecializationStrategy", config_.coreml_fast_prediction ? "FastPrediction" : "Default"},
        {"ProfileComputePlan", config_.coreml_profile_compute_plan ? "1" : "0"},
        {"AllowLowPrecisionAccumulationOnGPU",
         config_.coreml_allow_low_precision_accumulation ? "1" : "0"},
    };
    if (!config_.coreml_cache_dir.empty()) {
      std::error_code error;
      const auto cache = std::filesystem::absolute(config_.coreml_cache_dir, error);
      if (error) throw std::runtime_error("cannot resolve CoreML cache directory: " + error.message());
      std::filesystem::create_directories(cache, error);
      if (error) throw std::runtime_error("cannot create CoreML cache directory: " + error.message());
      coreml_options["ModelCacheDirectory"] = cache.string();
    }
    opts_.AppendExecutionProvider("CoreML", coreml_options);
    spdlog::info("ORT {} execution provider=CoreML model={} compute_units={} cache={} static_shapes={} "
                 "dimension_overrides={} fast_prediction={} low_precision_accumulation={}",
                 Ort::GetVersionString(), to_string(model_format),
                 to_string(config_.coreml_compute_units),
                 config_.coreml_cache_dir.empty() ? "disabled" : config_.coreml_cache_dir,
                 config_.coreml_require_static_input_shapes, config_.free_dimension_overrides.size(),
                 config_.coreml_fast_prediction,
                 config_.coreml_allow_low_precision_accumulation);
    return;
  }

  spdlog::info("ORT {} execution provider=CPU intra_threads={}", Ort::GetVersionString(), config_.intra_threads);
}

std::unique_ptr<Ort::Session> OrtContext::load(const std::string& path) {
  return std::make_unique<Ort::Session>(env_, path.c_str(), opts_);
}

SessionIo inspect(Ort::Session& session) {
  SessionIo io;
  Ort::AllocatorWithDefaultOptions alloc;
  auto in = session.GetInputNameAllocated(0, alloc);
  io.input_name = in.get();
  auto input_type = session.GetInputTypeInfo(0);
  auto input_info = input_type.GetTensorTypeAndShapeInfo();
  io.input_shape = input_info.GetShape();
  std::vector<const char*> input_symbols(io.input_shape.size());
  input_info.GetSymbolicDimensions(input_symbols.data(), input_symbols.size());
  const size_t nout = session.GetOutputCount();
  io.output_names.reserve(nout);
  io.output_shapes.reserve(nout);
  for (size_t i = 0; i < nout; ++i) {
    auto n = session.GetOutputNameAllocated(i, alloc);
    io.output_names.emplace_back(n.get());
    io.output_shapes.push_back(session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  std::string shape;
  for (auto d : io.input_shape) {
    if (!shape.empty()) shape += ",";
    shape += std::to_string(d);
  }
  std::string symbols;
  for (const char* symbol : input_symbols) {
    if (!symbols.empty()) symbols += ",";
    symbols += symbol ? symbol : "";
  }
  spdlog::info("ORT input={} shape=[{}] symbols=[{}] outs={}", io.input_name.c_str(), shape.c_str(),
               symbols.c_str(), io.output_names.size());
  for (size_t i = 0; i < io.output_names.size(); ++i) {
    std::string os;
    for (auto d : io.output_shapes[i]) {
      if (!os.empty()) os += ",";
      os += std::to_string(d);
    }
    spdlog::info("  out[{}] {} [{}]", i, io.output_names[i].c_str(), os.c_str());
  }
  return io;
}

void bind_io(SessionIo& io) {
  io.input_name_ptrs = {io.input_name.c_str()};
  io.output_name_ptrs.clear();
  io.output_name_ptrs.reserve(io.output_names.size());
  for (auto& s : io.output_names) io.output_name_ptrs.push_back(s.c_str());
}

}  // namespace hvax
