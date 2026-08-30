#include "hvax/config.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace hvax {

void print_usage() {
  std::cout
      << "hvaxd — accelerated InsightFace ingest + embedding search\n"
      << "Usage: hvaxd [options]\n"
      << "  --data-dir DIR          gallery dir (default ./data)\n"
      << "  --models-dir DIR        det_10g.onnx + w600k_r50.onnx (default ./models)\n"
      << "  --bind ADDR             default 127.0.0.1\n"
      << "  --port N                default 8080\n"
      << "  --det-size N            SCRFD letterbox; positive multiple of 32, default 640\n"
      << "  --threads N             ORT intra-op threads, default 8\n"
      << "  --cuda                  use CUDAExecutionProvider\n"
      << "  --cuda-device N         CUDA device ID, default 0\n"
      << "  --coreml                use CoreML (GPU/ANE selected by Core ML)\n"
      << "  --mps                   use CoreML CPU+GPU (Metal/MPS-oriented alias)\n"
      << "  --coreml-compute-units all|cpu-gpu|cpu-ane|cpu\n"
      << "  --coreml-model-format auto|mlprogram|neuralnetwork\n"
      << "  --coreml-cache-dir DIR  persistent compiled-model cache\n"
      << "  --coreml-low-precision  allow float16 GPU accumulation\n"
      << "  --coreml-profile        log Core ML operator placement\n"
      << "  --http-threads N        default 8\n"
      << "  --api-key STR           optional X-API-Key\n"
      << "  --dedup perceptual|sha256|off\n"
      << "  --phash-threshold N     default 10\n"
      << "  --dhash-threshold N     default 12\n"
      << "  --once IMAGE            detect+embed one file to stdout, no HTTP\n"
      << "  --help\n";
}

static bool eq(const char* a, const char* b) { return std::strcmp(a, b) == 0; }

Config parse_args(int argc, char** argv) {
  Config c;
  auto set_provider = [&](InferenceProvider provider, const char* flag) {
    if (c.inference.provider != InferenceProvider::cpu && c.inference.provider != provider)
      throw std::runtime_error(std::string(flag) + " conflicts with another execution provider");
    c.inference.provider = provider;
  };
  for (int i = 1; i < argc; ++i) {
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (eq(argv[i], "--help") || eq(argv[i], "-h")) c.print_help = true;
    else if (eq(argv[i], "--data-dir")) c.data_dir = need("--data-dir");
    else if (eq(argv[i], "--models-dir")) c.models_dir = need("--models-dir");
    else if (eq(argv[i], "--bind")) c.bind = need("--bind");
    else if (eq(argv[i], "--port")) c.port = std::stoi(need("--port"));
    else if (eq(argv[i], "--det-size")) c.det_size = std::stoi(need("--det-size"));
    else if (eq(argv[i], "--threads")) c.inference.intra_threads = std::stoi(need("--threads"));
    else if (eq(argv[i], "--cuda")) set_provider(InferenceProvider::cuda, "--cuda");
    else if (eq(argv[i], "--cuda-device")) {
      set_provider(InferenceProvider::cuda, "--cuda-device");
      c.inference.device_id = std::stoi(need("--cuda-device"));
    } else if (eq(argv[i], "--coreml")) {
      set_provider(InferenceProvider::coreml, "--coreml");
    } else if (eq(argv[i], "--mps")) {
      set_provider(InferenceProvider::coreml, "--mps");
      c.inference.coreml_compute_units = CoreMlComputeUnits::cpu_and_gpu;
    } else if (eq(argv[i], "--coreml-compute-units")) {
      set_provider(InferenceProvider::coreml, "--coreml-compute-units");
      c.inference.coreml_compute_units = parse_coreml_compute_units(need("--coreml-compute-units"));
    } else if (eq(argv[i], "--coreml-cache-dir")) {
      set_provider(InferenceProvider::coreml, "--coreml-cache-dir");
      c.inference.coreml_cache_dir = need("--coreml-cache-dir");
    } else if (eq(argv[i], "--coreml-model-format")) {
      set_provider(InferenceProvider::coreml, "--coreml-model-format");
      c.inference.coreml_model_format = parse_coreml_model_format(need("--coreml-model-format"));
    } else if (eq(argv[i], "--coreml-low-precision")) {
      set_provider(InferenceProvider::coreml, "--coreml-low-precision");
      c.inference.coreml_allow_low_precision_accumulation = true;
    } else if (eq(argv[i], "--coreml-profile")) {
      set_provider(InferenceProvider::coreml, "--coreml-profile");
      c.inference.coreml_profile_compute_plan = true;
    } else if (eq(argv[i], "--http-threads")) c.http_threads = std::stoi(need("--http-threads"));
    else if (eq(argv[i], "--api-key")) c.api_key = need("--api-key");
    else if (eq(argv[i], "--phash-threshold")) c.phash_threshold = std::stoi(need("--phash-threshold"));
    else if (eq(argv[i], "--dhash-threshold")) c.dhash_threshold = std::stoi(need("--dhash-threshold"));
    else if (eq(argv[i], "--once")) c.once_image = need("--once");
    else if (eq(argv[i], "--compact")) c.compact = true;
    else if (eq(argv[i], "--dedup")) {
      std::string v = need("--dedup");
      if (v == "off") c.dedup = DedupMode::off;
      else if (v == "sha256") c.dedup = DedupMode::sha256;
      else c.dedup = DedupMode::perceptual;
    } else {
      throw std::runtime_error(std::string("unknown flag: ") + argv[i]);
    }
  }
  if (c.inference.intra_threads <= 0) throw std::runtime_error("--threads must be positive");
  if (c.http_threads <= 0) throw std::runtime_error("--http-threads must be positive");
  if (c.det_size <= 0 || c.det_size % 32 != 0)
    throw std::runtime_error("--det-size must be a positive multiple of 32");
  if (c.inference.device_id < 0) throw std::runtime_error("CUDA device must be non-negative");
  if (c.inference.provider == InferenceProvider::coreml && c.inference.coreml_cache_dir.empty())
    c.inference.coreml_cache_dir = c.data_dir + "/coreml-cache";
  if (c.inference.provider == InferenceProvider::coreml)
    c.inference.expected_concurrency = c.once_image.empty() ? std::min(3, c.http_threads) : 1;
  return c;
}

}  // namespace hvax
