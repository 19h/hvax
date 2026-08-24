#include "hvax/config.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace hvax {

void print_usage() {
  std::cout
      << "hvaxd — CPU InsightFace ingest + embedding search\n"
      << "Usage: hvaxd [options]\n"
      << "  --data-dir DIR          gallery dir (default ./data)\n"
      << "  --models-dir DIR        det_10g.onnx + w600k_r50.onnx (default ./models)\n"
      << "  --bind ADDR             default 127.0.0.1\n"
      << "  --port N                default 8080\n"
      << "  --det-size N            SCRFD letterbox, default 640\n"
      << "  --threads N             ORT intra-op threads, default 8\n"
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
    else if (eq(argv[i], "--threads")) c.ort_intra_threads = std::stoi(need("--threads"));
    else if (eq(argv[i], "--http-threads")) c.http_threads = std::stoi(need("--http-threads"));
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
  return c;
}

}  // namespace hvax
