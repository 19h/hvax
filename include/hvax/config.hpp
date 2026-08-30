#pragma once

#include "hvax/infer/ort.hpp"
#include "hvax/types.hpp"

#include <cstdint>
#include <string>

namespace hvax {

struct Config {
  std::string data_dir = "./data";
  std::string models_dir = "./models";
  std::string bind = "127.0.0.1";
  int port = 8080;
  int det_size = 640;
  float det_thresh = 0.5f;
  float nms_thresh = 0.4f;
  InferenceOptions inference{.intra_threads = 8};
  int http_threads = 8;
  std::string api_key;
  DedupMode dedup = DedupMode::perceptual;
  int phash_threshold = 10;
  int dhash_threshold = 12;
  uint64_t exact_until = 100000;
  int default_k = 10;
  float default_min_score = 0.0f;
  size_t max_upload = 20 * 1024 * 1024;
  int64_t max_pixels = 100000000;
  std::string once_image;
  bool compact = false;
  bool print_help = false;
};

Config parse_args(int argc, char** argv);
void print_usage();

}  // namespace hvax
