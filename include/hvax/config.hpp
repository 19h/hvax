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
  int default_k = 32;
  float default_min_score = kDefaultMinScore;
  size_t max_upload = 20 * 1024 * 1024;
  int64_t max_pixels = 100000000;
  std::string once_image;
  bool compact = false;
  bool print_help = false;

  // Index gate: faces below either bound are stored with kLowQuality and left
  // out of the HNSW and of identity clustering.
  int index_min_face_px = kDefaultIndexMinFacePx;
  float index_min_det = kDefaultIndexMinDet;
  // Use the int8 matrix as the first pass of the exact tier, then rerank
  // candidates against f32 rows.
  bool i8_scan = true;
  // Range mode and saturated queries may grow the HNSW candidate pool up to
  // this many rows.
  int max_range = 4096;

  // Identity layer.
  float identity_join = kIdentityJoinCosine;
  float cluster_edge = kClusterEdgeCosine;
  float cluster_merge = kIdentityMergeCosine;
  int cluster_neighbors = kClusterNeighbors;
  int cluster_interval_s = 0;  // 0 = no background reclustering
  // Enumerating people (GET /v1/identities), triggering clustering over HTTP
  // and curating identities are off unless explicitly enabled. Looking up a
  // person reached from a search hit stays available either way.
  bool identity_browse = false;
  // Identity-management key: requests presenting it (X-Identity-Key, or
  // X-API-Key) may hide/unhide people, list hidden people and see hidden
  // faces. Falls back to $HVAX_IDENTITY_KEY.
  std::string identity_key;

  // One-shot maintenance modes; the daemon exits after running them.
  bool reindex = false;
  bool cluster_once = false;
  int eval_impostor_pairs = 0;  // >0: print the impostor evaluation as JSON and exit
};

Config parse_args(int argc, char** argv);
void print_usage();

}  // namespace hvax
