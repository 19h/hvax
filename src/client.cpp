#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "httplib.h"
#include "hvax/pipeline.hpp"
#include "hvax/processed.hpp"
#include "hvax/store/phash.hpp"
#include "hvax/util/hex.hpp"
#include "hvax/util/sha256.hpp"

namespace {

namespace fs = std::filesystem;

struct Options {
  std::string command;
  std::string server = "http://127.0.0.1:8080";
  bool server_set = false;
  std::string api_key;
  std::string bind = "127.0.0.1";
  int port = 8080;
  std::string models_dir = "./models";
  int det_size = 640;
  float det_thresh = 0.5f;
  float nms_thresh = 0.4f;
  int ort_threads = 1;
  int jobs = std::min(4u, std::max(1u, std::thread::hardware_concurrency()));
  bool cuda = false;
  int cuda_device = 0;
  bool recursive = true;
  bool cache_enabled = true;
  fs::path cache_dir;
  size_t max_bytes = 20 * 1024 * 1024;
  int64_t max_pixels = 40'000'000;
  std::vector<fs::path> inputs;
};

struct Endpoint {
  std::string origin;
  std::string processed_path;
  std::string check_path;
  std::string stats_path;
};

struct Counters {
  std::atomic<uint64_t> completed{0};
  std::atomic<uint64_t> cached{0};
  std::atomic<uint64_t> stored{0};
  std::atomic<uint64_t> duplicate{0};
  std::atomic<uint64_t> no_face{0};
  std::atomic<uint64_t> errors{0};
};

class PipelineInitializationError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class LazyPipeline {
 public:
  explicit LazyPipeline(const Options& options) : options_(options) {}

  hvax::Pipeline& get() {
    std::call_once(once_, [&] {
      try {
        pipeline_ = std::make_unique<hvax::Pipeline>(options_.models_dir, options_.det_size, options_.det_thresh,
                                                     options_.nms_thresh, options_.ort_threads, options_.cuda,
                                                     options_.cuda_device);
      } catch (const std::exception& error) {
        initialization_error_ = error.what();
      }
    });
    if (!initialization_error_.empty()) throw PipelineInitializationError(initialization_error_);
    return *pipeline_;
  }

  bool report_failure_once() { return !failure_reported_.exchange(true); }

 private:
  const Options& options_;
  std::once_flag once_;
  std::unique_ptr<hvax::Pipeline> pipeline_;
  std::string initialization_error_;
  std::atomic<bool> failure_reported_{false};
};

struct CachedPayload {
  std::string json;
  bool no_face = false;
};

fs::path default_cache_dir() {
  if (const char* value = std::getenv("HVAX_CACHE_DIR"); value && *value) return value;
  if (const char* value = std::getenv("XDG_CACHE_HOME"); value && *value) return fs::path(value) / "hvax";
  if (const char* value = std::getenv("HOME"); value && *value) return fs::path(value) / ".cache" / "hvax";
  return {};
}

std::string model_stamp(const fs::path& path) {
  std::error_code error;
  const auto size = fs::file_size(path, error);
  if (error) return path.filename().string() + ":missing";
  const auto modified = fs::last_write_time(path, error);
  const auto ticks = error ? int64_t{0} : modified.time_since_epoch().count();
  return path.filename().string() + ':' + std::to_string(size) + ':' + std::to_string(ticks);
}

class ProcessingCache {
 public:
  ProcessingCache(const Options& options, const Endpoint& endpoint) {
    if (!options.cache_enabled || options.cache_dir.empty()) return;
    std::ostringstream processing_identity;
    processing_identity << "hvax-processed-v1\n"
                        << model_stamp(fs::path(options.models_dir) / "det_10g.onnx") << '\n'
                        << model_stamp(fs::path(options.models_dir) / "w600k_r50.onnx") << '\n'
                        << options.det_size << '\n'
                        << options.det_thresh << '\n'
                        << options.nms_thresh << '\n';
    processed_root_ = options.cache_dir / "processed-v1" / fingerprint(processing_identity.str());

    // Remote-presence entries must never leak between galleries. Use the parsed,
    // normalized API destination rather than credentials or the user's spelling.
    const std::string remote_identity =
        "hvax-remote-v1\n" + endpoint.origin + '\n' + endpoint.processed_path + '\n' + endpoint.check_path + '\n';
    remote_root_ = options.cache_dir / "remote-v1" / fingerprint(remote_identity);
  }

  bool enabled() const { return !processed_root_.empty(); }

  std::optional<CachedPayload> load(const std::array<uint8_t, 32>& sha) const {
    if (!enabled()) return std::nullopt;
    try {
      std::ifstream input(path_for(processed_root_, sha), std::ios::binary);
      if (!input) return std::nullopt;
      std::string body((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
      const auto payload = nlohmann::json::parse(body);
      if (!payload.is_object() || !payload.contains("faces") || !payload["faces"].is_array()) return std::nullopt;
      return CachedPayload{std::move(body), payload["faces"].empty()};
    } catch (...) {
      return std::nullopt;
    }
  }

  void store(const std::array<uint8_t, 32>& sha, std::string_view payload) const {
    if (!enabled()) return;
    write(path_for(processed_root_, sha), processed_root_, payload);
  }

  std::optional<nlohmann::json> load_remote(const std::array<uint8_t, 32>& sha) const {
    if (remote_root_.empty()) return std::nullopt;
    try {
      std::ifstream input(path_for(remote_root_, sha), std::ios::binary);
      if (!input) return std::nullopt;
      auto marker = nlohmann::json::parse(input);
      if (!marker.is_object() || !marker.value("duplicate", false) || marker.value("process_required", true))
        return std::nullopt;
      return std::make_optional(std::move(marker));
    } catch (...) {
      return std::nullopt;
    }
  }

  void store_remote(const std::array<uint8_t, 32>& sha, const nlohmann::json& result) const {
    if (remote_root_.empty()) return;
    nlohmann::json marker = {
        {"duplicate", true},
        {"process_required", false},
    };
    std::string kind = result.value("duplicate_kind", "");
    if (!result.value("duplicate", false) || result.value("master_replaced", false)) kind = "sha256";
    marker["duplicate_kind"] = kind.empty() ? "remote" : kind;
    for (const char* field : {"image_id", "width", "height"}) {
      if (result.contains(field)) marker[field] = result[field];
    }
    write(path_for(remote_root_, sha), remote_root_, marker.dump());
  }

 private:
  static std::string fingerprint(std::string_view value) {
    const auto digest = hvax::sha256_bytes(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(value.data()), value.size()));
    return hvax::to_hex(digest);
  }

  static fs::path path_for(const fs::path& root, const std::array<uint8_t, 32>& sha) {
    const std::string hex = hvax::to_hex(sha);
    return root / hex.substr(0, 2) / (hex + ".json");
  }

  static void write(const fs::path& destination, const fs::path& root, std::string_view payload) {
    try {
      fs::create_directories(destination.parent_path());
      fs::permissions(root.parent_path().parent_path(), fs::perms::owner_all, fs::perm_options::replace);
      fs::permissions(root.parent_path(), fs::perms::owner_all, fs::perm_options::replace);
      fs::permissions(root, fs::perms::owner_all, fs::perm_options::replace);
      fs::permissions(destination.parent_path(), fs::perms::owner_all, fs::perm_options::replace);
      static std::atomic<uint64_t> sequence{0};
      const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
      auto temporary = destination;
      temporary += ".tmp." + std::to_string(nonce) + '.' + std::to_string(sequence.fetch_add(1));
      {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) return;
        output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
        if (!output) return;
      }
      fs::permissions(temporary, fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace);
      fs::rename(temporary, destination);
    } catch (...) {
    }
  }

  fs::path processed_root_;
  fs::path remote_root_;
};

void usage() {
  std::cout << "hvax — process images locally and store them in a remote hvax "
               "gallery\n"
            << "Usage: hvax ingest [options] FILE|DIR...\n"
            << "       hvax serve  [options]\n"
            << "\n"
            << "  --server URL           remote gallery URL (required by serve; ingest default "
               "http://127.0.0.1:8080)\n"
            << "  --api-key KEY          X-API-Key value (or HVAX_API_KEY)\n"
            << "  --models-dir DIR       det_10g.onnx + w600k_r50.onnx (default "
               "./models)\n"
            << "  --cuda                 use ONNX Runtime CUDAExecutionProvider\n"
            << "  --cuda-device N        CUDA device ID (default 0)\n"
            << "  --jobs N               concurrent process/upload jobs (default up "
               "to 4)\n"
            << "  --bind ADDRESS         serve bind address (default 127.0.0.1)\n"
            << "  --port N               serve port (default 8080)\n"
            << "  --threads N            ONNX Runtime threads per inference call "
               "(default 1)\n"
            << "  --det-size N           SCRFD input size (default 640)\n"
            << "  --no-recursive         do not descend into directory inputs\n"
            << "  --max-bytes-mib N      maximum encoded image size (default 20)\n"
            << "  --max-pixels N         maximum decoded pixels (default 40000000)\n"
            << "  --cache-dir DIR        processing cache (default $XDG_CACHE_HOME/hvax or ~/.cache/hvax)\n"
            << "  --no-cache             disable the processing cache\n"
            << "  --help\n";
}

const char* need_value(int& index, int argc, char** argv, const char* name) {
  if (index + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
  return argv[++index];
}

Options parse_options(int argc, char** argv) {
  Options out;
  out.cache_dir = default_cache_dir();
  if (const char* value = std::getenv("HVAX_SERVER")) {
    out.server = value;
    out.server_set = true;
  }
  if (const char* value = std::getenv("HVAX_API_KEY")) out.api_key = value;
  if (argc < 2 || (std::string(argv[1]) != "ingest" && std::string(argv[1]) != "serve")) {
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
      usage();
      std::exit(0);
    }
    throw std::runtime_error("expected the 'ingest' or 'serve' command");
  }
  out.command = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      usage();
      std::exit(0);
    } else if (arg == "--server") {
      out.server = need_value(i, argc, argv, "--server");
      out.server_set = true;
    } else if (arg == "--api-key") {
      out.api_key = need_value(i, argc, argv, "--api-key");
    } else if (arg == "--models-dir") {
      out.models_dir = need_value(i, argc, argv, "--models-dir");
    } else if (arg == "--cuda") {
      out.cuda = true;
    } else if (arg == "--cuda-device") {
      out.cuda = true;
      out.cuda_device = std::stoi(need_value(i, argc, argv, "--cuda-device"));
    } else if (arg == "--jobs") {
      out.jobs = std::stoi(need_value(i, argc, argv, "--jobs"));
    } else if (arg == "--bind") {
      out.bind = need_value(i, argc, argv, "--bind");
    } else if (arg == "--port") {
      out.port = std::stoi(need_value(i, argc, argv, "--port"));
    } else if (arg == "--threads") {
      out.ort_threads = std::stoi(need_value(i, argc, argv, "--threads"));
    } else if (arg == "--det-size") {
      out.det_size = std::stoi(need_value(i, argc, argv, "--det-size"));
    } else if (arg == "--no-recursive") {
      out.recursive = false;
    } else if (arg == "--max-bytes-mib") {
      out.max_bytes = static_cast<size_t>(std::stoull(need_value(i, argc, argv, "--max-bytes-mib"))) * 1024 * 1024;
    } else if (arg == "--max-pixels") {
      out.max_pixels = std::stoll(need_value(i, argc, argv, "--max-pixels"));
    } else if (arg == "--cache-dir") {
      out.cache_dir = need_value(i, argc, argv, "--cache-dir");
      out.cache_enabled = true;
    } else if (arg == "--no-cache") {
      out.cache_enabled = false;
    } else if (!arg.empty() && arg[0] == '-') {
      throw std::runtime_error("unknown option: " + arg);
    } else {
      out.inputs.emplace_back(arg);
    }
  }
  if (out.command == "ingest" && out.inputs.empty())
    throw std::runtime_error("at least one file or directory is required");
  if (out.command == "serve" && !out.inputs.empty()) throw std::runtime_error("serve does not accept file inputs");
  if (out.command == "serve" && !out.server_set)
    throw std::runtime_error("serve requires a remote --server URL (or HVAX_SERVER)");
  if (out.jobs <= 0 || out.ort_threads <= 0 || out.det_size <= 0 || out.max_bytes == 0 || out.max_pixels <= 0)
    throw std::runtime_error("numeric options must be positive");
  if (out.jobs > 256) throw std::runtime_error("--jobs must not exceed 256");
  if (out.port <= 0 || out.port > 65535) throw std::runtime_error("--port must be between 1 and 65535");
  if (out.cuda_device < 0) throw std::runtime_error("CUDA device must be non-negative");
  return out;
}

Endpoint parse_endpoint(std::string value) {
  if (value.find("://") == std::string::npos) value = "http://" + value;
  if (!value.starts_with("http://") && !value.starts_with("https://"))
    throw std::runtime_error("server URL must use http or https");
  const size_t scheme = value.find("://");
  const size_t path_pos = value.find('/', scheme + 3);
  Endpoint endpoint;
  endpoint.origin = path_pos == std::string::npos ? value : value.substr(0, path_pos);
  std::string base = path_pos == std::string::npos ? "" : value.substr(path_pos);
  if (base.find('?') != std::string::npos || base.find('#') != std::string::npos)
    throw std::runtime_error("server URL must not contain a query or fragment");
  while (base.size() > 1 && base.back() == '/') base.pop_back();
  constexpr std::string_view processed = "/v1/ingest/processed";
  constexpr std::string_view regular = "/v1/ingest";
  if (base.ends_with(processed)) {
    endpoint.processed_path = base;
    endpoint.check_path = base.substr(0, base.size() - std::string_view("/processed").size()) + "/check";
    endpoint.stats_path = base.substr(0, base.size() - std::string_view("/ingest/processed").size()) + "/stats";
  } else if (base.ends_with(regular)) {
    endpoint.processed_path = base + "/processed";
    endpoint.check_path = base + "/check";
    endpoint.stats_path = base.substr(0, base.size() - std::string_view("/ingest").size()) + "/stats";
  } else {
    endpoint.processed_path = base + std::string(processed);
    endpoint.check_path = base + "/v1/ingest/check";
    endpoint.stats_path = base + "/v1/stats";
  }
  return endpoint;
}

bool image_extension(const fs::path& path) {
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  static const std::set<std::string> extensions = {".avif", ".bmp", ".gif", ".heic", ".heif", ".jpeg",
                                                   ".jpg",  ".png", ".tif", ".tiff", ".webp"};
  return extensions.contains(ext);
}

std::vector<fs::path> collect_files(const Options& options) {
  std::set<fs::path> unique;
  for (const auto& input : options.inputs) {
    std::error_code error;
    if (fs::is_regular_file(input, error)) {
      unique.insert(input);
      continue;
    }
    if (!fs::is_directory(input, error)) {
      std::cerr << "skip: " << input << ": not a regular file or directory\n";
      continue;
    }
    if (!options.recursive) {
      for (fs::directory_iterator it(input, fs::directory_options::skip_permission_denied, error), end; it != end;
           it.increment(error)) {
        if (error) {
          error.clear();
          continue;
        }
        if (it->is_regular_file(error) && image_extension(it->path())) unique.insert(it->path());
      }
      continue;
    }
    for (fs::recursive_directory_iterator it(input, fs::directory_options::skip_permission_denied, error), end;
         it != end; it.increment(error)) {
      if (error) {
        error.clear();
        continue;
      }
      if (it->is_regular_file(error) && image_extension(it->path())) unique.insert(it->path());
    }
  }
  return {unique.begin(), unique.end()};
}

std::vector<uint8_t> read_file(const fs::path& path, size_t max_bytes) {
  std::error_code error;
  const auto size = fs::file_size(path, error);
  if (error) throw std::runtime_error("cannot stat file");
  if (size == 0) throw std::runtime_error("empty file");
  if (size > max_bytes) throw std::runtime_error("file exceeds --max-bytes-mib");
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open file");
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!input) throw std::runtime_error("cannot read file");
  return bytes;
}

std::string mime_for(std::span<const uint8_t> bytes) {
  switch (hvax::sniff_mime(bytes)) {
    case hvax::Mime::jpeg:
      return "image/jpeg";
    case hvax::Mime::png:
      return "image/png";
    case hvax::Mime::webp:
      return "image/webp";
    default:
      return "application/octet-stream";
  }
}

httplib::Headers remote_headers(const Options& options) {
  httplib::Headers headers = {{"Accept", "application/json"}, {"User-Agent", "hvax/0.1"}};
  if (!options.api_key.empty()) headers.emplace("X-API-Key", options.api_key);
  return headers;
}

void configure_client(httplib::Client& client) {
  client.enable_server_certificate_verification(true);
  client.set_keep_alive(true);
  client.set_follow_location(true);
  client.set_connection_timeout(std::chrono::seconds(10));
  client.set_read_timeout(std::chrono::minutes(5));
  client.set_write_timeout(std::chrono::minutes(5));
}

struct PreflightOutcome {
  bool process_required = true;
  nlohmann::json response;
};

PreflightOutcome preflight(httplib::Client& client, const Endpoint& endpoint, const Options& options,
                           std::span<const uint8_t> bytes, const cv::Mat& image) {
  const auto sha = hvax::sha256_bytes(bytes);
  const auto perceptual = hvax::hash_image(image);
  const nlohmann::json body = {
      {"sha256", hvax::to_hex(sha)},
      {"phash", hvax::to_hex64(perceptual.phash)},
      {"dhash", hvax::to_hex64(perceptual.dhash)},
      {"width", image.cols},
      {"height", image.rows},
  };
  auto check = client.Post(endpoint.check_path, remote_headers(options), body.dump(), "application/json");
  if (!check) throw std::runtime_error("HTTP preflight transport: " + httplib::to_string(check.error()));
  if (check->status == 404 || check->status == 405) return {};
  if (check->status != 200) {
    throw std::runtime_error("HTTP preflight " + std::to_string(check->status) +
                             (check->body.empty() ? "" : ": " + check->body));
  }
  PreflightOutcome outcome;
  outcome.response = nlohmann::json::parse(check->body);
  outcome.process_required = outcome.response.value("process_required", true);
  return outcome;
}

struct UploadOutcome {
  int status = 0;
  std::string body;
  std::string content_type = "application/json";
};

UploadOutcome upload_processed(httplib::Client& client, const Endpoint& endpoint, const Options& options,
                               std::span<const uint8_t> bytes, const std::string& filename,
                               const std::string& payload) {
  std::string image_body(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  httplib::MultipartFormDataItems items = {
      {"payload", payload, "payload.json", "application/json"},
      {"image", std::move(image_body), filename, mime_for(bytes)},
  };
  auto response = client.Post(endpoint.processed_path, remote_headers(options), items);
  if (!response) throw std::runtime_error("HTTP upload transport: " + httplib::to_string(response.error()));
  UploadOutcome outcome;
  outcome.status = response->status;
  outcome.body = std::move(response->body);
  const auto content_type = response->get_header_value("Content-Type");
  if (!content_type.empty()) outcome.content_type = content_type;
  return outcome;
}

void print_line(std::mutex& mutex, const fs::path& path, const std::string& status, const std::string& detail = {}) {
  std::lock_guard lock(mutex);
  std::cout << status << '\t' << path.string();
  if (!detail.empty()) std::cout << '\t' << detail;
  std::cout << '\n';
}

bool process_one(const fs::path& path, const Options& options, const Endpoint& endpoint, LazyPipeline& pipeline,
                 const ProcessingCache& cache, httplib::Client& client, Counters& counters, std::mutex& output_mu) {
  try {
    auto bytes = read_file(path, options.max_bytes);
    const auto sha = hvax::sha256_bytes(bytes);
    if (const auto remote = cache.load_remote(sha)) {
      counters.cached.fetch_add(1);
      counters.duplicate.fetch_add(1);
      print_line(output_mu, path, "cached-duplicate",
                 "image_id=" + std::to_string(remote->value("image_id", int64_t{0})) +
                     " remote=" + remote->value("duplicate_kind", "unknown"));
      return true;
    }
    const auto cached = cache.load(sha);
    if (cached && cached->no_face) {
      counters.cached.fetch_add(1);
      counters.no_face.fetch_add(1);
      print_line(output_mu, path, "cached-no-face");
      return true;
    }
    cv::Mat image = hvax::decode_image(bytes, options.max_pixels);
    if (image.empty()) throw std::runtime_error("unsupported or oversized image");
    const auto check = preflight(client, endpoint, options, bytes, image);
    if (!check.process_required) {
      cache.store_remote(sha, check.response);
      counters.duplicate.fetch_add(1);
      const std::string kind = check.response.value("duplicate_kind", "unknown");
      print_line(output_mu, path, "duplicate",
                 "image_id=" + std::to_string(check.response.value("image_id", int64_t{0})) + " preflight=" + kind);
      return true;
    }

    std::string payload;
    if (cached) {
      payload = cached->json;
      counters.cached.fetch_add(1);
    } else {
      payload = hvax::processed_payload_json(pipeline.get().run(image)).dump();
      cache.store(sha, payload);
    }
    auto response = upload_processed(client, endpoint, options, bytes, path.filename().string(), payload);
    if (response.status == 204) {
      counters.no_face.fetch_add(1);
      print_line(output_mu, path, cached ? "cached-no-face" : "no-face");
      return true;
    }
    if (response.status != 200) {
      throw std::runtime_error("HTTP " + std::to_string(response.status) +
                               (response.body.empty() ? "" : ": " + response.body));
    }
    const auto result = nlohmann::json::parse(response.body);
    cache.store_remote(sha, result);
    const bool duplicate = result.value("duplicate", false);
    const bool upgraded = result.value("master_replaced", false);
    if (duplicate)
      counters.duplicate.fetch_add(1);
    else
      counters.stored.fetch_add(1);
    const std::string status = upgraded ? "upgraded" : (duplicate ? "duplicate" : "stored");
    print_line(output_mu, path, status,
               "image_id=" + std::to_string(result.value("image_id", int64_t{0})) +
                   (cached ? " cache=hit" : ""));
    return true;
  } catch (const PipelineInitializationError& error) {
    if (pipeline.report_failure_once()) {
      counters.errors.fetch_add(1);
      print_line(output_mu, path, "fatal", error.what());
    }
    return false;
  } catch (const std::exception& error) {
    counters.errors.fetch_add(1);
    print_line(output_mu, path, "error", error.what());
    return true;
  }
}

void run_local_server(const Options& options, const Endpoint& endpoint, LazyPipeline& pipeline,
                      const ProcessingCache& cache) {
  httplib::Server server;
  server.new_task_queue = [jobs = options.jobs] { return new httplib::ThreadPool(jobs); };
  server.set_payload_max_length(options.max_bytes);

  server.Get("/health", [&](const httplib::Request&, httplib::Response& response) {
    response.set_content(nlohmann::json{{"status", "ok"},
                                        {"mode", "local-processor"},
                                        {"remote", options.server}}
                             .dump(),
                         "application/json");
  });

  server.Get("/v1/stats", [&](const httplib::Request&, httplib::Response& response) {
    try {
      httplib::Client remote(endpoint.origin);
      configure_client(remote);
      auto result = remote.Get(endpoint.stats_path, remote_headers(options));
      if (!result) throw std::runtime_error("HTTP stats transport: " + httplib::to_string(result.error()));
      response.status = result->status;
      const auto content_type = result->get_header_value("Content-Type");
      response.set_content(result->body, content_type.empty() ? "application/json" : content_type);
    } catch (const std::exception& error) {
      response.status = 502;
      response.set_content(nlohmann::json{{"error", error.what()}}.dump(), "application/json");
    }
  });

  server.Post("/v1/ingest", [&](const httplib::Request& request, httplib::Response& response) {
    try {
      if (request.body.empty()) {
        response.status = 400;
        response.set_content("{\"error\":\"empty body\"}", "application/json");
        return;
      }
      if (request.body.size() > options.max_bytes) {
        response.status = 413;
        response.set_content("{\"error\":\"image exceeds max upload size\"}", "application/json");
        return;
      }
      std::vector<uint8_t> bytes(request.body.begin(), request.body.end());
      const auto sha = hvax::sha256_bytes(bytes);
      if (const auto remote = cache.load_remote(sha)) {
        response.set_content(remote->dump(), "application/json");
        return;
      }
      const auto cached = cache.load(sha);
      if (cached && cached->no_face) {
        response.status = 204;
        return;
      }
      cv::Mat image = hvax::decode_image(bytes, options.max_pixels);
      if (image.empty()) {
        response.status = 415;
        response.set_content("{\"error\":\"not an image\"}", "application/json");
        return;
      }

      httplib::Client remote(endpoint.origin);
      configure_client(remote);
      const auto check = preflight(remote, endpoint, options, bytes, image);
      if (!check.process_required) {
        cache.store_remote(sha, check.response);
        response.set_content(check.response.dump(), "application/json");
        return;
      }

      std::string payload;
      if (cached) {
        payload = cached->json;
      } else {
        payload = hvax::processed_payload_json(pipeline.get().run(image)).dump();
        cache.store(sha, payload);
      }
      const auto uploaded = upload_processed(remote, endpoint, options, bytes, "browser-image", payload);
      response.status = uploaded.status;
      if (!uploaded.body.empty()) {
        if (uploaded.status == 200) cache.store_remote(sha, nlohmann::json::parse(uploaded.body));
        response.set_content(uploaded.body, uploaded.content_type);
      }
    } catch (const std::exception& error) {
      response.status = 502;
      response.set_content(nlohmann::json{{"error", error.what()}}.dump(), "application/json");
    }
  });

  server.Get("/", [&](const httplib::Request&, httplib::Response& response) {
    response.set_content("hvax local processor\nremote " + options.server + "\nPOST /v1/ingest\nGET /v1/stats\n",
                         "text/plain; charset=utf-8");
  });

  std::cout << "hvax local processor listening on http://" << options.bind << ':' << options.port << '\n'
            << "remote gallery " << options.server << '\n'
            << std::flush;
  if (!server.listen(options.bind, options.port)) throw std::runtime_error("failed to listen on local processor address");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const Endpoint endpoint = parse_endpoint(options.server);
    LazyPipeline pipeline(options);
    ProcessingCache cache(options, endpoint);
    if (options.command == "serve") {
      run_local_server(options, endpoint, pipeline, cache);
      return 0;
    }
    const auto files = collect_files(options);
    if (files.empty()) {
      std::cerr << "no candidate image files found\n";
      return 1;
    }

    std::atomic<size_t> next{0};
    std::atomic<bool> stop{false};
    Counters counters;
    std::mutex output_mu;
    const size_t worker_count = std::min(files.size(), static_cast<size_t>(options.jobs));
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (size_t i = 0; i < worker_count; ++i) {
      workers.emplace_back([&] {
        httplib::Client client(endpoint.origin);
        configure_client(client);
        while (!stop.load()) {
          const size_t index = next.fetch_add(1);
          if (index >= files.size()) break;
          if (!process_one(files[index], options, endpoint, pipeline, cache, client, counters, output_mu))
            stop.store(true);
          counters.completed.fetch_add(1);
        }
      });
    }
    for (auto& worker : workers) worker.join();

    std::cout << "summary\ttotal=" << files.size() << "\tstored=" << counters.stored.load()
              << "\tduplicate=" << counters.duplicate.load() << "\tno-face=" << counters.no_face.load()
              << "\tcached=" << counters.cached.load() << "\terrors=" << counters.errors.load()
              << "\taborted=" << files.size() - std::min<uint64_t>(files.size(), counters.completed.load()) << '\n';
    return counters.errors.load() == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "hvax: " << error.what() << "\n\n";
    usage();
    return 2;
  }
}
