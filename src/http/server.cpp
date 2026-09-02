#include "hvax/http/server.hpp"

#include <spdlog/spdlog.h>

#include <cmath>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "httplib.h"
#include "hvax/http/landing_html.hpp"
#include "hvax/processed.hpp"
#include "hvax/util/hex.hpp"

extern char** environ;

namespace hvax {
namespace {

constexpr size_t kMaxProcessedJson = 8 * 1024 * 1024;
constexpr size_t kMultipartOverhead = 1024 * 1024;
constexpr size_t kMaxTemplateReferences = 16;
constexpr int kMaxTemplateResults = 256;
constexpr int kMaxPdfPages = 64;

nlohmann::json bbox_json(const BBox& b) { return nlohmann::json::array({b.x1, b.y1, b.x2, b.y2}); }

nlohmann::json kps_json(const Landmark5& k) {
  nlohmann::json a = nlohmann::json::array();
  for (auto& p : k.xy) a.push_back(nlohmann::json::array({p[0], p[1]}));
  return a;
}

nlohmann::json face_json(const FaceView& f) {
  return {{"face_id", f.face_id},
          {"image_id", f.image_id},
          {"bbox", bbox_json(f.box)},
          {"det_score", f.det_score},
          {"landmarks", kps_json(f.kps)}};
}

nlohmann::json hit_json(const Hit& h) {
  return {{"face_id", h.face_id},
          {"image_id", h.image_id},
          {"sha256", to_hex(h.sha256)},
          {"score", h.score},
          {"bbox", bbox_json(h.box)},
          {"det_score", h.det_score}};
}

nlohmann::json detected_face_json(const DetectedFace& face, bool include_embedding) {
  nlohmann::json result = {{"bbox", bbox_json(face.box)},
                           {"det_score", face.det_score},
                           {"landmarks", kps_json(face.kps)}};
  if (include_embedding)
    result["embedding"] = std::vector<float>(face.embedding.begin(), face.embedding.end());
  return result;
}

std::string face_thumbnail_data_url(const cv::Mat& image, const BBox& box) {
  if (image.empty() || !std::isfinite(box.x1) || !std::isfinite(box.y1) ||
      !std::isfinite(box.x2) || !std::isfinite(box.y2)) {
    return {};
  }

  const float width = box.x2 - box.x1;
  const float height = box.y2 - box.y1;
  if (width <= 0.0f || height <= 0.0f) return {};

  const float side = std::max(width, height) * 1.5f;
  const float center_x = (box.x1 + box.x2) * 0.5f;
  const float center_y = (box.y1 + box.y2) * 0.5f;
  const int left = std::max(0, static_cast<int>(std::floor(center_x - side * 0.5f)));
  const int top = std::max(0, static_cast<int>(std::floor(center_y - side * 0.5f)));
  const int right = std::min(image.cols, static_cast<int>(std::ceil(center_x + side * 0.5f)));
  const int bottom = std::min(image.rows, static_cast<int>(std::ceil(center_y + side * 0.5f)));
  if (right <= left || bottom <= top) return {};

  cv::Mat thumbnail;
  cv::resize(image(cv::Rect(left, top, right - left, bottom - top)), thumbnail,
             cv::Size(160, 160), 0.0, 0.0, cv::INTER_AREA);
  std::vector<uint8_t> jpeg;
  if (!cv::imencode(".jpg", thumbnail, jpeg, {cv::IMWRITE_JPEG_QUALITY, 85}) || jpeg.empty()) {
    return {};
  }
  const std::string raw(reinterpret_cast<const char*>(jpeg.data()), jpeg.size());
  return "data:image/jpeg;base64," + httplib::detail::base64_encode(raw);
}

nlohmann::json ingest_json(const IngestResult& r) {
  nlohmann::json faces = nlohmann::json::array();
  for (auto& f : r.faces) faces.push_back(face_json(f));
  nlohmann::json j = {{"image_id", r.image_id}, {"sha256", to_hex(r.sha256)}, {"width", r.width},
                      {"height", r.height},     {"duplicate", r.duplicate},   {"master_replaced", r.master_replaced},
                      {"faces", faces}};
  if (!r.duplicate_kind.empty()) j["duplicate_kind"] = r.duplicate_kind;
  if (r.previous) {
    j["previous"] = {
        {"width", r.previous->width}, {"height", r.previous->height}, {"sha256", to_hex(r.previous->sha256)}};
  }
  return j;
}

void set_ingest_response(const IngestResult& r, httplib::Response& res) {
  if (r.status == IngestStatus::bad_image) {
    res.status = 415;
    res.set_content("{\"error\":\"not an image\"}", "application/json");
    return;
  }
  if (r.status == IngestStatus::ignored_no_face) {
    res.status = 204;
    return;
  }
  res.status = 200;
  res.set_content(ingest_json(r).dump(), "application/json");
}

bool check_key(const httplib::Request& req, const Config& cfg) {
  if (cfg.api_key.empty()) return true;
  auto it = req.headers.find("X-API-Key");
  return it != req.headers.end() && it->second == cfg.api_key;
}

std::vector<uint8_t> body_bytes(const httplib::Request& req) {
  if (!req.files.empty()) {
    const auto& f = req.files.begin()->second;
    return std::vector<uint8_t>(f.content.begin(), f.content.end());
  }
  return std::vector<uint8_t>(req.body.begin(), req.body.end());
}

bool is_pdf(std::span<const uint8_t> bytes) {
  constexpr std::string_view signature = "%PDF-";
  const size_t header_bytes = std::min<size_t>(bytes.size(), 1024);
  return std::search(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(header_bytes),
                     signature.begin(), signature.end()) !=
         bytes.begin() + static_cast<std::ptrdiff_t>(header_bytes);
}

class TempDirectory {
 public:
  TempDirectory() {
    const auto root = std::filesystem::temp_directory_path();
    const std::string pattern = (root / "hvax-pdf-XXXXXX").string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const char* created = ::mkdtemp(writable.data());
    if (!created) throw std::runtime_error("could not create temporary PDF directory");
    path_ = created;
  }

  ~TempDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

int pdf_page_number(const std::filesystem::path& path) {
  const std::string stem = path.stem().string();
  const size_t dash = stem.rfind('-');
  if (dash == std::string::npos || dash + 1 >= stem.size()) return 0;
  try {
    return std::stoi(stem.substr(dash + 1));
  } catch (...) {
    return 0;
  }
}

int run_pdf_tool(const char* program, std::vector<std::string> arguments) {
  std::vector<char*> argv;
  argv.reserve(arguments.size() + 2);
  argv.push_back(const_cast<char*>(program));
  for (auto& argument : arguments) argv.push_back(argument.data());
  argv.push_back(nullptr);
  posix_spawn_file_actions_t actions;
  const int actions_result = posix_spawn_file_actions_init(&actions);
  if (actions_result != 0) return actions_result;
  posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
  posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
  pid_t pid = 0;
  const int spawned = posix_spawnp(&pid, program, &actions, nullptr, argv.data(), ::environ);
  posix_spawn_file_actions_destroy(&actions);
  if (spawned != 0) return spawned;

  int status = 0;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (true) {
    const pid_t waited = waitpid(pid, &status, WNOHANG);
    if (waited == pid) break;
    if (waited < 0 && errno != EINTR) return errno;
    if (std::chrono::steady_clock::now() >= deadline) {
      ::kill(pid, SIGKILL);
      while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
      return ETIMEDOUT;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  if (!WIFEXITED(status)) return EIO;
  return WEXITSTATUS(status);
}

int render_pdf(const std::filesystem::path& input, const std::filesystem::path& output_prefix) {
  return run_pdf_tool("pdftoppm", {"-jpeg", "-r", "144", "-scale-to", "4096", "-f", "1", "-l",
                                      std::to_string(kMaxPdfPages + 1), input.string(),
                                      output_prefix.string()});
}

int extract_pdf_images(const std::filesystem::path& input, const std::filesystem::path& output_prefix) {
  return run_pdf_tool("pdfimages", {"-j", "-png", "-f", "1", "-l", std::to_string(kMaxPdfPages + 1),
                                     input.string(), output_prefix.string()});
}

std::vector<uint8_t> read_file(const std::filesystem::path& path, size_t limit) {
  std::error_code error;
  const auto size = std::filesystem::file_size(path, error);
  if (error || size == 0 || size > limit) return {};
  std::ifstream input(path, std::ios::binary);
  if (!input) return {};
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!input) return {};
  return bytes;
}

bool trim_white_margins(std::vector<uint8_t>& bytes, int64_t max_pixels) {
  const cv::Mat image = decode_image(bytes, max_pixels);
  if (image.empty() || image.type() != CV_8UC3) return false;

  std::vector<int> column_content(static_cast<size_t>(image.cols), 0);
  std::vector<int> row_content(static_cast<size_t>(image.rows), 0);
  constexpr uint8_t kWhiteThreshold = 245;
  for (int y = 0; y < image.rows; ++y) {
    const auto* row = image.ptr<cv::Vec3b>(y);
    for (int x = 0; x < image.cols; ++x) {
      const auto& pixel = row[x];
      if (pixel[0] >= kWhiteThreshold && pixel[1] >= kWhiteThreshold &&
          pixel[2] >= kWhiteThreshold) {
        continue;
      }
      column_content[static_cast<size_t>(x)]++;
      row_content[static_cast<size_t>(y)]++;
    }
  }

  const int minimum_column_content = std::max(3, image.rows / 100);
  const int minimum_row_content = std::max(3, image.cols / 100);
  int left = 0;
  while (left < image.cols && column_content[static_cast<size_t>(left)] < minimum_column_content) left++;
  int right = image.cols - 1;
  while (right >= left && column_content[static_cast<size_t>(right)] < minimum_column_content) right--;
  int top = 0;
  while (top < image.rows && row_content[static_cast<size_t>(top)] < minimum_row_content) top++;
  int bottom = image.rows - 1;
  while (bottom >= top && row_content[static_cast<size_t>(bottom)] < minimum_row_content) bottom--;
  if (right < left || bottom < top) return false;

  const int x_padding = std::max(4, image.cols / 100);
  const int y_padding = std::max(4, image.rows / 100);
  left = std::max(0, left - x_padding);
  top = std::max(0, top - y_padding);
  right = std::min(image.cols - 1, right + x_padding);
  bottom = std::min(image.rows - 1, bottom + y_padding);
  if (left == 0 && top == 0 && right == image.cols - 1 && bottom == image.rows - 1) return false;

  const cv::Rect content(left, top, right - left + 1, bottom - top + 1);
  std::vector<uint8_t> cropped;
  if (!cv::imencode(".jpg", image(content), cropped, {cv::IMWRITE_JPEG_QUALITY, 92}) || cropped.empty()) {
    return false;
  }
  bytes = std::move(cropped);
  return true;
}

struct PreparedPdfImage {
  std::vector<uint8_t> bytes;
  cv::Mat image;
  std::vector<DetectedFace> faces;
  int rotation = 0;
};

PreparedPdfImage prepare_pdf_image(Engine& engine, std::vector<uint8_t> bytes, int64_t max_pixels) {
  const cv::Mat original = decode_image(bytes, max_pixels);
  if (original.empty()) return {};

  PreparedPdfImage best;
  double best_confidence = -1.0;
  for (const int rotation : {0, 90, 180, 270}) {
    cv::Mat candidate;
    if (rotation == 0)
      candidate = original;
    else if (rotation == 90)
      cv::rotate(original, candidate, cv::ROTATE_90_CLOCKWISE);
    else if (rotation == 180)
      cv::rotate(original, candidate, cv::ROTATE_180);
    else
      cv::rotate(original, candidate, cv::ROTATE_90_COUNTERCLOCKWISE);

    auto faces = engine.debug_once(candidate);
    double confidence = 0.0;
    for (const auto& face : faces) confidence += face.det_score;
    if (faces.size() < best.faces.size() ||
        (faces.size() == best.faces.size() && confidence <= best_confidence)) {
      continue;
    }

    std::vector<uint8_t> candidate_bytes;
    if (rotation == 0) {
      candidate_bytes = bytes;
    } else if (!cv::imencode(".jpg", candidate, candidate_bytes, {cv::IMWRITE_JPEG_QUALITY, 95})) {
      continue;
    }
    best.bytes = std::move(candidate_bytes);
    best.image = candidate;
    best.faces = std::move(faces);
    best.rotation = rotation;
    best_confidence = confidence;
  }
  return best;
}

int header_int(const httplib::Request& req, const char* name, int def) {
  auto it = req.headers.find(name);
  if (it == req.headers.end()) return def;
  try {
    return std::stoi(it->second);
  } catch (...) {
    return def;
  }
}

float header_float(const httplib::Request& req, const char* name, float def) {
  auto it = req.headers.find(name);
  if (it == req.headers.end()) return def;
  try {
    return std::stof(it->second);
  } catch (...) {
    return def;
  }
}

std::optional<Embedding> parse_embedding_json(const nlohmann::json& j) {
  if (!j.contains("embedding") || !j["embedding"].is_array() || j["embedding"].size() != static_cast<size_t>(kDim))
    return std::nullopt;
  Embedding e{};
  for (int i = 0; i < kDim; ++i) e[static_cast<size_t>(i)] = j["embedding"][i].get<float>();
  return e;
}

bool query_flag(const httplib::Request& req, const char* name) {
  if (!req.has_param(name)) return false;
  const std::string value = req.get_param_value(name);
  return value.empty() || value == "1" || value == "true" || value == "yes";
}

bool parse_embedding_value(const nlohmann::json& value, Embedding& embedding, std::string& error) {
  if (!value.is_array() || value.size() != static_cast<size_t>(kDim)) {
    error = "each reference embedding must contain 512 numbers";
    return false;
  }
  double norm2 = 0.0;
  for (int i = 0; i < kDim; ++i) {
    const auto& component = value[static_cast<size_t>(i)];
    if (!component.is_number()) {
      error = "reference embeddings must contain only numbers";
      return false;
    }
    try {
      embedding[static_cast<size_t>(i)] = component.get<float>();
    } catch (...) {
      error = "reference embedding component is out of range";
      return false;
    }
    const float number = embedding[static_cast<size_t>(i)];
    if (!std::isfinite(number)) {
      error = "reference embeddings must contain finite numbers";
      return false;
    }
    norm2 += static_cast<double>(number) * number;
  }
  if (!std::isfinite(norm2) || norm2 <= 0.0) {
    error = "reference embedding norm must be non-zero and finite";
    return false;
  }
  return true;
}

bool parse_embedding_list(const nlohmann::json& root, const char* name, std::vector<Embedding>& out,
                          std::string& error) {
  if (!root.contains(name)) return true;
  const auto& values = root[name];
  if (!values.is_array()) {
    error = std::string(name) + " must be an array";
    return false;
  }
  out.reserve(values.size());
  for (const auto& value : values) {
    Embedding embedding{};
    if (!parse_embedding_value(value, embedding, error)) return false;
    out.push_back(embedding);
  }
  return true;
}

bool parse_face_ids(const nlohmann::json& root, const char* name, std::vector<int64_t>& out,
                    std::string& error) {
  if (!root.contains(name)) return true;
  const auto& values = root[name];
  if (!values.is_array()) {
    error = std::string(name) + " must be an array";
    return false;
  }
  std::unordered_set<int64_t> seen;
  out.reserve(values.size());
  for (const auto& value : values) {
    if (!value.is_number_integer() && !value.is_number_unsigned()) {
      error = std::string(name) + " must contain integer face IDs";
      return false;
    }
    int64_t face_id = -1;
    try {
      face_id = value.get<int64_t>();
    } catch (...) {
      error = std::string(name) + " contains an out-of-range face ID";
      return false;
    }
    if (face_id < 0) {
      error = std::string(name) + " must contain non-negative face IDs";
      return false;
    }
    if (seen.insert(face_id).second) out.push_back(face_id);
  }
  return true;
}

}  // namespace

bool trim_pdf_white_margins(std::vector<uint8_t>& bytes, int64_t max_pixels) {
  return trim_white_margins(bytes, max_pixels);
}

void run_server(Engine& engine) {
  const Config& cfg = engine.config();
  httplib::Server svr;
  svr.new_task_queue = [n = cfg.http_threads] { return new httplib::ThreadPool(n); };
  svr.set_payload_max_length(cfg.max_upload + kMaxProcessedJson + kMultipartOverhead);

  auto auth = [&](const httplib::Request& req, httplib::Response& res) {
    if (check_key(req, cfg)) return true;
    res.status = 401;
    res.set_content("{\"error\":\"unauthorized\"}", "application/json");
    return false;
  };

  auto no_store = [](httplib::Response& res) {
    res.set_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0");
    res.set_header("Pragma", "no-cache");
    res.set_header("Expires", "0");
  };

  auto wants_html = [](const httplib::Request& req) {
    const auto acc = req.get_header_value("Accept");
    if (acc.find("text/html") != std::string::npos) return true;
    const auto ua = req.get_header_value("User-Agent");
    return ua.find("Mozilla") != std::string::npos;
  };

  auto landing_plain = [&](const httplib::Request& req) {
    const auto& g = engine.gallery();
    std::string host = req.get_header_value("Host");
    if (host.empty()) host = cfg.bind + ":" + std::to_string(cfg.port);
    std::ostringstream o;
    o << "hvax — self-hosted cpu face gallery\n"
      << "status    online\n"
      << "images    " << g.live_images() << "\n"
      << "faces     " << g.live_faces() << "\n"
      << "embeds    " << g.embedding_rows() << "\n"
      << "index     " << (g.hnsw_active() ? "hnsw" : "exact") << "\n"
      << "jobs      " << cfg.http_threads << "\n"
      << "\n"
      << "GET   /health\n"
      << "GET   /metrics\n"
      << "GET   /v1/stats\n"
      << "POST  /v1/ingest\n"
      << "POST  /v1/ingest/pdf\n"
      << "POST  /v1/ingest/check\n"
      << "POST  /v1/ingest/processed\n"
      << "POST  /v1/query/image\n"
      << "POST  /v1/query/embedding\n"
      << "POST  /v1/query/template\n"
      << "\n"
      << "curl --data-binary @face.jpg \\\n"
      << "     -H 'Content-Type: image/jpeg' \\\n"
      << "     http://" << host << "/v1/ingest\n";
    return o.str();
  };

  svr.Get("/", [&](const httplib::Request& req, httplib::Response& res) {
    res.set_header("Cache-Control", "no-cache");
    if (wants_html(req)) {
      res.set_content(kLandingHtml, "text/html; charset=utf-8");
      return;
    }
    res.set_content(landing_plain(req), "text/plain; charset=utf-8");
  });

  svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
    no_store(res);
    nlohmann::json j = {{"status", "ok"},
                        {"faces", engine.gallery().live_faces()},
                        {"images", engine.gallery().live_images()},
                        {"hnsw", engine.gallery().hnsw_active()},
                        {"jobs", cfg.http_threads}};
    res.set_content(j.dump(), "application/json");
  });

  svr.Get("/metrics", [&](const httplib::Request&, httplib::Response& res) {
    no_store(res);
    res.set_content(engine.prometheus(), "text/plain; version=0.0.4");
  });

  svr.Get("/v1/stats", [&](const httplib::Request& req, httplib::Response& res) {
    no_store(res);
    if (!auth(req, res)) return;
    nlohmann::json j = {{"faces", engine.gallery().live_faces()},
                        {"images", engine.gallery().live_images()},
                        {"embedding_rows", engine.gallery().embedding_rows()},
                        {"hnsw", engine.gallery().hnsw_active()},
                        {"jobs", cfg.http_threads}};
    res.set_content(j.dump(), "application/json");
  });

  svr.Post("/v1/ingest", [&](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    auto bytes = body_bytes(req);
    if (bytes.empty()) {
      res.status = 400;
      res.set_content("{\"error\":\"empty body\"}", "application/json");
      return;
    }
    if (bytes.size() > cfg.max_upload) {
      res.status = 413;
      res.set_content("{\"error\":\"image exceeds max upload size\"}", "application/json");
      return;
    }
    auto r = engine.ingest(bytes);
    set_ingest_response(r, res);
  });

  svr.Post("/v1/ingest/pdf", [&](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    const bool detect_only = query_flag(req, "detect_only");
    const bool include_embedding = query_flag(req, "include_embedding");
    auto bytes = body_bytes(req);
    if (bytes.empty()) {
      res.status = 400;
      res.set_content("{\"error\":\"empty body\"}", "application/json");
      return;
    }
    if (bytes.size() > cfg.max_upload) {
      res.status = 413;
      res.set_content("{\"error\":\"PDF exceeds max upload size\"}", "application/json");
      return;
    }
    if (!is_pdf(bytes)) {
      res.status = 415;
      res.set_content("{\"error\":\"not a PDF\"}", "application/json");
      return;
    }

    try {
      TempDirectory temporary;
      const auto input_path = temporary.path() / "input.pdf";
      {
        std::ofstream input(input_path, std::ios::binary);
        input.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!input) throw std::runtime_error("could not stage PDF upload");
      }

      const auto output_prefix = temporary.path() / "page";
      const int rendered = render_pdf(input_path, output_prefix);
      if (rendered == ENOENT) {
        res.status = 501;
        res.set_content("{\"error\":\"PDF support requires pdftoppm (Poppler) on the server\"}",
                        "application/json");
        return;
      }
      if (rendered == ETIMEDOUT) {
        res.status = 422;
        res.set_content("{\"error\":\"PDF rendering timed out\"}", "application/json");
        return;
      }
      if (rendered != 0) {
        res.status = 422;
        res.set_content("{\"error\":\"PDF could not be rendered\"}", "application/json");
        return;
      }

      std::vector<std::filesystem::path> pages;
      for (const auto& entry : std::filesystem::directory_iterator(temporary.path())) {
        if (entry.is_regular_file() && entry.path().extension() == ".jpg" &&
            pdf_page_number(entry.path()) > 0) {
          pages.push_back(entry.path());
        }
      }
      std::sort(pages.begin(), pages.end(), [](const auto& left, const auto& right) {
        return pdf_page_number(left) < pdf_page_number(right);
      });
      if (pages.empty()) {
        res.status = 422;
        res.set_content("{\"error\":\"PDF contains no renderable pages\"}", "application/json");
        return;
      }
      if (pages.size() > kMaxPdfPages) {
        res.status = 422;
        res.set_content("{\"error\":\"PDF exceeds the 64-page limit\"}", "application/json");
        return;
      }

      std::vector<std::filesystem::path> embedded_images;
      const auto embedded_prefix = temporary.path() / "embedded";
      if (extract_pdf_images(input_path, embedded_prefix) == 0) {
        for (const auto& entry : std::filesystem::directory_iterator(temporary.path())) {
          const std::string extension = entry.path().extension().string();
          const std::string filename = entry.path().filename().string();
          if (entry.is_regular_file() && filename.starts_with("embedded-") &&
              (extension == ".jpg" || extension == ".jpeg" || extension == ".png")) {
            embedded_images.push_back(entry.path());
          }
        }
      }

      nlohmann::json results = nlohmann::json::array();
      for (const auto& page : pages) {
        const int number = pdf_page_number(page);
        const bool use_embedded_image = pages.size() == 1 && embedded_images.size() == 1;
        auto page_bytes = read_file(use_embedded_image ? embedded_images.front() : page, cfg.max_upload);
        if (page_bytes.empty()) {
          results.push_back({{"page", number}, {"status", "error"},
                             {"error", "rendered page exceeds the image upload limit"}});
          continue;
        }
        const bool cropped = trim_pdf_white_margins(page_bytes, cfg.max_pixels);
        IngestResult result;
        int rotation = 0;
        std::vector<DetectedFace> detected_faces;
        cv::Mat detected_image;
        if (use_embedded_image) {
          auto prepared = prepare_pdf_image(engine, std::move(page_bytes), cfg.max_pixels);
          rotation = prepared.rotation;
          if (prepared.image.empty()) {
            result.status = IngestStatus::bad_image;
          } else if (detect_only) {
            detected_image = prepared.image;
            detected_faces = std::move(prepared.faces);
          } else {
            result = engine.ingest_processed(prepared.bytes, prepared.image, prepared.faces);
          }
        } else if (detect_only) {
          detected_image = decode_image(page_bytes, cfg.max_pixels);
          if (!detected_image.empty()) detected_faces = engine.debug_once(detected_image);
        } else {
          result = engine.ingest(page_bytes);
        }
        if (detect_only) {
          nlohmann::json faces = nlohmann::json::array();
          for (const auto& face : detected_faces) {
            auto face_result = detected_face_json(face, include_embedding);
            const std::string thumbnail = face_thumbnail_data_url(detected_image, face.box);
            if (!thumbnail.empty()) face_result["thumbnail"] = thumbnail;
            faces.push_back(std::move(face_result));
          }
          results.push_back({{"page", number},
                             {"status", faces.empty() ? "no_face" : "detected"},
                             {"cropped", cropped},
                             {"rotation", rotation},
                             {"source", use_embedded_image ? "embedded_image" : "page"},
                             {"faces", std::move(faces)}});
          continue;
        }
        if (result.status == IngestStatus::ignored_no_face) {
          results.push_back({{"page", number}, {"status", "no_face"}, {"cropped", cropped},
                             {"rotation", rotation}, {"source", use_embedded_image ? "embedded_image" : "page"}});
        } else if (result.status == IngestStatus::bad_image) {
          results.push_back({{"page", number}, {"status", "error"},
                             {"cropped", cropped},
                             {"rotation", rotation},
                             {"source", use_embedded_image ? "embedded_image" : "page"},
                             {"error", "rendered page is not a valid image"}});
        } else {
          results.push_back({{"page", number}, {"status", "stored"},
                             {"cropped", cropped},
                             {"rotation", rotation},
                             {"source", use_embedded_image ? "embedded_image" : "page"},
                             {"result", ingest_json(result)}});
        }
      }
      res.set_content(nlohmann::json{{"pages", results}}.dump(), "application/json");
    } catch (const std::exception& error) {
      res.status = 500;
      res.set_content(nlohmann::json{{"error", error.what()}}.dump(), "application/json");
    }
  });

  svr.Post("/v1/ingest/check", [&](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    if (req.body.empty() || req.body.size() > 4096) {
      res.status = 400;
      res.set_content("{\"error\":\"small JSON body required\"}", "application/json");
      return;
    }
    try {
      const auto body = nlohmann::json::parse(req.body);
      if (!body.is_object() || !body.contains("sha256") || !body.contains("phash") || !body.contains("dhash") ||
          !body.contains("width") || !body.contains("height") || !body["sha256"].is_string() ||
          !body["phash"].is_string() || !body["dhash"].is_string() || !body["width"].is_number_integer() ||
          !body["height"].is_number_integer()) {
        throw std::runtime_error("sha256, phash, dhash, width, and height are required");
      }
      std::array<uint8_t, 32> sha{};
      uint64_t phash = 0;
      uint64_t dhash = 0;
      const int width = body["width"].get<int>();
      const int height = body["height"].get<int>();
      if (!sha256_from_string(body["sha256"].get<std::string>(), sha) ||
          !hex64_from_string(body["phash"].get<std::string>(), phash) ||
          !hex64_from_string(body["dhash"].get<std::string>(), dhash) || width <= 0 || height <= 0 ||
          static_cast<int64_t>(width) * height > cfg.max_pixels) {
        throw std::runtime_error("invalid hash or image dimensions");
      }
      const auto check = engine.check_ingest(sha, phash, dhash, width, height);
      nlohmann::json result = {{"duplicate", check.duplicate}, {"process_required", check.process_required}};
      if (check.duplicate) {
        result["duplicate_kind"] = check.duplicate_kind;
        result["image_id"] = check.image_id;
        result["width"] = check.width;
        result["height"] = check.height;
      }
      res.set_content(result.dump(), "application/json");
    } catch (const std::exception& error) {
      res.status = 422;
      res.set_content(nlohmann::json{{"error", error.what()}}.dump(), "application/json");
    }
  });

  svr.Post("/v1/ingest/processed", [&](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    if (!req.has_file("image") || !req.has_file("payload")) {
      res.status = 400;
      res.set_content("{\"error\":\"multipart fields 'image' and 'payload' are required\"}", "application/json");
      return;
    }
    const auto image_part = req.get_file_value("image");
    const auto payload_part = req.get_file_value("payload");
    if (image_part.content.empty()) {
      res.status = 400;
      res.set_content("{\"error\":\"empty image\"}", "application/json");
      return;
    }
    if (image_part.content.size() > cfg.max_upload) {
      res.status = 413;
      res.set_content("{\"error\":\"image exceeds max upload size\"}", "application/json");
      return;
    }
    if (payload_part.content.size() > kMaxProcessedJson) {
      res.status = 413;
      res.set_content("{\"error\":\"processed payload is too large\"}", "application/json");
      return;
    }

    std::vector<DetectedFace> faces;
    std::string error;
    if (!parse_processed_payload(payload_part.content, faces, error)) {
      res.status = 422;
      res.set_content(nlohmann::json{{"error", error}}.dump(), "application/json");
      return;
    }
    std::vector<uint8_t> bytes(image_part.content.begin(), image_part.content.end());
    cv::Mat image = decode_image(bytes, cfg.max_pixels);
    if (image.empty()) {
      res.status = 415;
      res.set_content("{\"error\":\"not an image\"}", "application/json");
      return;
    }
    if (!validate_processed_faces(faces, image.cols, image.rows, error)) {
      res.status = 422;
      res.set_content(nlohmann::json{{"error", error}}.dump(), "application/json");
      return;
    }
    set_ingest_response(engine.ingest_processed(bytes, image, faces), res);
  });

  svr.Post("/v1/query/embedding", [&](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    const int k = header_int(req, "X-K", engine.config().default_k);
    const float min_s = header_float(req, "X-Min-Score", engine.config().default_min_score);
    Embedding e{};
    if (req.body.size() == static_cast<size_t>(kDim) * sizeof(float)) {
      std::memcpy(e.data(), req.body.data(), sizeof(float) * kDim);
    } else {
      try {
        auto j = nlohmann::json::parse(req.body);
        auto p = parse_embedding_json(j);
        if (!p) {
          res.status = 400;
          res.set_content("{\"error\":\"embedding must be 512 floats\"}", "application/json");
          return;
        }
        e = *p;
      } catch (...) {
        res.status = 400;
        res.set_content("{\"error\":\"bad embedding\"}", "application/json");
        return;
      }
    }
    auto hits = engine.query_embedding(e, k, min_s);
    nlohmann::json arr = nlohmann::json::array();
    for (auto& h : hits) arr.push_back(hit_json(h));
    res.set_content(nlohmann::json{{"hits", arr}}.dump(), "application/json");
  });

  svr.Post("/v1/query/embedding/batch", [&](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    const int k = header_int(req, "X-K", engine.config().default_k);
    const float min_s = header_float(req, "X-Min-Score", engine.config().default_min_score);
    int nq = header_int(req, "X-Count", 0);
    const size_t need = static_cast<size_t>(kDim) * sizeof(float);
    if (nq <= 0) nq = static_cast<int>(req.body.size() / need);
    if (nq <= 0 || req.body.size() < static_cast<size_t>(nq) * need) {
      res.status = 400;
      res.set_content("{\"error\":\"batch body must be N*2048 bytes\"}", "application/json");
      return;
    }
    auto* p = reinterpret_cast<const float*>(req.body.data());
    auto batches =
        engine.query_embedding_batch(std::span<const float>(p, static_cast<size_t>(nq * kDim)), nq, k, min_s);
    nlohmann::json arr = nlohmann::json::array();
    for (auto& hits : batches) {
      nlohmann::json h = nlohmann::json::array();
      for (auto& x : hits) h.push_back(hit_json(x));
      arr.push_back(h);
    }
    res.set_content(nlohmann::json{{"results", arr}}.dump(), "application/json");
  });

  svr.Post("/v1/query/template", [&](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    try {
      const auto payload = nlohmann::json::parse(req.body);
      if (!payload.is_object()) {
        res.status = 400;
        res.set_content("{\"error\":\"template body must be a JSON object\"}", "application/json");
        return;
      }

      std::vector<Embedding> positive_embeddings;
      std::vector<Embedding> negative_embeddings;
      std::vector<int64_t> positive_face_ids;
      std::vector<int64_t> negative_face_ids;
      std::string error;
      if (!parse_embedding_list(payload, "positive_embeddings", positive_embeddings, error) ||
          !parse_embedding_list(payload, "negative_embeddings", negative_embeddings, error) ||
          !parse_face_ids(payload, "positive_face_ids", positive_face_ids, error) ||
          !parse_face_ids(payload, "negative_face_ids", negative_face_ids, error)) {
        res.status = 400;
        res.set_content(nlohmann::json{{"error", error}}.dump(), "application/json");
        return;
      }

      if (positive_embeddings.size() + positive_face_ids.size() == 0) {
        res.status = 422;
        res.set_content("{\"error\":\"at least one positive reference is required\"}", "application/json");
        return;
      }
      if (positive_embeddings.size() + positive_face_ids.size() > kMaxTemplateReferences ||
          negative_embeddings.size() + negative_face_ids.size() > kMaxTemplateReferences) {
        res.status = 422;
        res.set_content("{\"error\":\"template supports at most 16 positive and 16 negative references\"}",
                        "application/json");
        return;
      }
      const std::unordered_set<int64_t> positive_ids(positive_face_ids.begin(), positive_face_ids.end());
      for (const int64_t face_id : negative_face_ids) {
        if (positive_ids.contains(face_id)) {
          res.status = 422;
          res.set_content("{\"error\":\"a face cannot be both a positive and negative reference\"}",
                          "application/json");
          return;
        }
      }

      const int k = header_int(req, "X-K", engine.config().default_k);
      const float min_s = header_float(req, "X-Min-Score", engine.config().default_min_score);
      if (k > kMaxTemplateResults || !std::isfinite(min_s)) {
        res.status = 422;
        res.set_content("{\"error\":\"X-K must be at most 256 and X-Min-Score must be finite\"}",
                        "application/json");
        return;
      }
      auto hits = engine.query_template(positive_embeddings, positive_face_ids, negative_embeddings,
                                        negative_face_ids, k, min_s);
      nlohmann::json result = nlohmann::json::array();
      for (const auto& hit : hits) result.push_back(hit_json(hit));
      res.set_content(nlohmann::json{{"hits", result}}.dump(), "application/json");
    } catch (const nlohmann::json::exception&) {
      res.status = 400;
      res.set_content("{\"error\":\"bad template JSON\"}", "application/json");
    } catch (const std::exception& error) {
      res.status = 422;
      res.set_content(nlohmann::json{{"error", error.what()}}.dump(), "application/json");
    }
  });

  svr.Post("/v1/query/image", [&](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    auto bytes = body_bytes(req);
    if (bytes.size() > cfg.max_upload) {
      res.status = 413;
      res.set_content("{\"error\":\"image exceeds max upload size\"}", "application/json");
      return;
    }
    const int k = header_int(req, "X-K", engine.config().default_k);
    const float min_s = header_float(req, "X-Min-Score", engine.config().default_min_score);
    const bool detect_only = query_flag(req, "detect_only");
    const bool include_embedding = query_flag(req, "include_embedding");
    auto groups = engine.query_image(bytes, k, min_s, detect_only);
    if (groups.empty()) {
      res.status = 204;
      return;
    }
    nlohmann::json queries = nlohmann::json::array();
    for (auto& [face, hits] : groups) {
      nlohmann::json hj = nlohmann::json::array();
      for (auto& h : hits) hj.push_back(hit_json(h));
      nlohmann::json query = detected_face_json(face, include_embedding);
      query["hits"] = std::move(hj);
      queries.push_back(std::move(query));
    }
    res.set_content(nlohmann::json{{"queries", queries}}.dump(), "application/json");
  });

  svr.Get(R"(/v1/faces/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    try {
      const int64_t id = std::stoll(req.matches[1]);
      auto f = engine.get_face(id);
      auto j = face_json(f);
      if (req.has_param("include_embedding")) {
        Embedding e{};
        if (engine.get_face_embedding(id, e)) j["embedding"] = std::vector<float>(e.begin(), e.end());
      }
      res.set_content(j.dump(), "application/json");
    } catch (...) {
      res.status = 404;
      res.set_content("{\"error\":\"not found\"}", "application/json");
    }
  });

  svr.Get(R"(/v1/images/([0-9a-fA-F]{64})/meta)", [&](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    try {
      std::array<uint8_t, 32> sha{};
      if (!sha256_from_string(req.matches[1].str(), sha)) throw std::runtime_error("bad image hash");
      auto im = engine.get_image(sha);
      nlohmann::json j = {{"image_id", im.image_id}, {"sha256", to_hex(im.sha256)},   {"width", im.width},
                          {"height", im.height},     {"mime", engine.image_mime(sha)}, {"nbytes", im.nbytes},
                          {"face_ids", im.face_ids}};
      res.set_content(j.dump(), "application/json");
    } catch (...) {
      res.status = 404;
      res.set_content("{\"error\":\"not found\"}", "application/json");
    }
  });

  svr.Get(R"(/v1/images/([0-9a-fA-F]{64}))", [&](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    try {
      std::array<uint8_t, 32> sha{};
      if (!sha256_from_string(req.matches[1].str(), sha)) throw std::runtime_error("bad image hash");
      auto path = engine.image_file(sha);
      std::error_code ec;
      if (!std::filesystem::is_regular_file(path, ec)) {
        res.status = 404;
        res.set_content("{\"error\":\"file missing\"}", "application/json");
        return;
      }
      res.set_header("Cache-Control", "private, max-age=31536000, immutable");
      res.set_file_content(path.string(), engine.image_mime(sha));
    } catch (...) {
      res.status = 404;
      res.set_content("{\"error\":\"not found\"}", "application/json");
    }
  });

  svr.Delete(R"(/v1/images/([0-9a-fA-F]{64}))", [&](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    try {
      std::array<uint8_t, 32> sha{};
      if (!sha256_from_string(req.matches[1].str(), sha) || !engine.delete_image(sha)) {
        res.status = 404;
        res.set_content("{\"error\":\"not found\"}", "application/json");
        return;
      }
      res.status = 204;
    } catch (...) {
      res.status = 404;
      res.set_content("{\"error\":\"not found\"}", "application/json");
    }
  });

  spdlog::info("hvaxd listening on http://{}:{}", cfg.bind.c_str(), cfg.port);
  if (!svr.listen(cfg.bind, cfg.port)) {
    throw std::runtime_error("listen failed");
  }
}

}  // namespace hvax
