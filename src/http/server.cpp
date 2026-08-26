#include "hvax/http/server.hpp"

#include <spdlog/spdlog.h>

#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

#include "httplib.h"
#include "hvax/http/landing_html.hpp"
#include "hvax/processed.hpp"
#include "hvax/util/hex.hpp"

namespace hvax {
namespace {

constexpr size_t kMaxProcessedJson = 8 * 1024 * 1024;
constexpr size_t kMultipartOverhead = 1024 * 1024;

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

}  // namespace

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
      << "\n"
      << "GET   /health\n"
      << "GET   /metrics\n"
      << "GET   /v1/stats\n"
      << "POST  /v1/ingest\n"
      << "POST  /v1/ingest/check\n"
      << "POST  /v1/ingest/processed\n"
      << "POST  /v1/query/image\n"
      << "POST  /v1/query/embedding\n"
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
                        {"hnsw", engine.gallery().hnsw_active()}};
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
                        {"hnsw", engine.gallery().hnsw_active()}};
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
    auto groups = engine.query_image(bytes, k, min_s);
    if (groups.empty()) {
      res.status = 204;
      return;
    }
    nlohmann::json queries = nlohmann::json::array();
    for (auto& [face, hits] : groups) {
      nlohmann::json hj = nlohmann::json::array();
      for (auto& h : hits) hj.push_back(hit_json(h));
      queries.push_back({{"bbox", bbox_json(face.box)},
                         {"det_score", face.det_score},
                         {"landmarks", kps_json(face.kps)},
                         {"hits", hj}});
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
      std::ifstream in(path, std::ios::binary);
      if (!in) {
        res.status = 404;
        res.set_content("{\"error\":\"file missing\"}", "application/json");
        return;
      }
      std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      res.set_content(body, engine.image_mime(sha));
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
