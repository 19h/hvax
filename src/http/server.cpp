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
constexpr uint64_t kMaxPageLimit = 500;

nlohmann::json bbox_json(const BBox& b) { return nlohmann::json::array({b.x1, b.y1, b.x2, b.y2}); }

nlohmann::json kps_json(const Landmark5& k) {
  nlohmann::json a = nlohmann::json::array();
  for (auto& p : k.xy) a.push_back(nlohmann::json::array({p[0], p[1]}));
  return a;
}

nlohmann::json identity_ref_json(int64_t id) {
  if (id < 0) return nullptr;
  return id;
}

nlohmann::json face_json(const FaceView& f) {
  return {{"face_id", f.face_id},
          {"image_id", f.image_id},
          {"bbox", bbox_json(f.box)},
          {"det_score", f.det_score},
          {"landmarks", kps_json(f.kps)},
          {"quality", f.quality},
          {"low_quality", (f.flags & kLowQuality) != 0},
          {"identity_id", identity_ref_json(f.identity_id)},
          {"created_at", f.created_at}};
}

nlohmann::json hit_json(const Hit& h) {
  nlohmann::json j = {{"face_id", h.face_id},
                      {"image_id", h.image_id},
                      {"sha256", to_hex(h.sha256)},
                      {"score", h.score},
                      {"bbox", bbox_json(h.box)},
                      {"det_score", h.det_score},
                      {"quality", h.quality},
                      {"low_quality", (h.flags & kLowQuality) != 0},
                      {"identity_id", identity_ref_json(h.identity_id)}};
  if (h.collapsed > 0) j["collapsed"] = h.collapsed;
  return j;
}

nlohmann::json identity_json(const IdentityView& v) {
  return {{"identity_id", v.identity_id},
          {"size", v.size},
          {"n_images", v.n_images},
          {"rep_face_id", v.rep_face_id},
          {"name", v.name},
          {"pinned", (v.flags & kIdentityPinned) != 0},
          {"cohesion", v.cohesion},
          {"first_seen", v.first_seen},
          {"last_seen", v.last_seen},
          {"created_at", v.created_at},
          {"updated_at", v.updated_at}};
}

nlohmann::json identity_hit_json(const IdentityHit& h) {
  nlohmann::json j = {{"identity_id", h.identity_id}, {"score", h.score},         {"size", h.size},
                      {"n_images", h.n_images},       {"rep_face_id", h.rep_face_id}, {"name", h.name}};
  if (h.best_face) j["best_face"] = hit_json(*h.best_face);
  return j;
}

nlohmann::json cluster_report_json(const ClusterReport& r) {
  return {{"faces_eligible", r.faces_eligible},
          {"faces_assigned", r.faces_assigned},
          {"identities", r.identities},
          {"identities_pinned", r.identities_pinned},
          {"clusters_before_merge", r.clusters_before_merge},
          {"merged", r.merged},
          {"singletons_dropped", r.singletons_dropped},
          {"largest", r.largest},
          {"edges", r.edges},
          {"iterations", r.iterations},
          {"knn_ms", r.knn_ms},
          {"cw_ms", r.cw_ms},
          {"merge_ms", r.merge_ms},
          {"apply_ms", r.apply_ms},
          {"total_ms", r.total_ms}};
}

nlohmann::json impostor_json(const ImpostorReport& r) {
  auto table = [](const std::vector<std::pair<float, double>>& v) {
    nlohmann::json a = nlohmann::json::array();
    for (auto& [t, x] : v) a.push_back({{"threshold", t}, {"value", x}});
    return a;
  };
  return {{"images_used", r.images_used},
          {"same_image_pairs", r.same_image_pairs},
          {"random_pairs", r.random_pairs},
          {"same_image_mean", r.same_image_mean},
          {"same_image_sd", r.same_image_sd},
          {"random_mean", r.random_mean},
          {"random_sd", r.random_sd},
          {"random_mode", r.random_mode},
          {"impostor_ceiling", r.impostor_ceiling},
          {"same_image_far", table(r.same_image_far)},
          {"random_above", table(r.random_above)},
          {"random_same_identity_excess", table(r.random_excess)}};
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

bool header_flag(const httplib::Request& req, const char* name) {
  auto it = req.headers.find(name);
  if (it == req.headers.end()) return false;
  const std::string& v = it->second;
  return v == "1" || v == "true" || v == "yes" || v == "on";
}

std::string lower(std::string s) {
  for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

uint64_t param_u64(const httplib::Request& req, const char* name, uint64_t def, uint64_t max) {
  if (!req.has_param(name)) return def;
  try {
    const long long v = std::stoll(req.get_param_value(name));
    if (v < 0) return def;
    return std::min<uint64_t>(static_cast<uint64_t>(v), max);
  } catch (...) {
    return def;
  }
}

// Search headers shared by the query endpoints:
//   X-K, X-Min-Score, X-Mode: faces|range|identity, X-Group-By: identity,
//   X-Include-Low-Quality: 1
struct QueryOpts {
  SearchOptions search;
  bool identity_mode = false;
};

QueryOpts query_opts(const httplib::Request& req, const Config& cfg) {
  QueryOpts q;
  q.search.k = header_int(req, "X-K", cfg.default_k);
  q.search.min_score = header_float(req, "X-Min-Score", cfg.default_min_score);
  q.search.include_low_quality = header_flag(req, "X-Include-Low-Quality");
  const std::string mode = lower(req.get_header_value("X-Mode"));
  if (mode == "range") q.search.range = true;
  else if (mode == "identity" || mode == "identities" || mode == "people") q.identity_mode = true;
  if (lower(req.get_header_value("X-Group-By")) == "identity") q.search.group_by_identity = true;
  return q;
}

std::optional<Embedding> parse_embedding_json(const nlohmann::json& j) {
  if (!j.contains("embedding") || !j["embedding"].is_array() || j["embedding"].size() != static_cast<size_t>(kDim))
    return std::nullopt;
  Embedding e{};
  for (int i = 0; i < kDim; ++i) e[static_cast<size_t>(i)] = j["embedding"][i].get<float>();
  return e;
}

std::optional<Embedding> parse_embedding_body(const httplib::Request& req) {
  Embedding e{};
  if (req.body.size() == static_cast<size_t>(kDim) * sizeof(float)) {
    std::memcpy(e.data(), req.body.data(), sizeof(float) * kDim);
    return e;
  }
  try {
    auto j = nlohmann::json::parse(req.body);
    return parse_embedding_json(j);
  } catch (...) {
    return std::nullopt;
  }
}

void json_error(httplib::Response& res, int status, const std::string& msg) {
  res.status = status;
  res.set_content(nlohmann::json{{"error", msg}}.dump(), "application/json");
}

nlohmann::json stats_json(const Engine& engine) {
  const auto& g = engine.gallery();
  return {{"faces", g.live_faces()},
          {"images", g.live_images()},
          {"embedding_rows", g.embedding_rows()},
          {"indexed_faces", g.indexed_faces()},
          {"low_quality_faces", g.low_quality_faces()},
          {"unassigned_faces", g.unassigned_faces()},
          {"identities", g.identity_count()},
          {"largest_identity", g.largest_identity()},
          {"clustering", g.clustering()},
          {"hnsw", g.hnsw_active()},
          {"i8_kernel", i8_kernel_name()},
          {"index_min_face_px", engine.config().index_min_face_px},
          {"index_min_det", engine.config().index_min_det},
          {"default_min_score", engine.config().default_min_score},
          {"identity_join", engine.config().identity_join}};
}

}  // namespace

void register_routes(Engine& engine, httplib::Server& svr) {
  const Config& cfg = engine.config();
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
      << "status      online\n"
      << "images      " << g.live_images() << "\n"
      << "faces       " << g.live_faces() << "\n"
      << "embeds      " << g.embedding_rows() << "\n"
      << "identities  " << g.identity_count() << "\n"
      << "index       " << (g.hnsw_active() ? "hnsw" : "exact") << "\n"
      << "\n"
      << "GET    /health\n"
      << "GET    /metrics\n"
      << "GET    /v1/stats\n"
      << "POST   /v1/ingest\n"
      << "POST   /v1/ingest/check\n"
      << "POST   /v1/ingest/processed\n"
      << "POST   /v1/query/image            (X-Mode: faces|range|identity, X-Group-By: identity)\n"
      << "POST   /v1/query/embedding\n"
      << "GET    /v1/identities\n"
      << "GET    /v1/identities/:id\n"
      << "GET    /v1/identities/:id/faces\n"
      << "GET    /v1/identities/:id/cooccurring\n"
      << "GET    /v1/identities/:id/timeline\n"
      << "GET    /v1/faces/:id/crop\n"
      << "\n"
      << "curl --data-binary @face.jpg \\\n"
      << "     -H 'Content-Type: image/jpeg' \\\n"
      << "     http://" << host << "/v1/ingest\n";
    return o.str();
  };

  svr.Get("/", [&, wants_html, landing_plain](const httplib::Request& req, httplib::Response& res) {
    res.set_header("Cache-Control", "no-cache");
    if (wants_html(req)) {
      res.set_content(kLandingHtml, "text/html; charset=utf-8");
      return;
    }
    res.set_content(landing_plain(req), "text/plain; charset=utf-8");
  });

  svr.Get("/health", [&, no_store](const httplib::Request&, httplib::Response& res) {
    no_store(res);
    nlohmann::json j = {{"status", "ok"},
                        {"faces", engine.gallery().live_faces()},
                        {"images", engine.gallery().live_images()},
                        {"identities", engine.gallery().identity_count()},
                        {"hnsw", engine.gallery().hnsw_active()}};
    res.set_content(j.dump(), "application/json");
  });

  svr.Get("/metrics", [&, no_store](const httplib::Request&, httplib::Response& res) {
    no_store(res);
    res.set_content(engine.prometheus(), "text/plain; version=0.0.4");
  });

  svr.Get("/v1/stats", [&, auth, no_store](const httplib::Request& req, httplib::Response& res) {
    no_store(res);
    if (!auth(req, res)) return;
    res.set_content(stats_json(engine).dump(), "application/json");
  });

  svr.Post("/v1/ingest", [&, auth](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    auto bytes = body_bytes(req);
    if (bytes.empty()) return json_error(res, 400, "empty body");
    if (bytes.size() > cfg.max_upload) return json_error(res, 413, "image exceeds max upload size");
    auto r = engine.ingest(bytes);
    set_ingest_response(r, res);
  });

  svr.Post("/v1/ingest/check", [&, auth](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    if (req.body.empty() || req.body.size() > 4096) return json_error(res, 400, "small JSON body required");
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
      json_error(res, 422, error.what());
    }
  });

  svr.Post("/v1/ingest/processed", [&, auth](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    if (!req.has_file("image") || !req.has_file("payload"))
      return json_error(res, 400, "multipart fields 'image' and 'payload' are required");
    const auto image_part = req.get_file_value("image");
    const auto payload_part = req.get_file_value("payload");
    if (image_part.content.empty()) return json_error(res, 400, "empty image");
    if (image_part.content.size() > cfg.max_upload) return json_error(res, 413, "image exceeds max upload size");
    if (payload_part.content.size() > kMaxProcessedJson) return json_error(res, 413, "processed payload is too large");

    std::vector<DetectedFace> faces;
    std::string error;
    if (!parse_processed_payload(payload_part.content, faces, error)) return json_error(res, 422, error);
    std::vector<uint8_t> bytes(image_part.content.begin(), image_part.content.end());
    cv::Mat image = decode_image(bytes, cfg.max_pixels);
    if (image.empty()) return json_error(res, 415, "not an image");
    if (!validate_processed_faces(faces, image.cols, image.rows, error)) return json_error(res, 422, error);
    set_ingest_response(engine.ingest_processed(bytes, image, faces), res);
  });

  svr.Post("/v1/query/embedding", [&, auth](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    const QueryOpts q = query_opts(req, cfg);
    auto e = parse_embedding_body(req);
    if (!e) return json_error(res, 400, "embedding must be 512 floats");
    if (q.identity_mode) {
      auto hits = engine.query_embedding_identities(*e, q.search.k, q.search.min_score);
      nlohmann::json arr = nlohmann::json::array();
      for (auto& h : hits) arr.push_back(identity_hit_json(h));
      res.set_content(nlohmann::json{{"identities", arr}}.dump(), "application/json");
      return;
    }
    auto hits = engine.query_embedding(*e, q.search);
    nlohmann::json arr = nlohmann::json::array();
    for (auto& h : hits) arr.push_back(hit_json(h));
    res.set_content(nlohmann::json{{"hits", arr}}.dump(), "application/json");
  });

  svr.Post("/v1/query/embedding/batch", [&, auth](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    const QueryOpts q = query_opts(req, cfg);
    int nq = header_int(req, "X-Count", 0);
    const size_t need = static_cast<size_t>(kDim) * sizeof(float);
    if (nq <= 0) nq = static_cast<int>(req.body.size() / need);
    if (nq <= 0 || req.body.size() < static_cast<size_t>(nq) * need)
      return json_error(res, 400, "batch body must be N*2048 bytes");
    auto* p = reinterpret_cast<const float*>(req.body.data());
    auto batches = engine.query_embedding_batch(std::span<const float>(p, static_cast<size_t>(nq * kDim)), nq,
                                                q.search);
    nlohmann::json arr = nlohmann::json::array();
    for (auto& hits : batches) {
      nlohmann::json h = nlohmann::json::array();
      for (auto& x : hits) h.push_back(hit_json(x));
      arr.push_back(h);
    }
    res.set_content(nlohmann::json{{"results", arr}}.dump(), "application/json");
  });

  svr.Post("/v1/query/image", [&, auth](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    auto bytes = body_bytes(req);
    if (bytes.size() > cfg.max_upload) return json_error(res, 413, "image exceeds max upload size");
    const QueryOpts q = query_opts(req, cfg);
    nlohmann::json queries = nlohmann::json::array();
    if (q.identity_mode) {
      auto groups = engine.query_image_identities(bytes, q.search.k, q.search.min_score);
      if (groups.empty()) {
        res.status = 204;
        return;
      }
      for (auto& [face, hits] : groups) {
        nlohmann::json hj = nlohmann::json::array();
        for (auto& h : hits) hj.push_back(identity_hit_json(h));
        queries.push_back({{"bbox", bbox_json(face.box)},
                           {"det_score", face.det_score},
                           {"landmarks", kps_json(face.kps)},
                           {"quality", face_quality(face.det_score, face.box)},
                           {"identities", hj}});
      }
    } else {
      auto groups = engine.query_image(bytes, q.search);
      if (groups.empty()) {
        res.status = 204;
        return;
      }
      for (auto& [face, hits] : groups) {
        nlohmann::json hj = nlohmann::json::array();
        for (auto& h : hits) hj.push_back(hit_json(h));
        queries.push_back({{"bbox", bbox_json(face.box)},
                           {"det_score", face.det_score},
                           {"landmarks", kps_json(face.kps)},
                           {"quality", face_quality(face.det_score, face.box)},
                           {"hits", hj}});
      }
    }
    res.set_content(nlohmann::json{{"queries", queries}}.dump(), "application/json");
  });

  // ---- faces ----

  svr.Get(R"(/v1/faces/(\d+)/crop)", [&, auth](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    CropOptions opts;
    opts.size = static_cast<int>(param_u64(req, "size", static_cast<uint64_t>(opts.size), 1024));
    if (req.has_param("pad")) {
      try {
        opts.pad = std::clamp(std::stof(req.get_param_value("pad")), 0.f, 2.f);
      } catch (...) {
      }
    }
    opts.jpeg_quality = static_cast<int>(param_u64(req, "q", static_cast<uint64_t>(opts.jpeg_quality), 100));
    std::vector<uint8_t> jpeg;
    int64_t id = -1;
    try {
      id = std::stoll(req.matches[1]);
    } catch (...) {
    }
    if (id < 0 || !engine.face_crop(id, opts, jpeg)) return json_error(res, 404, "not found");
    // A face crop only changes when its master is replaced; allow client caching.
    res.set_header("Cache-Control", "private, max-age=86400");
    res.set_content(std::string(jpeg.begin(), jpeg.end()), "image/jpeg");
  });

  svr.Get(R"(/v1/faces/(\d+))", [&, auth](const httplib::Request& req, httplib::Response& res) {
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
      json_error(res, 404, "not found");
    }
  });

  // ---- identities ----

  svr.Get("/v1/identities", [&, auth, no_store](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    no_store(res);
    IdentitySort sort = IdentitySort::size;
    const std::string s = lower(req.get_param_value("sort"));
    if (s == "recent") sort = IdentitySort::recent;
    else if (s == "id") sort = IdentitySort::id;
    const uint64_t offset = param_u64(req, "offset", 0, UINT64_MAX);
    const uint64_t limit = param_u64(req, "limit", 50, kMaxPageLimit);
    auto list = engine.gallery().list_identities(sort, offset, limit);
    nlohmann::json arr = nlohmann::json::array();
    for (auto& v : list) arr.push_back(identity_json(v));
    res.set_content(nlohmann::json{{"identities", arr},
                                   {"total", engine.gallery().identity_count()},
                                   {"offset", offset},
                                   {"limit", limit}}
                        .dump(),
                    "application/json");
  });

  svr.Post("/v1/identities/cluster", [&, auth](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    auto r = engine.recluster();
    if (!r) return json_error(res, 409, "clustering already running");
    res.set_content(cluster_report_json(*r).dump(), "application/json");
  });

  svr.Post("/v1/identities/merge", [&, auth](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    std::vector<int64_t> ids;
    try {
      auto j = nlohmann::json::parse(req.body);
      for (auto& v : j.at("ids")) ids.push_back(v.get<int64_t>());
    } catch (...) {
      return json_error(res, 400, "body must be {\"ids\": [..]}");
    }
    auto target = engine.gallery().merge_identities(ids);
    if (!target) return json_error(res, 422, "need two or more distinct live identities");
    auto v = engine.gallery().identity(*target);
    res.set_content(nlohmann::json{{"identity", v ? identity_json(*v) : nullptr}, {"merged", ids.size()}}.dump(),
                    "application/json");
  });

  svr.Get(R"(/v1/identities/(\d+))", [&, auth, no_store](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    no_store(res);
    const int64_t id = std::stoll(req.matches[1]);
    auto v = engine.gallery().identity(id);
    if (!v) return json_error(res, 404, "not found");
    auto j = identity_json(*v);
    nlohmann::json co = nlohmann::json::array();
    for (auto& c : engine.gallery().cooccurring(id, 10))
      co.push_back({{"identity_id", c.identity_id}, {"shared_images", c.shared_images}});
    j["cooccurring"] = co;
    res.set_content(j.dump(), "application/json");
  });

  svr.Get(R"(/v1/identities/(\d+)/faces)", [&, auth, no_store](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    no_store(res);
    const int64_t id = std::stoll(req.matches[1]);
    if (!engine.gallery().identity(id)) return json_error(res, 404, "not found");
    const FaceSort sort = lower(req.get_param_value("sort")) == "time" ? FaceSort::time : FaceSort::score;
    const uint64_t offset = param_u64(req, "offset", 0, UINT64_MAX);
    const uint64_t limit = param_u64(req, "limit", 50, kMaxPageLimit);
    std::vector<float> scores;
    auto faces = engine.gallery().identity_faces(id, sort, offset, limit, &scores);
    nlohmann::json arr = nlohmann::json::array();
    for (size_t i = 0; i < faces.size(); ++i) {
      auto j = face_json(faces[i]);
      j["score"] = scores[i];
      arr.push_back(j);
    }
    res.set_content(nlohmann::json{{"identity_id", id}, {"faces", arr}, {"offset", offset}, {"limit", limit}}.dump(),
                    "application/json");
  });

  svr.Get(R"(/v1/identities/(\d+)/cooccurring)", [&, auth, no_store](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    no_store(res);
    const int64_t id = std::stoll(req.matches[1]);
    if (!engine.gallery().identity(id)) return json_error(res, 404, "not found");
    const uint64_t limit = param_u64(req, "limit", 50, kMaxPageLimit);
    nlohmann::json arr = nlohmann::json::array();
    for (auto& c : engine.gallery().cooccurring(id, limit)) {
      nlohmann::json j = {{"identity_id", c.identity_id}, {"shared_images", c.shared_images}};
      if (auto v = engine.gallery().identity(c.identity_id)) {
        j["size"] = v->size;
        j["rep_face_id"] = v->rep_face_id;
        j["name"] = v->name;
      }
      arr.push_back(j);
    }
    res.set_content(nlohmann::json{{"identity_id", id}, {"cooccurring", arr}}.dump(), "application/json");
  });

  svr.Get(R"(/v1/identities/(\d+)/timeline)", [&, auth, no_store](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    no_store(res);
    const int64_t id = std::stoll(req.matches[1]);
    if (!engine.gallery().identity(id)) return json_error(res, 404, "not found");
    const std::string b = lower(req.get_param_value("bucket"));
    int64_t bucket_ms = 86400000;
    if (b == "hour") bucket_ms = 3600000;
    else if (b == "week") bucket_ms = 7 * 86400000LL;
    nlohmann::json arr = nlohmann::json::array();
    for (auto& t : engine.gallery().timeline(id, bucket_ms)) arr.push_back({{"start_ms", t.start_ms}, {"faces", t.faces}});
    res.set_content(nlohmann::json{{"identity_id", id}, {"bucket_ms", bucket_ms}, {"buckets", arr}}.dump(),
                    "application/json");
  });

  svr.Post(R"(/v1/identities/(\d+)/split)", [&, auth](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    const int64_t id = std::stoll(req.matches[1]);
    std::vector<int64_t> faces;
    try {
      auto j = nlohmann::json::parse(req.body);
      for (auto& v : j.at("face_ids")) faces.push_back(v.get<int64_t>());
    } catch (...) {
      return json_error(res, 400, "body must be {\"face_ids\": [..]}");
    }
    auto nid = engine.gallery().split_identity(id, faces);
    if (!nid) return json_error(res, 422, "faces must belong to the identity and leave at least one behind");
    auto a = engine.gallery().identity(id);
    auto b = engine.gallery().identity(*nid);
    res.set_content(nlohmann::json{{"identity", a ? identity_json(*a) : nullptr},
                                   {"new_identity", b ? identity_json(*b) : nullptr}}
                        .dump(),
                    "application/json");
  });

  svr.Patch(R"(/v1/identities/(\d+))", [&, auth](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    const int64_t id = std::stoll(req.matches[1]);
    try {
      auto j = nlohmann::json::parse(req.body);
      if (j.contains("name")) {
        if (!engine.gallery().rename_identity(id, j["name"].get<std::string>())) return json_error(res, 404, "not found");
      }
      if (j.contains("assign_face_ids")) {
        for (auto& v : j["assign_face_ids"])
          if (!engine.gallery().assign_face(v.get<int64_t>(), id)) return json_error(res, 422, "cannot assign face");
      }
    } catch (...) {
      return json_error(res, 400, "body must be JSON with name and/or assign_face_ids");
    }
    auto v = engine.gallery().identity(id);
    if (!v) return json_error(res, 404, "not found");
    res.set_content(identity_json(*v).dump(), "application/json");
  });

  svr.Delete(R"(/v1/identities/(\d+))", [&, auth](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    const int64_t id = std::stoll(req.matches[1]);
    if (!engine.gallery().dissolve_identity(id)) return json_error(res, 404, "not found");
    res.status = 204;
  });

  svr.Get("/v1/eval/impostor", [&, auth, no_store](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    no_store(res);
    const uint64_t pairs = param_u64(req, "pairs", 100000, 5000000);
    const uint64_t seed = param_u64(req, "seed", 1, UINT64_MAX);
    res.set_content(impostor_json(engine.impostor_eval(pairs, seed)).dump(), "application/json");
  });

  // ---- images ----

  svr.Get(R"(/v1/images/([0-9a-fA-F]{64})/meta)", [&, auth](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    try {
      std::array<uint8_t, 32> sha{};
      if (!sha256_from_string(req.matches[1].str(), sha)) throw std::runtime_error("bad image hash");
      auto im = engine.get_image(sha);
      nlohmann::json j = {{"image_id", im.image_id},
                          {"sha256", to_hex(im.sha256)},
                          {"width", im.width},
                          {"height", im.height},
                          {"mime", engine.image_mime(sha)},
                          {"nbytes", im.nbytes},
                          {"face_ids", im.face_ids},
                          {"identities", engine.gallery().identities_of(im.image_id)},
                          {"repeat_identity", (im.flags & kImageRepeatIdentity) != 0},
                          {"created_at", im.created_at}};
      res.set_content(j.dump(), "application/json");
    } catch (...) {
      json_error(res, 404, "not found");
    }
  });

  svr.Get(R"(/v1/images/([0-9a-fA-F]{64}))", [&, auth](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    try {
      std::array<uint8_t, 32> sha{};
      if (!sha256_from_string(req.matches[1].str(), sha)) throw std::runtime_error("bad image hash");
      auto path = engine.image_file(sha);
      std::ifstream in(path, std::ios::binary);
      if (!in) return json_error(res, 404, "file missing");
      std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      res.set_content(body, engine.image_mime(sha));
    } catch (...) {
      json_error(res, 404, "not found");
    }
  });

  svr.Delete(R"(/v1/images/([0-9a-fA-F]{64}))", [&, auth](const httplib::Request& req, httplib::Response& res) {
    if (!auth(req, res)) return;
    try {
      std::array<uint8_t, 32> sha{};
      if (!sha256_from_string(req.matches[1].str(), sha) || !engine.delete_image(sha))
        return json_error(res, 404, "not found");
      res.status = 204;
    } catch (...) {
      json_error(res, 404, "not found");
    }
  });
}

void run_server(Engine& engine) {
  const Config& cfg = engine.config();
  httplib::Server svr;
  svr.new_task_queue = [n = cfg.http_threads] { return new httplib::ThreadPool(n); };
  register_routes(engine, svr);
  engine.start_background_cluster();
  spdlog::info("hvaxd listening on http://{}:{}", cfg.bind.c_str(), cfg.port);
  if (!svr.listen(cfg.bind, cfg.port)) {
    throw std::runtime_error("listen failed");
  }
}

}  // namespace hvax
