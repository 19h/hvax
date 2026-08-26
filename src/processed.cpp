#include "hvax/processed.hpp"

#include "hvax/embed/arcface.hpp"

#include <algorithm>
#include <cmath>

#include <nlohmann/json.hpp>

namespace hvax {
namespace {

bool finite(float value) { return std::isfinite(value); }

bool read_float_array(const nlohmann::json& value, size_t size, float* out) {
  if (!value.is_array() || value.size() != size) return false;
  for (size_t i = 0; i < size; ++i) {
    if (!value[i].is_number()) return false;
    const float v = value[i].get<float>();
    if (!finite(v)) return false;
    out[i] = v;
  }
  return true;
}

}  // namespace

nlohmann::json processed_payload_json(const std::vector<DetectedFace>& faces) {
  nlohmann::json out = {{"version", kProcessedPayloadVersion},
                        {"model", "insightface-buffalo_l"},
                        {"embedding_dim", kDim},
                        {"faces", nlohmann::json::array()}};
  for (const auto& face : faces) {
    nlohmann::json landmarks = nlohmann::json::array();
    for (const auto& point : face.kps.xy) landmarks.push_back({point[0], point[1]});
    out["faces"].push_back({{"bbox", {face.box.x1, face.box.y1, face.box.x2, face.box.y2}},
                            {"det_score", face.det_score},
                            {"landmarks", std::move(landmarks)},
                            {"embedding", std::vector<float>(face.embedding.begin(), face.embedding.end())}});
  }
  return out;
}

bool parse_processed_payload(std::string_view payload, std::vector<DetectedFace>& faces, std::string& error) {
  faces.clear();
  try {
    const auto root = nlohmann::json::parse(payload);
    if (!root.is_object() || root.value("version", 0) != kProcessedPayloadVersion) {
      error = "unsupported processed payload version";
      return false;
    }
    if (root.value("embedding_dim", 0) != kDim) {
      error = "embedding_dim must be 512";
      return false;
    }
    if (root.value("model", std::string{}) != "insightface-buffalo_l") {
      error = "model must be insightface-buffalo_l";
      return false;
    }
    if (!root.contains("faces") || !root["faces"].is_array()) {
      error = "faces must be an array";
      return false;
    }
    if (root["faces"].size() > kMaxProcessedFaces) {
      error = "too many faces";
      return false;
    }

    faces.reserve(root["faces"].size());
    for (size_t index = 0; index < root["faces"].size(); ++index) {
      const auto& value = root["faces"][index];
      if (!value.is_object()) {
        error = "face must be an object";
        return false;
      }
      DetectedFace face;
      float bbox[4];
      if (!value.contains("bbox") || !read_float_array(value["bbox"], 4, bbox)) {
        error = "bbox must contain 4 finite numbers";
        return false;
      }
      face.box = {bbox[0], bbox[1], bbox[2], bbox[3]};
      if (!value.contains("det_score") || !value["det_score"].is_number()) {
        error = "det_score must be a number";
        return false;
      }
      face.det_score = value["det_score"].get<float>();
      if (!finite(face.det_score) || face.det_score < 0.f || face.det_score > 1.f) {
        error = "det_score must be finite and between 0 and 1";
        return false;
      }
      if (!value.contains("landmarks") || !value["landmarks"].is_array() || value["landmarks"].size() != 5) {
        error = "landmarks must contain 5 points";
        return false;
      }
      for (size_t i = 0; i < 5; ++i) {
        float point[2];
        if (!read_float_array(value["landmarks"][i], 2, point)) {
          error = "each landmark must contain 2 finite numbers";
          return false;
        }
        face.kps.xy[i] = {point[0], point[1]};
      }
      if (!value.contains("embedding") ||
          !read_float_array(value["embedding"], static_cast<size_t>(kDim), face.embedding.data())) {
        error = "embedding must contain 512 finite numbers";
        return false;
      }
      faces.push_back(std::move(face));
    }
  } catch (const std::exception& ex) {
    error = std::string("invalid processed payload: ") + ex.what();
    return false;
  }
  return true;
}

bool validate_processed_faces(std::vector<DetectedFace>& faces, int width, int height, std::string& error) {
  if (width <= 0 || height <= 0) {
    error = "invalid image dimensions";
    return false;
  }
  const float max_x = static_cast<float>(width);
  const float max_y = static_cast<float>(height);
  for (auto& face : faces) {
    if (face.box.x2 <= 0.f || face.box.y2 <= 0.f || face.box.x1 >= max_x || face.box.y1 >= max_y) {
      error = "face bbox does not intersect the image";
      return false;
    }
    face.box.x1 = std::clamp(face.box.x1, 0.f, max_x);
    face.box.y1 = std::clamp(face.box.y1, 0.f, max_y);
    face.box.x2 = std::clamp(face.box.x2, 0.f, max_x);
    face.box.y2 = std::clamp(face.box.y2, 0.f, max_y);
    if (face.box.x2 <= face.box.x1 || face.box.y2 <= face.box.y1) {
      error = "face bbox must have positive area";
      return false;
    }
    for (auto& point : face.kps.xy) {
      point[0] = std::clamp(point[0], 0.f, max_x);
      point[1] = std::clamp(point[1], 0.f, max_y);
    }
    double norm2 = 0;
    for (float value : face.embedding) norm2 += static_cast<double>(value) * value;
    if (!std::isfinite(norm2) || norm2 < 1e-12) {
      error = "embedding norm must be non-zero and finite";
      return false;
    }
    l2_normalize(face.embedding.data());
  }
  return true;
}

}  // namespace hvax
