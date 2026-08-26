#pragma once

#include "hvax/types.hpp"

#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace hvax {

inline constexpr int kProcessedPayloadVersion = 1;
inline constexpr size_t kMaxProcessedFaces = 1024;

nlohmann::json processed_payload_json(const std::vector<DetectedFace>& faces);

// Parse the versioned wire payload. This validates shape, types, finite
// coordinates/scores, and embedding length. Image-relative bounds are checked
// separately once the server has decoded the submitted image.
bool parse_processed_payload(std::string_view payload, std::vector<DetectedFace>& faces, std::string& error);

// Validate image-relative geometry, clamp detector coordinates to the decoded
// image, and L2-normalize embeddings before they enter persistent storage.
bool validate_processed_faces(std::vector<DetectedFace>& faces, int width, int height, std::string& error);

}  // namespace hvax
