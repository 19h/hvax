#pragma once

#include <nlohmann/json.hpp>

namespace hvax {

// Sets `jobs` on object bodies; leaves arrays and scalars unchanged.
inline nlohmann::json attach_jobs(nlohmann::json body, int jobs) {
  if (body.is_object()) body["jobs"] = jobs;
  return body;
}

}  // namespace hvax
