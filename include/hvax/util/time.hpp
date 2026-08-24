#pragma once

#include <chrono>
#include <cstdint>

namespace hvax {

inline int64_t unix_ms() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

}  // namespace hvax
