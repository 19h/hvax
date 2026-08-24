#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace hvax {

inline char hex_nibble(unsigned v) {
  return static_cast<char>(v < 10 ? '0' + v : 'a' + (v - 10));
}

inline std::string to_hex(std::span<const uint8_t> bytes) {
  std::string out;
  out.resize(bytes.size() * 2);
  for (size_t i = 0; i < bytes.size(); ++i) {
    out[i * 2] = hex_nibble(bytes[i] >> 4);
    out[i * 2 + 1] = hex_nibble(bytes[i] & 0xf);
  }
  return out;
}

inline std::array<uint8_t, 32> sha_from_hex(std::string_view s) {
  std::array<uint8_t, 32> out{};
  auto nibble = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
  };
  if (s.size() < 64) return out;
  for (int i = 0; i < 32; ++i) {
    out[static_cast<size_t>(i)] =
        static_cast<uint8_t>((nibble(s[static_cast<size_t>(i * 2)]) << 4) |
                             nibble(s[static_cast<size_t>(i * 2 + 1)]));
  }
  return out;
}

}  // namespace hvax
