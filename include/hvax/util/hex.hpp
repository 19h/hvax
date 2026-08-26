#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace hvax {

inline char hex_nibble(unsigned v) { return static_cast<char>(v < 10 ? '0' + v : 'a' + (v - 10)); }

inline std::string to_hex(std::span<const uint8_t> bytes) {
  std::string out;
  out.resize(bytes.size() * 2);
  for (size_t i = 0; i < bytes.size(); ++i) {
    out[i * 2] = hex_nibble(bytes[i] >> 4);
    out[i * 2 + 1] = hex_nibble(bytes[i] & 0xf);
  }
  return out;
}

inline std::string to_hex64(uint64_t value) {
  std::string out(16, '0');
  for (size_t i = 0; i < out.size(); ++i) {
    const unsigned shift = static_cast<unsigned>((out.size() - i - 1) * 4);
    out[i] = hex_nibble(static_cast<unsigned>((value >> shift) & 0xf));
  }
  return out;
}

inline bool hex64_from_string(std::string_view value, uint64_t& out) {
  if (value.size() != 16) return false;
  uint64_t parsed = 0;
  for (char c : value) {
    unsigned nibble = 0;
    if (c >= '0' && c <= '9')
      nibble = static_cast<unsigned>(c - '0');
    else if (c >= 'a' && c <= 'f')
      nibble = static_cast<unsigned>(c - 'a' + 10);
    else if (c >= 'A' && c <= 'F')
      nibble = static_cast<unsigned>(c - 'A' + 10);
    else
      return false;
    parsed = (parsed << 4) | nibble;
  }
  out = parsed;
  return true;
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
        static_cast<uint8_t>((nibble(s[static_cast<size_t>(i * 2)]) << 4) | nibble(s[static_cast<size_t>(i * 2 + 1)]));
  }
  return out;
}

inline bool sha256_from_string(std::string_view value, std::array<uint8_t, 32>& out) {
  if (value.size() != 64) return false;
  auto nibble = [](char c, uint8_t& result) {
    if (c >= '0' && c <= '9')
      result = static_cast<uint8_t>(c - '0');
    else if (c >= 'a' && c <= 'f')
      result = static_cast<uint8_t>(c - 'a' + 10);
    else if (c >= 'A' && c <= 'F')
      result = static_cast<uint8_t>(c - 'A' + 10);
    else
      return false;
    return true;
  };
  std::array<uint8_t, 32> parsed{};
  for (size_t i = 0; i < parsed.size(); ++i) {
    uint8_t high = 0;
    uint8_t low = 0;
    if (!nibble(value[i * 2], high) || !nibble(value[i * 2 + 1], low)) return false;
    parsed[i] = static_cast<uint8_t>((high << 4) | low);
  }
  out = parsed;
  return true;
}

}  // namespace hvax
