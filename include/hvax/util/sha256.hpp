#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>

#include <openssl/evp.h>

namespace hvax {

inline std::array<uint8_t, 32> sha256_bytes(std::span<const uint8_t> data) {
  std::array<uint8_t, 32> out{};
  EVP_MD_CTX* ctx = EVP_MD_CTX_new();
  if (!ctx) throw std::runtime_error("EVP_MD_CTX_new failed");
  if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1 ||
      EVP_DigestUpdate(ctx, data.data(), data.size()) != 1) {
    EVP_MD_CTX_free(ctx);
    throw std::runtime_error("SHA-256 failed");
  }
  unsigned int len = 32;
  if (EVP_DigestFinal_ex(ctx, out.data(), &len) != 1) {
    EVP_MD_CTX_free(ctx);
    throw std::runtime_error("SHA-256 final failed");
  }
  EVP_MD_CTX_free(ctx);
  return out;
}

}  // namespace hvax
