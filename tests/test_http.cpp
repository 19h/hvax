#include <gtest/gtest.h>

#include <filesystem>
#include <string_view>

#include "hvax/http/landing_html.hpp"
#include "hvax/pipeline.hpp"
#include "hvax/util/hex.hpp"

TEST(Landing, HtmlDocument) {
  const std::string_view html(kLandingHtml);
  EXPECT_NE(html.find("<!DOCTYPE html>"), std::string_view::npos);
  EXPECT_NE(html.find("hvax"), std::string_view::npos);
  EXPECT_NE(html.find("/v1/ingest"), std::string_view::npos);
  EXPECT_NE(html.find("/v1/ingest/check"), std::string_view::npos);
  EXPECT_NE(html.find("/v1/ingest/processed"), std::string_view::npos);
  EXPECT_NE(html.find("/v1/query/image"), std::string_view::npos);
}

TEST(PipelineHelpers, SniffMime) {
  const uint8_t jpeg[] = {0xff, 0xd8, 0xff, 0x00};
  EXPECT_EQ(hvax::sniff_mime(jpeg), hvax::Mime::jpeg);
  const uint8_t png[] = {0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};
  EXPECT_EQ(hvax::sniff_mime(png), hvax::Mime::png);
}

TEST(HttpHelpers, HashHexRoundTrip) {
  constexpr uint64_t expected = 0xfedcba9876543210ULL;
  const auto encoded = hvax::to_hex64(expected);
  EXPECT_EQ(encoded, "fedcba9876543210");
  uint64_t decoded = 0;
  EXPECT_TRUE(hvax::hex64_from_string(encoded, decoded));
  EXPECT_EQ(decoded, expected);
  EXPECT_FALSE(hvax::hex64_from_string("not-a-valid-hash", decoded));
}
