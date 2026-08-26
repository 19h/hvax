#include "hvax/http/landing_html.hpp"
#include "hvax/pipeline.hpp"

#include <filesystem>
#include <string_view>

#include <gtest/gtest.h>

TEST(Landing, HtmlDocument) {
  const std::string_view html(kLandingHtml);
  EXPECT_NE(html.find("<!DOCTYPE html>"), std::string_view::npos);
  EXPECT_NE(html.find("hvax"), std::string_view::npos);
  EXPECT_NE(html.find("/v1/ingest"), std::string_view::npos);
  EXPECT_NE(html.find("/v1/query/image"), std::string_view::npos);
}

TEST(PipelineHelpers, SniffMime) {
  const uint8_t jpeg[] = {0xff, 0xd8, 0xff, 0x00};
  EXPECT_EQ(hvax::sniff_mime(jpeg), hvax::Mime::jpeg);
  const uint8_t png[] = {0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};
  EXPECT_EQ(hvax::sniff_mime(png), hvax::Mime::png);
}
