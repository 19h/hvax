#include "hvax/pipeline.hpp"

#include <filesystem>

#include <gtest/gtest.h>

TEST(PipelineHelpers, SniffMime) {
  const uint8_t jpeg[] = {0xff, 0xd8, 0xff, 0x00};
  EXPECT_EQ(hvax::sniff_mime(jpeg), hvax::Mime::jpeg);
  const uint8_t png[] = {0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};
  EXPECT_EQ(hvax::sniff_mime(png), hvax::Mime::png);
}
