#include <gtest/gtest.h>

#include <filesystem>
#include <string_view>

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

#include "httplib.h"
#include "hvax/config.hpp"
#include "hvax/http/jobs.hpp"
#include "hvax/http/landing_html.hpp"
#include "hvax/http/server.hpp"
#include "hvax/pipeline.hpp"
#include "hvax/util/hex.hpp"

TEST(Landing, HtmlDocument) {
  const std::string_view html(kLandingHtml);
  EXPECT_NE(html.find("<!DOCTYPE html>"), std::string_view::npos);
  EXPECT_NE(html.find("hvax"), std::string_view::npos);
  EXPECT_NE(html.find("/v1/ingest"), std::string_view::npos);
  EXPECT_NE(html.find("/v1/ingest/pdf"), std::string_view::npos);
  EXPECT_NE(html.find("/v1/ingest/check"), std::string_view::npos);
  EXPECT_NE(html.find("/v1/ingest/processed"), std::string_view::npos);
  EXPECT_NE(html.find("/v1/query/image"), std::string_view::npos);
  EXPECT_NE(html.find("/v1/query/template"), std::string_view::npos);
  EXPECT_NE(html.find("value=\"32\""), std::string_view::npos);
  EXPECT_NE(html.find("type=\"file\" accept=\"image/*,application/pdf,.pdf\" multiple"),
            std::string_view::npos);
  EXPECT_NE(html.find("sendIngestBatch"), std::string_view::npos);
  EXPECT_NE(html.find("/v1/ingest/pdf?detect_only=1&include_embedding=1"), std::string_view::npos);
  EXPECT_NE(html.find("hasFaceThumbnails"), std::string_view::npos);
  EXPECT_NE(html.find("has-thumb"), std::string_view::npos);
  EXPECT_NE(html.find("not my person"), std::string_view::npos);
  EXPECT_NE(html.find("zoomable"), std::string_view::npos);
  EXPECT_NE(html.find("IntersectionObserver"), std::string_view::npos);
  EXPECT_NE(html.find("MAX_IMAGE_REQUESTS = 4"), std::string_view::npos);
}

TEST(Config, DefaultSearchSizeIsThirtyTwo) { EXPECT_EQ(hvax::Config{}.default_k, 32); }

TEST(Http, ListenBacklogHandlesThumbnailBursts) { EXPECT_GE(CPPHTTPLIB_LISTEN_BACKLOG, 128); }

TEST(PipelineHelpers, SniffMime) {
  const uint8_t jpeg[] = {0xff, 0xd8, 0xff, 0x00};
  EXPECT_EQ(hvax::sniff_mime(jpeg), hvax::Mime::jpeg);
  const uint8_t png[] = {0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};
  EXPECT_EQ(hvax::sniff_mime(png), hvax::Mime::png);
}

TEST(Config, DefaultPixelLimitAcceptsHighResolutionPhotos) {
  EXPECT_GE(hvax::Config{}.max_pixels, int64_t{8480} * 5664);
}

TEST(HttpHelpers, AttachJobsOverwritesRemoteField) {
  const auto out = hvax::attach_jobs(nlohmann::json{{"faces", 1}, {"jobs", 8}}, 4);
  EXPECT_EQ(out.at("jobs").get<int>(), 4);
  EXPECT_EQ(out.at("faces").get<int>(), 1);
}

TEST(HttpHelpers, AttachJobsIgnoresNonObject) {
  const auto out = hvax::attach_jobs(nlohmann::json::array({1, 2}), 4);
  EXPECT_TRUE(out.is_array());
  EXPECT_EQ(out.size(), 2);
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

TEST(HttpHelpers, PdfWhiteMarginsAreTrimmed) {
  cv::Mat page(160, 200, CV_8UC3, cv::Scalar(255, 255, 255));
  page(cv::Rect(50, 40, 100, 80)).setTo(cv::Scalar(20, 40, 80));
  page.at<cv::Vec3b>(2, 2) = cv::Vec3b(0, 0, 0);
  page.at<cv::Vec3b>(157, 197) = cv::Vec3b(0, 0, 0);
  std::vector<uint8_t> encoded;
  ASSERT_TRUE(cv::imencode(".png", page, encoded));

  ASSERT_TRUE(hvax::trim_pdf_white_margins(encoded, 1'000'000));
  const cv::Mat cropped = cv::imdecode(encoded, cv::IMREAD_COLOR);
  ASSERT_FALSE(cropped.empty());
  EXPECT_LT(cropped.cols, page.cols);
  EXPECT_LT(cropped.rows, page.rows);
  EXPECT_GE(cropped.cols, 100);
  EXPECT_GE(cropped.rows, 80);
}

TEST(HttpHelpers, FullBleedPdfPageIsNotTrimmed) {
  cv::Mat page(80, 100, CV_8UC3, cv::Scalar(20, 40, 80));
  std::vector<uint8_t> encoded;
  ASSERT_TRUE(cv::imencode(".png", page, encoded));
  const auto original = encoded;

  EXPECT_FALSE(hvax::trim_pdf_white_margins(encoded, 1'000'000));
  EXPECT_EQ(encoded, original);
}
