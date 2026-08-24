#include "hvax/store/phash.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <gtest/gtest.h>

static cv::Mat make_photo(int seed, int w, int h) {
  cv::Mat img(h, w, CV_8UC3);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      img.at<cv::Vec3b>(y, x) = cv::Vec3b(
          static_cast<uint8_t>((x * 13 + seed) & 255),
          static_cast<uint8_t>((y * 7 + seed * 3) & 255),
          static_cast<uint8_t>((x * y + seed * 11) & 255));
    }
  }
  cv::GaussianBlur(img, img, cv::Size(5, 5), 1.2);
  cv::circle(img, {w / 3, h / 3}, w / 6, {40, 80, 200}, -1);
  cv::rectangle(img, {w / 2, h / 2}, {w - 10, h - 10}, {200, 40, 40}, -1);
  return img;
}

TEST(Phash, ResizeAndTranscodeStayClose) {
  cv::Mat src = make_photo(1, 640, 480);
  auto a = hvax::hash_image(src);
  cv::Mat small;
  cv::resize(src, small, cv::Size(320, 240));
  auto b = hvax::hash_image(small);
  EXPECT_LE(hvax::hamming64(a.phash, b.phash), 10);
  EXPECT_LE(hvax::hamming64(a.dhash, b.dhash), 12);

  std::vector<uint8_t> png;
  cv::imencode(".png", src, png);
  cv::Mat decoded = cv::imdecode(png, cv::IMREAD_COLOR);
  auto c = hvax::hash_image(decoded);
  EXPECT_LE(hvax::hamming64(a.phash, c.phash), 6);
  EXPECT_LE(hvax::hamming64(a.dhash, c.dhash), 6);
}

TEST(Phash, DifferentImagesAreFar) {
  auto a = hvax::hash_image(make_photo(1, 400, 400));
  auto b = hvax::hash_image(make_photo(99, 400, 400));
  EXPECT_GT(hvax::hamming64(a.phash, b.phash) + hvax::hamming64(a.dhash, b.dhash), 16);
}
