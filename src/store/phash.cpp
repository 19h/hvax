#include "hvax/store/phash.hpp"

#include <opencv2/imgproc.hpp>

namespace hvax {

PerceptualHash hash_image(const cv::Mat& bgr) {
  PerceptualHash h;
  if (bgr.empty()) return h;
  cv::Mat gray;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

  cv::Mat small32;
  cv::resize(gray, small32, cv::Size(32, 32), 0, 0, cv::INTER_AREA);
  cv::Mat f32;
  small32.convertTo(f32, CV_32F);
  cv::Mat dct;
  cv::dct(f32, dct);

  float acc = 0;
  for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 8; ++x)
      if (x || y) acc += dct.at<float>(y, x);
  const float avg = acc / 63.f;
  uint64_t ph = 0;
  int bit = 0;
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      if (x == 0 && y == 0) continue;
      if (dct.at<float>(y, x) > avg) ph |= (1ull << bit);
      ++bit;
    }
  }
  h.phash = ph;

  cv::Mat small9;
  cv::resize(gray, small9, cv::Size(9, 8), 0, 0, cv::INTER_AREA);
  uint64_t dh = 0;
  int db = 0;
  for (int y = 0; y < 8; ++y) {
    const uint8_t* row = small9.ptr<uint8_t>(y);
    for (int x = 0; x < 8; ++x) {
      if (row[x] > row[x + 1]) dh |= (1ull << db);
      ++db;
    }
  }
  h.dhash = dh;
  return h;
}

}  // namespace hvax
