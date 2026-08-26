#include "hvax/align/umeyama.hpp"
#include "hvax/embed/arcface.hpp"
#include "hvax/store/gallery.hpp"
#include "hvax/util/sha256.hpp"

#include <atomic>
#include <filesystem>
#include <unistd.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <gtest/gtest.h>

namespace {

std::filesystem::path tmpdir() {
  static std::atomic<int> seq{0};
  auto p = std::filesystem::temp_directory_path() /
           ("hvax-test-" + std::to_string(::getpid()) + "-" + std::to_string(seq.fetch_add(1)));
  std::filesystem::create_directories(p);
  return p;
}

cv::Mat photo(int seed, int w, int h) {
  cv::Mat img(h, w, CV_8UC3);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      img.at<cv::Vec3b>(y, x) =
          cv::Vec3b(static_cast<uint8_t>(x + seed), static_cast<uint8_t>(y + seed * 2), 80);
  cv::circle(img, {w / 2, h / 2}, w / 4, {30, 90, 220}, -1);
  return img;
}

std::vector<uint8_t> encode_jpg(const cv::Mat& im, int q = 90) {
  std::vector<uint8_t> buf;
  cv::imencode(".jpg", im, buf, {cv::IMWRITE_JPEG_QUALITY, q});
  return buf;
}

hvax::DetectedFace dummy_face(const cv::Mat& im, int seed) {
  hvax::DetectedFace f;
  f.box = {10, 10, static_cast<float>(im.cols - 10), static_cast<float>(im.rows - 10)};
  f.det_score = 0.9f;
  f.kps = hvax::arcface_dst();
  for (int i = 0; i < hvax::kDim; ++i) f.embedding[static_cast<size_t>(i)] = (i == seed % hvax::kDim) ? 1.f : 0.f;
  hvax::l2_normalize(f.embedding.data());
  return f;
}

}  // namespace

TEST(Gallery, InsertSearchNoSql) {
  auto dir = tmpdir();
  hvax::Config cfg;
  cfg.data_dir = dir.string();
  hvax::Gallery g(cfg);
  auto img = photo(1, 200, 200);
  auto bytes = encode_jpg(img);
  auto sha = hvax::sha256_bytes(bytes);
  auto ph = hvax::hash_image(img);
  auto face = dummy_face(img, 3);
  const int64_t id = g.insert(bytes, img, sha, hvax::Mime::jpeg, ph, {face});
  EXPECT_EQ(g.live_images(), 1u);
  EXPECT_EQ(g.live_faces(), 1u);
  auto hits = g.search(face.embedding.data(), 5, -1.f);
  ASSERT_FALSE(hits.empty());
  EXPECT_EQ(hits[0].image_id, id);
  EXPECT_EQ(hits[0].sha256, sha);
  EXPECT_GT(hits[0].score, 0.99f);
  std::filesystem::remove_all(dir);
}

TEST(Gallery, MasterReplaceKeepsId) {
  auto dir = tmpdir();
  hvax::Config cfg;
  cfg.data_dir = dir.string();
  hvax::Gallery g(cfg);
  auto small = photo(7, 80, 80);
  auto large = photo(7, 240, 240);
  auto b1 = encode_jpg(small);
  auto b2 = encode_jpg(large);
  auto f1 = dummy_face(small, 1);
  auto f2 = dummy_face(large, 1);
  auto id = g.insert(b1, small, hvax::sha256_bytes(b1), hvax::Mime::jpeg, hvax::hash_image(small), {f1});
  auto faces = g.upgrade(id, b2, large, hvax::sha256_bytes(b2), hvax::Mime::jpeg, hvax::hash_image(large), {f2});
  EXPECT_EQ(g.live_images(), 1u);
  auto im = g.image(id);
  EXPECT_EQ(im.width, 240);
  EXPECT_EQ(im.height, 240);
  ASSERT_FALSE(faces.empty());
  EXPECT_EQ(faces[0].face_id, 0);
  EXPECT_NEAR(faces[0].box.x2, 230.f, 1.f);
  std::filesystem::remove_all(dir);
}

TEST(Gallery, PerceptualFind) {
  auto dir = tmpdir();
  hvax::Config cfg;
  cfg.data_dir = dir.string();
  hvax::Gallery g(cfg);
  auto img = photo(3, 160, 160);
  auto bytes = encode_jpg(img);
  g.insert(bytes, img, hvax::sha256_bytes(bytes), hvax::Mime::jpeg, hvax::hash_image(img), {dummy_face(img, 2)});
  cv::Mat down;
  cv::resize(img, down, cv::Size(80, 80));
  auto h = hvax::hash_image(down);
  auto hit = g.find_perceptual(h.phash, h.dhash);
  ASSERT_TRUE(hit.has_value());
  EXPECT_TRUE(hit->hard);
  std::filesystem::remove_all(dir);
}
