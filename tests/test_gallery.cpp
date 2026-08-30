#include "hvax/align/umeyama.hpp"
#include "hvax/embed/arcface.hpp"
#include "hvax/store/gallery.hpp"
#include "hvax/util/sha256.hpp"

#include <atomic>
#include <cmath>
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

hvax::DetectedFace mixed_face(const cv::Mat& im, std::initializer_list<std::pair<int, float>> components) {
  auto face = dummy_face(im, 0);
  face.embedding.fill(0.f);
  for (const auto& [dimension, value] : components) face.embedding[static_cast<size_t>(dimension)] = value;
  hvax::l2_normalize(face.embedding.data());
  return face;
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

TEST(Gallery, TemplateSearchReturnsUniqueImages) {
  auto dir = tmpdir();
  hvax::Config cfg;
  cfg.data_dir = dir.string();
  hvax::Gallery g(cfg);
  auto first = photo(21, 200, 200);
  auto second = photo(22, 200, 200);
  const auto reference = dummy_face(first, 3).embedding;
  const auto first_bytes = encode_jpg(first);
  const auto first_id = g.insert(first_bytes, first, hvax::sha256_bytes(first_bytes), hvax::Mime::jpeg,
                                 hvax::hash_image(first), {dummy_face(first, 3), dummy_face(first, 3)});
  const auto second_bytes = encode_jpg(second);
  const auto second_id = g.insert(second_bytes, second, hvax::sha256_bytes(second_bytes), hvax::Mime::jpeg,
                                  hvax::hash_image(second), {dummy_face(second, 3)});

  const std::array<hvax::Embedding, 1> positives{reference};
  const auto hits = g.search_template(positives, {}, {}, 10, 0.5f);
  ASSERT_EQ(hits.size(), 2u);
  EXPECT_NE(hits[0].image_id, hits[1].image_id);
  EXPECT_TRUE(hits[0].image_id == first_id || hits[0].image_id == second_id);
  EXPECT_TRUE(hits[1].image_id == first_id || hits[1].image_id == second_id);
  std::filesystem::remove_all(dir);
}

TEST(Gallery, TemplateNegativesConservativelyDownrankAndExclude) {
  auto dir = tmpdir();
  hvax::Config cfg;
  cfg.data_dir = dir.string();
  hvax::Gallery g(cfg);
  auto distractor_image = photo(31, 200, 200);
  auto target_image = photo(32, 200, 200);
  const auto distractor_bytes = encode_jpg(distractor_image);
  const auto target_bytes = encode_jpg(target_image);
  const auto distractor_id = g.insert(distractor_bytes, distractor_image, hvax::sha256_bytes(distractor_bytes),
                                      hvax::Mime::jpeg, hvax::hash_image(distractor_image),
                                      {mixed_face(distractor_image, {{0, 0.6f}, {1, 0.8f}})});
  const auto target_id = g.insert(target_bytes, target_image, hvax::sha256_bytes(target_bytes), hvax::Mime::jpeg,
                                  hvax::hash_image(target_image),
                                  {mixed_face(target_image, {{0, 0.55f}, {2, std::sqrt(1.f - 0.55f * 0.55f)}})});
  std::array<hvax::Embedding, 1> positives{mixed_face(target_image, {{0, 1.f}}).embedding};
  std::array<hvax::Embedding, 1> negatives{mixed_face(target_image, {{1, 1.f}}).embedding};

  auto hits = g.search_template(positives, {}, {}, 2, -1.f);
  ASSERT_EQ(hits.size(), 2u);
  EXPECT_EQ(hits[0].image_id, distractor_id);
  hits = g.search_template(positives, negatives, {}, 2, -1.f);
  ASSERT_EQ(hits.size(), 2u);
  EXPECT_EQ(hits[0].image_id, target_id);

  const std::array<int64_t, 1> excluded{target_id};
  hits = g.search_template(positives, negatives, excluded, 2, -1.f);
  ASSERT_EQ(hits.size(), 1u);
  EXPECT_EQ(hits[0].image_id, distractor_id);
  std::filesystem::remove_all(dir);
}

TEST(Gallery, TemplateSearchUsesHnswCandidates) {
  auto dir = tmpdir();
  hvax::Config cfg;
  cfg.data_dir = dir.string();
  cfg.exact_until = 0;
  hvax::Gallery g(cfg);
  auto image = photo(41, 200, 200);
  const auto bytes = encode_jpg(image);
  const auto image_id = g.insert(bytes, image, hvax::sha256_bytes(bytes), hvax::Mime::jpeg,
                                 hvax::hash_image(image), {dummy_face(image, 9)});
  const std::array<hvax::Embedding, 1> positives{dummy_face(image, 9).embedding};
  const auto hits = g.search_template(positives, {}, {}, 32, 0.5f);
  ASSERT_EQ(hits.size(), 1u);
  EXPECT_EQ(hits[0].image_id, image_id);
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
