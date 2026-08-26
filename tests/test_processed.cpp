#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <cmath>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>

#include "hvax/engine.hpp"
#include "hvax/processed.hpp"
#include "hvax/store/phash.hpp"
#include "hvax/util/sha256.hpp"

namespace {

std::filesystem::path processed_tmpdir() {
  static std::atomic<int> sequence{0};
  auto path = std::filesystem::temp_directory_path() /
              ("hvax-processed-test-" + std::to_string(::getpid()) + "-" + std::to_string(sequence.fetch_add(1)));
  std::filesystem::create_directories(path);
  return path;
}

hvax::DetectedFace processed_face() {
  hvax::DetectedFace face;
  face.box = {-2.f, 4.f, 70.f, 80.f};
  face.det_score = 0.9f;
  for (size_t i = 0; i < face.kps.xy.size(); ++i)
    face.kps.xy[i] = {10.f + static_cast<float>(i), 20.f + static_cast<float>(i)};
  face.embedding[7] = 2.f;
  return face;
}

std::vector<uint8_t> processed_jpeg(cv::Mat& image) {
  image = cv::Mat(96, 96, CV_8UC3, cv::Scalar(30, 80, 160));
  std::vector<uint8_t> bytes;
  cv::imencode(".jpg", image, bytes);
  return bytes;
}

}  // namespace

TEST(ProcessedPayload, RoundTripValidatesAndNormalizes) {
  const auto encoded = hvax::processed_payload_json({processed_face()}).dump();
  std::vector<hvax::DetectedFace> faces;
  std::string error;
  ASSERT_TRUE(hvax::parse_processed_payload(encoded, faces, error)) << error;
  ASSERT_EQ(faces.size(), 1u);
  ASSERT_TRUE(hvax::validate_processed_faces(faces, 64, 64, error)) << error;
  EXPECT_FLOAT_EQ(faces[0].box.x1, 0.f);
  EXPECT_FLOAT_EQ(faces[0].box.x2, 64.f);
  double norm2 = 0;
  for (float value : faces[0].embedding) norm2 += static_cast<double>(value) * value;
  EXPECT_NEAR(norm2, 1.0, 1e-6);
}

TEST(ProcessedPayload, RejectsInvalidEmbedding) {
  auto payload = hvax::processed_payload_json({processed_face()});
  payload["faces"][0]["embedding"] = std::vector<float>(hvax::kDim, 0.f);
  std::vector<hvax::DetectedFace> faces;
  std::string error;
  ASSERT_TRUE(hvax::parse_processed_payload(payload.dump(), faces, error));
  EXPECT_FALSE(hvax::validate_processed_faces(faces, 100, 100, error));
  EXPECT_NE(error.find("norm"), std::string::npos);
}

TEST(ProcessedPayload, RejectsIncompatibleModel) {
  auto payload = hvax::processed_payload_json({processed_face()});
  payload["model"] = "different-embedding-space";
  std::vector<hvax::DetectedFace> faces;
  std::string error;
  EXPECT_FALSE(hvax::parse_processed_payload(payload.dump(), faces, error));
  EXPECT_NE(error.find("model"), std::string::npos);
}

TEST(ProcessedIngest, StoresAndDeduplicatesWithoutLoadingModels) {
  const auto dir = processed_tmpdir();
  hvax::Config config;
  config.data_dir = dir.string();
  config.models_dir = (dir / "models-do-not-exist").string();
  config.dedup = hvax::DedupMode::sha256;
  hvax::Engine engine(config);

  cv::Mat image;
  const auto bytes = processed_jpeg(image);
  std::vector<hvax::DetectedFace> faces{processed_face()};
  std::string error;
  ASSERT_TRUE(hvax::validate_processed_faces(faces, image.cols, image.rows, error)) << error;

  const auto first = engine.ingest_processed(bytes, image, faces);
  EXPECT_EQ(first.status, hvax::IngestStatus::stored);
  EXPECT_FALSE(first.duplicate);
  ASSERT_EQ(first.faces.size(), 1u);
  const auto hits = engine.query_embedding(faces[0].embedding, 5, -1.f);
  ASSERT_FALSE(hits.empty());
  EXPECT_EQ(hits[0].image_id, first.image_id);
  EXPECT_EQ(hits[0].sha256, first.sha256);

  const auto second = engine.ingest_processed(bytes, image, faces);
  EXPECT_EQ(second.status, hvax::IngestStatus::duplicate);
  EXPECT_TRUE(second.duplicate);
  EXPECT_EQ(second.duplicate_kind, "sha256");

  std::filesystem::remove_all(dir);
}

TEST(ProcessedIngest, PreflightSkipsOnlySafeDuplicates) {
  const auto dir = processed_tmpdir();
  hvax::Config config;
  config.data_dir = dir.string();
  config.models_dir = (dir / "models-do-not-exist").string();
  config.dedup = hvax::DedupMode::perceptual;
  hvax::Engine engine(config);

  cv::Mat image;
  const auto bytes = processed_jpeg(image);
  const auto sha = hvax::sha256_bytes(bytes);
  const auto perceptual = hvax::hash_image(image);
  const auto stored = engine.ingest_processed(bytes, image, {processed_face()});
  ASSERT_EQ(stored.status, hvax::IngestStatus::stored);

  const auto exact = engine.check_ingest(sha, perceptual.phash, perceptual.dhash, image.cols, image.rows);
  EXPECT_TRUE(exact.duplicate);
  EXPECT_FALSE(exact.process_required);
  EXPECT_EQ(exact.duplicate_kind, "sha256");
  EXPECT_EQ(exact.image_id, stored.image_id);

  auto different_sha = sha;
  different_sha[0] ^= 0xff;
  const auto perceptual_duplicate =
      engine.check_ingest(different_sha, perceptual.phash, perceptual.dhash, image.cols / 2, image.rows / 2);
  EXPECT_TRUE(perceptual_duplicate.duplicate);
  EXPECT_FALSE(perceptual_duplicate.process_required);
  EXPECT_EQ(perceptual_duplicate.duplicate_kind, "perceptual");

  const auto possible_upgrade =
      engine.check_ingest(different_sha, perceptual.phash, perceptual.dhash, image.cols * 2, image.rows * 2);
  EXPECT_FALSE(possible_upgrade.duplicate);
  EXPECT_TRUE(possible_upgrade.process_required);

  const auto unknown = engine.check_ingest(different_sha, ~perceptual.phash, ~perceptual.dhash, image.cols, image.rows);
  EXPECT_FALSE(unknown.duplicate);
  EXPECT_TRUE(unknown.process_required);

  std::filesystem::remove_all(dir);
}
