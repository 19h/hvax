#include "hvax/embed/arcface.hpp"
#include "hvax/index/exact_scan.hpp"

#include <algorithm>
#include <cmath>
#include <random>

#include <gtest/gtest.h>

namespace {

std::vector<float> random_rows(int n, std::mt19937& rng) {
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> db(static_cast<size_t>(n) * hvax::kDim);
  for (auto& x : db) x = nd(rng);
  for (int i = 0; i < n; ++i) hvax::l2_normalize(db.data() + i * hvax::kDim);
  return db;
}

std::vector<std::pair<float, int>> naive_ranking(const std::vector<float>& db, int n, const float* q) {
  std::vector<std::pair<float, int>> naive;
  for (int i = 0; i < n; ++i) {
    float s = 0;
    const float* row = db.data() + i * hvax::kDim;
    for (int d = 0; d < hvax::kDim; ++d) s += q[d] * row[d];
    naive.push_back({s, i});
  }
  std::sort(naive.begin(), naive.end(), [](auto& a, auto& b) { return a.first > b.first; });
  return naive;
}

}  // namespace

TEST(ExactScan, AgreesWithNaive) {
  constexpr int n = 200;
  std::mt19937 rng(1);
  auto db = random_rows(n, rng);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> q(hvax::kDim);
  for (auto& x : q) x = nd(rng);
  hvax::l2_normalize(q.data());

  auto hits = hvax::exact_topk_f32(db.data(), n, q.data(), 5, nullptr, -2.f);
  ASSERT_EQ(hits.size(), 5u);
  auto naive = naive_ranking(db, n, q.data());
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(hits[static_cast<size_t>(i)].row, static_cast<uint64_t>(naive[static_cast<size_t>(i)].second));
    EXPECT_NEAR(hits[static_cast<size_t>(i)].score, naive[static_cast<size_t>(i)].first, 1e-4);
  }
}

TEST(ExactScan, SkipsTombstones) {
  std::vector<float> db(3 * hvax::kDim, 0.f);
  db[0] = 1.f;
  db[hvax::kDim + 1] = 1.f;
  db[2 * hvax::kDim] = 1.f;
  hvax::l2_normalize(db.data());
  hvax::l2_normalize(db.data() + hvax::kDim);
  hvax::l2_normalize(db.data() + 2 * hvax::kDim);
  std::vector<uint32_t> flags = {0, hvax::kTombstone, 0};
  auto hits = hvax::exact_topk_f32(db.data(), 3, db.data(), 3, flags.data(), -2.f);
  ASSERT_EQ(hits.size(), 2u);
  EXPECT_NE(hits[0].row, 1u);
  EXPECT_NE(hits[1].row, 1u);
}

TEST(ExactScan, SkipMaskHonoursLowQuality) {
  std::vector<float> db(2 * hvax::kDim, 0.f);
  db[0] = 1.f;
  db[hvax::kDim] = 1.f;
  std::vector<uint32_t> flags = {hvax::kLowQuality, 0};
  auto hidden = hvax::exact_topk_f32(db.data(), 2, db.data(), 2, flags.data(), -2.f);
  ASSERT_EQ(hidden.size(), 1u);
  EXPECT_EQ(hidden[0].row, 1u);
  auto shown = hvax::exact_topk_f32(db.data(), 2, db.data(), 2, flags.data(), -2.f, hvax::kTombstone);
  EXPECT_EQ(shown.size(), 2u);
}

TEST(Int8, QuantizeUsesTheFullRange) {
  std::vector<float> v(hvax::kDim, 0.f);
  v[0] = 0.26f;   // largest component seen in the 1.1M-face gallery
  v[1] = -0.26f;
  v[2] = 0.5f;    // beyond the range: must clamp, not wrap
  v[3] = -0.001f;
  std::vector<int8_t> q(hvax::kDim);
  hvax::quantize_i8(v.data(), q.data());
  EXPECT_EQ(q[0], static_cast<int8_t>(std::lrintf(0.26f * hvax::kI8Scale)));
  EXPECT_EQ(q[1], -q[0]);
  EXPECT_EQ(q[2], 127);
  EXPECT_EQ(q[3], 0);
  EXPECT_GT(q[0], 120) << "a 0.26 component should use nearly all of the int8 range";
}

TEST(Int8, DotMatchesScalar) {
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> ud(-127, 127);
  for (int trial = 0; trial < 50; ++trial) {
    std::vector<int8_t> a(hvax::kDim), b(hvax::kDim);
    for (int i = 0; i < hvax::kDim; ++i) {
      a[static_cast<size_t>(i)] = static_cast<int8_t>(ud(rng));
      b[static_cast<size_t>(i)] = static_cast<int8_t>(ud(rng));
    }
    int32_t naive = 0;
    for (int i = 0; i < hvax::kDim; ++i)
      naive += static_cast<int32_t>(a[static_cast<size_t>(i)]) * static_cast<int32_t>(b[static_cast<size_t>(i)]);
    EXPECT_EQ(hvax::dot512_i8(a.data(), b.data()), naive) << "kernel " << hvax::i8_kernel_name();
  }
  // extremes: all -127 * -127 must not overflow int16 accumulation paths
  std::vector<int8_t> lo(hvax::kDim, -127);
  EXPECT_EQ(hvax::dot512_i8(lo.data(), lo.data()), 127 * 127 * hvax::kDim);
}

TEST(Int8, CandidateScanRerankedIsExact) {
  constexpr int n = 4000;
  std::mt19937 rng(3);
  auto db = random_rows(n, rng);
  std::vector<int8_t> db8(static_cast<size_t>(n) * hvax::kDim);
  for (int i = 0; i < n; ++i) hvax::quantize_i8(db.data() + i * hvax::kDim, db8.data() + i * hvax::kDim);

  std::normal_distribution<float> nd(0.f, 1.f);
  int exact_top5 = 0;
  constexpr int trials = 40;
  for (int t = 0; t < trials; ++t) {
    // queries near a database row so there is real structure to rank
    std::vector<float> q(db.begin() + (t * 97 % n) * hvax::kDim, db.begin() + (t * 97 % n + 1) * hvax::kDim);
    for (auto& x : q) x += 0.6f * nd(rng) / std::sqrt(static_cast<float>(hvax::kDim));
    hvax::l2_normalize(q.data());
    std::vector<int8_t> q8(hvax::kDim);
    hvax::quantize_i8(q.data(), q8.data());

    auto cand = hvax::exact_topk_i8(db8.data(), n, q8.data(), 20, nullptr, -2.f);
    ASSERT_EQ(cand.size(), 20u);
    // approximate scores are within quantization error of the true cosine
    for (auto& c : cand) {
      float s = 0;
      for (int d = 0; d < hvax::kDim; ++d) s += q[static_cast<size_t>(d)] * db[c.row * hvax::kDim + static_cast<size_t>(d)];
      EXPECT_NEAR(c.score, s, 0.01f);
      c.score = s;
    }
    std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.score > b.score; });
    auto naive = naive_ranking(db, n, q.data());
    bool same = true;
    for (int i = 0; i < 5; ++i)
      if (cand[static_cast<size_t>(i)].row != static_cast<uint64_t>(naive[static_cast<size_t>(i)].second)) same = false;
    exact_top5 += same;
  }
  EXPECT_EQ(exact_top5, trials) << "int8 candidates + f32 rerank must reproduce the exact top-5";
}
