#include "hvax/embed/arcface.hpp"
#include "hvax/index/exact_scan.hpp"

#include <algorithm>
#include <random>

#include <gtest/gtest.h>

TEST(ExactScan, AgreesWithNaive) {
  constexpr int n = 200;
  std::mt19937 rng(1);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> db(static_cast<size_t>(n * hvax::kDim));
  for (auto& x : db) x = nd(rng);
  for (int i = 0; i < n; ++i) hvax::l2_normalize(db.data() + i * hvax::kDim);
  std::vector<float> q(hvax::kDim);
  for (auto& x : q) x = nd(rng);
  hvax::l2_normalize(q.data());

  auto hits = hvax::exact_topk_f32(db.data(), n, q.data(), 5, nullptr, -2.f);
  ASSERT_EQ(hits.size(), 5u);

  std::vector<std::pair<float, int>> naive;
  for (int i = 0; i < n; ++i) {
    float s = 0;
    const float* row = db.data() + i * hvax::kDim;
    for (int d = 0; d < hvax::kDim; ++d) s += q[static_cast<size_t>(d)] * row[d];
    naive.push_back({s, i});
  }
  std::sort(naive.begin(), naive.end(), [](auto& a, auto& b) { return a.first > b.first; });
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
