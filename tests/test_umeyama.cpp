#include "hvax/align/umeyama.hpp"

#include <cmath>

#include <gtest/gtest.h>

TEST(Umeyama, IdentityOnTemplate) {
  auto M = hvax::estimate_norm(hvax::arcface_dst(), 112);
  EXPECT_NEAR(M(0, 0), 1.f, 1e-4);
  EXPECT_NEAR(M(1, 1), 1.f, 1e-4);
  EXPECT_NEAR(M(0, 1), 0.f, 1e-4);
  EXPECT_NEAR(M(1, 0), 0.f, 1e-4);
  EXPECT_NEAR(M(0, 2), 0.f, 1e-3);
  EXPECT_NEAR(M(1, 2), 0.f, 1e-3);
}

TEST(Umeyama, Translation) {
  hvax::Landmark5 src = hvax::arcface_dst();
  for (auto& p : src.xy) {
    p[0] += 10.f;
    p[1] += -4.f;
  }
  auto M = hvax::estimate_norm(src, 112);
  // x' = x - 10, y' = y + 4
  EXPECT_NEAR(M(0, 0), 1.f, 1e-4);
  EXPECT_NEAR(M(1, 1), 1.f, 1e-4);
  EXPECT_NEAR(M(0, 2), -10.f, 1e-2);
  EXPECT_NEAR(M(1, 2), 4.f, 1e-2);
}
