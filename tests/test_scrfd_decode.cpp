#include "hvax/detect/scrfd.hpp"

#include <gtest/gtest.h>

TEST(Nms, SuppressesOverlap) {
  std::vector<hvax::BBox> boxes = {
      {0, 0, 10, 10},
      {1, 1, 11, 11},
      {50, 50, 60, 60},
  };
  std::vector<float> scores = {0.9f, 0.8f, 0.7f};
  auto keep = hvax::nms_boxes(boxes, scores, 0.4f);
  ASSERT_EQ(keep.size(), 2u);
  EXPECT_EQ(keep[0], 0);
  EXPECT_EQ(keep[1], 2);
}
