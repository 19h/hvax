#include "hvax/align/umeyama.hpp"

#include <cmath>

#include <opencv2/imgproc.hpp>

namespace hvax {

cv::Matx23f estimate_norm(const Landmark5& lmk, int image_size) {
  const float ratio = static_cast<float>(image_size) / 112.0f;
  cv::Mat src(5, 2, CV_64F);
  cv::Mat dst(5, 2, CV_64F);
  const auto& tmpl = arcface_dst();
  for (int i = 0; i < 5; ++i) {
    src.at<double>(i, 0) = lmk.xy[static_cast<size_t>(i)][0];
    src.at<double>(i, 1) = lmk.xy[static_cast<size_t>(i)][1];
    dst.at<double>(i, 0) = tmpl.xy[static_cast<size_t>(i)][0] * ratio;
    dst.at<double>(i, 1) = tmpl.xy[static_cast<size_t>(i)][1] * ratio;
  }

  double mu_sx = 0, mu_sy = 0, mu_dx = 0, mu_dy = 0;
  for (int i = 0; i < 5; ++i) {
    mu_sx += src.at<double>(i, 0);
    mu_sy += src.at<double>(i, 1);
    mu_dx += dst.at<double>(i, 0);
    mu_dy += dst.at<double>(i, 1);
  }
  mu_sx /= 5.0;
  mu_sy /= 5.0;
  mu_dx /= 5.0;
  mu_dy /= 5.0;
  cv::Mat src_c = src.clone();
  cv::Mat dst_c = dst.clone();
  for (int i = 0; i < 5; ++i) {
    src_c.at<double>(i, 0) -= mu_sx;
    src_c.at<double>(i, 1) -= mu_sy;
    dst_c.at<double>(i, 0) -= mu_dx;
    dst_c.at<double>(i, 1) -= mu_dy;
  }

  double var_src = 0;
  for (int i = 0; i < 5; ++i) {
    var_src += src_c.at<double>(i, 0) * src_c.at<double>(i, 0) + src_c.at<double>(i, 1) * src_c.at<double>(i, 1);
  }
  var_src /= 5.0;

  cv::Mat cov = dst_c.t() * src_c / 5.0;  // 2x2
  cv::Mat w, u, vt;
  cv::SVD::compute(cov, w, u, vt, cv::SVD::FULL_UV);

  cv::Mat d = cv::Mat::eye(2, 2, CV_64F);
  double det = cv::determinant(cov);
  if (det < 0) d.at<double>(1, 1) = -1;
  // also handle reflection via U V
  double det_uv = cv::determinant(u) * cv::determinant(vt);
  if (det_uv < 0) d.at<double>(1, 1) = -1;

  cv::Mat r = u * d * vt;
  double scale = (w.at<double>(0) * d.at<double>(0, 0) + w.at<double>(1) * d.at<double>(1, 1)) / var_src;
  cv::Mat t = (cv::Mat_<double>(2, 1) << mu_dx, mu_dy) - scale * r * (cv::Mat_<double>(2, 1) << mu_sx, mu_sy);

  cv::Matx23f M;
  M(0, 0) = static_cast<float>(scale * r.at<double>(0, 0));
  M(0, 1) = static_cast<float>(scale * r.at<double>(0, 1));
  M(0, 2) = static_cast<float>(t.at<double>(0));
  M(1, 0) = static_cast<float>(scale * r.at<double>(1, 0));
  M(1, 1) = static_cast<float>(scale * r.at<double>(1, 1));
  M(1, 2) = static_cast<float>(t.at<double>(1));
  return M;
}

cv::Mat norm_crop(const cv::Mat& bgr, const Landmark5& lmk, int image_size) {
  cv::Matx23f M = estimate_norm(lmk, image_size);
  cv::Mat out;
  cv::warpAffine(bgr, out, M, cv::Size(image_size, image_size), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                 cv::Scalar(0, 0, 0));
  return out;
}

}  // namespace hvax
