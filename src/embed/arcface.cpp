#include "hvax/embed/arcface.hpp"

#include <cmath>
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

namespace hvax {

void l2_normalize(float* v, int n) {
  double acc = 0;
  for (int i = 0; i < n; ++i) acc += static_cast<double>(v[i]) * v[i];
  const float inv = acc > 0 ? static_cast<float>(1.0 / std::sqrt(acc)) : 1.f;
  for (int i = 0; i < n; ++i) v[i] *= inv;
}

float dot512(const float* a, const float* b) {
#if defined(__AVX2__)
  __m256 acc = _mm256_setzero_ps();
  for (int i = 0; i < 512; i += 8) {
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
  }
  __m128 lo = _mm256_castps256_ps128(acc);
  __m128 hi = _mm256_extractf128_ps(acc, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
#elif defined(__ARM_NEON)
  float32x4_t acc0 = vdupq_n_f32(0.f);
  float32x4_t acc1 = vdupq_n_f32(0.f);
  float32x4_t acc2 = vdupq_n_f32(0.f);
  float32x4_t acc3 = vdupq_n_f32(0.f);
  for (int i = 0; i < 512; i += 16) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
  }
  return vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
#else
  float s = 0;
  for (int i = 0; i < 512; ++i) s += a[i] * b[i];
  return s;
#endif
}

namespace {

void blob_nchw_112(const cv::Mat& bgr, float* dst) {
  const int h = bgr.rows;
  const int w = bgr.cols;
  const int hw = h * w;
  constexpr float mean = 127.5f;
  constexpr float stdv = 127.5f;
  constexpr float scale = 1.0f / stdv;
  constexpr float bias = -mean * scale;
  for (int y = 0; y < h; ++y) {
    const cv::Vec3b* row = bgr.ptr<cv::Vec3b>(y);
    for (int x = 0; x < w; ++x) {
      const int i = y * w + x;
      dst[0 * hw + i] = static_cast<float>(row[x][2]) * scale + bias;
      dst[1 * hw + i] = static_cast<float>(row[x][1]) * scale + bias;
      dst[2 * hw + i] = static_cast<float>(row[x][0]) * scale + bias;
    }
  }
}

}  // namespace

ArcFace::ArcFace(OrtContext& ctx, const std::string& model_path) : ctx_(&ctx) {
  session_ = ctx.load(model_path);
  io_ = inspect(*session_);
  bind_io(io_);
  if (io_.input_shape.size() >= 4 && io_.input_shape[2] > 0) {
    input_size_ = static_cast<int>(io_.input_shape[2]);
  }
  dynamic_batch_ = !io_.input_shape.empty() && io_.input_shape[0] < 0;
  spdlog::info("ArcFace input_size={} dynamic_batch={}", input_size_, dynamic_batch_);
}

Embedding ArcFace::run_blob(std::vector<float>& blob, int n) {
  const int64_t shape[] = {n, 3, input_size_, input_size_};
  Ort::Value input =
      Ort::Value::CreateTensor<float>(ctx_->cpu_mem(), blob.data(), blob.size(), shape, 4);
  auto outputs = session_->Run(Ort::RunOptions{nullptr}, io_.input_name_ptrs.data(), &input, 1,
                               io_.output_name_ptrs.data(), io_.output_name_ptrs.size());
  if (outputs.empty() || outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() !=
                             static_cast<size_t>(n) * kDim)
    throw std::runtime_error("ArcFace output tensor size does not match its input batch");
  const float* y = outputs[0].GetTensorData<float>();
  Embedding e{};
  std::copy(y, y + kDim, e.begin());
  l2_normalize(e.data());
  return e;
}

Embedding ArcFace::embed(const cv::Mat& aligned_bgr_112) {
  thread_local std::vector<float> blob;
  blob.resize(static_cast<size_t>(3 * input_size_ * input_size_));
  cv::Mat resized = aligned_bgr_112;
  if (aligned_bgr_112.rows != input_size_ || aligned_bgr_112.cols != input_size_) {
    cv::resize(aligned_bgr_112, resized, cv::Size(input_size_, input_size_));
  }
  blob_nchw_112(resized, blob.data());
  return run_blob(blob, 1);
}

std::vector<Embedding> ArcFace::embed_batch(const std::vector<cv::Mat>& aligned) {
  std::vector<Embedding> out;
  out.reserve(aligned.size());
  if (aligned.empty()) return out;
  if (!dynamic_batch_ || aligned.size() == 1) {
    for (const auto& im : aligned) out.push_back(embed(im));
    return out;
  }
  const int n = static_cast<int>(aligned.size());
  const int plane = 3 * input_size_ * input_size_;
  thread_local std::vector<float> blob;
  blob.resize(static_cast<size_t>(n * plane));
  for (int i = 0; i < n; ++i) {
    cv::Mat resized = aligned[static_cast<size_t>(i)];
    if (resized.rows != input_size_ || resized.cols != input_size_) {
      cv::resize(aligned[static_cast<size_t>(i)], resized, cv::Size(input_size_, input_size_));
    }
    blob_nchw_112(resized, blob.data() + static_cast<size_t>(i * plane));
  }
  const int64_t shape[] = {n, 3, input_size_, input_size_};
  Ort::Value input =
      Ort::Value::CreateTensor<float>(ctx_->cpu_mem(), blob.data(), blob.size(), shape, 4);
  auto outputs = session_->Run(Ort::RunOptions{nullptr}, io_.input_name_ptrs.data(), &input, 1,
                               io_.output_name_ptrs.data(), io_.output_name_ptrs.size());
  if (outputs.empty() || outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() !=
                             static_cast<size_t>(n) * kDim)
    throw std::runtime_error("ArcFace output tensor size does not match its input batch");
  const float* y = outputs[0].GetTensorData<float>();
  for (int i = 0; i < n; ++i) {
    Embedding e{};
    std::copy(y + i * kDim, y + (i + 1) * kDim, e.begin());
    l2_normalize(e.data());
    out.push_back(e);
  }
  return out;
}

}  // namespace hvax
