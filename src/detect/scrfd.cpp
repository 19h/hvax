#include "hvax/detect/scrfd.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

namespace hvax {

namespace {

cv::Mat letterbox(const cv::Mat& img, int size, float& det_scale, int& new_w, int& new_h) {
  const int h = img.rows;
  const int w = img.cols;
  const float im_ratio = static_cast<float>(h) / static_cast<float>(w);
  const float model_ratio = 1.0f;  // square input
  if (im_ratio > model_ratio) {
    new_h = size;
    new_w = static_cast<int>(static_cast<float>(new_h) / im_ratio);
  } else {
    new_w = size;
    new_h = static_cast<int>(static_cast<float>(new_w) * im_ratio);
  }
  det_scale = static_cast<float>(new_h) / static_cast<float>(h);
  thread_local cv::Mat out;
  out.create(size, size, CV_8UC3);
  out.setTo(cv::Scalar::all(0));
  cv::Mat resized = out(cv::Rect(0, 0, new_w, new_h));
  cv::resize(img, resized, resized.size(), 0, 0, cv::INTER_LINEAR);
  return out;
}

void blob_nchw(const cv::Mat& bgr, float mean, float std, std::vector<float>& out) {
  const int h = bgr.rows;
  const int w = bgr.cols;
  out.resize(static_cast<size_t>(3 * h * w));
  const int hw = h * w;
  const float scale = 1.0f / std;
  const float bias = -mean * scale;
  for (int y = 0; y < h; ++y) {
    const cv::Vec3b* row = bgr.ptr<cv::Vec3b>(y);
    for (int x = 0; x < w; ++x) {
      const int i = y * w + x;
      // swapRB: R,G,B from BGR
      out[static_cast<size_t>(0 * hw + i)] = static_cast<float>(row[x][2]) * scale + bias;
      out[static_cast<size_t>(1 * hw + i)] = static_cast<float>(row[x][1]) * scale + bias;
      out[static_cast<size_t>(2 * hw + i)] = static_cast<float>(row[x][0]) * scale + bias;
    }
  }
}

BBox distance2bbox(float cx, float cy, const float* d) {
  return BBox{cx - d[0], cy - d[1], cx + d[2], cy + d[3]};
}

Landmark5 distance2kps(float cx, float cy, const float* d) {
  Landmark5 k;
  for (int i = 0; i < 5; ++i) {
    k.xy[static_cast<size_t>(i)][0] = cx + d[i * 2];
    k.xy[static_cast<size_t>(i)][1] = cy + d[i * 2 + 1];
  }
  return k;
}

}  // namespace

std::vector<int> nms_boxes(const std::vector<BBox>& boxes, const std::vector<float>& scores, float thresh) {
  const int n = static_cast<int>(boxes.size());
  std::vector<int> order(static_cast<size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return scores[static_cast<size_t>(a)] > scores[static_cast<size_t>(b)]; });
  std::vector<float> areas(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    const auto& b = boxes[static_cast<size_t>(i)];
    areas[static_cast<size_t>(i)] = (b.x2 - b.x1 + 1.f) * (b.y2 - b.y1 + 1.f);
  }
  std::vector<int> keep;
  std::vector<char> dead(static_cast<size_t>(n), 0);
  for (int oi = 0; oi < n; ++oi) {
    int i = order[static_cast<size_t>(oi)];
    if (dead[static_cast<size_t>(i)]) continue;
    keep.push_back(i);
    const auto& bi = boxes[static_cast<size_t>(i)];
    for (int oj = oi + 1; oj < n; ++oj) {
      int j = order[static_cast<size_t>(oj)];
      if (dead[static_cast<size_t>(j)]) continue;
      const auto& bj = boxes[static_cast<size_t>(j)];
      const float xx1 = std::max(bi.x1, bj.x1);
      const float yy1 = std::max(bi.y1, bj.y1);
      const float xx2 = std::min(bi.x2, bj.x2);
      const float yy2 = std::min(bi.y2, bj.y2);
      const float w = std::max(0.f, xx2 - xx1 + 1.f);
      const float h = std::max(0.f, yy2 - yy1 + 1.f);
      const float inter = w * h;
      const float ovr = inter / (areas[static_cast<size_t>(i)] + areas[static_cast<size_t>(j)] - inter);
      if (ovr > thresh) dead[static_cast<size_t>(j)] = 1;
    }
  }
  return keep;
}

Scrfd::Scrfd(OrtContext& ctx, const std::string& model_path, int det_size, float det_thresh, float nms_thresh)
    : ctx_(&ctx), det_size_(det_size), det_thresh_(det_thresh), nms_thresh_(nms_thresh) {
  if (det_size_ <= 0 || det_size_ % 32 != 0)
    throw std::invalid_argument("SCRFD detector size must be a positive multiple of 32");
  session_ = ctx.load(model_path);
  io_ = inspect(*session_);
  bind_io(io_);
  if (io_.input_shape.size() != 4) throw std::runtime_error("SCRFD input must be rank 4");
  if (ctx.coreml_enabled() &&
      (io_.input_shape[0] != 1 || io_.input_shape[1] != 3 || io_.input_shape[2] != det_size_ ||
       io_.input_shape[3] != det_size_)) {
    throw std::runtime_error("CoreML SCRFD spatial specialization failed: expected [1,3," +
                             std::to_string(det_size_) + "," + std::to_string(det_size_) + "]");
  }
  if (io_.output_shapes.empty()) throw std::runtime_error("SCRFD has no outputs");
  batched_ = io_.output_shapes[0].size() == 3;
  const size_t nout = io_.output_names.size();
  if (nout == 6) {
    fmc_ = 3;
    strides_ = {8, 16, 32};
    num_anchors_ = 2;
    use_kps_ = false;
  } else if (nout == 9) {
    fmc_ = 3;
    strides_ = {8, 16, 32};
    num_anchors_ = 2;
    use_kps_ = true;
  } else if (nout == 10) {
    fmc_ = 5;
    strides_ = {8, 16, 32, 64, 128};
    num_anchors_ = 1;
    use_kps_ = false;
  } else if (nout == 15) {
    fmc_ = 5;
    strides_ = {8, 16, 32, 64, 128};
    num_anchors_ = 1;
    use_kps_ = true;
  } else {
    spdlog::warn("SCRFD unexpected output count {}, assuming 9-output kps pack", nout);
    fmc_ = 3;
    use_kps_ = nout >= 9;
    num_anchors_ = 2;
  }
  if (det_size_ % *std::max_element(strides_.begin(), strides_.end()) != 0)
    throw std::invalid_argument("SCRFD detector size must be divisible by its maximum feature stride");
  anchor_centers_.reserve(strides_.size());
  for (const int stride : strides_) {
    const int height = det_size_ / stride;
    const int width = det_size_ / stride;
    std::vector<std::array<float, 2>> centers;
    centers.reserve(static_cast<size_t>(height * width * num_anchors_));
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int anchor = 0; anchor < num_anchors_; ++anchor)
          centers.push_back({static_cast<float>(x * stride), static_cast<float>(y * stride)});
      }
    }
    anchor_centers_.push_back(std::move(centers));
  }
  spdlog::info("SCRFD batched={} kps={} fmc={} anchors={} det_size={}", batched_, use_kps_, fmc_,
               num_anchors_, det_size_);
}

Scrfd::ForwardOut Scrfd::forward(const cv::Mat& padded_bgr) {
  thread_local std::vector<float> blob;
  blob_nchw(padded_bgr, 127.5f, 128.0f, blob);
  const int64_t shape[] = {1, 3, padded_bgr.rows, padded_bgr.cols};
  Ort::Value input = Ort::Value::CreateTensor<float>(ctx_->cpu_mem(), blob.data(), blob.size(), shape, 4);
  auto outputs = session_->Run(Ort::RunOptions{nullptr}, io_.input_name_ptrs.data(), &input, 1,
                               io_.output_name_ptrs.data(), io_.output_name_ptrs.size());

  ForwardOut out;
  out.scores.reserve(256);
  out.boxes.reserve(256);
  out.kps.reserve(256);

  for (size_t si = 0; si < strides_.size(); ++si) {
    const int stride = strides_[si];
    if (si >= outputs.size() || si + static_cast<size_t>(fmc_) >= outputs.size()) break;

    auto& score_t = outputs[si];
    auto& bbox_t = outputs[si + static_cast<size_t>(fmc_)];
    const float* scores = score_t.GetTensorData<float>();
    const float* bboxes = bbox_t.GetTensorData<float>();
    const float* kps = nullptr;
    if (use_kps_ && si + static_cast<size_t>(fmc_) * 2 < outputs.size()) {
      kps = outputs[si + static_cast<size_t>(fmc_) * 2].GetTensorData<float>();
    }

    const auto& ssh = io_.output_shapes[si];
    const auto& anc = anchor_centers_[si];
    const size_t n = anc.size();

    const size_t score_width =
        ssh.size() == 2 ? static_cast<size_t>(std::max<int64_t>(ssh[1], 1)) : 1;
    for (size_t i = 0; i < n; ++i) {
      const float sc = scores[i * score_width];
      if (sc < det_thresh_) continue;
      const float* bd = bboxes + i * 4;
      float dist[4] = {bd[0] * stride, bd[1] * stride, bd[2] * stride, bd[3] * stride};
      BBox box = distance2bbox(anc[i][0], anc[i][1], dist);
      out.scores.push_back(sc);
      out.boxes.push_back(box);
      if (kps) {
        const float* kd = kps + i * 10;
        float kdist[10];
        for (int k = 0; k < 10; ++k) kdist[k] = kd[k] * stride;
        out.kps.push_back(distance2kps(anc[i][0], anc[i][1], kdist));
      } else {
        out.kps.push_back(Landmark5{});
      }
    }
  }
  return out;
}

std::vector<DetectedFace> Scrfd::detect(const cv::Mat& bgr) {
  float det_scale = 1.f;
  int nw = 0, nh = 0;
  cv::Mat pad = letterbox(bgr, det_size_, det_scale, nw, nh);
  auto raw = forward(pad);
  if (raw.scores.empty()) return {};

  std::vector<BBox> boxes;
  std::vector<float> scores;
  std::vector<Landmark5> kps;
  boxes.reserve(raw.scores.size());
  scores.reserve(raw.scores.size());
  kps.reserve(raw.scores.size());
  for (size_t i = 0; i < raw.scores.size(); ++i) {
    BBox b = raw.boxes[i];
    b.x1 /= det_scale;
    b.y1 /= det_scale;
    b.x2 /= det_scale;
    b.y2 /= det_scale;
    boxes.push_back(b);
    scores.push_back(raw.scores[i]);
    Landmark5 k = raw.kps[i];
    for (auto& p : k.xy) {
      p[0] /= det_scale;
      p[1] /= det_scale;
    }
    kps.push_back(k);
  }
  auto keep = nms_boxes(boxes, scores, nms_thresh_);
  std::vector<DetectedFace> faces;
  faces.reserve(keep.size());
  for (int i : keep) {
    DetectedFace f;
    f.box = boxes[static_cast<size_t>(i)];
    f.det_score = scores[static_cast<size_t>(i)];
    f.kps = kps[static_cast<size_t>(i)];
    faces.push_back(f);
  }
  return faces;
}

}  // namespace hvax
