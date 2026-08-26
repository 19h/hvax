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
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
  cv::Mat out = cv::Mat::zeros(size, size, CV_8UC3);
  resized.copyTo(out(cv::Rect(0, 0, new_w, new_h)));
  return out;
}

void blob_nchw(const cv::Mat& bgr, float mean, float std, std::vector<float>& out) {
  const int h = bgr.rows;
  const int w = bgr.cols;
  out.resize(static_cast<size_t>(3 * h * w));
  const int hw = h * w;
  for (int y = 0; y < h; ++y) {
    const cv::Vec3b* row = bgr.ptr<cv::Vec3b>(y);
    for (int x = 0; x < w; ++x) {
      const int i = y * w + x;
      // swapRB: R,G,B from BGR
      out[static_cast<size_t>(0 * hw + i)] = (static_cast<float>(row[x][2]) - mean) / std;
      out[static_cast<size_t>(1 * hw + i)] = (static_cast<float>(row[x][1]) - mean) / std;
      out[static_cast<size_t>(2 * hw + i)] = (static_cast<float>(row[x][0]) - mean) / std;
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
  session_ = ctx.load(model_path);
  io_ = inspect(*session_);
  bind_io(io_);
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
  spdlog::info("SCRFD batched={} kps={} fmc={} anchors={} det_size={}", batched_, use_kps_, fmc_,
               num_anchors_, det_size_);
}

const std::vector<std::array<float, 2>>& Scrfd::centers(int height, int width, int stride) {
  std::lock_guard lock(center_cache_mu_);
  auto key = std::make_tuple(height, width, stride);
  auto it = center_cache_.find(key);
  if (it != center_cache_.end()) return it->second;
  std::vector<std::array<float, 2>> c;
  c.reserve(static_cast<size_t>(height * width * num_anchors_));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int a = 0; a < num_anchors_; ++a) {
        c.push_back({static_cast<float>(x * stride), static_cast<float>(y * stride)});
      }
    }
  }
  auto [ins, _] = center_cache_.emplace(key, std::move(c));
  return ins->second;
}

Scrfd::ForwardOut Scrfd::forward(const cv::Mat& padded_bgr) {
  std::vector<float> blob;
  blob_nchw(padded_bgr, 127.5f, 128.0f, blob);
  const int64_t shape[] = {1, 3, padded_bgr.rows, padded_bgr.cols};
  Ort::Value input = Ort::Value::CreateTensor<float>(ctx_->cpu_mem(), blob.data(), blob.size(), shape, 4);
  auto outputs = session_->Run(Ort::RunOptions{nullptr}, io_.input_name_ptrs.data(), &input, 1,
                               io_.output_name_ptrs.data(), io_.output_name_ptrs.size());

  ForwardOut out;
  const int input_h = padded_bgr.rows;
  const int input_w = padded_bgr.cols;

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

    auto ssh = score_t.GetTensorTypeAndShapeInfo().GetShape();
    const int fh = input_h / stride;
    const int fw = input_w / stride;
    const auto& anc = centers(fh, fw, stride);
    const size_t n = anc.size();

    // scores layout: batched [1,N,1] or [N,1] or [N]
    size_t score_stride = 1;
    if (ssh.size() >= 3) score_stride = 1;  // last dim 1
    else if (ssh.size() == 2 && ssh[1] > 1) score_stride = static_cast<size_t>(ssh[1]);

    for (size_t i = 0; i < n; ++i) {
      float sc = scores[i * (ssh.size() >= 3 ? 1 : (ssh.size() == 2 ? 1 : 1))];
      if (ssh.size() >= 3) {
        // [1, N, 1]
        sc = scores[i];
      } else if (ssh.size() == 2) {
        sc = scores[i * static_cast<size_t>(std::max<int64_t>(ssh[1], 1))];
        if (ssh[1] == 1) sc = scores[i];
      }
      (void)score_stride;
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

  std::vector<int> order(raw.scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return raw.scores[static_cast<size_t>(a)] > raw.scores[static_cast<size_t>(b)]; });

  std::vector<BBox> boxes;
  std::vector<float> scores;
  std::vector<Landmark5> kps;
  boxes.reserve(order.size());
  scores.reserve(order.size());
  kps.reserve(order.size());
  for (int i : order) {
    BBox b = raw.boxes[static_cast<size_t>(i)];
    b.x1 /= det_scale;
    b.y1 /= det_scale;
    b.x2 /= det_scale;
    b.y2 /= det_scale;
    boxes.push_back(b);
    scores.push_back(raw.scores[static_cast<size_t>(i)]);
    Landmark5 k = raw.kps[static_cast<size_t>(i)];
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
