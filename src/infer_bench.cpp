#include "hvax/pipeline.hpp"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

double percentile(std::vector<double> values, double q) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const double index = q * static_cast<double>(values.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(index));
  const size_t hi = static_cast<size_t>(std::ceil(index));
  const double fraction = index - static_cast<double>(lo);
  return values[lo] * (1.0 - fraction) + values[hi] * fraction;
}

double intersection_over_union(const hvax::BBox& a, const hvax::BBox& b) {
  const double x1 = std::max<double>(a.x1, b.x1);
  const double y1 = std::max<double>(a.y1, b.y1);
  const double x2 = std::min<double>(a.x2, b.x2);
  const double y2 = std::min<double>(a.y2, b.y2);
  const double intersection = std::max(0.0, x2 - x1) * std::max(0.0, y2 - y1);
  const double area_a = std::max(0.0, static_cast<double>(a.x2 - a.x1)) *
                        std::max(0.0, static_cast<double>(a.y2 - a.y1));
  const double area_b = std::max(0.0, static_cast<double>(b.x2 - b.x1)) *
                        std::max(0.0, static_cast<double>(b.y2 - b.y1));
  const double denominator = area_a + area_b - intersection;
  return denominator > 0.0 ? intersection / denominator : 0.0;
}

std::vector<size_t> maximum_iou_assignment(const std::vector<hvax::DetectedFace>& reference,
                                           const std::vector<hvax::DetectedFace>& candidate) {
  const size_t n = std::max(reference.size(), candidate.size());
  if (n == 0) return {};
  std::vector<double> row_potential(n + 1);
  std::vector<double> column_potential(n + 1);
  std::vector<size_t> column_match(n + 1);
  std::vector<size_t> predecessor(n + 1);
  for (size_t row = 1; row <= n; ++row) {
    column_match[0] = row;
    size_t current_column = 0;
    std::vector<double> minimum(n + 1, std::numeric_limits<double>::infinity());
    std::vector<bool> used(n + 1, false);
    do {
      used[current_column] = true;
      const size_t current_row = column_match[current_column];
      double delta = std::numeric_limits<double>::infinity();
      size_t next_column = 0;
      for (size_t column = 1; column <= n; ++column) {
        if (used[column]) continue;
        const size_t reference_index = current_row - 1;
        const size_t candidate_index = column - 1;
        double cost = 0.0;
        if (reference_index < reference.size() && candidate_index < candidate.size()) {
          cost = 1.0 - intersection_over_union(reference[reference_index].box,
                                                candidate[candidate_index].box);
        } else if (reference_index < reference.size() || candidate_index < candidate.size()) {
          cost = 2.0;
        }
        const double reduced_cost =
            cost - row_potential[current_row] - column_potential[column];
        if (reduced_cost < minimum[column]) {
          minimum[column] = reduced_cost;
          predecessor[column] = current_column;
        }
        if (minimum[column] < delta) {
          delta = minimum[column];
          next_column = column;
        }
      }
      for (size_t column = 0; column <= n; ++column) {
        if (used[column]) {
          row_potential[column_match[column]] += delta;
          column_potential[column] -= delta;
        } else {
          minimum[column] -= delta;
        }
      }
      current_column = next_column;
    } while (column_match[current_column] != 0);
    do {
      const size_t previous_column = predecessor[current_column];
      column_match[current_column] = column_match[previous_column];
      current_column = previous_column;
    } while (current_column != 0);
  }
  std::vector<size_t> assignment(reference.size(), candidate.size());
  for (size_t column = 1; column <= n; ++column) {
    const size_t row = column_match[column];
    if (row > 0 && row <= reference.size() && column <= candidate.size())
      assignment[row - 1] = column - 1;
  }
  return assignment;
}

struct ComparisonStats {
  size_t images = 0;
  size_t reference_faces = 0;
  size_t candidate_faces = 0;
  size_t matched_faces = 0;
  size_t count_mismatches = 0;
  double min_iou = std::numeric_limits<double>::infinity();
  double max_bbox_error = 0.0;
  double max_landmark_error = 0.0;
  double max_score_error = 0.0;
  double min_embedding_cosine = std::numeric_limits<double>::infinity();
  double max_embedding_l2 = 0.0;
  std::vector<hvax::Embedding> reference_embeddings;
  std::vector<hvax::Embedding> candidate_embeddings;
};

void accumulate_comparison(const std::vector<hvax::DetectedFace>& reference,
                           const std::vector<hvax::DetectedFace>& candidate,
                           ComparisonStats& stats) {
  ++stats.images;
  stats.reference_faces += reference.size();
  stats.candidate_faces += candidate.size();
  if (reference.size() != candidate.size()) ++stats.count_mismatches;
  const auto assignment = maximum_iou_assignment(reference, candidate);
  for (size_t reference_index = 0; reference_index < reference.size(); ++reference_index) {
    const auto& a = reference[reference_index];
    const size_t best = assignment[reference_index];
    if (best == candidate.size()) continue;
    ++stats.matched_faces;
    const auto& b = candidate[best];
    stats.min_iou = std::min(stats.min_iou, intersection_over_union(a.box, b.box));
    stats.max_bbox_error =
        std::max({stats.max_bbox_error, std::abs(static_cast<double>(a.box.x1 - b.box.x1)),
                  std::abs(static_cast<double>(a.box.y1 - b.box.y1)),
                  std::abs(static_cast<double>(a.box.x2 - b.box.x2)),
                  std::abs(static_cast<double>(a.box.y2 - b.box.y2))});
    for (size_t point = 0; point < a.kps.xy.size(); ++point) {
      stats.max_landmark_error =
          std::max({stats.max_landmark_error,
                    std::abs(static_cast<double>(a.kps.xy[point][0] - b.kps.xy[point][0])),
                    std::abs(static_cast<double>(a.kps.xy[point][1] - b.kps.xy[point][1]))});
    }
    stats.max_score_error =
        std::max(stats.max_score_error, std::abs(static_cast<double>(a.det_score - b.det_score)));
    stats.min_embedding_cosine =
        std::min(stats.min_embedding_cosine,
                 static_cast<double>(hvax::dot512(a.embedding.data(), b.embedding.data())));
    double squared_l2 = 0.0;
    for (size_t dimension = 0; dimension < a.embedding.size(); ++dimension) {
      const double difference =
          static_cast<double>(a.embedding[dimension]) - static_cast<double>(b.embedding[dimension]);
      squared_l2 += difference * difference;
    }
    stats.max_embedding_l2 = std::max(stats.max_embedding_l2, std::sqrt(squared_l2));
    stats.reference_embeddings.push_back(a.embedding);
    stats.candidate_embeddings.push_back(b.embedding);
  }
}

double max_pairwise_cosine_error(const ComparisonStats& stats) {
  double maximum = 0.0;
  for (size_t i = 0; i < stats.reference_embeddings.size(); ++i) {
    for (size_t j = i + 1; j < stats.reference_embeddings.size(); ++j) {
      const double reference = hvax::dot512(stats.reference_embeddings[i].data(),
                                            stats.reference_embeddings[j].data());
      const double candidate = hvax::dot512(stats.candidate_embeddings[i].data(),
                                            stats.candidate_embeddings[j].data());
      maximum = std::max(maximum, std::abs(reference - candidate));
    }
  }
  return maximum;
}

std::vector<std::filesystem::path> comparison_inputs(const std::filesystem::path& input) {
  namespace fs = std::filesystem;
  std::error_code error;
  if (fs::is_regular_file(input, error)) return {input};
  if (!fs::is_directory(input, error)) throw std::runtime_error("comparison input is not a file or directory");
  std::vector<fs::path> paths;
  for (fs::recursive_directory_iterator it(input, fs::directory_options::skip_permission_denied, error), end;
       it != end; it.increment(error)) {
    if (error) {
      error.clear();
      continue;
    }
    if (!it->is_regular_file(error)) continue;
    std::string extension = it->path().extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char byte) { return static_cast<char>(std::tolower(byte)); });
    if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
        extension == ".webp")
      paths.push_back(it->path());
  }
  std::sort(paths.begin(), paths.end());
  if (paths.empty()) throw std::runtime_error("comparison directory contains no supported images");
  return paths;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace hvax;
  if (argc < 4) {
    std::cerr << "Usage: hvax-infer-bench MODELS_DIR IMAGE_OR_DIR "
                 "cpu|coreml|mps|coreml-ane|coreml-cpu|compare|compare-low-precision [RUNS] [WARMUP]\n";
    return 2;
  }

  try {
    InferenceOptions inference;
    inference.intra_threads = 1;
    const std::string provider = argv[3];
    const bool compare = provider == "compare";
    const bool compare_low_precision = provider == "compare-low-precision";
    if (provider == "coreml" || compare) {
      inference.provider = InferenceProvider::coreml;
      inference.coreml_compute_units = CoreMlComputeUnits::all;
    } else if (provider == "mps" || compare_low_precision) {
      inference.provider = InferenceProvider::coreml;
      inference.coreml_compute_units = CoreMlComputeUnits::cpu_and_gpu;
    } else if (provider == "coreml-ane") {
      inference.provider = InferenceProvider::coreml;
      inference.coreml_compute_units = CoreMlComputeUnits::cpu_and_neural_engine;
    } else if (provider == "coreml-cpu") {
      inference.provider = InferenceProvider::coreml;
      inference.coreml_compute_units = CoreMlComputeUnits::cpu_only;
    } else if (provider != "cpu") {
      throw std::runtime_error(
          "provider must be cpu, coreml, mps, coreml-ane, coreml-cpu, compare, or compare-low-precision");
    }
    if (const char* cache = std::getenv("HVAX_COREML_CACHE_DIR")) inference.coreml_cache_dir = cache;
    if (const char* threads = std::getenv("HVAX_ORT_THREADS")) inference.intra_threads = std::stoi(threads);
    if (const char* low_precision = std::getenv("HVAX_COREML_LOW_PRECISION"))
      inference.coreml_allow_low_precision_accumulation = std::string_view(low_precision) == "1";
    if (const char* model_format = std::getenv("HVAX_COREML_MODEL_FORMAT"))
      inference.coreml_model_format = parse_coreml_model_format(model_format);
    if (const char* fast_prediction = std::getenv("HVAX_COREML_FAST_PREDICTION"))
      inference.coreml_fast_prediction = std::string_view(fast_prediction) == "1";
    const int concurrency = std::getenv("HVAX_BENCH_CONCURRENCY")
                                ? std::stoi(std::getenv("HVAX_BENCH_CONCURRENCY"))
                                : 1;
    inference.expected_concurrency = concurrency;
    const int runs = argc > 4 ? std::stoi(argv[4]) : 20;
    const int warmup = argc > 5 ? std::stoi(argv[5]) : 3;
    if (runs <= 0 || warmup < 0 || concurrency <= 0)
      throw std::runtime_error("RUNS/concurrency must be positive and WARMUP non-negative");

    if (compare || compare_low_precision) {
      InferenceOptions reference_inference = inference;
      InferenceOptions candidate_inference = inference;
      std::string reference_name = "CPU";
      std::string candidate_name = "CoreML";
      if (compare) {
        reference_inference = {};
        reference_inference.intra_threads = 8;
      } else {
        reference_name = "CoreML-float32-accumulation";
        candidate_name = "CoreML-float16-accumulation";
        reference_inference.coreml_allow_low_precision_accumulation = false;
        candidate_inference.coreml_allow_low_precision_accumulation = true;
        if (!inference.coreml_cache_dir.empty()) {
          reference_inference.coreml_cache_dir =
              (std::filesystem::path(inference.coreml_cache_dir) / "float32").string();
          candidate_inference.coreml_cache_dir =
              (std::filesystem::path(inference.coreml_cache_dir) / "float16").string();
        }
      }
      Pipeline reference_pipeline(argv[1], 640, 0.5f, 0.4f, reference_inference);
      Pipeline candidate_pipeline(argv[1], 640, 0.5f, 0.4f, candidate_inference);
      ComparisonStats stats;
      for (const auto& path : comparison_inputs(argv[2])) {
        cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
        if (image.empty()) throw std::runtime_error("cannot decode comparison image: " + path.string());
        accumulate_comparison(reference_pipeline.run(image), candidate_pipeline.run(image), stats);
      }
      if (stats.matched_faces == 0) {
        stats.min_iou = std::numeric_limits<double>::quiet_NaN();
        stats.min_embedding_cosine = std::numeric_limits<double>::quiet_NaN();
      }
      std::cout << std::fixed << std::setprecision(8)
                << "reference=" << reference_name << " candidate=" << candidate_name
                << " images=" << stats.images << " reference_faces=" << stats.reference_faces
                << " candidate_faces=" << stats.candidate_faces << " matched=" << stats.matched_faces
                << " count_mismatches=" << stats.count_mismatches << " min_iou=" << stats.min_iou
                << " max_bbox_abs_px=" << stats.max_bbox_error
                << " max_landmark_abs_px=" << stats.max_landmark_error
                << " max_detection_score_abs=" << stats.max_score_error
                << " min_embedding_cosine=" << stats.min_embedding_cosine
                << " max_embedding_l2=" << stats.max_embedding_l2
                << " max_pairwise_cosine_abs=" << max_pairwise_cosine_error(stats) << '\n';
      return stats.count_mismatches == 0 ? 0 : 1;
    }

    cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR);
    if (image.empty()) throw std::runtime_error("cannot decode benchmark image");
    if (const char* tile_x_value = std::getenv("HVAX_BENCH_TILE_X")) {
      const int tile_x = std::stoi(tile_x_value);
      if (tile_x <= 0) throw std::runtime_error("HVAX_BENCH_TILE_X must be positive");
      std::vector<cv::Mat> tiles(static_cast<size_t>(tile_x), image);
      cv::hconcat(tiles, image);
    }

    const auto init_start = std::chrono::steady_clock::now();
    Pipeline pipeline(argv[1], 640, 0.5f, 0.4f, inference);
    const auto init_end = std::chrono::steady_clock::now();
    for (int i = 0; i < warmup; ++i) (void)pipeline.run(image);

    std::vector<size_t> face_counts(static_cast<size_t>(runs));
    std::vector<double> milliseconds(static_cast<size_t>(runs));
    std::vector<double> detection_ms(static_cast<size_t>(runs));
    std::vector<double> alignment_ms(static_cast<size_t>(runs));
    std::vector<double> embedding_ms(static_cast<size_t>(runs));
    std::atomic<int> next{0};
    std::barrier gate(concurrency + 1);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(concurrency));
    for (int worker = 0; worker < concurrency; ++worker) {
      workers.emplace_back([&] {
        gate.arrive_and_wait();
        for (;;) {
          const int i = next.fetch_add(1, std::memory_order_relaxed);
          if (i >= runs) break;
          const auto start = std::chrono::steady_clock::now();
          PipelineTimings timings;
          face_counts[static_cast<size_t>(i)] = pipeline.run(image, &timings).size();
          const auto end = std::chrono::steady_clock::now();
          milliseconds[static_cast<size_t>(i)] =
              std::chrono::duration<double, std::milli>(end - start).count();
          detection_ms[static_cast<size_t>(i)] = timings.detection_ms;
          alignment_ms[static_cast<size_t>(i)] = timings.alignment_ms;
          embedding_ms[static_cast<size_t>(i)] = timings.embedding_ms;
        }
      });
    }
    const auto throughput_start = std::chrono::steady_clock::now();
    gate.arrive_and_wait();
    for (auto& worker : workers) worker.join();
    const auto throughput_end = std::chrono::steady_clock::now();
    const double wall_seconds =
        std::chrono::duration<double>(throughput_end - throughput_start).count();
    const auto [min_faces, max_faces] = std::minmax_element(face_counts.begin(), face_counts.end());
    if (*min_faces != *max_faces)
      throw std::runtime_error("face count changed between benchmark runs");
    const size_t faces = face_counts.front();
    const double sum = std::accumulate(milliseconds.begin(), milliseconds.end(), 0.0);
    const double mean = sum / static_cast<double>(milliseconds.size());
    double squared_deviation_sum = 0.0;
    for (const double value : milliseconds) {
      const double deviation = value - mean;
      squared_deviation_sum += deviation * deviation;
    }
    const double standard_deviation =
        runs > 1 ? std::sqrt(squared_deviation_sum / static_cast<double>(runs - 1)) : 0.0;
    const double mean_ci95 = 1.96 * standard_deviation / std::sqrt(static_cast<double>(runs));
    std::cout << std::fixed << std::setprecision(3)
              << "provider=" << to_string(inference.provider)
              << " model_format=" << to_string(inference.coreml_model_format)
              << " compute_units=" << to_string(inference.coreml_compute_units)
              << " image=" << image.cols << 'x' << image.rows
              << " faces=" << faces << " runs=" << runs << " warmup=" << warmup
              << " concurrency=" << concurrency << '\n'
              << "session_init_ms=" << std::chrono::duration<double, std::milli>(init_end - init_start).count()
              << " mean_ms=" << mean << " p50_ms=" << percentile(milliseconds, 0.50)
              << " p95_ms=" << percentile(milliseconds, 0.95)
              << " min_ms=" << *std::min_element(milliseconds.begin(), milliseconds.end())
              << " max_ms=" << *std::max_element(milliseconds.begin(), milliseconds.end())
              << " stddev_ms=" << standard_deviation << " mean_ci95_ms=" << mean_ci95
              << " throughput_images_s=" << static_cast<double>(runs) / wall_seconds << '\n'
              << "mean_detection_ms="
              << std::accumulate(detection_ms.begin(), detection_ms.end(), 0.0) / runs
              << " mean_alignment_ms="
              << std::accumulate(alignment_ms.begin(), alignment_ms.end(), 0.0) / runs
              << " mean_embedding_ms="
              << std::accumulate(embedding_ms.begin(), embedding_ms.end(), 0.0) / runs << '\n';
  } catch (const std::exception& error) {
    std::cerr << "hvax-infer-bench: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
