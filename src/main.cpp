#include "hvax/config.hpp"
#include "hvax/engine.hpp"
#include "hvax/http/server.hpp"
#include "hvax/pipeline.hpp"
#include "hvax/util/hex.hpp"

#include <csignal>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

namespace {
int g_lockfd = -1;
hvax::Engine* g_engine = nullptr;

void on_signal(int) {
  if (g_engine) {
    try {
      g_engine->gallery().flush();
    } catch (...) {
    }
  }
  _exit(0);
}
}  // namespace

int main(int argc, char** argv) {
  using namespace hvax;
  spdlog::set_level(spdlog::level::info);
  Config cfg;
  try {
    cfg = parse_args(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    print_usage();
    return 2;
  }
  if (cfg.print_help) {
    print_usage();
    return 0;
  }

  std::filesystem::create_directories(cfg.data_dir);
  g_lockfd = ::open((std::filesystem::path(cfg.data_dir) / "hvax.lock").c_str(), O_CREAT | O_RDWR, 0644);
  if (g_lockfd >= 0) {
    if (flock(g_lockfd, LOCK_EX | LOCK_NB) != 0) {
      spdlog::error("another hvaxd holds {}", cfg.data_dir.c_str());
      return 1;
    }
  }

  try {
    Engine engine(cfg);
    g_engine = &engine;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    if (cfg.reindex) {
      const auto r = engine.reindex();
      std::cout << nlohmann::json{{"rows", r.rows},
                                  {"live", r.live},
                                  {"low_quality", r.low_quality},
                                  {"indexed", r.indexed},
                                  {"unassigned_by_gate", r.unassigned_by_gate},
                                  {"requantized", r.requantized},
                                  {"ms", r.ms}}
                       .dump(2)
                << "\n";
      if (!cfg.cluster_once && cfg.eval_impostor_pairs == 0) return 0;
    }
    if (cfg.cluster_once) {
      const auto r = engine.recluster();
      if (!r) {
        spdlog::error("clustering already running");
        return 1;
      }
      std::cout << nlohmann::json{{"faces_eligible", r->faces_eligible},
                                  {"faces_assigned", r->faces_assigned},
                                  {"identities", r->identities},
                                  {"identities_pinned", r->identities_pinned},
                                  {"clusters_before_merge", r->clusters_before_merge},
                                  {"merged", r->merged},
                                  {"singletons_dropped", r->singletons_dropped},
                                  {"largest", r->largest},
                                  {"edges", r->edges},
                                  {"iterations", r->iterations},
                                  {"knn_ms", r->knn_ms},
                                  {"cw_ms", r->cw_ms},
                                  {"merge_ms", r->merge_ms},
                                  {"apply_ms", r->apply_ms},
                                  {"total_ms", r->total_ms}}
                       .dump(2)
                << "\n";
      if (cfg.eval_impostor_pairs == 0) return 0;
    }
    if (cfg.eval_impostor_pairs > 0) {
      const auto r = engine.impostor_eval(static_cast<uint64_t>(cfg.eval_impostor_pairs));
      auto table = [](const std::vector<std::pair<float, double>>& v) {
        nlohmann::json a = nlohmann::json::array();
        for (auto& [t, x] : v) a.push_back({{"threshold", t}, {"value", x}});
        return a;
      };
      std::cout << nlohmann::json{{"images_used", r.images_used},
                                  {"same_image_pairs", r.same_image_pairs},
                                  {"random_pairs", r.random_pairs},
                                  {"same_image_mean", r.same_image_mean},
                                  {"same_image_sd", r.same_image_sd},
                                  {"random_mean", r.random_mean},
                                  {"random_sd", r.random_sd},
                                  {"random_mode", r.random_mode},
                                  {"impostor_ceiling", r.impostor_ceiling},
                                  {"same_image_far", table(r.same_image_far)},
                                  {"random_above", table(r.random_above)},
                                  {"random_same_identity_excess", table(r.random_excess)}}
                       .dump(2)
                << "\n";
      return 0;
    }

    if (!cfg.once_image.empty()) {
      cv::Mat img = cv::imread(cfg.once_image, cv::IMREAD_COLOR);
      if (img.empty()) {
        spdlog::error("cannot read {}", cfg.once_image.c_str());
        return 1;
      }
      auto faces = engine.debug_once(img);
      nlohmann::json arr = nlohmann::json::array();
      for (auto& f : faces) {
        nlohmann::json kps = nlohmann::json::array();
        for (auto& p : f.kps.xy) kps.push_back({p[0], p[1]});
        double n2 = 0;
        for (float x : f.embedding) n2 += static_cast<double>(x) * x;
        arr.push_back({{"bbox", {f.box.x1, f.box.y1, f.box.x2, f.box.y2}},
                       {"det_score", f.det_score},
                       {"landmarks", kps},
                       {"emb_l2", n2}});
      }
      std::cout << nlohmann::json{{"nfaces", faces.size()}, {"faces", arr}}.dump(2) << "\n";
      return 0;
    }

    run_server(engine);
  } catch (const std::exception& e) {
    spdlog::error("fatal: {}", e.what());
    return 1;
  }
  return 0;
}
