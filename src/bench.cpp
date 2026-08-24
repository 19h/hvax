#include "hvax/embed/arcface.hpp"
#include "hvax/index/exact_scan.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char** argv) {
  using namespace hvax;
  uint64_t n = 100000;
  int k = 10;
  int queries = 1000;
  if (argc > 1) n = std::stoull(argv[1]);
  if (argc > 2) queries = std::stoi(argv[2]);
  if (argc > 3) k = std::stoi(argv[3]);

  std::mt19937 rng(42);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> db(n * static_cast<uint64_t>(kDim));
  for (auto& x : db) x = nd(rng);
  for (uint64_t i = 0; i < n; ++i) l2_normalize(db.data() + i * kDim);

  std::vector<float> q(static_cast<size_t>(queries * kDim));
  for (auto& x : q) x = nd(rng);
  for (int i = 0; i < queries; ++i) l2_normalize(q.data() + i * kDim);

  // warmup
  exact_topk_f32(db.data(), std::min<uint64_t>(n, 1024), q.data(), k, nullptr, -2.f);

  auto t0 = std::chrono::steady_clock::now();
  double acc = 0;
  for (int i = 0; i < queries; ++i) {
    auto hits = exact_topk_f32(db.data(), n, q.data() + i * kDim, k, nullptr, -2.f);
    if (!hits.empty()) acc += hits[0].score;
  }
  auto t1 = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double qps = queries / (ms / 1000.0);
  std::cout << "n=" << n << " queries=" << queries << " k=" << k << "\n";
  std::cout << "total_ms=" << ms << " per_query_ms=" << (ms / queries) << " qps=" << qps << "\n";
  std::cout << "mean_top1=" << (acc / queries) << "\n";
  return 0;
}
