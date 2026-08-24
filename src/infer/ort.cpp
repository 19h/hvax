#include "hvax/infer/ort.hpp"

#include <stdexcept>

#include <spdlog/spdlog.h>

namespace hvax {

OrtContext::OrtContext(int intra_threads)
    : env_(ORT_LOGGING_LEVEL_WARNING, "hvax"),
      cpu_mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
  opts_.SetIntraOpNumThreads(intra_threads > 0 ? intra_threads : 1);
  opts_.SetInterOpNumThreads(1);
  opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  opts_.EnableMemPattern();
  opts_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
}

std::unique_ptr<Ort::Session> OrtContext::load(const std::string& path) {
  return std::make_unique<Ort::Session>(env_, path.c_str(), opts_);
}

SessionIo inspect(Ort::Session& session) {
  SessionIo io;
  Ort::AllocatorWithDefaultOptions alloc;
  auto in = session.GetInputNameAllocated(0, alloc);
  io.input_name = in.get();
  io.input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  const size_t nout = session.GetOutputCount();
  io.output_names.reserve(nout);
  io.output_shapes.reserve(nout);
  for (size_t i = 0; i < nout; ++i) {
    auto n = session.GetOutputNameAllocated(i, alloc);
    io.output_names.emplace_back(n.get());
    io.output_shapes.push_back(session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  std::string shape;
  for (auto d : io.input_shape) {
    if (!shape.empty()) shape += ",";
    shape += std::to_string(d);
  }
  spdlog::info("ORT input={} shape=[{}] outs={}", io.input_name.c_str(), shape.c_str(), io.output_names.size());
  for (size_t i = 0; i < io.output_names.size(); ++i) {
    std::string os;
    for (auto d : io.output_shapes[i]) {
      if (!os.empty()) os += ",";
      os += std::to_string(d);
    }
    spdlog::info("  out[{}] {} [{}]", i, io.output_names[i].c_str(), os.c_str());
  }
  return io;
}

void bind_io(SessionIo& io) {
  io.input_name_ptrs = {io.input_name.c_str()};
  io.output_name_ptrs.clear();
  io.output_name_ptrs.reserve(io.output_names.size());
  for (auto& s : io.output_names) io.output_name_ptrs.push_back(s.c_str());
}

}  // namespace hvax
