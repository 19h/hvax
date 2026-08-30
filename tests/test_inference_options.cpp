#include "hvax/infer/ort.hpp"
#include "hvax/config.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

hvax::Config parse(std::initializer_list<const char*> arguments) {
  std::vector<std::string> storage;
  storage.reserve(arguments.size());
  for (const char* argument : arguments) storage.emplace_back(argument);
  std::vector<char*> argv;
  argv.reserve(storage.size());
  for (auto& argument : storage) argv.push_back(argument.data());
  return hvax::parse_args(static_cast<int>(argv.size()), argv.data());
}

}  // namespace

TEST(InferenceOptions, ParsesCoreMlComputeUnits) {
  using hvax::CoreMlComputeUnits;
  EXPECT_EQ(hvax::parse_coreml_compute_units("all"), CoreMlComputeUnits::all);
  EXPECT_EQ(hvax::parse_coreml_compute_units("cpu-gpu"), CoreMlComputeUnits::cpu_and_gpu);
  EXPECT_EQ(hvax::parse_coreml_compute_units("gpu"), CoreMlComputeUnits::cpu_and_gpu);
  EXPECT_EQ(hvax::parse_coreml_compute_units("cpu-ane"), CoreMlComputeUnits::cpu_and_neural_engine);
  EXPECT_EQ(hvax::parse_coreml_compute_units("ane"), CoreMlComputeUnits::cpu_and_neural_engine);
  EXPECT_EQ(hvax::parse_coreml_compute_units("cpu"), CoreMlComputeUnits::cpu_only);
  EXPECT_THROW((void)hvax::parse_coreml_compute_units("invalid"), std::invalid_argument);
}

TEST(InferenceOptions, StableProviderNames) {
  EXPECT_STREQ(hvax::to_string(hvax::InferenceProvider::cpu), "CPU");
  EXPECT_STREQ(hvax::to_string(hvax::InferenceProvider::cuda), "CUDA");
  EXPECT_STREQ(hvax::to_string(hvax::InferenceProvider::coreml), "CoreML");
}

TEST(InferenceOptions, ParsesCoreMlModelFormat) {
  using hvax::CoreMlModelFormat;
  EXPECT_EQ(hvax::parse_coreml_model_format("auto"), CoreMlModelFormat::automatic);
  EXPECT_EQ(hvax::parse_coreml_model_format("mlprogram"), CoreMlModelFormat::ml_program);
  EXPECT_EQ(hvax::parse_coreml_model_format("neuralnetwork"), CoreMlModelFormat::neural_network);
  EXPECT_THROW((void)hvax::parse_coreml_model_format("invalid"), std::invalid_argument);
}

TEST(InferenceOptions, DaemonCoreMlFlags) {
  const auto config = parse({"hvaxd", "--data-dir", "/tmp/hvax-test", "--coreml",
                             "--coreml-compute-units", "cpu-ane", "--coreml-model-format", "mlprogram",
                             "--threads", "3"});
  EXPECT_EQ(config.inference.provider, hvax::InferenceProvider::coreml);
  EXPECT_EQ(config.inference.coreml_compute_units, hvax::CoreMlComputeUnits::cpu_and_neural_engine);
  EXPECT_EQ(config.inference.coreml_model_format, hvax::CoreMlModelFormat::ml_program);
  EXPECT_EQ(config.inference.intra_threads, 3);
  EXPECT_EQ(config.inference.expected_concurrency, 3);
  EXPECT_EQ(config.inference.coreml_cache_dir, "/tmp/hvax-test/coreml-cache");
}

TEST(InferenceOptions, DaemonMpsAliasAndValidation) {
  const auto config = parse({"hvaxd", "--mps"});
  EXPECT_EQ(config.inference.provider, hvax::InferenceProvider::coreml);
  EXPECT_EQ(config.inference.coreml_compute_units, hvax::CoreMlComputeUnits::cpu_and_gpu);
  EXPECT_THROW((void)parse({"hvaxd", "--cuda", "--coreml"}), std::runtime_error);
  EXPECT_THROW((void)parse({"hvaxd", "--coreml-compute-units", "invalid"}), std::invalid_argument);
  EXPECT_THROW((void)parse({"hvaxd", "--threads", "0"}), std::runtime_error);
  EXPECT_THROW((void)parse({"hvaxd", "--http-threads", "0"}), std::runtime_error);
  EXPECT_THROW((void)parse({"hvaxd", "--det-size", "641"}), std::runtime_error);
  EXPECT_EQ(parse({"hvaxd", "--det-size", "320"}).det_size, 320);
  EXPECT_EQ(parse({"hvaxd", "--coreml", "--once", "image.jpg"}).inference.expected_concurrency, 1);
}

TEST(InferenceOptions, ValidatesFreeDimensionOverrides) {
  hvax::InferenceOptions invalid_name;
  invalid_name.free_dimension_overrides.emplace_back("", 640);
  EXPECT_THROW((void)hvax::OrtContext(invalid_name), std::invalid_argument);

  hvax::InferenceOptions invalid_value;
  invalid_value.free_dimension_overrides.emplace_back("?", 0);
  EXPECT_THROW((void)hvax::OrtContext(invalid_value), std::invalid_argument);
}
