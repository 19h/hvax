cmake_minimum_required(VERSION 3.24)
project(hvax VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(HVAX_MARCH_NATIVE "Enable -march=native" ON)
option(HVAX_BUILD_TESTS "Build unit tests" ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter -Wno-c11-extensions)
if(HVAX_MARCH_NATIVE)
  add_compile_options(-O3 -march=native)
endif()

# --- OpenCV (Ubuntu splits the CMake package; find libs directly) ---
find_path(OpenCV_INCLUDE_DIR opencv2/core.hpp HINTS /usr/include/opencv4 REQUIRED)
find_library(OpenCV_CORE_LIB opencv_core REQUIRED)
find_library(OpenCV_IMGPROC_LIB opencv_imgproc REQUIRED)
find_library(OpenCV_IMGCODECS_LIB opencv_imgcodecs REQUIRED)

# --- system ---
find_package(OpenSSL REQUIRED)
find_package(spdlog REQUIRED)
find_package(Threads REQUIRED)

find_path(NLOHMANN_JSON_INCLUDE nlohmann/json.hpp REQUIRED)

# --- ONNX Runtime (prebuilt tarball in-tree) ---
set(ORT_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.20.1")
if(NOT EXISTS "${ORT_ROOT}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "ONNX Runtime not found at ${ORT_ROOT}. Extract the official linux-x64 CPU tarball there.")
endif()

set(HVAX_SOURCES
  src/config.cpp
  src/infer/ort.cpp
  src/detect/scrfd.cpp
  src/align/umeyama.cpp
  src/embed/arcface.cpp
  src/store/phash.cpp
  src/store/gallery.cpp
  src/index/exact_scan.cpp
  src/index/hnsw_index.cpp
  src/pipeline.cpp
  src/engine.cpp
  src/http/server.cpp
)

add_library(hvax_lib STATIC ${HVAX_SOURCES})
target_include_directories(hvax_lib
  PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${ORT_ROOT}/include
    ${OpenCV_INCLUDE_DIR}
    ${NLOHMANN_JSON_INCLUDE}
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/usearch
)
target_compile_definitions(hvax_lib PUBLIC
  USEARCH_USE_SIMSIMD=0
  USEARCH_USE_FP16LIB=0
  CPPHTTPLIB_NO_EXCEPTIONS=0
  FMT_USE_CONSTEVAL=0
  SPDLOG_NO_EXCEPTIONS
)
target_link_libraries(hvax_lib
  PUBLIC
    ${OpenCV_CORE_LIB}
    ${OpenCV_IMGPROC_LIB}
    ${OpenCV_IMGCODECS_LIB}
    OpenSSL::Crypto
    spdlog::spdlog
    Threads::Threads
    ${ORT_ROOT}/lib/libonnxruntime.so
)
set_target_properties(hvax_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(hvaxd src/main.cpp)
target_link_libraries(hvaxd PRIVATE hvax_lib)
set_target_properties(hvaxd PROPERTIES
  BUILD_RPATH "${ORT_ROOT}/lib"
  INSTALL_RPATH "${ORT_ROOT}/lib"
)

add_executable(hvax-bench src/bench.cpp)
target_link_libraries(hvax-bench PRIVATE hvax_lib)
set_target_properties(hvax-bench PROPERTIES
  BUILD_RPATH "${ORT_ROOT}/lib"
  INSTALL_RPATH "${ORT_ROOT}/lib"
)

if(HVAX_BUILD_TESTS)
  find_package(GTest REQUIRED)
  add_executable(hvax_tests
    tests/test_umeyama.cpp
    tests/test_phash.cpp
    tests/test_exact_scan.cpp
    tests/test_scrfd_decode.cpp
    tests/test_gallery.cpp
    tests/test_http.cpp
  )
  target_link_libraries(hvax_tests PRIVATE hvax_lib GTest::gtest GTest::gtest_main)
  target_compile_definitions(hvax_tests PRIVATE
    HVAX_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures"
    HVAX_MODELS_DIR="${CMAKE_SOURCE_DIR}/models"
  )
  set_target_properties(hvax_tests PROPERTIES
    BUILD_RPATH "${ORT_ROOT}/lib"
    INSTALL_RPATH "${ORT_ROOT}/lib"
  )
  enable_testing()
  add_test(NAME hvax_tests COMMAND hvax_tests)
endif()

message(STATUS "hvax: OpenCV include ${OpenCV_INCLUDE_DIR}")
message(STATUS "hvax: ORT ${ORT_ROOT}")
