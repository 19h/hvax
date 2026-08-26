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

# --- ONNX Runtime (CPU tarball in-tree; point at a GPU build for --cuda) ---
set(HVAX_ORT_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.20.1" CACHE PATH
    "ONNX Runtime distribution root (use an onnxruntime-gpu build for CUDA)")
set(ORT_ROOT "${HVAX_ORT_ROOT}")
if(NOT EXISTS "${ORT_ROOT}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "ONNX Runtime not found at ${ORT_ROOT}. Set HVAX_ORT_ROOT to an extracted CPU or GPU distribution.")
endif()

set(HVAX_LANDING_HTML "${CMAKE_SOURCE_DIR}/landing/index.html")
set(HVAX_LANDING_HPP "${CMAKE_BINARY_DIR}/generated/hvax/http/landing_html.hpp")
set(HVAX_EMBED_SCRIPT "${CMAKE_SOURCE_DIR}/cmake/embed_text.cmake")

execute_process(
  COMMAND ${CMAKE_COMMAND}
    -DINPUT=${HVAX_LANDING_HTML}
    -DOUTPUT=${HVAX_LANDING_HPP}
    -DVAR=kLandingHtml
    -P ${HVAX_EMBED_SCRIPT}
  RESULT_VARIABLE HVAX_EMBED_RC
)
if(NOT HVAX_EMBED_RC EQUAL 0)
  message(FATAL_ERROR "failed to embed landing/index.html")
endif()

add_custom_command(
  OUTPUT ${HVAX_LANDING_HPP}
  COMMAND ${CMAKE_COMMAND}
    -DINPUT=${HVAX_LANDING_HTML}
    -DOUTPUT=${HVAX_LANDING_HPP}
    -DVAR=kLandingHtml
    -P ${HVAX_EMBED_SCRIPT}
  DEPENDS ${HVAX_LANDING_HTML} ${HVAX_EMBED_SCRIPT}
  COMMENT "Embedding landing/index.html"
)

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
  src/processed.cpp
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
    ${CMAKE_BINARY_DIR}/generated
)
set_source_files_properties(src/http/server.cpp PROPERTIES OBJECT_DEPENDS ${HVAX_LANDING_HPP})
add_custom_target(hvax_embed_landing DEPENDS ${HVAX_LANDING_HPP})
add_dependencies(hvax_lib hvax_embed_landing)
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

add_executable(hvax src/client.cpp)
target_link_libraries(hvax PRIVATE hvax_lib OpenSSL::SSL)
target_compile_definitions(hvax PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
set_target_properties(hvax PROPERTIES
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
    tests/test_processed.cpp
  )
  target_link_libraries(hvax_tests PRIVATE hvax_lib GTest::gtest GTest::gtest_main)
  target_compile_definitions(hvax_tests PRIVATE
    HVAX_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures"
    HVAX_MODELS_DIR="${CMAKE_SOURCE_DIR}/models"
  )
  add_dependencies(hvax_tests hvax_embed_landing)
  set_target_properties(hvax_tests PROPERTIES
    BUILD_RPATH "${ORT_ROOT}/lib"
    INSTALL_RPATH "${ORT_ROOT}/lib"
  )
  enable_testing()
  add_test(NAME hvax_tests COMMAND hvax_tests)
endif()

message(STATUS "hvax: OpenCV include ${OpenCV_INCLUDE_DIR}")
message(STATUS "hvax: ORT ${ORT_ROOT}")
