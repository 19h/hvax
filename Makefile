SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := all
.NOTPARALLEL:

CMAKE ?= cmake
CTEST ?= ctest
GENERATOR ?= Ninja
BUILD_DIR ?= $(CURDIR)/build
BUILD_TYPE ?= RelWithDebInfo
BUILD_TESTS ?= ON
MARCH_NATIVE ?= ON
JOBS ?= $(shell if command -v nproc >/dev/null 2>&1; then nproc; else sysctl -n hw.logicalcpu 2>/dev/null || echo 4; fi)
CMAKE_ARGS ?=
SANITIZERS ?=

MODELS_DIR ?= $(CURDIR)/models
DATA_DIR ?= $(CURDIR)/data
RUN_ARGS ?=
IMAGE ?= $(CURDIR)/tests/fixtures/lena.jpg
BENCH_RUNS ?= 100
BENCH_WARMUP ?= 10
ASAN_BUILD_DIR ?= $(CURDIR)/build-asan
DEBUG_BUILD_DIR ?= $(CURDIR)/build-debug
RELEASE_BUILD_DIR ?= $(CURDIR)/build-release

.PHONY: all setup bootstrap deps check-tools runtime models configure build test check \
        verify rebuild debug release asan run once bench extension clean distclean help

all: build

## Install platform packages, fetch artifacts, build, and test.
setup: deps
	@$(MAKE) test

## Fetch/verify runtime and models, then configure without installing packages.
bootstrap: models configure

## Install build dependencies with Homebrew or apt-get.
deps:
	@os="$$(uname -s)"; \
	case "$$os" in \
	  Darwin) \
	    command -v brew >/dev/null 2>&1 || { echo "Homebrew is required for 'make deps'" >&2; exit 2; }; \
	    brew install cmake ninja opencv openssl spdlog nlohmann-json googletest poppler; \
	    ;; \
	  Linux) \
	    command -v apt-get >/dev/null 2>&1 || { echo "apt-get is required for 'make deps' on Linux" >&2; exit 2; }; \
	    if [[ "$$(id -u)" -eq 0 ]]; then \
	      apt=(apt-get); \
	    else \
	      command -v sudo >/dev/null 2>&1 || { echo "sudo or a root shell is required for 'make deps'" >&2; exit 2; }; \
	      apt=(sudo apt-get); \
	    fi; \
	    "$${apt[@]}" update; \
	    "$${apt[@]}" install -y build-essential cmake ninja-build curl ca-certificates \
	      libopencv-dev libssl-dev libspdlog-dev nlohmann-json3-dev libgtest-dev poppler-utils; \
	    ;; \
	  *) \
	    echo "unsupported host for automatic dependency installation: $$os" >&2; \
	    exit 2; \
	    ;; \
	esac

## Validate required tools and the minimum CMake version.
check-tools:
	@for tool in "$(CMAKE)" curl c++; do \
	  command -v "$$tool" >/dev/null 2>&1 || { echo "missing required tool: $$tool" >&2; exit 2; }; \
	done; \
	if [[ "$(GENERATOR)" == "Ninja" ]]; then \
	  command -v ninja >/dev/null 2>&1 || { echo "missing required tool: ninja" >&2; exit 2; }; \
	fi; \
	version="$$($(CMAKE) --version | awk 'NR == 1 { print $$3 }')"; \
	IFS=. read -r major minor patch <<<"$$version"; \
	if (( major < 3 || (major == 3 && minor < 24) )); then \
	  echo "CMake 3.24 or newer is required; found $$version" >&2; \
	  exit 2; \
	fi

## Fetch or verify the platform ONNX Runtime distribution.
runtime: check-tools
	@host="$$(uname -s)/$$(uname -m)"; \
	case "$$host" in \
	  Darwin/arm64) \
	    ./scripts/download_onnxruntime.sh; \
	    ;; \
	  Linux/x86_64|Linux/amd64) \
	    root="$(CURDIR)/third_party/onnxruntime-linux-x64-1.20.1"; \
	    test -f "$$root/include/onnxruntime_cxx_api.h"; \
	    test -f "$$root/lib/libonnxruntime.so"; \
	    echo "ONNX Runtime verified in $$root"; \
	    ;; \
	  *) \
	    echo "unsupported host architecture: $$host" >&2; \
	    exit 2; \
	    ;; \
	esac

## Fetch or checksum-verify the pinned detector and recognizer models.
models: check-tools
	@HVAX_MODELS_DIR="$(MODELS_DIR)" ./scripts/download_models.sh

## Generate the CMake build tree.
configure: runtime
	@mkdir -p "$(BUILD_DIR)"; \
	generator_args=(-G "$(GENERATOR)"); \
	if [[ -f "$(BUILD_DIR)/CMakeCache.txt" ]]; then \
	  configured_generator="$$(awk -F= '$$1 == "CMAKE_GENERATOR:INTERNAL" { print $$2 }' "$(BUILD_DIR)/CMakeCache.txt")"; \
	  if [[ -n "$$configured_generator" && "$$configured_generator" != "$(GENERATOR)" ]]; then \
	    echo "build tree uses '$$configured_generator', not '$(GENERATOR)'; run make distclean" >&2; \
	    exit 2; \
	  fi; \
	  generator_args=(); \
	fi; \
	extra_args=(); \
	if [[ -n "$(SANITIZERS)" ]]; then \
	  extra_args+=("-DCMAKE_CXX_FLAGS=-fsanitize=$(SANITIZERS) -fno-omit-frame-pointer"); \
	  extra_args+=("-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=$(SANITIZERS)"); \
	fi; \
	$(CMAKE) -S "$(CURDIR)" -B "$(BUILD_DIR)" "$${generator_args[@]}" \
	  -DCMAKE_BUILD_TYPE="$(BUILD_TYPE)" \
	  -DHVAX_BUILD_TESTS="$(BUILD_TESTS)" \
	  -DHVAX_MARCH_NATIVE="$(MARCH_NATIVE)" \
	  "$${extra_args[@]}" $(CMAKE_ARGS)

## Fetch/verify artifacts and compile all binaries.
build: models configure
	@$(CMAKE) --build "$(BUILD_DIR)" --parallel "$(JOBS)"

## Build and run the CTest suite.
test: build
	@$(CTEST) --test-dir "$(BUILD_DIR)" --build-config "$(BUILD_TYPE)" --output-on-failure

check: test
verify: test

## Clean and rebuild the selected build tree.
rebuild:
	@$(MAKE) clean
	@$(MAKE) build

## Build a separate unoptimized Debug tree.
debug:
	@$(MAKE) BUILD_DIR="$(DEBUG_BUILD_DIR)" BUILD_TYPE=Debug MARCH_NATIVE=OFF build

## Build a separate optimized Release tree.
release:
	@$(MAKE) BUILD_DIR="$(RELEASE_BUILD_DIR)" BUILD_TYPE=Release build

## Build and test with AddressSanitizer and UndefinedBehaviorSanitizer.
asan:
	@$(MAKE) BUILD_DIR="$(ASAN_BUILD_DIR)" BUILD_TYPE=Debug MARCH_NATIVE=OFF \
	  SANITIZERS=address,undefined test

## Start hvaxd. Pass backend/options through RUN_ARGS, e.g. RUN_ARGS=--coreml.
run: build
	@mkdir -p "$(DATA_DIR)"
	@exec "$(BUILD_DIR)/hvaxd" --data-dir "$(DATA_DIR)" --models-dir "$(MODELS_DIR)" $(RUN_ARGS)

## Run one image through detection and embedding.
once: build
	@test -n "$(IMAGE)" || { echo "set IMAGE=/path/to/image" >&2; exit 2; }
	@"$(BUILD_DIR)/hvaxd" --models-dir "$(MODELS_DIR)" --once "$(IMAGE)" $(RUN_ARGS)

## Run the end-to-end inference benchmark.
bench: build
	@test -n "$(IMAGE)" || { echo "set IMAGE=/path/to/image" >&2; exit 2; }
	@"$(BUILD_DIR)/hvax-infer-bench" "$(MODELS_DIR)" "$(IMAGE)" cpu "$(BENCH_RUNS)" "$(BENCH_WARMUP)"

## Install and build the optional browser extension.
extension:
	@command -v npm >/dev/null 2>&1 || { echo "missing required tool: npm" >&2; exit 2; }
	@npm --prefix extension ci
	@npm --prefix extension run typecheck
	@npm --prefix extension run build

## Remove compiled files while retaining the configured build tree.
clean:
	@target="$(abspath $(BUILD_DIR))"; \
	case "$$target" in \
	  "$(CURDIR)"/*) ;; \
	  *) echo "refusing to clean build directory outside $(CURDIR): $$target" >&2; exit 2 ;; \
	esac; \
	if [[ -f "$$target/CMakeCache.txt" ]]; then \
	  $(CMAKE) --build "$$target" --target clean; \
	else \
	  echo "nothing to clean in $$target"; \
	fi

## Remove the selected build tree after verifying it is inside this repository.
distclean:
	@target="$(abspath $(BUILD_DIR))"; \
	case "$$target" in \
	  "$(CURDIR)"/*) ;; \
	  *) echo "refusing to remove build directory outside $(CURDIR): $$target" >&2; exit 2 ;; \
	esac; \
	if [[ "$$target" == "$(CURDIR)" ]]; then \
	  echo "refusing to remove repository root" >&2; \
	  exit 2; \
	fi; \
	$(CMAKE) -E remove_directory "$$target"

## Show targets and configurable variables.
help:
	@echo "hvax Make targets"
	@echo "  make                     fetch/verify artifacts and build"
	@echo "  make setup               install packages, build, and test"
	@echo "  make bootstrap           fetch runtime/models and configure"
	@echo "  make models|runtime      fetch or verify external artifacts"
	@echo "  make build|test|verify   build or run the test suite"
	@echo "  make clean build         clean and rebuild"
	@echo "  make rebuild             clean and rebuild"
	@echo "  make debug|release|asan  create a separate build tree"
	@echo "  make run                 start hvaxd"
	@echo "  make once IMAGE=FILE     process one image"
	@echo "  make bench IMAGE=FILE    run CPU inference benchmark"
	@echo "  make extension           install and build extension dependencies"
	@echo "  make clean|distclean     clean outputs or remove the build tree"
	@echo
	@echo "Variables: BUILD_DIR, BUILD_TYPE, BUILD_TESTS, MARCH_NATIVE, JOBS,"
	@echo "           GENERATOR, CMAKE_ARGS, MODELS_DIR, DATA_DIR, RUN_ARGS,"
	@echo "           IMAGE, BENCH_RUNS, BENCH_WARMUP"
	@echo 'Example: make run RUN_ARGS="--coreml --coreml-profile"'
