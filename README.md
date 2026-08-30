# hvax

hvax is a self-hosted face gallery for ingesting images and searching them by
facial similarity. Its server and companion CLI can process images on CPU,
NVIDIA CUDA, or Apple CoreML/Metal before completed embeddings are stored.
Images, metadata, and 512-dimensional embeddings are stored directly on disk—no
Python runtime or database required.

> [!IMPORTANT]
> InsightFace distributes the `buffalo_l` model weights for **non-commercial
> research only**. The hvax source code is MIT-licensed; the model weights are
> not. See [License](#license).

## Highlights

- SCRFD-10G face detection and ArcFace R50 embeddings via ONNX Runtime
- Raw-image, single-embedding, and batch-embedding search
- Exact cosine search for galleries below 100,000 embedding rows
- USearch HNSW candidate search with exact float32 reranking at that threshold
  and above
- Memory-mapped metadata and embedding files instead of SQL
- SHA-256 and optional perceptual image deduplication
- Stable image IDs when a duplicate is replaced by a higher-resolution master
- Prometheus metrics and a health endpoint
- Validated processed-ingest API for using the server as storage only
- Parallel CLI with recursive directory scanning and optional CUDA or CoreML
  inference
- Optional Chrome extension for ingesting images seen while browsing

## How it works

```text
image -> SCRFD detector -> aligned face crops -> ArcFace embeddings -> gallery
                                                                    |
query image/embedding -> normalized 512-float vector -> cosine search + rerank
```

Each detected face gets one L2-normalized, 512-dimensional float32 embedding.
The `face_id` is also the embedding row, so searching does not require a query
planner or a metadata join on the hot path.

Images with no detected faces are ignored. Byte-identical images resolve to the
same `image_id`; with perceptual deduplication enabled, resized and recompressed
copies can resolve to it as well. If a perceptual duplicate has more pixels than
the stored master, hvax replaces the master image and recomputes its faces while
keeping the `image_id` stable.

## Requirements

hvax targets Linux x86-64 and Apple-silicon macOS. ONNX Runtime 1.20.1 for Linux
x86-64 is bundled in `third_party/`; the checksum-pinned macOS arm64 runtime is
downloaded separately.

- C++20 compiler
- CMake 3.24 or newer
- OpenCV (`core`, `imgproc`, and `imgcodecs`)
- OpenSSL
- spdlog
- nlohmann/json
- GoogleTest when building the test suite (enabled by default)
- Ninja when using the included CMake preset

On Ubuntu, the system packages are typically:

```bash
sudo apt install build-essential cmake ninja-build curl \
  libopencv-dev libssl-dev libspdlog-dev nlohmann-json3-dev libgtest-dev
```

Check that your distribution supplies CMake 3.24 or newer before configuring.

On Apple-silicon macOS with Homebrew:

```bash
brew install cmake ninja opencv openssl spdlog nlohmann-json googletest
./scripts/download_onnxruntime.sh
```

The macOS bootstrap installs the official ONNX Runtime arm64 package. Official
macOS arm64 packages include `CoreMLExecutionProvider`; hvax rejects `--coreml`
if that provider is absent instead of falling back silently. See the
[ONNX Runtime CoreML provider documentation](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
and [macOS build documentation](https://onnxruntime.ai/docs/build/inferencing.html).

## Quick start

Install platform dependencies, fetch and verify ONNX Runtime and the two pinned
InsightFace models, build, and test with one command:

```bash
make setup
```

Subsequent `make` invocations reverify required artifacts and build only changed
sources. Start the server with:

```bash
make run
```

On Apple silicon, use `make run RUN_ARGS=--coreml`.

`make setup` uses Homebrew on macOS and `apt-get` on Linux. When dependencies
are already installed, `make bootstrap`, `make`, and `make test` perform setup,
build, and validation without modifying system packages. Run `make help` for
debug, release, sanitizer, benchmark, extension, and guarded cleanup targets.

### Make workflow

| Command | Operation |
|---|---|
| `make` or `make build` | Verify runtime/models, configure, and incrementally build |
| `make setup` | Install platform packages, then build and test |
| `make models` | Fetch missing models, repair checksum failures atomically, or verify existing models |
| `make runtime` | Fetch or verify the platform ONNX Runtime distribution |
| `make test` | Build and run CTest |
| `make clean build` or `make rebuild` | Remove compiled outputs and rebuild |
| `make debug`, `make release` | Build separate debug or release trees |
| `make asan` | Build and test with AddressSanitizer and UndefinedBehaviorSanitizer |
| `make once IMAGE=face.jpg RUN_ARGS=--coreml` | Run one image through the pipeline |
| `make distclean` | Remove the selected build tree after a repository-boundary check |

Build variables can be overridden without editing the Makefile, for example:

```bash
make build BUILD_DIR=build-custom BUILD_TYPE=Debug JOBS=8
make run RUN_ARGS="--coreml --coreml-profile"
```

The server listens on `http://127.0.0.1:8080` by default. In another terminal:

```bash
curl http://127.0.0.1:8080/health

curl --data-binary @face.jpg \
  -H 'Content-Type: image/jpeg' \
  http://127.0.0.1:8080/v1/ingest
```

A successful ingest returns the image metadata and one record per detected
face:

```json
{
  "image_id": 1,
  "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "width": 1280,
  "height": 853,
  "duplicate": false,
  "master_replaced": false,
  "faces": [
    {
      "face_id": 0,
      "image_id": 1,
      "bbox": [421.2, 171.8, 725.4, 566.1],
      "det_score": 0.94,
      "landmarks": [
        [503.1, 319.4],
        [637.8, 315.2],
        [572.6, 393.7],
        [520.4, 468.9],
        [628.7, 465.3]
      ]
    }
  ]
}
```

No-face images return `204 No Content` and are not stored.

## Process locally, store remotely

`hvax` can recursively scan files and directory trees, run SCRFD and ArcFace on
the local machine, and submit the image plus completed face metadata to a remote
gallery:

```bash
./build/hvax ingest \
  --server https://hv.ax \
  --models-dir ./models \
  --jobs 4 \
  ~/Pictures ./another-image.jpg
```

Directory inputs are recursive by default. `--jobs` controls concurrent decode,
inference, and upload work; `--no-recursive` limits directory inputs to one
level. The command prints one result per file and a stored/duplicate/no-face/error
summary. `HVAX_SERVER` and `HVAX_API_KEY` may be used instead of their flags.

Before running face detection, the CLI sends the image's SHA-256, pHash, dHash,
and dimensions to `/v1/ingest/check`. Exact duplicates and strong perceptual
duplicates whose existing master is at least as large skip inference and upload.
Potential higher-resolution upgrades and borderline perceptual matches are still
processed so the server can apply embedding confirmation and master replacement.
Model loading is lazy, so a scan containing only known images never initializes
ONNX Runtime.

Local inference results are cached by image SHA-256 under
`$XDG_CACHE_HOME/hvax` or `~/.cache/hvax`. This includes no-face results, which
the remote gallery intentionally does not store, and completed face payloads
that can be retried without rerunning ONNX. Cache entries are automatically
namespaced by model file metadata and detector settings. Use `--cache-dir DIR`
to choose another location or `--no-cache` to disable it. Cache files contain
face embeddings and are created with owner-only permissions.

Successful uploads and remote duplicate responses are cached separately by the
image SHA-256. That cache namespace includes a hash of the normalized remote
server URL and API paths, so a hit from one gallery is never reused for another.
These entries skip decoding, inference, and network requests on later scans.
Use `--no-cache` (or remove that server's `remote-v1` cache directory) if the
remote gallery has been reset or its images were deleted.

The server lazily loads its ONNX models, so a process that only receives
`/v1/ingest/processed` requests does not need the models at all. It still decodes
each image and independently computes SHA-256, pHash, and dHash before applying
the normal deduplication and gallery persistence rules.

### Local processing server

Run the CLI as a loopback-only ingest server when the browser extension should
use local CPU/GPU/ANE inference while keeping the gallery remote:

```bash
./build/hvax serve \
  --server https://hv.ax \
  --models-dir ./models \
  --jobs 4 \
  --bind 127.0.0.1 \
  --port 8080
```

Add `--cuda`, `--coreml`, or `--mps` for accelerated inference. The local
processor accepts ordinary image bytes at `POST /v1/ingest`, checks the remote
gallery before inference, and sends the image plus completed embeddings to
`/v1/ingest/processed`. `GET /v1/stats`
proxies the remote gallery's authoritative counts. It binds only to loopback by
default; do not expose it to untrusted networks.

### CUDA client build

The bundled ONNX Runtime distribution is CPU-only. To enable `--cuda`, configure
against a matching ONNX Runtime GPU distribution:

```bash
curl -fL -o /tmp/onnxruntime-gpu.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-gpu-1.20.1.tgz
tar -xzf /tmp/onnxruntime-gpu.tgz -C third_party

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DHVAX_ORT_ROOT="$PWD/third_party/onnxruntime-linux-x64-gpu-1.20.1"
cmake --build build -j

./build/hvax ingest \
  --server https://hv.ax \
  --models-dir ./models \
  --cuda --cuda-device 0 --jobs 4 \
  ~/Pictures
```

The GPU distribution's CUDA and cuDNN runtime requirements must be installed.
If `--cuda` is requested from a CPU-only build, the CLI exits with an explicit
provider-unavailable error rather than silently falling back to CPU.

### Apple CoreML/Metal acceleration

`--coreml` uses Core ML with all compatible Apple compute units. The default
`auto` uses MLProgram for SCRFD plus NeuralNetwork format for ArcFace when one
inference is expected in flight. With multiple workers it uses NeuralNetwork
format for both models because that detector scales better under concurrent
execution.

The distributed SCRFD model has symbolic spatial axes. For CoreML, hvax
specializes both axes to `--det-size` in the ONNX Runtime session, requires
static inputs, and verifies the resulting detector input is exactly
`[1,3,N,N]`. At `N=640`, the resulting optimized node count, CoreML coverage,
and outputs matched a separately exported static ONNX model without adding a
generated model artifact. Detector sizes
must be positive multiples of the maximum feature-pyramid stride, 32 pixels;
the daemon and local processor reject other values before inference.
`--mps` is an alias for `--coreml --coreml-compute-units cpu-gpu`, excluding the
Neural Engine. ONNX Runtime exposes CoreML rather than a PyTorch-style MPS
execution provider; this alias selects Core ML's CPU-and-GPU compute-unit mode.
Explicit controls are available:

```text
--coreml-compute-units all|cpu-gpu|cpu-ane|cpu
--coreml-model-format auto|mlprogram|neuralnetwork
--coreml-cache-dir DIR
--coreml-low-precision
--coreml-profile
```

Compiled Core ML models are cached persistently and namespaced by model path,
size, modification time, detector size, and static-specialization version. The
daemon defaults to `DATA_DIR/coreml-cache`; the local processor defaults to
`$XDG_CACHE_HOME/hvax/coreml-models` or `~/.cache/hvax/coreml-models`.
Changing either model normally selects a new namespace. Remove the cache if a
model was replaced while preserving its path, size, and modification time.

The default CoreML admission limit is three simultaneous inferences, bounded by
the configured HTTP worker count. On the measured M4 Max host, throughput
stopped increasing beyond three while tail latency continued to increase. The
local processor therefore defaults to `--jobs 3` under CoreML; an explicit
`--jobs` value is retained.

`--coreml-low-precision` permits float16 GPU accumulation and remains disabled
unless explicitly requested. In a 43-image deterministic stress corpus with
133 detected faces, float32 and float16 accumulation produced equal face counts
and a minimum matched-box IoU of 0.99999989. Box, landmark, and detection-score
differences reported zero at 10⁻⁸ output precision; minimum corresponding-face
embedding cosine similarity was 0.99991477, maximum embedding L2 distance was
0.01305318, and the maximum absolute shift among all pairwise embedding cosine
scores was 0.00282283. Repeated 200-run CPU+GPU measurements observed
105.5–105.6 images/s with float32 accumulation and 108.4–109.6 images/s with
float16 accumulation, a paired increase of 2.65%–3.84%. The option therefore
exposes the measured throughput/numerical tradeoff without changing the default
numerical mode. `--coreml-profile` emits
Core ML partition and compute-plan diagnostics, including exact CPU fallback
nodes.

Static specialization changes detector partitioning as follows on ONNX Runtime
1.29.0. The dynamic detector assigned 142/153 optimized nodes to MLProgram and
133/153 to NeuralNetwork. After specialization and constant folding,
MLProgram assigns 137/137 detector nodes to one CoreML partition. NeuralNetwork
assigns 134/137; the only CPU nodes are `AveragePool_36`, `AveragePool_67`, and
`AveragePool_84`. A semantics-equivalent test export changed their `ceil_mode`
from 1 to 0 for even feature-map dimensions and obtained full NeuralNetwork
capture, but concurrency-3 throughput decreased from 263.4 to 193.9 images/s
(-26.4%). The production graph retains those three CPU nodes; its observed
throughput was 35.8% higher than the fully captured variant in that sweep.

Measured on an Apple M4 Max, macOS 27.0, ONNX Runtime 1.29.0, detector size 640,
and the 512×512 Lena fixture (300 steady-state runs after warmup unless stated):

| Backend | Concurrency | Observed mean latency | Observed throughput |
|---|---:|---:|---:|
| ORT CPU, 8 threads (200 runs) | 1 | 44.388 ms | 22.529 images/s |
| CoreML auto, static detector | 1 | 6.831 ms | 146.383 images/s |
| CoreML auto, static detector | 3 | 11.621 ms | 257.265 images/s |

The static CoreML single-flight throughput ratio was 6.50× relative to 8-thread
CPU for this input. Approximate 95% confidence half-widths for mean latency were
0.057 ms for CoreML and 0.326 ms for CPU. Across the
43-image corpus, CPU and static CoreML produced 133/133 matched faces with no
count mismatches, minimum IoU 0.99999847, maximum box and landmark absolute
error 0.00012207 pixels, maximum detection-score absolute error 1.2×10⁻⁷,
minimum corresponding-face embedding cosine similarity 0.99973571, and maximum
pairwise cosine-score shift 0.00651477. These empirical bounds characterize
this host, model pair, and corpus; use `hvax-infer-bench` on the deployment
distribution.

## Search

### By image

hvax detects every face in the probe image and returns a separate hit list for
each one:

```bash
curl --data-binary @probe.jpg \
  -H 'Content-Type: image/jpeg' \
  -H 'X-K: 10' \
  -H 'X-Min-Score: 0.35' \
  http://127.0.0.1:8080/v1/query/image
```

The response has the shape `{ "queries": [{ "bbox": ..., "hits": [...] }] }`.
A probe with no detected faces returns `204 No Content`.

### By embedding

Send either 2,048 raw bytes containing 512 little-endian float32 values:

```bash
curl --data-binary @query.f32 \
  -H 'Content-Type: application/octet-stream' \
  -H 'X-K: 10' \
  -H 'X-Min-Score: 0.35' \
  http://127.0.0.1:8080/v1/query/embedding
```

Or send JSON with one `embedding` field whose value is an array of exactly 512
numbers.

Input embeddings are L2-normalized by the server. Hits are sorted by descending
cosine similarity and include `face_id`, `image_id`, `sha256`, `score`, `bbox`, and
`det_score`.

For batch search, concatenate `N` raw embeddings and use
`POST /v1/query/embedding/batch`. hvax infers `N` from the body length, or you
can set it explicitly with `X-Count`.

## HTTP API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness and gallery summary |
| `GET` | `/metrics` | Prometheus text metrics |
| `GET` | `/v1/stats` | Image, face, embedding-row, and index counts |
| `POST` | `/v1/ingest` | Detect faces and add an image to the gallery |
| `POST` | `/v1/ingest/check` | Check SHA-256 and perceptual hashes before processing |
| `POST` | `/v1/ingest/processed` | Store an image with client-computed faces and embeddings |
| `POST` | `/v1/query/image` | Search every face found in an image |
| `POST` | `/v1/query/embedding` | Search one raw or JSON embedding |
| `POST` | `/v1/query/embedding/batch` | Search concatenated raw embeddings |
| `GET` | `/v1/faces/:id` | Fetch face metadata |
| `GET` | `/v1/faces/:id?include_embedding=1` | Fetch face metadata and its embedding |
| `GET` | `/v1/images/:sha256/meta` | Fetch image metadata and face IDs |
| `GET` | `/v1/images/:sha256` | Download the stored master image |
| `DELETE` | `/v1/images/:sha256` | Delete an image and tombstone its faces |

Image downloads and metadata are addressed only by the master image's full
SHA-256. Search hits return that value in `sha256`; numeric image routes are not
registered. This prevents trivial gallery scraping by incrementing image IDs,
but it is not a substitute for authentication when image contents or hashes are
already known. Set an API key if gallery access itself must be restricted.

`/health`, `/metrics`, and `/v1/stats` explicitly return a
`Cache-Control: no-store` response header; clients should not reuse gallery
counts from an HTTP cache.

Search endpoints accept these optional headers:

| Header | Default | Meaning |
|---|---:|---|
| `X-K` | `10` | Maximum hits per query face or embedding |
| `X-Min-Score` | `0.0` | Minimum cosine-similarity score |
| `X-Count` | inferred | Number of embeddings in a batch body |

Ingest and image-query bodies may be raw encoded images or multipart uploads.
The maximum request size is 20 MiB, and decoded images above 40 megapixels are
rejected. JPEG, PNG, and WebP are recognized for stored MIME metadata, subject
to the codecs available in OpenCV.

Processed ingest requires multipart fields named `image` and `payload`. The
JSON payload is versioned and has this shape:

```json
{
  "version": 1,
  "model": "insightface-buffalo_l",
  "embedding_dim": 512,
  "faces": [
    {
      "bbox": [421.2, 171.8, 725.4, 566.1],
      "det_score": 0.94,
      "landmarks": [[503.1, 319.4], [637.8, 315.2], [572.6, 393.7], [520.4, 468.9], [628.7, 465.3]],
      "embedding": ["512 finite float values"]
    }
  ]
}
```

`embedding` must contain numbers rather than the descriptive string shown
above. The server requires the `insightface-buffalo_l` model identifier,
validates geometry and scores, rejects zero or non-finite
embeddings, clamps coordinates to the decoded image, and L2-normalizes every
embedding. Submit a prepared payload directly with:

```bash
curl -F 'image=@face.jpg' \
  -F 'payload=@payload.json;type=application/json' \
  http://127.0.0.1:8080/v1/ingest/processed
```

Common response statuses:

| Status | Meaning |
|---:|---|
| `200` | Request succeeded |
| `204` | No face was found, or a delete succeeded |
| `400` | Empty ingest body or malformed embedding input |
| `401` | Missing or incorrect API key |
| `404` | Image or face ID does not exist |
| `413` | The submitted image or processed payload is too large |
| `415` | Ingest body could not be decoded as an image |
| `422` | A processed-ingest payload failed validation |

### Authentication and exposure

Set an API key to protect all `/v1/*` endpoints:

```bash
./build/hvaxd --api-key 'change-me'

curl -H 'X-API-Key: change-me' http://127.0.0.1:8080/v1/stats
```

`/health` and `/metrics` remain unauthenticated. hvax binds to localhost by
default; if you expose it on another interface, put it behind TLS and suitable
network access controls. Face embeddings and source images are sensitive data,
so use hvax only with the knowledge and permission of the people involved and
in accordance with applicable law.

## Deduplication

The default `--dedup perceptual` mode applies two image hashes:

- SHA-256 collapses byte-identical input.
- pHash and dHash identify likely resized or recompressed copies. Borderline
  perceptual matches are confirmed using face embeddings before merging.

The ingest response reports `duplicate`, `duplicate_kind` (`sha256` or
`perceptual`), and `master_replaced`. A replacement also includes the previous
master's dimensions and SHA-256 under `previous`.

Use `--dedup sha256` when only byte-identical images should be merged. Exact
SHA-256 matches are always reused, including when `--dedup off` is selected;
`off` disables the perceptual stage.

Tune perceptual matching with `--phash-threshold` and `--dhash-threshold`.
Larger values merge less visually similar images and therefore increase the
risk of false positives.

## Configuration

```text
--data-dir DIR          gallery directory                    (default: ./data)
--models-dir DIR        directory containing both ONNX files (default: ./models)
--bind ADDR             listen address                       (default: 127.0.0.1)
--port N                listen port                          (default: 8080)
--det-size N            SCRFD size, positive multiple of 32  (default: 640)
--threads N             ONNX Runtime intra-op threads        (default: 8)
--cuda                  use CUDAExecutionProvider
--cuda-device N         CUDA device ID                       (default: 0)
--coreml                use all compatible Apple compute units
--mps                   use CoreML CPU+GPU only
--coreml-compute-units  all, cpu-gpu, cpu-ane, or cpu        (default: all)
--coreml-model-format   auto, mlprogram, or neuralnetwork    (default: auto)
--coreml-cache-dir DIR  persistent compiled-model cache
--coreml-low-precision  permit float16 GPU accumulation      (default: off)
--coreml-profile        log Core ML operator placement       (default: off)
--http-threads N        HTTP worker threads                  (default: 8)
--api-key STRING        require X-API-Key on /v1/*           (default: disabled)
--dedup MODE            perceptual, sha256, or off           (default: perceptual)
--phash-threshold N     pHash Hamming-distance threshold     (default: 10)
--dhash-threshold N     dHash Hamming-distance threshold     (default: 12)
--once IMAGE            process one image and exit
--help                  show command help
```

`--threads` controls CPU execution and CPU fallback nodes. CoreML measurements
should sweep this value when the selected models contain unsupported operators.

### Inspect one image without starting HTTP

```bash
./build/hvaxd \
  --models-dir ./models \
  --data-dir /tmp/hvax-once \
  --once tests/fixtures/lena.jpg
```

This prints detected boxes, landmarks, confidence scores, and embedding norms as
JSON. It still opens the specified data directory but does not ingest the image.

## Storage and search

The gallery is append-oriented and memory-mapped. Only one `hvaxd` process may
open a given data directory for writing; `hvax.lock` enforces that constraint.

| Path | Purpose |
|---|---|
| `images/<aa>/<sha256>` | Content-addressed master images |
| `images.slots` | Packed image records |
| `faces.slots` | Packed face records; `face_id` equals embedding row |
| `embeddings.f32` | Aligned, normalized 512-float embeddings |
| `embeddings.i8` | Quantized companion embeddings |
| `index.usearch` | Persistent HNSW index |
| `hvax.lock` | Single-writer process lock |

Below 100,000 embedding rows, search is an exact float32 inner-product scan.
At and above 100,000 rows, USearch HNSW produces candidates and hvax reranks
them using the original float32 embeddings. Deleted rows remain as tombstones,
so `embedding_rows` can be larger than the live face count.

## Chrome extension

The optional Manifest V3 extension watches pages for existing and dynamically
inserted images, then sends them to `/v1/ingest`:

```bash
cd extension
npm install
npm run build
```

In `chrome://extensions`, enable Developer mode, choose **Load unpacked**, and
select `extension/dist`. Configure the endpoint, API key, minimum image size,
and CSS-background capture from the popup or options page.

The extension requests access to all HTTP and HTTPS pages and can transmit the
images displayed on those pages to your configured hvax server. Review its
permissions and use it only in browsing contexts where that collection is
appropriate. See [extension/README.md](extension/README.md) for details.

## Development

Run the C++ test suite:

```bash
ctest --test-dir build --output-on-failure
```

Build without tests:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DHVAX_BUILD_TESTS=OFF
cmake --build build -j
```

Run the exact-search benchmark with `100000` rows, `1000` queries, and top-10
results:

```bash
./build/hvax-bench 100000 1000 10
```

Run end-to-end inference latency and provider-equivalence checks:

```bash
./build/hvax-infer-bench models tests/fixtures/lena.jpg cpu 100 10
HVAX_COREML_CACHE_DIR=.cache/coreml \
  ./build/hvax-infer-bench models tests/fixtures/lena.jpg coreml 100 10
HVAX_COREML_CACHE_DIR=.cache/coreml \
  ./build/hvax-infer-bench models tests/fixtures/lena.jpg compare
HVAX_COREML_CACHE_DIR=.cache/coreml \
  ./build/hvax-infer-bench models path/to/image-corpus compare-low-precision
```

The two comparison modes accept either one image or a recursively scanned
directory of JPEG, PNG, or WebP files. Faces are matched by maximum-total-IoU
bipartite assignment rather than output order. The report includes count
mismatches, box/landmark/score absolute errors, corresponding-face embedding
cosine and L2 distances, and maximum absolute change over all pairwise
embedding cosine scores. For `R` reference faces, `C` candidate faces, and `F`
matched embeddings, matching costs `O(max(R,C)^3)` time and `O(max(R,C))`
auxiliary space per image; the 512-dimensional all-pairs comparison costs
`O(512 F^2)` time and `O(512 F)` retained embedding space.

`HVAX_BENCH_CONCURRENCY`, `HVAX_BENCH_TILE_X`, `HVAX_ORT_THREADS`,
`HVAX_COREML_MODEL_FORMAT`, `HVAX_COREML_FAST_PREDICTION`, and
`HVAX_COREML_LOW_PRECISION` control benchmark sweeps without changing production
defaults.

Check the extension separately with:

```bash
npm --prefix extension run typecheck
npm --prefix extension run build
```

## License

hvax source code is available under the [MIT License](LICENSE).

The downloaded `det_10g.onnx` and `w600k_r50.onnx` weights are separately
licensed by InsightFace for non-commercial research. They are not covered by
the MIT license. Contact the model publisher for commercial licensing.
