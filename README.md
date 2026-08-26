# hvax

hvax is a self-hosted face gallery for ingesting images and searching them by
facial similarity. Its server runs on CPU, while the companion CLI can process
images on CPU or CUDA before sending completed embeddings to the server. Images,
metadata, and 512-dimensional embeddings are stored directly on disk—no Python
runtime or database required.

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
- Parallel CLI with recursive directory scanning and optional CUDA inference
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

hvax currently targets **Linux x86-64** because ONNX Runtime 1.20.1 for that
platform is bundled in `third_party/`.

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

## Quick start

Download the two checksum-pinned InsightFace models, build the binaries, and
start the server:

```bash
./scripts/download_models.sh
cmake --preset rel
cmake --build build -j
./build/hvaxd --data-dir ./data --models-dir ./models
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
use the local CPU/GPU while keeping the gallery remote:

```bash
./build/hvax serve \
  --server https://hv.ax \
  --models-dir ./models \
  --jobs 4 \
  --bind 127.0.0.1 \
  --port 8080
```

Add `--cuda` for a GPU-enabled build. The local processor accepts ordinary image
bytes at `POST /v1/ingest`, checks the remote gallery before inference, and sends
the image plus completed embeddings to `/v1/ingest/processed`. `GET /v1/stats`
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
--det-size N            SCRFD letterbox size                 (default: 640)
--threads N             ONNX Runtime intra-op threads        (default: 8)
--http-threads N        HTTP worker threads                  (default: 8)
--api-key STRING        require X-API-Key on /v1/*           (default: disabled)
--dedup MODE            perceptual, sha256, or off           (default: perceptual)
--phash-threshold N     pHash Hamming-distance threshold     (default: 10)
--dhash-threshold N     dHash Hamming-distance threshold     (default: 12)
--once IMAGE            process one image and exit
--help                  show command help
```

For `--threads`, a good starting point is the number of physical performance
cores available to the process.

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
