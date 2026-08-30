# hvax ingest — Chrome extension

TypeScript Manifest V3 extension. It watches every page for `<img>` (and optional CSS `background-image`) nodes, including ones inserted after load, and POSTs the bytes to a configured hvax ingest URL.

## Build

```bash
cd extension
npm install
npm run build
```

Load unpacked in Chrome: `chrome://extensions` → Developer mode → **Load unpacked** → select `extension/dist`.

## Configure

Popup or Options:

- **gallery URL** — remote gallery used for authoritative stats and, by default, ingest
- **local processor URL** — optional `hvax serve` address such as `http://127.0.0.1:8080`; when set, image bytes go there while stats still come directly from the gallery
- **API key** — sent as `X-API-Key` if you started `hvaxd --api-key`
- min width/height — skip favicons and tracking pixels (default 64)
- capture CSS backgrounds — off if you only want `<img>`
- skip SVG

Start a local CPU processor with:

```bash
./build/hvax serve --server https://hv.ax --models-dir ./models --jobs 4
```

Add `--cuda` when the binary was built against ONNX Runtime GPU. The processor
preflights hashes against the gallery, skips known images, computes faces locally,
and uploads the finished payload. It is reachable from the extension service
worker on `127.0.0.1`; pages themselves never talk to hvax. The extension reads
`jobs` from `GET /health` on the ingest target (the local processor URL when
set, otherwise the gallery) and uses that as its POST concurrency, falling back
to 2 if the field is absent.

## What it captures

- existing `img` / `srcset` / `picture > source`
- `src` and `srcset` changes (lazy load)
- nodes added later (`MutationObserver`, including open shadow roots)
- `load` events on images that were incomplete at scan time
- CSS `background-image` URLs when enabled

Same-origin and `data:` images are read in the content script (page cookies
apply) and forwarded as bytes. Cross-origin fetches include matching site
credentials; if a CDN rejects the extension request, hvax retries from the
originating page frame and finally tries to encode the already-rendered image.
This supports authenticated image hosts such as private Google Photos URLs when
their normal page context is required. Canvas extraction remains unavailable
when a host supplies neither fetch access nor CORS permission. `blob:` images are decoded by Chrome and
normalized to JPEG first, so browser-supported formats also work when the
server's OpenCV build cannot decode the original container. Image bytes cross
the content-script boundary as base64 because Chrome extension messages are
JSON-serialized. Cross-origin URLs are fetched by the service worker.

Reloading or updating the extension automatically replaces its content script
in already-open web tabs. Unsupported browser-internal pages still require a
normal navigation before the extension can run there.

hvax 204 (no face) is treated as success: the image was seen and ignored by the gallery.

The popup reads authoritative image, face, embedding-row, and search-index stats
from the server's `/v1/stats` endpoint. Transfer and error counters remain local
to the current browser session. Its live `pending` count includes both queued
and currently processing images.
