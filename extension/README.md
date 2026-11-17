# hvax ingest — Chrome and Safari extension

TypeScript Manifest V3 extension, also packaged as a native Safari app (see
[Safari](#safari-macos)). It watches every page for `<img>` (and optional CSS `background-image`) nodes, including ones inserted after load, and POSTs the bytes to a configured hvax ingest URL.

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
- skip backend images — on by default; an image whose URL is served by the
  gallery or local-processor host (including its subdomains) is ignored, so
  browsing the gallery itself never sends its own images back to
  `/v1/ingest` or re-indexes them

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

Images whose URL belongs to the configured gallery or processor host are
skipped before they are fetched (this check also runs again in the service
worker), so pages such as the hvax gallery do not re-ingest images that are
already stored.

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
So is a 422 with `"rejected": "too_small"`: the gallery refuses images below its
`--min-image-side` (350 px by default). The extension remembers the limit from the
first refusal and stops fetching images it knows to be smaller.

The popup reads authoritative image, face, embedding-row, and search-index stats
from the server's `/v1/stats` endpoint. Transfer and error counters remain local
to the current browser session. Its live `pending` count includes both queued
and currently processing images.

## Safari (macOS)

The same sources build into a Safari Web Extension wrapped in a small native
macOS app. It needs Xcode 27 or newer to build and runs on macOS 14 / Safari 17
or newer.

```bash
cd extension
npm install
npm run safari
```

`make safari-extension` from the repository root does the same after a clean
`npm ci` and a typecheck. The app lands in
`safari/build/Build/Products/Release/hvax ingest.app`.

By default the build is only signed to run locally, which Safari treats as
unsigned: turn on **Allow unsigned extensions** in Safari's developer settings,
and again after every Safari restart. To sign with your Apple Development
certificate instead, pass your team ID:

```bash
npm run safari -- DEVELOPMENT_TEAM=ABCDE12345 CODE_SIGN_IDENTITY="Apple Development"
# or: make safari-extension SAFARI_TEAM=ABCDE12345
```

Then:

1. Open `hvax ingest.app` once. It shows whether the extension is on and has a
   button that opens Safari's Extensions settings.
2. Enable **hvax ingest** there.
3. Choose **Always Allow on Every Website**. Safari grants site access per
   host; without it the extension cannot read page images, fetch them from
   their CDNs, or reach the gallery.

The popup, options page, and capture behaviour are the ones described above.
Differences from Chrome:

- The background code runs as a non-persistent module page instead of a service
  worker (`esbuild.mjs --safari` rewrites the manifest).
- The toolbar shows a monochrome glyph (`public/icons/toolbar-icon.svg`).
- Safari keeps `storage.sync` on the device, so settings do not follow you to
  another Mac.
- Safari versions without `match_about_blank` ignore that key and do not scan
  `about:blank` frames.

Layout:

- `safari/App` — the SwiftUI container app.
- `safari/Extension` — the app-extension target Safari loads.
- `safari/hvax-ingest.xcodeproj` — bundles the files in `dist-safari/`, so run
  `npm run build:safari` before building from Xcode. A new file in the web
  bundle must be added to the `Resources` group in `project.xcproj`.

`npm run test:safari` runs the background fallback test against the Safari
bundle.
