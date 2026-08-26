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

- **server URL** — default `http://127.0.0.1:8080`; a base path or legacy full `/v1/ingest` URL also works (`hvaxd` must be running, bind not limited to localhost if you ingest from another machine)
- **API key** — sent as `X-API-Key` if you started `hvaxd --api-key`
- min width/height — skip favicons and tracking pixels (default 64)
- capture CSS backgrounds — off if you only want `<img>`
- skip SVG

`hvaxd` on `127.0.0.1` is reachable from the extension service worker (host permissions). Pages themselves never talk to hvax; the worker does.

## What it captures

- existing `img` / `srcset` / `picture > source`
- `src` and `srcset` changes (lazy load)
- nodes added later (`MutationObserver`, including open shadow roots)
- `load` events on images that were incomplete at scan time
- CSS `background-image` URLs when enabled

Same-origin, `blob:`, and `data:` images are read in the content script (page cookies apply) and forwarded as bytes. Cross-origin URLs are fetched by the service worker.

hvax 204 (no face) is treated as success: the image was seen and ignored by the gallery.

The popup reads authoritative image, face, embedding-row, and search-index stats
from the server's `/v1/stats` endpoint. Transfer and error counters remain local
to the current browser session.
