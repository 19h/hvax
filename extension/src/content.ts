import { DEFAULT_SETTINGS, type IngestBytesMessage, type IngestUrlMessage, type Settings } from "./types";

const seen = new Set<string>();
let settings: Settings = { ...DEFAULT_SETTINGS };
const observedRoots = new WeakSet<Node>();
const observers = new Set<MutationObserver>();
let alive = true;

type ContentInstance = { stop: () => void };
const instanceKey = "__hvaxContentInstance";
const contentGlobal = globalThis as typeof globalThis & { [instanceKey]?: ContentInstance };

function onStorageChanged(changes: Record<string, chrome.storage.StorageChange>, area: string): void {
  if (!alive || area !== "sync") return;
  const was = settings.enabled;
  settings = { ...settings, ...Object.fromEntries(Object.entries(changes).map(([k, v]) => [k, v.newValue])) };
  if (!was && settings.enabled) walk(document);
}

function onImageLoad(ev: Event): void {
  if (alive && ev.target instanceof HTMLImageElement) considerImg(ev.target);
}

function shutdown(): void {
  if (!alive) return;
  alive = false;
  observers.forEach((observer) => observer.disconnect());
  observers.clear();
  try {
    chrome.storage?.onChanged.removeListener(onStorageChanged);
  } catch {
    // A reloaded extension invalidates this API before the old page world is
    // torn down. The DOM cleanup above must still complete without an uncaught
    // exception.
  }
  document.removeEventListener("load", onImageLoad, true);
  if (contentGlobal[instanceKey]?.stop === shutdown) delete contentGlobal[instanceKey];
}

contentGlobal[instanceKey]?.stop();
contentGlobal[instanceKey] = { stop: shutdown };

function normalizeUrl(raw: string): string {
  const t = raw.trim().replace(/^url\((['"]?)(.*)\1\)$/, "$2");
  if (!t || t === "none" || t.startsWith("chrome-extension://") || t.startsWith("about:")) return "";
  try {
    return new URL(t, document.baseURI).href;
  } catch {
    return "";
  }
}

function looksSvg(url: string): boolean {
  return /\.svg(\?|#|$)/i.test(url) || url.startsWith("data:image/svg");
}

function already(url: string): boolean {
  if (!url || seen.has(url)) return true;
  seen.add(url);
  return false;
}

function sameOrigin(url: string): boolean {
  try {
    return new URL(url).origin === location.origin;
  } catch {
    return false;
  }
}

async function sendMessage(msg: IngestUrlMessage | IngestBytesMessage): Promise<boolean> {
  if (!alive) return false;
  try {
    if (!chrome.runtime?.id) {
      shutdown();
      return false;
    }
    await chrome.runtime.sendMessage(msg);
    return true;
  } catch (err) {
    if (!chrome.runtime?.id || (err instanceof Error && err.message.includes("Extension context invalidated"))) {
      shutdown();
    }
    return false;
  }
}

function sendUrl(url: string, width: number, height: number): void {
  const msg: IngestUrlMessage = {
    type: "ingest-url",
    url,
    pageUrl: location.href,
    width,
    height,
  };
  void sendMessage(msg);
}

async function transcodeBlob(blob: Blob, image?: HTMLImageElement): Promise<Blob> {
  let bitmap: ImageBitmap | undefined;
  let width = image?.naturalWidth ?? 0;
  let height = image?.naturalHeight ?? 0;
  if (!image || width === 0 || height === 0) {
    bitmap = await createImageBitmap(blob);
    width = bitmap.width;
    height = bitmap.height;
  }

  // Stay below hvaxd's default 40-megapixel decode limit and common canvas limits.
  const maxPixels = 39_000_000;
  const scale = Math.min(1, Math.sqrt(maxPixels / Math.max(1, width * height)), 16_384 / Math.max(width, height));
  const outWidth = Math.max(1, Math.floor(width * scale));
  const outHeight = Math.max(1, Math.floor(height * scale));
  const canvas = new OffscreenCanvas(outWidth, outHeight);
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    bitmap?.close();
    throw new Error("2D canvas unavailable");
  }
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, outWidth, outHeight);
  if (image) ctx.drawImage(image, 0, 0, outWidth, outHeight);
  else if (bitmap) ctx.drawImage(bitmap, 0, 0, outWidth, outHeight);
  bitmap?.close();
  return canvas.convertToBlob({ type: "image/jpeg", quality: 0.92 });
}

async function sendBytes(url: string, width: number, height: number, image?: HTMLImageElement): Promise<void> {
  let res: Response;
  try {
    res = await fetch(url, { credentials: "include", cache: "force-cache" });
    if (!res.ok) {
      sendUrl(url, width, height);
      return;
    }
  } catch {
    sendUrl(url, width, height);
    return;
  }

  let body = await res.blob();
  if (url.startsWith("blob:")) {
    try {
      // Chrome may display AVIF, animated WebP, SVG, or other blob-backed
      // formats that the server's OpenCV build cannot decode. Normalize the
      // browser-decoded pixels to a regular JPEG before sending them.
      body = await transcodeBlob(body, image);
    } catch {
      // Preserve the old behavior as a fallback if this browser cannot render
      // the blob through an OffscreenCanvas.
    }
  }
  if (body.size === 0 || body.size > settings.maxBytes) return;
  const msg: IngestBytesMessage = {
    type: "ingest-bytes",
    sourceUrl: url,
    pageUrl: location.href,
    mime: body.type || "application/octet-stream",
    width,
    height,
    bytes: await body.arrayBuffer(),
  };
  await sendMessage(msg);
}

function dispatch(url: string, width: number, height: number, image?: HTMLImageElement): void {
  if (already(url)) return;
  if (settings.skipSvg && looksSvg(url)) return;
  if (width > 0 && height > 0 && (width < settings.minWidth || height < settings.minHeight)) return;
  if (url.startsWith("blob:") || url.startsWith("data:") || sameOrigin(url)) {
    void sendBytes(url, width, height, image);
  } else {
    sendUrl(url, width, height);
  }
}

function considerImg(img: HTMLImageElement): void {
  if (!settings.enabled) return;
  if (!img.complete || img.naturalWidth === 0) {
    img.addEventListener("load", () => considerImg(img), { once: true });
    return;
  }
  const url = normalizeUrl(img.currentSrc || img.src);
  if (!url) return;
  dispatch(url, img.naturalWidth, img.naturalHeight, img);
}

function considerSrcset(el: HTMLSourceElement | HTMLImageElement): void {
  const srcset = el.getAttribute("srcset");
  if (!srcset) return;
  let best = "";
  let bestW = 0;
  for (const part of srcset.split(",")) {
    const bits = part.trim().split(/\s+/);
    const u = bits[0] ? normalizeUrl(bits[0]) : "";
    if (!u) continue;
    const desc = bits[1] ?? "";
    const w = desc.endsWith("w") ? parseInt(desc, 10) : 0;
    if (w >= bestW) {
      bestW = w;
      best = u;
    } else if (!best) {
      best = u;
    }
  }
  if (best) dispatch(best, bestW, bestW, el instanceof HTMLImageElement ? el : undefined);
}

const BG_URL = /url\(\s*(['"]?)(.*?)\1\s*\)/g;

function considerBackground(el: Element): void {
  if (!settings.captureBackgrounds) return;
  let css = "";
  try {
    css = getComputedStyle(el).backgroundImage;
  } catch {
    return;
  }
  if (!css || css === "none") return;
  BG_URL.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = BG_URL.exec(css))) {
    const url = normalizeUrl(m[2] ?? "");
    if (url) dispatch(url, 0, 0);
  }
}

function walk(root: ParentNode): void {
  root.querySelectorAll("img").forEach((n) => considerImg(n));
  root.querySelectorAll("source[srcset], img[srcset]").forEach((n) => considerSrcset(n as HTMLSourceElement));
  if (settings.captureBackgrounds) {
    root.querySelectorAll("*").forEach((el) => considerBackground(el));
  }
  root.querySelectorAll("*").forEach((el) => {
    const sr = (el as HTMLElement).shadowRoot;
    if (sr) {
      walk(sr);
      observe(sr);
    }
  });
}

function observe(root: Node): void {
  if (!alive || observedRoots.has(root)) return;
  observedRoots.add(root);
  const mo = new MutationObserver((mutations) => {
    if (!alive || !settings.enabled) return;
    for (const m of mutations) {
      if (m.type === "attributes" && m.target instanceof HTMLImageElement) {
        considerImg(m.target);
        continue;
      }
      if (m.type === "attributes" && m.target instanceof Element) {
        considerBackground(m.target);
        continue;
      }
      for (const node of m.addedNodes) {
        if (node instanceof HTMLImageElement) considerImg(node);
        else if (node instanceof Element) {
          node.querySelectorAll("img").forEach((img) => considerImg(img));
          const sr = (node as HTMLElement).shadowRoot;
          if (sr) {
            walk(sr);
            observe(sr);
          }
        }
      }
    }
  });
  observers.add(mo);
  mo.observe(root, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["src", "srcset", "style", "class"],
  });
}

async function boot(): Promise<void> {
  try {
    settings = await chrome.storage.sync.get(DEFAULT_SETTINGS).then((s) => ({ ...DEFAULT_SETTINGS, ...s }));
  } catch {
    shutdown();
    return;
  }
  if (!alive) return;
  chrome.storage.onChanged.addListener(onStorageChanged);
  observe(document.documentElement);
  document.addEventListener("load", onImageLoad, true);
  if (settings.enabled) walk(document);
}

void boot();
