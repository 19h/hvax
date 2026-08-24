import { DEFAULT_SETTINGS, type IngestBytesMessage, type IngestUrlMessage, type Settings } from "./types";

const seen = new Set<string>();
let settings: Settings = { ...DEFAULT_SETTINGS };
const observedRoots = new WeakSet<Node>();

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

function sendUrl(url: string, width: number, height: number): void {
  const msg: IngestUrlMessage = {
    type: "ingest-url",
    url,
    pageUrl: location.href,
    width,
    height,
  };
  void chrome.runtime.sendMessage(msg).catch(() => undefined);
}

async function sendBytes(url: string, width: number, height: number): Promise<void> {
  try {
    const res = await fetch(url, { credentials: "include", cache: "force-cache" });
    if (!res.ok) {
      sendUrl(url, width, height);
      return;
    }
    const mime = res.headers.get("content-type") || "application/octet-stream";
    const buf = await res.arrayBuffer();
    if (buf.byteLength === 0 || buf.byteLength > settings.maxBytes) return;
    const msg: IngestBytesMessage = {
      type: "ingest-bytes",
      sourceUrl: url,
      pageUrl: location.href,
      mime: mime.split(";")[0]!.trim(),
      width,
      height,
      bytes: buf,
    };
    void chrome.runtime.sendMessage(msg).catch(() => undefined);
  } catch {
    sendUrl(url, width, height);
  }
}

function dispatch(url: string, width: number, height: number): void {
  if (already(url)) return;
  if (settings.skipSvg && looksSvg(url)) return;
  if (width > 0 && height > 0 && (width < settings.minWidth || height < settings.minHeight)) return;
  if (url.startsWith("blob:") || url.startsWith("data:") || sameOrigin(url)) {
    void sendBytes(url, width, height);
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
  dispatch(url, img.naturalWidth, img.naturalHeight);
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
  if (best) dispatch(best, bestW, bestW);
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
  if (observedRoots.has(root)) return;
  observedRoots.add(root);
  const mo = new MutationObserver((mutations) => {
    if (!settings.enabled) return;
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
  mo.observe(root, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["src", "srcset", "style", "class"],
  });
}

async function boot(): Promise<void> {
  settings = await chrome.storage.sync.get(DEFAULT_SETTINGS).then((s) => ({ ...DEFAULT_SETTINGS, ...s }));
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "sync") return;
    const was = settings.enabled;
    settings = { ...settings, ...Object.fromEntries(Object.entries(changes).map(([k, v]) => [k, v.newValue])) };
    if (!was && settings.enabled) walk(document);
  });
  observe(document.documentElement);
  document.addEventListener(
    "load",
    (ev) => {
      if (ev.target instanceof HTMLImageElement) considerImg(ev.target);
    },
    true,
  );
  if (settings.enabled) walk(document);
}

void boot();
