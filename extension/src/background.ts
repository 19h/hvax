import { bumpStats, loadSettings, loadStats, saveStats } from "./settings";
import { EMPTY_STATS, type ExtMessage, type StatusResponse } from "./types";

const inFlight = new Set<string>();
const posted = new Set<string>();
let chain: Promise<void> = Promise.resolve();
const MAX_PARALLEL = 2;
let active = 0;
const waiters: Array<() => void> = [];

function acquire(): Promise<void> {
  if (active < MAX_PARALLEL) {
    active += 1;
    return Promise.resolve();
  }
  return new Promise((resolve) => waiters.push(resolve));
}

function release(): void {
  const next = waiters.shift();
  if (next) next();
  else active = Math.max(0, active - 1);
}

async function updateBadge(): Promise<void> {
  const s = await loadStats();
  const n = s.stored + s.duplicates;
  await chrome.action.setBadgeBackgroundColor({ color: "#c45c26" });
  await chrome.action.setBadgeText({ text: n > 0 ? String(n) : "" });
}

async function postBody(body: ArrayBuffer, mime: string, sourceUrl: string): Promise<void> {
  const settings = await loadSettings();
  if (!settings.enabled) return;
  if (body.byteLength === 0 || body.byteLength > settings.maxBytes) return;

  await bumpStats({ seen: 1, posted: 1 });
  const headers: Record<string, string> = {
    "Content-Type": mime || "application/octet-stream",
  };
  if (settings.apiKey) headers["X-API-Key"] = settings.apiKey;

  let res: Response;
  try {
    res = await fetch(settings.endpoint, { method: "POST", headers, body });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    await bumpStats({ errors: 1, lastError: msg, lastStatus: "network" });
    await updateBadge();
    return;
  }

  if (res.status === 204) {
    await bumpStats({ ignored: 1, lastStatus: "204 no face", lastError: "" });
  } else if (res.status === 200) {
    let kind = "stored";
    try {
      const j = (await res.json()) as { duplicate?: boolean; duplicate_kind?: string; master_replaced?: boolean };
      if (j.duplicate) {
        kind = j.master_replaced ? "upgraded" : j.duplicate_kind ?? "duplicate";
        await bumpStats({ duplicates: 1, lastStatus: kind, lastError: "" });
      } else {
        await bumpStats({ stored: 1, lastStatus: "stored", lastError: "" });
      }
    } catch {
      await bumpStats({ stored: 1, lastStatus: "stored", lastError: "" });
    }
  } else if (res.status === 415 || res.status === 400) {
    await bumpStats({ errors: 1, lastStatus: String(res.status), lastError: `reject ${sourceUrl}` });
  } else {
    await bumpStats({ errors: 1, lastStatus: String(res.status), lastError: `HTTP ${res.status}` });
  }
  await updateBadge();
}

async function ingestUrl(url: string, width: number, height: number): Promise<void> {
  const settings = await loadSettings();
  if (width > 0 && height > 0 && (width < settings.minWidth || height < settings.minHeight)) return;
  if (inFlight.has(url) || posted.has(url)) return;
  inFlight.add(url);
  await acquire();
  try {
    const res = await fetch(url, { credentials: "omit", cache: "force-cache" });
    if (!res.ok) throw new Error(`fetch image ${res.status}`);
    const mime = (res.headers.get("content-type") || "application/octet-stream").split(";")[0]!.trim();
    const body = await res.arrayBuffer();
    posted.add(url);
    await postBody(body, mime, url);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    await bumpStats({ seen: 1, errors: 1, lastError: msg, lastStatus: "fetch" });
    await updateBadge();
  } finally {
    inFlight.delete(url);
    release();
  }
}

async function ingestBytes(sourceUrl: string, mime: string, bytes: ArrayBuffer): Promise<void> {
  if (inFlight.has(sourceUrl) || posted.has(sourceUrl)) return;
  inFlight.add(sourceUrl);
  await acquire();
  try {
    posted.add(sourceUrl);
    await postBody(bytes, mime, sourceUrl);
  } finally {
    inFlight.delete(sourceUrl);
    release();
  }
}

chrome.runtime.onMessage.addListener((raw: ExtMessage, _sender, sendResponse) => {
  if (raw.type === "get-status") {
    void Promise.all([loadSettings(), loadStats()]).then(([settings, stats]) => {
      const resp: StatusResponse = { settings, stats };
      sendResponse(resp);
    });
    return true;
  }
  if (raw.type === "reset-stats") {
    void saveStats({ ...EMPTY_STATS }).then(() => updateBadge()).then(() => sendResponse({ ok: true }));
    return true;
  }
  if (raw.type === "ingest-url") {
    chain = chain.then(() => ingestUrl(raw.url, raw.width, raw.height)).catch(() => undefined);
    return false;
  }
  if (raw.type === "ingest-bytes") {
    chain = chain.then(() => ingestBytes(raw.sourceUrl, raw.mime, raw.bytes)).catch(() => undefined);
    return false;
  }
  return false;
});

chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "sync" && changes.enabled) {
    void chrome.action.setBadgeText({ text: changes.enabled.newValue ? "" : "off" });
  }
});
