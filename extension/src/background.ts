import { resolveApiUrl } from "./api";
import { bumpStats, loadSettings, loadStats, saveStats } from "./settings";
import { EMPTY_STATS, type ExtMessage, type ServerStats, type Settings, type StatusResponse } from "./types";

const inFlight = new Set<string>();
const posted = new Set<string>();
let chain: Promise<void> = Promise.resolve();
let pending = 0;
const MAX_PARALLEL = 2;
let active = 0;
const waiters: Array<() => void> = [];

function enqueue(task: () => Promise<void>): void {
  pending += 1;
  chain = chain
    .then(task)
    .catch(() => undefined)
    .finally(() => {
      pending = Math.max(0, pending - 1);
    });
}

async function injectIntoOpenTabs(): Promise<void> {
  const tabs = await chrome.tabs.query({});
  await Promise.allSettled(
    tabs.map(async (tab) => {
      if (tab.id === undefined) return;
      await chrome.scripting.executeScript({
        target: { tabId: tab.id, allFrames: true },
        files: ["content.js"],
      });
    }),
  );
}

async function fetchServerStats(settings: Settings): Promise<ServerStats> {
  const headers: Record<string, string> = { Accept: "application/json" };
  if (settings.apiKey) headers["X-API-Key"] = settings.apiKey;
  const url = new URL(resolveApiUrl(settings.endpoint, "/v1/stats"));
  url.searchParams.set("_hvax_ts", Date.now().toString());

  const res = await fetch(url.href, {
    headers,
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);

  const stats = (await res.json()) as Partial<ServerStats>;
  if (
    typeof stats.faces !== "number" ||
    typeof stats.images !== "number" ||
    typeof stats.embedding_rows !== "number" ||
    typeof stats.hnsw !== "boolean"
  ) {
    throw new Error("invalid stats response");
  }
  return stats as ServerStats;
}

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
    const ingestEndpoint = settings.processorEndpoint || settings.endpoint;
    res = await fetch(resolveApiUrl(ingestEndpoint, "/v1/ingest"), { method: "POST", headers, body });
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
    await bumpStats({
      errors: 1,
      lastStatus: String(res.status),
      lastError: `reject ${mime || "unknown"} ${body.byteLength}B ${sourceUrl}`,
    });
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

function decodeBase64(data: string): ArrayBuffer {
  const binary = atob(data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

async function ingestBytes(sourceUrl: string, mime: string, dataBase64: string): Promise<void> {
  if (inFlight.has(sourceUrl) || posted.has(sourceUrl)) return;
  inFlight.add(sourceUrl);
  await acquire();
  try {
    const bytes = decodeBase64(dataBase64);
    posted.add(sourceUrl);
    await postBody(bytes, mime, sourceUrl);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    await bumpStats({ seen: 1, errors: 1, lastError: msg, lastStatus: "decode" });
    await updateBadge();
  } finally {
    inFlight.delete(sourceUrl);
    release();
  }
}

chrome.runtime.onMessage.addListener((raw: ExtMessage, _sender, sendResponse) => {
  if (raw.type === "get-status") {
    void Promise.all([loadSettings(), loadStats()]).then(async ([settings, stats]) => {
      let serverStats: ServerStats | null = null;
      let serverError = "";
      try {
        serverStats = await fetchServerStats(settings);
      } catch (err) {
        serverError = err instanceof Error ? err.message : String(err);
      }
      const resp: StatusResponse = { settings, stats, pending, serverStats, serverError };
      sendResponse(resp);
    });
    return true;
  }
  if (raw.type === "reset-stats") {
    void saveStats({ ...EMPTY_STATS }).then(() => updateBadge()).then(() => sendResponse({ ok: true }));
    return true;
  }
  if (raw.type === "ingest-url") {
    enqueue(() => ingestUrl(raw.url, raw.width, raw.height));
    return false;
  }
  if (raw.type === "ingest-bytes") {
    enqueue(() => ingestBytes(raw.sourceUrl, raw.mime, raw.dataBase64));
    return false;
  }
  return false;
});

chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "sync" && changes.enabled) {
    void chrome.action.setBadgeText({ text: changes.enabled.newValue ? "" : "off" });
  }
});

chrome.runtime.onInstalled.addListener(() => {
  // Declarative content scripts only run on future navigations. Reinject the
  // freshly installed version into existing tabs so an extension reload does
  // not leave them permanently attached to an invalidated runtime context.
  void injectIntoOpenTabs().catch(() => undefined);
});
