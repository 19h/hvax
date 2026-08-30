import { DEFAULT_SETTINGS, type StatusResponse } from "./types";

const enabled = document.querySelector<HTMLInputElement>("#enabled")!;
const endpoint = document.querySelector<HTMLInputElement>("#endpoint")!;
const processorEndpoint = document.querySelector<HTMLInputElement>("#processorEndpoint")!;
const statusEl = document.querySelector<HTMLParagraphElement>("#status")!;
const serverStatsEl = document.querySelector<HTMLDListElement>("#server-stats")!;
const statsEl = document.querySelector<HTMLDListElement>("#stats")!;
const saveBtn = document.querySelector<HTMLButtonElement>("#save")!;
const resetBtn = document.querySelector<HTMLButtonElement>("#reset")!;
const optionsBtn = document.querySelector<HTMLButtonElement>("#options")!;
let endpointDirty = false;
let processorEndpointDirty = false;

function renderRows(el: HTMLDListElement, rows: Array<[string, number | string]>): void {
  el.innerHTML = "";
  for (const [key, value] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    dd.textContent = String(value);
    el.append(dt, dd);
  }
}

function render(s: StatusResponse): void {
  enabled.checked = s.settings.enabled;
  if (!endpointDirty) endpoint.value = s.settings.endpoint;
  if (!processorEndpointDirty) processorEndpoint.value = s.settings.processorEndpoint;
  statusEl.textContent = s.serverError
    ? `server: ${s.serverError}`
    : s.stats.lastError
    ? `error: ${s.stats.lastError}`
    : s.stats.lastStatus
      ? s.stats.lastStatus
      : "idle";
  statusEl.classList.toggle("err", Boolean(s.serverError || s.stats.lastError));

  const serverRows: Array<[string, number | string]> = s.serverStats
    ? [
        ["images", s.serverStats.images],
        ["faces", s.serverStats.faces],
        ["embedding rows", s.serverStats.embedding_rows],
        ["search", s.serverStats.hnsw ? "HNSW" : "exact"],
      ]
    : [["status", "unavailable"]];
  renderRows(serverStatsEl, serverRows);

  renderRows(statsEl, [
    ["pending", s.pending],
    ["jobs", s.jobs],
    ["seen", s.stats.seen],
    ["posted", s.stats.posted],
    ["stored", s.stats.stored],
    ["dup", s.stats.duplicates],
    ["no face", s.stats.ignored],
    ["errors", s.stats.errors],
  ]);
}

async function refresh(): Promise<void> {
  const s = (await chrome.runtime.sendMessage({ type: "get-status" })) as StatusResponse;
  render(s);
}

enabled.addEventListener("change", () => {
  void chrome.storage.sync.set({ enabled: enabled.checked });
});

endpoint.addEventListener("input", () => {
  endpointDirty = true;
});

processorEndpoint.addEventListener("input", () => {
  processorEndpointDirty = true;
});

saveBtn.addEventListener("click", () => {
  const url = endpoint.value.trim() || DEFAULT_SETTINGS.endpoint;
  const processorUrl = processorEndpoint.value.trim();
  void chrome.storage.sync.set({ endpoint: url, processorEndpoint: processorUrl }).then(() => {
    endpointDirty = false;
    processorEndpointDirty = false;
    return refresh();
  });
});

resetBtn.addEventListener("click", () => {
  void chrome.runtime.sendMessage({ type: "reset-stats" }).then(() => refresh());
});

optionsBtn.addEventListener("click", () => {
  void chrome.runtime.openOptionsPage();
});

void refresh();
setInterval(() => void refresh(), 1500);
