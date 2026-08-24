import { DEFAULT_SETTINGS, type StatusResponse } from "./types";

const enabled = document.querySelector<HTMLInputElement>("#enabled")!;
const endpoint = document.querySelector<HTMLInputElement>("#endpoint")!;
const statusEl = document.querySelector<HTMLParagraphElement>("#status")!;
const statsEl = document.querySelector<HTMLDListElement>("#stats")!;
const saveBtn = document.querySelector<HTMLButtonElement>("#save")!;
const resetBtn = document.querySelector<HTMLButtonElement>("#reset")!;
const optionsBtn = document.querySelector<HTMLButtonElement>("#options")!;

function render(s: StatusResponse): void {
  enabled.checked = s.settings.enabled;
  endpoint.value = s.settings.endpoint;
  statusEl.textContent = s.stats.lastError
    ? `error: ${s.stats.lastError}`
    : s.stats.lastStatus
      ? s.stats.lastStatus
      : "idle";
  statusEl.classList.toggle("err", Boolean(s.stats.lastError));
  statsEl.innerHTML = "";
  const rows: Array<[string, number]> = [
    ["seen", s.stats.seen],
    ["posted", s.stats.posted],
    ["stored", s.stats.stored],
    ["dup", s.stats.duplicates],
    ["no face", s.stats.ignored],
    ["errors", s.stats.errors],
  ];
  for (const [k, v] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    dd.textContent = String(v);
    statsEl.append(dt, dd);
  }
}

async function refresh(): Promise<void> {
  const s = (await chrome.runtime.sendMessage({ type: "get-status" })) as StatusResponse;
  render(s);
}

enabled.addEventListener("change", () => {
  void chrome.storage.sync.set({ enabled: enabled.checked });
});

saveBtn.addEventListener("click", () => {
  const url = endpoint.value.trim() || DEFAULT_SETTINGS.endpoint;
  void chrome.storage.sync.set({ endpoint: url }).then(() => refresh());
});

resetBtn.addEventListener("click", () => {
  void chrome.runtime.sendMessage({ type: "reset-stats" }).then(() => refresh());
});

optionsBtn.addEventListener("click", () => {
  void chrome.runtime.openOptionsPage();
});

void refresh();
setInterval(() => void refresh(), 1500);
