import { DEFAULT_SETTINGS, type Settings } from "./types";

const form = document.querySelector<HTMLFormElement>("#form")!;
const saved = document.querySelector<HTMLParagraphElement>("#saved")!;

const fields = {
  enabled: document.querySelector<HTMLInputElement>("#enabled")!,
  endpoint: document.querySelector<HTMLInputElement>("#endpoint")!,
  apiKey: document.querySelector<HTMLInputElement>("#apiKey")!,
  minWidth: document.querySelector<HTMLInputElement>("#minWidth")!,
  minHeight: document.querySelector<HTMLInputElement>("#minHeight")!,
  maxBytesMb: document.querySelector<HTMLInputElement>("#maxBytesMb")!,
  captureBackgrounds: document.querySelector<HTMLInputElement>("#captureBackgrounds")!,
  skipSvg: document.querySelector<HTMLInputElement>("#skipSvg")!,
};

async function load(): Promise<void> {
  const s = (await chrome.storage.sync.get(DEFAULT_SETTINGS)) as Settings;
  fields.enabled.checked = s.enabled;
  fields.endpoint.value = s.endpoint;
  fields.apiKey.value = s.apiKey;
  fields.minWidth.value = String(s.minWidth);
  fields.minHeight.value = String(s.minHeight);
  fields.maxBytesMb.value = String(Math.round(s.maxBytes / (1024 * 1024)));
  fields.captureBackgrounds.checked = s.captureBackgrounds;
  fields.skipSvg.checked = s.skipSvg;
}

form.addEventListener("submit", (ev) => {
  ev.preventDefault();
  const maxMb = Math.max(1, Number(fields.maxBytesMb.value) || 20);
  const next: Settings = {
    enabled: fields.enabled.checked,
    endpoint: fields.endpoint.value.trim() || DEFAULT_SETTINGS.endpoint,
    apiKey: fields.apiKey.value.trim(),
    minWidth: Math.max(1, Number(fields.minWidth.value) || 64),
    minHeight: Math.max(1, Number(fields.minHeight.value) || 64),
    maxBytes: maxMb * 1024 * 1024,
    captureBackgrounds: fields.captureBackgrounds.checked,
    skipSvg: fields.skipSvg.checked,
  };
  void chrome.storage.sync.set(next).then(() => {
    saved.textContent = "saved";
    setTimeout(() => {
      saved.textContent = "";
    }, 1500);
  });
});

void load();
