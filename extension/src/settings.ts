import { DEFAULT_SETTINGS, EMPTY_STATS, type Settings, type Stats } from "./types";

export async function loadSettings(): Promise<Settings> {
  const stored = await chrome.storage.sync.get(DEFAULT_SETTINGS);
  return { ...DEFAULT_SETTINGS, ...stored };
}

export async function saveSettings(partial: Partial<Settings>): Promise<void> {
  await chrome.storage.sync.set(partial);
}

// Safari sizes stored data in characters but the value being replaced in
// bytes. Once a stored string holds a non-ASCII character, a shorter write
// underflows that sum and is rejected with "Exceeded storage quota". Stored
// text is therefore kept ASCII, with everything else (and "\") as \uXXXX.
function toAscii(text: string): string {
  return text.replace(/[\\\u0080-\uffff]/g, (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, "0")}`);
}

function fromAscii(text: string): string {
  return text.replace(/\\u([0-9a-f]{4})/g, (_, hex: string) => String.fromCharCode(parseInt(hex, 16)));
}

export async function loadStats(): Promise<Stats> {
  const { stats } = await chrome.storage.session.get("stats");
  const loaded: Stats = { ...EMPTY_STATS, ...(stats as Partial<Stats> | undefined) };
  return { ...loaded, lastError: fromAscii(loaded.lastError), lastStatus: fromAscii(loaded.lastStatus) };
}

export async function saveStats(stats: Stats): Promise<void> {
  const stored: Stats = { ...stats, lastError: toAscii(stats.lastError), lastStatus: toAscii(stats.lastStatus) };
  await chrome.storage.session.set({ stats: stored });
}

export async function bumpStats(patch: Partial<Stats>): Promise<Stats> {
  const cur = await loadStats();
  const next: Stats = { ...cur };
  for (const [k, v] of Object.entries(patch)) {
    const key = k as keyof Stats;
    if (typeof next[key] === "number" && typeof v === "number") {
      (next[key] as number) += v;
    } else if (typeof v === "string") {
      (next[key] as string) = v;
    }
  }
  await saveStats(next);
  return next;
}
