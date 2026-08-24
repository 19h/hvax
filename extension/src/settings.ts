import { DEFAULT_SETTINGS, EMPTY_STATS, type Settings, type Stats } from "./types";

export async function loadSettings(): Promise<Settings> {
  const stored = await chrome.storage.sync.get(DEFAULT_SETTINGS);
  return { ...DEFAULT_SETTINGS, ...stored };
}

export async function saveSettings(partial: Partial<Settings>): Promise<void> {
  await chrome.storage.sync.set(partial);
}

export async function loadStats(): Promise<Stats> {
  const { stats } = await chrome.storage.session.get("stats");
  return { ...EMPTY_STATS, ...(stats as Partial<Stats> | undefined) };
}

export async function saveStats(stats: Stats): Promise<void> {
  await chrome.storage.session.set({ stats });
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
