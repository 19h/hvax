import assert from "node:assert/strict";

// Safari rejects storage.session writes as over quota once a stored string
// holds a non-ASCII character, so the stored stats must stay ASCII while the
// status text shown in the popup keeps its original characters.
const imageUrl = "https://cdn.example/photo.jpg";
const expectedStatus = "stored · 1 face · 1 known (#7)";
let messageListener;
let sessionStats;

globalThis.chrome = {
  action: {
    setBadgeBackgroundColor: async () => {},
    setBadgeText: async () => {},
  },
  runtime: {
    onInstalled: { addListener: () => {} },
    onMessage: { addListener: (listener) => { messageListener = listener; } },
  },
  scripting: { executeScript: async () => {} },
  storage: {
    onChanged: { addListener: () => {} },
    sync: { get: async (defaults) => ({ ...defaults, endpoint: "https://hv.ax" }) },
    session: {
      get: async () => ({ stats: sessionStats }),
      set: async ({ stats }) => { sessionStats = stats; },
    },
  },
  tabs: { query: async () => [], sendMessage: async () => ({ ok: false }) },
};

const json = (body) =>
  new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });

globalThis.fetch = async (input) => {
  const url = String(input);
  if (url === "https://hv.ax/health") return json({ jobs: 2 });
  if (url.startsWith("https://hv.ax/v1/stats")) {
    return json({ faces: 1, images: 1, embedding_rows: 1, hnsw: false });
  }
  if (url === imageUrl) {
    return new Response(new Uint8Array([1, 2, 3]), { status: 200, headers: { "Content-Type": "image/jpeg" } });
  }
  if (url === "https://hv.ax/v1/ingest") return json({ image_id: 1, faces: [{ face_id: 1, identity_id: 7 }] });
  throw new Error(`unexpected fetch ${url}`);
};

const dist = process.argv[2] ?? "dist";
await import(`../${dist}/background.js?stats-ascii-test=${Date.now()}`);

messageListener(
  { type: "ingest-url", url: imageUrl, pageUrl: "https://photos.example/", width: 640, height: 480 },
  { tab: { id: 1 }, frameId: 0 },
  () => {},
);

for (let attempt = 0; attempt < 100 && sessionStats?.stored !== 1; attempt += 1) {
  await new Promise((resolve) => setTimeout(resolve, 5));
}

assert.equal(sessionStats?.stored, 1);
assert.match(JSON.stringify(sessionStats), /^[\x20-\x7e]*$/);

const status = await new Promise((resolve) => messageListener({ type: "get-status" }, {}, resolve));
assert.equal(status.stats.lastStatus, expectedStatus);
