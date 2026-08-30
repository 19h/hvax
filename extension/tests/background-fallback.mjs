import assert from "node:assert/strict";

const imageUrl = "https://private.example/image.jpg";
let messageListener;
let frameRequest;
let postedBody;
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
  tabs: {
    query: async () => [],
    sendMessage: async (tabId, message, options) => {
      frameRequest = { tabId, message, options };
      return { ok: true, mime: "image/jpeg", dataBase64: "AQID" };
    },
  },
};

globalThis.fetch = async (input, init = {}) => {
  const url = String(input);
  if (url === "https://hv.ax/health") {
    return new Response(JSON.stringify({ jobs: 2 }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }
  if (url === imageUrl) {
    assert.equal(init.credentials, "include");
    return new Response("forbidden", { status: 403 });
  }
  if (url === "https://hv.ax/v1/ingest") {
    postedBody = init.body;
    return new Response(null, { status: 204 });
  }
  throw new Error(`unexpected fetch ${url}`);
};

await import(`../dist/background.js?fallback-test=${Date.now()}`);
assert.equal(typeof messageListener, "function");

messageListener(
  { type: "ingest-url", url: imageUrl, pageUrl: "https://photos.example/", width: 640, height: 480 },
  { tab: { id: 17 }, frameId: 3 },
  () => {},
);

for (let attempt = 0; attempt < 100 && !postedBody; attempt += 1) {
  await new Promise((resolve) => setTimeout(resolve, 5));
}

assert.deepEqual(frameRequest, {
  tabId: 17,
  message: { type: "capture-image", url: imageUrl },
  options: { frameId: 3 },
});
assert.ok(postedBody instanceof ArrayBuffer);
assert.deepEqual([...new Uint8Array(postedBody)], [1, 2, 3]);
assert.equal(sessionStats.ignored, 1);
