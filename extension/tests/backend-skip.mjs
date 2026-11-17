import assert from "node:assert/strict";

const backendImage = "https://hv.ax/v1/images/42/content";
const backendSubdomainImage = "https://static.hv.ax/crop.jpg";
const foreignImage = "https://cdn.example/photo.jpg";
let messageListener;
let sessionStats;
const fetched = [];
let posts = 0;

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
    sendMessage: async () => ({ ok: false, error: "unexpected page fallback" }),
  },
};

globalThis.fetch = async (input, init = {}) => {
  const url = String(input);
  fetched.push(url);
  if (url === "https://hv.ax/health") {
    return new Response(JSON.stringify({ jobs: 2 }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }
  if (url === foreignImage) {
    return new Response(new Uint8Array([1, 2, 3]), {
      status: 200,
      headers: { "Content-Type": "image/jpeg" },
    });
  }
  if (url === "https://hv.ax/v1/ingest") {
    assert.ok(init.body instanceof ArrayBuffer);
    posts += 1;
    return new Response(null, { status: 204 });
  }
  throw new Error(`unexpected fetch ${url}`);
};

await import(`../dist/background.js?backend-skip-test=${Date.now()}`);
assert.equal(typeof messageListener, "function");

for (const url of [backendImage, backendSubdomainImage, foreignImage]) {
  messageListener(
    { type: "ingest-url", url, pageUrl: "https://hv.ax/", width: 640, height: 480 },
    { tab: { id: 17 }, frameId: 0 },
    () => {},
  );
}

for (let attempt = 0; attempt < 100 && postedCount() < 1; attempt += 1) {
  await new Promise((resolve) => setTimeout(resolve, 5));
}

function postedCount() {
  return sessionStats?.posted ?? 0;
}

assert.equal(sessionStats?.seen, 1);
assert.equal(sessionStats?.posted, 1);
assert.equal(sessionStats?.ignored, 1);
assert.equal(posts, 1);
assert.ok(!fetched.includes(backendImage), "backend image must not be fetched");
assert.ok(!fetched.includes(backendSubdomainImage), "backend subdomain image must not be fetched");
assert.ok(fetched.includes(foreignImage), "foreign image must still be ingested");
