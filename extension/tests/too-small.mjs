import assert from "node:assert/strict";

// The gallery refuses images below its minimum side with a 422. That counts as
// ignored, not as an error, and teaches the extension the limit so smaller
// images are no longer fetched or uploaded.
const origin = "https://cdn.example";
let messageListener;
let sessionStats;
const fetched = [];
const uploads = [];

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

const json = (body, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

globalThis.fetch = async (input, init) => {
  const url = String(input);
  if (url === "https://hv.ax/health") return json({ jobs: 2 });
  if (url.startsWith("https://hv.ax/v1/stats")) {
    return json({ faces: 1, images: 1, embedding_rows: 1, hnsw: false });
  }
  if (url.startsWith(origin)) {
    fetched.push(url);
    const size = Number(new URL(url).searchParams.get("bytes"));
    return new Response(new Uint8Array(size), { status: 200, headers: { "Content-Type": "image/jpeg" } });
  }
  if (url === "https://hv.ax/v1/ingest") {
    uploads.push(init.body.byteLength);
    // One byte stands for a small image, anything longer for a large one.
    if (init.body.byteLength === 1) {
      return json(
        { error: "image is smaller than 350x350", rejected: "too_small", width: 300, height: 200, min_side: 350 },
        422,
      );
    }
    return json({ image_id: 1, faces: [] });
  }
  throw new Error(`unexpected fetch ${url}`);
};

const dist = process.argv[2] ?? "dist";
await import(`../${dist}/background.js?too-small-test=${Date.now()}`);

async function send(message, done) {
  messageListener(message, { tab: { id: 1 }, frameId: 0 }, () => {});
  for (let attempt = 0; attempt < 100 && !done(); attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}
const urlMessage = (name, bytes, width, height) => ({
  type: "ingest-url",
  url: `${origin}/${name}.jpg?bytes=${bytes}`,
  pageUrl: "https://photos.example/",
  width,
  height,
});

// The first small image is uploaded and refused.
await send(urlMessage("first", 1, 300, 200), () => sessionStats?.ignored === 1);
assert.equal(sessionStats.ignored, 1);
assert.equal(sessionStats.errors, 0);
assert.equal(sessionStats.stored, 0);
assert.equal(sessionStats.lastError, "");
assert.equal(uploads.length, 1);

const { stats } = await new Promise((resolve) => messageListener({ type: "get-status" }, {}, resolve));
assert.equal(stats.lastStatus, "too small 300x200 (min 350)");

// A later one known to be below the learned limit is not even fetched ...
await send(urlMessage("second", 1, 349, 2000), () => false);
await send(
  {
    type: "ingest-bytes",
    sourceUrl: `${origin}/third.jpg`,
    pageUrl: "https://photos.example/",
    mime: "image/jpeg",
    width: 2000,
    height: 100,
    dataBase64: "AA==",
  },
  () => false,
);
assert.deepEqual(fetched, [`${origin}/first.jpg?bytes=1`]);
assert.equal(uploads.length, 1);
assert.equal(sessionStats.ignored, 1);

// ... while images at or above it, and ones of unknown size, still go through.
await send(urlMessage("large", 8, 350, 350), () => sessionStats?.stored === 1);
await send(urlMessage("unknown", 9, 0, 0), () => sessionStats?.stored === 2);
assert.equal(sessionStats.stored, 2);
assert.deepEqual(uploads, [1, 8, 9]);

console.log("too-small: ok");
