#!/usr/bin/env node
import { cpSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as esbuild from "esbuild";

const root = dirname(fileURLToPath(import.meta.url));
const dist = join(root, "dist");
const watch = process.argv.includes("--watch");

mkdirSync(dist, { recursive: true });
mkdirSync(join(dist, "icons"), { recursive: true });

const common = {
  bundle: true,
  sourcemap: true,
  target: "chrome114",
  logLevel: "info",
};

const content = {
  ...common,
  entryPoints: [join(root, "src/content.ts")],
  outfile: join(dist, "content.js"),
  format: "iife",
};

const background = {
  ...common,
  entryPoints: [join(root, "src/background.ts")],
  outfile: join(dist, "background.js"),
  format: "esm",
};

const popup = {
  ...common,
  entryPoints: [join(root, "src/popup.ts")],
  outfile: join(dist, "popup.js"),
  format: "iife",
};

const options = {
  ...common,
  entryPoints: [join(root, "src/options.ts")],
  outfile: join(dist, "options.js"),
  format: "iife",
};

function copyStatic() {
  for (const f of ["manifest.json"]) {
    cpSync(join(root, f), join(dist, f));
  }
  for (const f of ["popup.html", "popup.css", "options.html", "options.css"]) {
    cpSync(join(root, "public", f), join(dist, f));
  }
  cpSync(join(root, "public/icons"), join(dist, "icons"), { recursive: true });
  const pkg = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
  const manifest = JSON.parse(readFileSync(join(dist, "manifest.json"), "utf8"));
  manifest.version = pkg.version;
  writeFileSync(join(dist, "manifest.json"), JSON.stringify(manifest, null, 2));
}

if (watch) {
  copyStatic();
  const ctxs = await Promise.all(
    [content, background, popup, options].map((opts) => esbuild.context(opts)),
  );
  await Promise.all(ctxs.map((c) => c.watch()));
  console.log("watching");
} else {
  await Promise.all([content, background, popup, options].map((opts) => esbuild.build(opts)));
  copyStatic();
}
