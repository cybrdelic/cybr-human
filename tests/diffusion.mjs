import assert from "node:assert/strict";
import fs from "node:fs";
import { spawnSync } from "node:child_process";
import { chromium } from "playwright";
import { appURL } from "./runtime.mjs";
const browser = await chromium.launch({
  channel: "msedge",
  headless: true,
  args: ["--enable-unsafe-webgpu"],
});
try {
  const page = await browser.newPage({
      viewport: { width: 1280, height: 900 },
    }),
    errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error" && !m.text().includes("404"))
      errors.push(m.text().slice(0, 1500));
  });
  await page.goto(appURL);
  await page.waitForFunction(
    () =>
      window.__fullHeadFEM?.ready && window.__fullHeadFEM.settlement?.converged,
    {},
    { timeout: 45000 },
  );
  await page.waitForTimeout(500);
  const prefix = "output/verification/";
  for (const [name, radius] of [
    ["transport-zero", 0],
    ["transport-on", 1.2],
    ["transport-return", 0],
  ]) {
    const count = await page.evaluate((radius) => {
      const d = window.__fullHeadFEM;
      d.skinDiffusion.radiusMm = radius;
      d.setView("front");
      return d.skinDiffusion.renders;
    }, radius);
    await page.waitForFunction(
      (n) => window.__fullHeadFEM.skinDiffusion.renders > n,
      count,
    );
    await page.locator("canvas").screenshot({ path: prefix + name + ".png" });
  }
  const diagnostic = await page.evaluate(() => ({
    sss: window.__fullHeadFEM.skinDiffusion,
    errors: window.__fullHeadFEM.errors,
    build: window.__tissueRuntime.build,
    eyeSamples: window.__fullHeadFEM.eyeSamplePixels(),
  }));
  fs.writeFileSync(
    prefix + "diffusion.json",
    JSON.stringify({ diagnostic, errors }, null, 2),
  );
  assert(!errors.length && !diagnostic.errors.length);
  assert(diagnostic.sss.supported);
  const pixels = spawnSync("python", ["tests/render_pixels.py"], {
    encoding: "utf8",
  });
  if (pixels.status !== 0) throw Error(pixels.stderr || pixels.stdout);
  console.log(pixels.stdout.trim());
} finally {
  await browser.close();
}
