import { appURL } from "./runtime.mjs";
import fs from "node:fs";
import assert from "node:assert/strict";
import { chromium } from "../node_modules/playwright/index.mjs";
let browser;
try {
  browser = await chromium.launch({
    channel: "msedge",
    headless: true,
    args: ["--enable-unsafe-webgpu"],
  });
  const p = await browser.newPage({ viewport: { width: 1280, height: 900 } }),
    errors = [];
  p.on("pageerror", (e) => errors.push(e.message));
  p.on("console", (m) => {
    if (m.type() === "error" && !m.text().includes("404"))
      errors.push(m.text().slice(0, 1000));
  });
  await p.goto(appURL);
  try {
    await p.waitForFunction(
      () => window.__fullHeadFEM?.ready || window.__fullHeadFEM?.errors.length,
    );
  } catch (e) {
    console.log(JSON.stringify({ errors }));
    throw e;
  }
  assert(!errors.length);
  await p.evaluate(() => {
    const d = window.__fullHeadFEM;
    d.grabPoint([0, 0.088, 0.055], [0, -0.004, 0]);
    d.resume();
  });
  await p.waitForFunction(
    () =>
      window.__fullHeadFEM.samples.length >= 20 ||
      window.__fullHeadFEM.errors.length,
    {},
    { timeout: 30000 },
  );
  await p.evaluate(() => window.__fullHeadFEM.pause());
  await p.waitForFunction(
    () =>
      window.__fullHeadFEM.fineSkinPatch?.ready ||
      window.__fullHeadFEM.fineSkinPatch?.errors.length,
    {},
    { timeout: 30000 },
  );
  await p.waitForTimeout(1000);
  const result = await p.evaluate(() => ({
    fine: window.__fullHeadFEM.fineSkinPatch,
    last: window.__fullHeadFEM.samples.at(-1),
    errors: window.__fullHeadFEM.errors,
  }));
  fs.writeFileSync(
    "output/verification/fine-skin-browser-check.json",
    JSON.stringify({ result, errors }, null, 2),
  );
  await p.screenshot({
    path: "output/verification/fine-skin-compressed.png",
  });
  await p.locator("#foreheadDetail").click();
  await p.waitForTimeout(100);
  await p.screenshot({ path: "output/verification/fine-skin-detail.png" });
  await p.locator("#fineHeatmap").check();
  await p.waitForTimeout(100);
  await p.screenshot({
    path: "output/verification/fine-skin-height-map.png",
  });
  console.log(JSON.stringify({ result, errors }));
  assert(!errors.length && !result.errors.length && !result.fine.errors.length);
  assert(result.last.minJ > 0.2);
  assert(
    result.fine.report.maxHeightM > 0.00005,
    "Compression did not produce wrinkles",
  );
} finally {
  await browser?.close();
}
