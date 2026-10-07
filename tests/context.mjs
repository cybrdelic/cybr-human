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
  await p.goto(appURL);
  await p.waitForFunction(
    () => window.__fullHeadFEM?.ready,
    {},
    { timeout: 45000 },
  );
  await p.locator("#fat").fill("50");
  await p.waitForFunction(
    () => window.__fullHeadFEM.settlement?.converged,
    {},
    { timeout: 45000 },
  );
  const before = await p.evaluate(() =>
    Array.from(window.__fullHeadFEM.positions),
  );
  await p.evaluate(() => {
    window.lossTest = document
      .querySelector("canvas")
      .getContext("webgl2")
      .getExtension("WEBGL_lose_context");
    window.lossTest.loseContext();
  });
  await p.waitForFunction(() => window.__fullHeadFEM.graphicsSuspended);
  await p.evaluate(() => window.lossTest.restoreContext());
  await p.waitForFunction(
    () => window.__fullHeadFEM.contextRestorations === 1,
    {},
    { timeout: 30000 },
  );
  await p.screenshot({ path: "output/verification/context-restored.png" });
  const state = await p.evaluate(() => ({
    positions: Array.from(window.__fullHeadFEM.positions),
    errors: window.__fullHeadFEM.errors,
    suspended: window.__fullHeadFEM.graphicsSuspended,
    restorations: window.__fullHeadFEM.contextRestorations,
  }));
  const error = Math.max(
    ...state.positions.map((v, i) => Math.abs(v - before[i])),
  );
  assert.equal(error, 0);
  assert.deepEqual(errors, []);
  assert.deepEqual(state.errors, []);
  await p.locator("#sag").fill("50");
  await p.waitForFunction(
    () => window.__fullHeadFEM.settlement?.converged,
    {},
    { timeout: 45000 },
  );
  const last = await p.evaluate(() => window.__fullHeadFEM.samples.at(-1));
  assert(last.minJ > 0.2 && last.staticResidualN < 0.001);
  fs.writeFileSync(
    "output/verification/context-check.json",
    JSON.stringify(
      {
        maximumCheckpointErrorM: error,
        last,
        restorations: state.restorations,
        errors,
      },
      null,
      2,
    ),
  );
  console.log(
    JSON.stringify({
      maximumCheckpointErrorM: error,
      last,
      restorations: state.restorations,
      errors,
    }),
  );
} finally {
  await browser?.close();
}
