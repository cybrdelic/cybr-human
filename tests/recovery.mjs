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
  assert.equal(
    await p.evaluate(() => window.__fullHeadFEM.engine),
    "WebGPU implicit FEM",
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
    window.__fullHeadFEM.solver.device.destroy();
    window.__fullHeadFEM.resume();
  });
  await p.waitForFunction(
    () => window.__fullHeadFEM.recovery || window.__fullHeadFEM.errors.length,
    {},
    { timeout: 60000 },
  );
  await p.evaluate(() => window.__fullHeadFEM.pause());
  const restored = await p.evaluate(() => ({
    positions: Array.from(window.__fullHeadFEM.positions),
    recovery: window.__fullHeadFEM.recovery,
    engine: window.__fullHeadFEM.engine,
    errors: window.__fullHeadFEM.errors,
  }));
  assert(restored.recovery?.positionsPreserved);
  assert.equal(restored.engine, "CPU worker implicit FEM");
  const error = Math.max(
    ...restored.positions.map((v, i) => Math.abs(v - before[i])),
  );
  assert(error < 1e-5, "Recovery reset the solved face");
  assert.deepEqual(restored.errors, []);
  // Verify the recovered worker can solve a new control request.
  await p.locator("#fat").fill("60");
  await p.waitForFunction(
    () => window.__fullHeadFEM.settlement || window.__fullHeadFEM.errors.length,
    {},
    { timeout: 120000 },
  );
  const state = await p.evaluate(() => ({
    recovery: window.__fullHeadFEM.recovery,
    last: window.__fullHeadFEM.samples.at(-1),
    settlement: window.__fullHeadFEM.settlement,
    runtime: window.__tissueRuntime,
    errors: window.__fullHeadFEM.errors,
  }));
  assert.equal(state.last.applied.fatPercent, 60);
  assert(state.last.minJ > 0.2);
  assert.deepEqual(errors, []);
  assert.deepEqual(state.errors, []);
  fs.writeFileSync(
    "output/verification/recovery-check.json",
    JSON.stringify({ maximumCheckpointErrorM: error, ...state }, null, 2),
  );
  await p.screenshot({ path: "output/verification/recovery.png" });
  console.log(JSON.stringify({ maximumCheckpointErrorM: error, ...state }));
} finally {
  await browser?.close();
}
