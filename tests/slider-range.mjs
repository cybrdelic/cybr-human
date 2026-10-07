import { appURL } from "./runtime.mjs";
import fs from "node:fs";
import assert from "node:assert/strict";
import { chromium } from "../node_modules/playwright/index.mjs";
let b;
try {
  b = await chromium.launch({
    channel: "msedge",
    headless: true,
    args: ["--enable-unsafe-webgpu"],
  });
  const p = await b.newPage({ viewport: { width: 1280, height: 900 } }),
    errors = [];
  p.on("pageerror", (e) => errors.push(e.message));
  await p.goto(appURL);
  await p.waitForFunction(
    () => window.__fullHeadFEM?.ready,
    {},
    { timeout: 45000 },
  );
  await p.screenshot({
    path: "output/verification/slider-range-before.png",
  });
  const ids = await p.evaluate(() =>
      Array.from({ length: 188 }, (_, i) =>
        Math.floor(
          (i * (window.__fullHeadFEM.coverage.mainSkinVertices - 1)) / 187,
        ),
      ),
    ),
    baseline = await p.evaluate(
      (ids) => window.__fullHeadFEM.sampleSurface(ids),
      ids,
    );
  await p.locator("#fat").fill("200");
  await p.waitForFunction(
    () =>
      window.__fullHeadFEM.errors.length ||
      (window.__fullHeadFEM.settlement?.converged &&
        window.__fullHeadFEM.samples.at(-1)?.applied.fatPercent === 200),
    {},
    { timeout: 45000 },
  );
  const data = await p.evaluate((ids) => {
    const d = window.__fullHeadFEM;
    return {
      engine: d.engine,
      last: d.samples.at(-1),
      settlement: d.settlement,
      surface: d.sampleSurface(ids),
      errors: d.errors,
    };
  }, ids);
  data.maximumVisibleDeltaMm = Math.max(
    ...data.surface.map(
      (s, i) =>
        1000 *
        Math.hypot(...s.position.map((v, k) => v - baseline[i].position[k])),
    ),
  );
  assert(!errors.length && !data.errors.length);
  assert(data.last.minJ > 0.2);
  assert(data.settlement.converged && data.last.staticResidualN < 0.001);
  assert(
    data.maximumVisibleDeltaMm > 4,
    "200% volume growth failed to produce a meaningful shape change",
  );
  await p.screenshot({ path: "output/verification/slider-range-200.png" });
  fs.writeFileSync(
    "output/verification/slider-range-check.json",
    JSON.stringify({ data, errors }, null, 2),
  );
  console.log(
    JSON.stringify({
      engine: data.engine,
      last: data.last,
      settlement: data.settlement,
      maximumVisibleDeltaMm: data.maximumVisibleDeltaMm,
      errors,
    }),
  );
} finally {
  await b?.close();
}
