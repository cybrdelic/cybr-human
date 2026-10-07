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
    errors = [],
    cases = [];
  p.on("pageerror", (e) => errors.push(e.message));
  p.on("console", (m) => {
    if (m.type() === "error" && !m.text().includes("404"))
      errors.push(m.text());
  });
  await p.goto(appURL);
  try {
    await p.waitForFunction(
      () =>
        (window.__fullHeadFEM?.ready &&
          document.querySelector("#fat").disabled === false) ||
        window.__fullHeadFEM?.errors.length,
      {},
      { timeout: 30000 },
    );
  } catch (e) {
    console.log(
      JSON.stringify({
        pageErrors: errors,
        state: await p.evaluate(() => ({
          ready: window.__fullHeadFEM?.ready,
          errors: window.__fullHeadFEM?.errors,
          message: document.querySelector("#message")?.textContent,
          status: document.querySelector("#tissueStatus")?.textContent,
        })),
      }),
    );
    throw e;
  }
  await p.screenshot({ path: "output/verification/before.png" });
  for (const [id, value] of [
    ["fat", "50"],
    ["sag", "50"],
  ]) {
    await p.locator("#" + id).fill(value);
    await p.waitForFunction(
      ({ id, value }) => {
        const d = window.__fullHeadFEM;
        return (
          d.errors.length ||
          (d.samples.length > 0 &&
            Math.abs(
              d.samples.at(-1).applied[
                id === "fat" ? "fatPercent" : "sagPercent"
              ] - value,
            ) < 0.1)
        );
      },
      { id, value: Number(value) },
      { timeout: 30000 },
    );
    await p.waitForFunction(
      () =>
        window.__fullHeadFEM.errors.length ||
        window.__fullHeadFEM.settlement?.converged,
      {},
      { timeout: 45000 },
    );
    const data = await p.evaluate(() => {
      const d = window.__fullHeadFEM;
      return {
        errors: d.errors,
        samples: d.samples.slice(-12),
        coverage: d.coverage,
        adapter: d.adapter,
        frames: d.solver.frames,
        surface: d.sampleSurface(
          Array.from({ length: 188 }, (_, i) =>
            Math.floor((i * (d.coverage.mainSkinVertices - 1)) / 187),
          ),
        ),
      };
    });
    data.displacementMm = data.surface.map(
      (p) => Math.hypot(...p.position.map((v, j) => v - p.rest[j])) * 1000,
    );
    data.maxSkinDisplacementMm = Math.max(...data.displacementMm);
    assert(data.maxSkinDisplacementMm > 0.1, "No visible-scale skin movement");
    assert(!data.errors.length);
    assert(
      data.samples.every((s) => s.minJ > 0.2 && Number.isFinite(s.residualN)),
    );
    assert(
      data.samples.at(-1).staticResidualN < 0.001,
      "Control did not settle statically",
    );
    assert(data.samples.at(-1).lineAccepted, "Final line search rejected");
    cases.push({ id, value, data });
    await p.screenshot({ path: `output/verification/${id}-${value}.png` });
  }
  await p.evaluate(() => {
    window.__fullHeadFEM.grabPoint([0, 0.06, 0.06], [0, -0.003, 0.002]);
    window.__fullHeadFEM.resume();
  });
  const start = await p.evaluate(() => window.__fullHeadFEM.solver.frames);
  await p.waitForFunction(
    (f) =>
      window.__fullHeadFEM.errors.length ||
      window.__fullHeadFEM.solver.frames >= f + 16,
    start,
    { timeout: 30000 },
  );
  await p.evaluate(() => window.__fullHeadFEM.pause());
  await p.screenshot({ path: "output/verification/skin-pull.png" });
  const pull = await p.evaluate(() => ({
    errors: window.__fullHeadFEM.errors,
    samples: window.__fullHeadFEM.samples.slice(-16),
  }));
  assert(!pull.errors.length);
  assert(pull.samples.every((s) => s.minJ > 0.2));
  assert(pull.samples.at(-1).residualN < 0.01);
  await p.evaluate(() => window.__fullHeadFEM.setView("profile"));
  await p.waitForTimeout(100);
  await p.screenshot({ path: "output/verification/profile.png" });
  await p.locator("#release").click();
  const releaseStart = await p.evaluate(
    () => window.__fullHeadFEM.solver.frames,
  );
  await p.waitForFunction(
    (f) =>
      window.__fullHeadFEM.errors.length ||
      (window.__fullHeadFEM.solver.frames > f &&
        !window.__fullHeadFEM.samples.at(-1).grabbed &&
        window.__fullHeadFEM.settlement),
    releaseStart,
    { timeout: 30000 },
  );
  await p.waitForTimeout(250);
  const release = await p.evaluate(() => window.__fullHeadFEM.samples.at(-1));
  assert(!release.grabbed, "Drag was not released");
  assert(release.minJ > 0.2 && release.residualN < 0.01);
  assert(!errors.length);
  fs.writeFileSync(
    "output/verification/browser-check.json",
    JSON.stringify({ cases, pull, release, errors }, null, 2),
  );
  console.log(
    JSON.stringify({
      cases: cases.map((c) => ({
        id: c.id,
        value: c.value,
        last: c.data.samples.at(-1),
        maxSkinDisplacementMm: c.data.maxSkinDisplacementMm,
        adapter: c.data.adapter,
        errors: c.data.errors,
      })),
      pull: pull.samples.at(-1),
      release,
      errors,
    }),
  );
} finally {
  await browser?.close();
}
