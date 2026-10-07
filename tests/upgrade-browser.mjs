// Run only while owning the shared GPU/browser slot. One browser, serial cases.
import { chromium } from "playwright";
import fs from "node:fs";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
const origin = process.env.TISSUE_BASE_URL || "http://127.0.0.1:8779";
const output = "output/verification/upgrade"; fs.mkdirSync(output, { recursive: true });
const candidate = JSON.parse(fs.readFileSync("output/runtime-candidate.json")).build;
const baseline = JSON.parse(fs.readFileSync("output/runtime-current.json")).build;
const browser = await chromium.launch({ channel: "msedge", headless: true, args: ["--enable-unsafe-webgpu"] });
const results = [];
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
async function capture(page, label) {
  const bytes = await page.screenshot({ path: `${output}/${label}.png` });
  return { label, sha256: hash(bytes) };
}
try {
  for (const variant of ["baseline", "candidate-geometric", "candidate-smooth"]) {
    const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 1 });
    const page = await context.newPage(), errors = [], requests = [];
    page.on("pageerror", error => errors.push(error.message));
    page.on("request", request => requests.push(new URL(request.url()).pathname));
    const url = new URL("/neutral-tissue.html", origin);
    const isCandidate = variant !== "baseline";
    if (isCandidate) url.searchParams.set("candidate", "1");
    if (variant.endsWith("geometric")) url.searchParams.set("normals", "geometric");
    url.searchParams.set("expectedBuild", isCandidate ? candidate : baseline);
    const begin = performance.now();
    await page.goto(url.href);
    await page.waitForFunction(() => window.__fullHeadFEM?.ready, {}, { timeout: 60000 });
    const readyMs = performance.now() - begin;
    const anatomyURL = await page.evaluate(async () => {
      const model = await (await fetch(window.__tissueRuntime.tissueBase + "model.json")).json();
      return new URL(model.internal_anatomy_url, location.href).pathname;
    });
    if (isCandidate) assert(!requests.includes(anatomyURL), "Anatomy fetched during candidate startup");
    await page.waitForFunction(() => window.__fullHeadFEM.settlement?.converged, {}, { timeout: 60000 });
    const pictures = [];
    for (const state of ["neutral", "passive-25"]) {
      if (state !== "neutral") {
        await page.locator("#fat").fill("25"); await page.locator("#sag").fill("25");
        await page.waitForFunction(() => window.__fullHeadFEM.settlement?.converged && window.__fullHeadFEM.samples.at(-1).applied.sagPercent === 25, {}, { timeout: 60000 });
      }
      for (const light of isCandidate ? ["studio", "side", "overhead"] : ["studio"]) {
        if (isCandidate) await page.evaluate(light => window.__fullHeadFEM.setInspectionLighting(light), light);
        await page.locator('[data-view="three-quarter"]').click();
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        pictures.push(await capture(page, `${variant}-${state}-${light}`));
      }
    }
    const settled = await page.evaluate(() => {
      const d = window.__fullHeadFEM;
      return { last: d.samples.at(-1), settlement: d.settlement,
        surface: d.sampleSurface(Array.from({ length: 188 }, (_, i) => Math.floor(i * (d.coverage.mainSkinVertices - 1) / 187))) };
    });
    if (isCandidate) await page.evaluate(() => window.__fullHeadFEM.setInspectionLighting("studio"));
    await page.locator("#running").check();
    await page.waitForTimeout(2000);
    const start = await page.evaluate(() => ({ time: performance.now(), simulation: window.__fullHeadFEM.clock.simulationSeconds, samples: window.__fullHeadFEM.solver.frames }));
    await page.waitForTimeout(12000);
    const dynamic = await page.evaluate(start => {
      const d = window.__fullHeadFEM;
      return { wallSeconds: (performance.now() - start.time) / 1000,
        simulationSeconds: d.clock.simulationSeconds - start.simulation,
        solverUpdates: d.solver.frames - start.samples, clock: d.clock,
        measurement: d.measurement?.(), samples: d.samples.slice(-240), startup: d.startup,
        adapter: d.adapter, errors: d.errors };
    }, start);
    await page.locator("#running").uncheck();
    if (isCandidate) {
      await page.route(anatomyURL, route => route.abort());
      await page.locator("#anatomy").check();
      await page.waitForFunction(() => window.__fullHeadFEM.anatomyLoading.phase === "failed");
      assert.equal(await page.locator("#anatomy").isChecked(), false);
      await page.unroute(anatomyURL);
      await page.locator("#anatomy").check();
      await page.waitForFunction(() => window.__fullHeadFEM.anatomyLoading.phase === "ready", {}, { timeout: 45000 });
      pictures.push(await capture(page, `${variant}-anatomy-retry`));
      await page.locator("#anatomy").uncheck();
    }
    assert.deepEqual(errors, []); assert.deepEqual(dynamic.errors, []);
    assert(dynamic.samples.every(sample => sample.minJ > 0 && Number.isFinite(sample.residualN)));
    results.push({ variant, build: isCandidate ? candidate : baseline, readyMs, settled, dynamic, pictures });
    await context.close();
    console.log(JSON.stringify({ variant, readyMs, simulationSeconds: dynamic.simulationSeconds, wallSeconds: dynamic.wallSeconds }));
  }
  const reference = results[0].settled.surface;
  for (const result of results.slice(1)) {
    result.maxSettledSurfaceDifferenceM = Math.max(...result.settled.surface.map((point, i) => Math.hypot(...point.position.map((v, d) => v - reference[i].position[d]))));
    assert(result.maxSettledSurfaceDifferenceM < 2e-6, "Baseline mechanical positions differ by over 2 micrometres");
  }
  fs.writeFileSync(`${output}/results.json`, JSON.stringify({ baseline, candidate, results,
    limits: "Single bounded ordered run; render-submission counts are not displayed FPS. No physical Android or biological calibration acceptance." }, null, 2));
} finally { await browser.close(); }
