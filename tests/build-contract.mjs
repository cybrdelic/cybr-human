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
  const results = {};
  // Corrupt the delivered GLB, without touching the source or saved snapshot.
  const corrupt = await browser.newPage();
  await corrupt.route("**/assets/output/**/head.glb", async (route) => {
    const response = await route.fetch(),
      body = await response.body();
    body[body.length - 1] ^= 1;
    await route.fulfill({ response, body });
  });
  await corrupt.goto(appURL);
  await corrupt.waitForFunction(
    () => window.__fullHeadFEM?.errors.length,
    {},
    { timeout: 45000 },
  );
  results.tamper = await corrupt.evaluate(() => ({
    errors: window.__fullHeadFEM.errors,
    disabled: document.querySelector("#fat").disabled,
  }));
  assert(
    results.tamper.errors.some((e) => e.includes("Rendered skin differs")),
  );
  assert(results.tamper.disabled);
  await corrupt.close();
  // A fine-detail worker failure must remain visible after bulk solves.
  const page = await browser.newPage(),
    requested = [];
  page.on("request", (r) => requested.push(r.url()));
  await page.route("**/wrinkle_worker.js", (route) =>
    route.fulfill({
      contentType: "text/javascript",
      body: "self.onmessage=()=>self.postMessage({error:'Injected fine-worker failure'});",
    }),
  );
  await page.goto(appURL);
  await page.waitForFunction(
    () => window.__fullHeadFEM?.ready,
    {},
    { timeout: 45000 },
  );
  await page.locator("#fat").fill("25");
  await page.waitForFunction(
    () => window.__fullHeadFEM.settlement?.converged,
    {},
    { timeout: 45000 },
  );
  results.fineFailure = await page.evaluate(() => ({
    failed: window.__fullHeadFEM.fineSkinPatch.failed,
    status: document.querySelector("#tissueStatus").textContent,
    disabled: document.querySelector("#fineHeatmap").disabled,
  }));
  assert(results.fineFailure.failed && results.fineFailure.disabled);
  assert(results.fineFailure.status.includes("Injected fine-worker failure"));
  await page.locator("#fat").fill("200");
  await page.locator("#resetShape").click();
  await page.waitForFunction(
    () =>
      window.__fullHeadFEM.resetCount &&
      window.__fullHeadFEM.settlement?.converged,
    {},
    { timeout: 45000 },
  );
  results.reset = await page.evaluate(() => ({
    last: window.__fullHeadFEM.samples.at(-1),
    errors: window.__fullHeadFEM.errors,
    build: window.__tissueRuntime.build,
  }));
  assert.equal(results.reset.last.applied.fatPercent, 0);
  assert.equal(results.reset.last.applied.sagPercent, 0);
  assert.deepEqual(results.reset.errors, []);
  const runtimePrefix = "/output/runtime/" + results.reset.build + "/";
  results.requests = requested
    .filter((url) => /\.(?:js|wgsl|glb|bin|f32|json)(?:\?|$)/.test(url))
    .map((url) => new URL(url).pathname);
  assert(
    results.requests.every(
      (path) =>
        path.includes("/vendor/") ||
        path.endsWith("/src/boot.js") ||
        path.endsWith("/runtime-current.json") ||
        path.endsWith("/runtime-candidate.json") ||
        path.startsWith(runtimePrefix),
    ),
    "Mutable code or asset escaped the build snapshot",
  );
  fs.writeFileSync(
    "output/verification/build-contract-check.json",
    JSON.stringify(results, null, 2),
  );
  console.log(
    JSON.stringify({
      tamper: results.tamper,
      fineFailure: results.fineFailure.failed,
      reset: results.reset.last,
      build: results.reset.build,
      requests: results.requests.length,
    }),
  );
} finally {
  await browser?.close();
}
