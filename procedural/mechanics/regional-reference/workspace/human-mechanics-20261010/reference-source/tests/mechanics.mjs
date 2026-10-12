import assert from "node:assert/strict";
import fs from "node:fs";
import { chromium } from "playwright";
import { pointerPath, expectedBuild, solverURL } from "./runtime.mjs";
const browser = await chromium.launch({
  channel: "msedge",
  headless: true,
  args: ["--enable-unsafe-webgpu"],
});
try {
  const page = await browser.newPage();
  await page.goto(solverURL);
  const result = await page.evaluate(
    async ({ pointerPath, expectedBuild }) => {
      const runtime = await (await fetch(pointerPath)).json();
      if (expectedBuild && runtime.build !== expectedBuild)
        throw Error("Wrong verification build");
      const base = runtime.entry.slice(0, runtime.entry.lastIndexOf("/") + 1);
      const { loadModel } = await import(base + "model.js");
      const { model, arrays } = await loadModel(runtime.tissueBase);
      const { LayeredFEM } = await import(base + "layered_fem.js");
      const { CPUHeadFEM } = await import(base + "tissue_cpu_backend.js");
      const solver = new LayeredFEM(model, arrays),
        rest = arrays.rest;
      const rotation = 0.35,
        c = Math.cos(rotation),
        s = Math.sin(rotation);
      const rotated = rest.slice();
      for (let i = 0; i < rest.length; i += 3) {
        rotated[i] = c * rest[i] - s * rest[i + 1] + 0.003;
        rotated[i + 1] = s * rest[i] + c * rest[i + 1] - 0.002;
        rotated[i + 2] += 0.001;
      }
      const rigid = solver.evaluate(rotated, Infinity, false);
      const deformed = rest.map((v, i) => v * [1.015, 0.975, 1.01][i % 3]);
      solver.volumeTargets = Float64Array.from(
        arrays.fatWeights,
        (w) => 1 + 0.2 * w,
      );
      const state = solver.evaluate(deformed, Infinity, true),
        gradient = solver.g.slice();
      let gradientRelativeError = 0,
        maxAbsoluteErrorN = 0;
      for (const n of [71, 367, 991, 1489, 2047, 2371])
        for (let axis = 0; axis < 3; axis++) {
          const i = n * 3 + axis,
            h = 2e-8,
            original = deformed[i];
          deformed[i] = original + h;
          const plus = solver.evaluate(deformed, Infinity, false).energy;
          deformed[i] = original - h;
          const minus = solver.evaluate(deformed, Infinity, false).energy;
          deformed[i] = original;
          const fd = (plus - minus) / (2 * h),
            error = Math.abs(fd - gradient[i]);
          maxAbsoluteErrorN = Math.max(maxAbsoluteErrorN, error);
          gradientRelativeError = Math.max(
            gradientRelativeError,
            error / Math.max(0.01, Math.abs(fd), Math.abs(gradient[i])),
          );
        }
      const a = await CPUHeadFEM.create(model, arrays, "substep contract"),
        b = await CPUHeadFEM.create(model, arrays, "substep contract");
      let substepError = 0,
        stats;
      try {
        const parameters = { gravity: 1, fatPercent: 25, sagPercent: 30 },
          options = { dt: 1 / 120, newton: 2, cg: 24 };
        const together = await a.step(parameters, { ...options, substeps: 2 });
        await b.step(parameters, { ...options, substeps: 1 });
        const separate = await b.step(parameters, { ...options, substeps: 1 });
        substepError = Math.max(
          ...together.packed.map((v, i) => Math.abs(v - separate.packed[i])),
        );
        stats = together.stats;
      } finally {
        a.dispose();
        b.dispose();
      }
      return {
        build: runtime.build,
        rigidEnergyJ: rigid.energy,
        rigidMinJ: rigid.minJ,
        gradientRelativeError,
        maxAbsoluteErrorN,
        minJ: state.minJ,
        substepError,
        stats,
      };
    },
    { pointerPath, expectedBuild },
  );
  fs.writeFileSync(
    "output/verification/mechanics.json",
    JSON.stringify(result, null, 2),
  );
  assert(
    Math.abs(result.rigidEnergyJ) < 1e-8,
    "Rigid motion changed elastic energy",
  );
  assert(
    result.gradientRelativeError < 0.002,
    "FEM force is not the energy derivative",
  );
  assert(
    result.substepError === 0,
    "CPU ignored or changed the requested substeps",
  );
  assert.equal(result.stats.simulatedSeconds, 1 / 60);
  console.log(JSON.stringify(result));
} finally {
  await browser.close();
}
