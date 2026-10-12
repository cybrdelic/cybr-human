import assert from "node:assert/strict";
import fs from "node:fs";
import { chromium } from "playwright";
import { pointerPath, expectedBuild, solverURL } from "./runtime.mjs";
const runtime = JSON.parse(
  fs.readFileSync(pointerPath.replace(/^\//, ""), "utf8"),
);
if (expectedBuild) assert.equal(runtime.build, expectedBuild);
const { LayeredFEM, compressionBarrier } = await import(
  new URL(
    ".." + runtime.entry.replace(/viewer\.js$/, "layered_fem.js"),
    import.meta.url,
  )
);
let firstError = 0,
  secondError = 0;
for (const J of [0.21, 0.25, 0.35, 0.5, 0.599]) {
  const h = 1e-6,
    b = compressionBarrier(J),
    a = compressionBarrier(J - h),
    c = compressionBarrier(J + h);
  firstError = Math.max(
    firstError,
    Math.abs((c[0] - a[0]) / (2 * h) - b[1]) / Math.max(1, Math.abs(b[1])),
  );
  secondError = Math.max(
    secondError,
    Math.abs((c[1] - a[1]) / (2 * h) - b[2]) / Math.max(1, Math.abs(b[2])),
  );
  assert(b[0] >= 0 && b[2] >= 0);
}
assert(firstError < 1e-6 && secondError < 1e-6);
assert.deepEqual(compressionBarrier(0.6), [0, 0, 0]);
assert(!Number.isFinite(compressionBarrier(0.2)[0]));
const arrays = {
  rest: new Float64Array([0, 0, 0, 0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01]),
  nodes: new Uint32Array([0, 1, 2, 3]),
  gradients: new Float64Array([
    -100, -100, -100, 100, 0, 0, 0, 100, 0, 0, 0, 100,
  ]),
  material: new Float64Array([1e-6 / 6, 10000, 80000]),
  mass: new Float64Array([0.001, 0.001, 0.001, 0.001]),
};
const fem = new LayeredFEM({ schema: "cybr-layered-fem-v1", nodes: 4 }, arrays);
const x = arrays.rest.slice();
x[11] *= 0.35;
const state = fem.evaluate(x, Infinity, true),
  g = fem.g.slice();
let forceError = 0;
for (let i = 0; i < x.length; i++) {
  const v = x[i],
    h = 1e-8;
  x[i] = v + h;
  const plus = fem.evaluate(x, Infinity, false).energy;
  x[i] = v - h;
  const minus = fem.evaluate(x, Infinity, false).energy;
  x[i] = v;
  forceError = Math.max(
    forceError,
    Math.abs((plus - minus) / (2 * h) - g[i]) / Math.max(0.01, Math.abs(g[i])),
  );
}
assert(forceError < 1e-5, "Barrier force differs from energy derivative");
const browser = await chromium.launch({
  channel: "msedge",
  headless: true,
  args: ["--enable-unsafe-webgpu"],
});
let gpuForceError;
try {
  const page = await browser.newPage();
  await page.goto(solverURL);
  gpuForceError = await page.evaluate(
    async ({ entry, reference }) => {
      const { GPUHeadFEM } = await import(
        entry.replace(/viewer\.js$/, "full_head_fem_gpu.js")
      );
      const a = {
        rest: new Float64Array([0, 0, 0, 0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01]),
        nodes: new Uint32Array([0, 1, 2, 3]),
        gradients: new Float64Array([
          -100, -100, -100, 100, 0, 0, 0, 100, 0, 0, 0, 100,
        ]),
        material: new Float64Array([1e-6 / 6, 10000, 80000]),
        mass: new Float64Array([0.001, 0.001, 0.001, 0.001]),
        faces: new Uint32Array([0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2]),
        fatWeights: new Float64Array(1),
        fixedNodes: new Float64Array(4),
        contactPlanes: new Float64Array(28),
      };
      for (let i = 0; i < 4; i++)
        a.contactPlanes.set([0, -1, 0, 0, 1, 0, 0.00001], i * 7);
      const engine = await GPUHeadFEM.create(
        { nodes: 4, tetrahedra: 1, surface_nodes: 4 },
        a,
      );
      try {
        const nodes = await engine.readState();
        nodes[3 * 64 + 4 + 2] = -0.0065;
        engine.device.queue.writeBuffer(engine.buffers[0], 0, nodes);
        await engine.step(
          { gravity: 0 },
          { quasiStatic: true, newton: 0, cg: 0, substeps: 1 },
        );
        const state = await engine.readState();
        let error = 0;
        for (let i = 0; i < 12; i++)
          error = Math.max(
            error,
            Math.abs(
              state[Math.floor(i / 3) * 64 + 20 + (i % 3)] - reference[i],
            ) / Math.max(0.01, Math.abs(reference[i])),
          );
        if (engine.errors.length) throw Error(engine.errors.join("\n"));
        return error;
      } finally {
        engine.dispose();
      }
    },
    { entry: runtime.entry, reference: Array.from(g) },
  );
  assert(
    gpuForceError < 0.002,
    "GPU compression force differs from CPU energy derivative",
  );
} finally {
  await browser.close();
}
console.log(
  JSON.stringify({
    firstError,
    secondError,
    forceError,
    gpuForceError,
    minJ: state.minJ,
  }),
);
