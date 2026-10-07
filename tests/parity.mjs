import { pointerPath, expectedBuild, solverURL } from "./runtime.mjs";
import fs from "node:fs";
import { chromium } from "../node_modules/playwright/index.mjs";
const b = await chromium.launch({
  channel: "msedge",
  headless: true,
  args: ["--enable-unsafe-webgpu"],
});
const p = await b.newPage();
const errors = [];
p.on("console", (m) => {
  if (m.type() === "error") errors.push(m.text().slice(0, 1000));
});
await p.goto(solverURL);
try {
  const result = await p.evaluate(
    async ({ pointerPath, expectedBuild }) => {
      const runtime = await (await fetch(pointerPath)).json();
      if (expectedBuild && runtime.build !== expectedBuild)
        throw Error("Wrong verification build");
      const moduleBase = runtime.entry.slice(
        0,
        runtime.entry.lastIndexOf("/") + 1,
      );
      const { GPUHeadFEM } = await import(moduleBase + "full_head_fem_gpu.js");
      const model = await (
          await fetch(runtime.tissueBase + "model.json")
        ).json(),
        arrays = {};
      for (const [k, v] of Object.entries(model.assets)) {
        const b = await (
          await fetch(v.url.startsWith("/") ? v.url : "/" + v.url)
        ).arrayBuffer();
        arrays[k] = new (
          ["nodes", "faces", "layers", "groups"].includes(k)
            ? Uint32Array
            : Float64Array
        )(b);
      }
      const engine = await GPUHeadFEM.create(model, arrays);
      window.gpuTest = { engine, arrays, model };
      const { LayeredFEM, installTissueBoundary } = await import(
        moduleBase + "layered_fem.js"
      );
      const qa = {};
      for (const [k, v] of Object.entries(arrays))
        qa[k] =
          v instanceof Float64Array
            ? Float64Array.from(Float32Array.from(v))
            : v;
      const cpu = new LayeredFEM(model, qa);
      cpu.gravity = [0, -9.81, 0];
      installTissueBoundary(cpu, model, qa);
      const samples = [];
      for (let i = 0; i < 4; i++) {
        const result = await engine.step(
          { gravity: 1 },
          { substeps: 1, newton: 2, cg: 24, block: false },
        );
        samples.push(result.stats);
        cpu.step(1 / 120, {
          iterations: 2,
          cgIterations: 24,
          cgTolerance: 0.01,
          forceTolerance: 0,
        });
      }
      const gpu = await engine.readState();
      let maxError = 0;
      for (let n = 0; n < model.nodes; n++)
        for (let k = 0; k < 3; k++)
          maxError = Math.max(
            maxError,
            Math.abs(gpu[n * 64 + 4 + k] + gpu[n * 64 + k] - cpu.x[n * 3 + k]),
          );
      let staticResult, staticCPU;
      for (let i = 0; i < 12; i++) {
        staticResult = await engine.step(
          { gravity: 1 },
          { quasiStatic: true, substeps: 1, newton: 4, cg: 256, block: false },
        );
        staticCPU = cpu.step(Infinity, {
          iterations: 4,
          cgIterations: 256,
          cgTolerance: 0.01,
          forceTolerance: 1e-4,
        });
        if (
          staticResult.stats.staticResidualN < 0.001 &&
          staticCPU.residual < 0.001
        )
          break;
      }
      const staticGPU = await engine.readState();
      let staticError = 0;
      for (let n = 0; n < model.nodes; n++)
        for (let k = 0; k < 3; k++)
          staticError = Math.max(
            staticError,
            Math.abs(
              staticGPU[n * 64 + 4 + k] +
                staticGPU[n * 64 + k] -
                cpu.x[n * 3 + k],
            ),
          );
      return {
        samples,
        maxPositionErrorM: maxError,
        cpuResidual: cpu.last.residual,
        quasiStaticPositionErrorM: staticError,
        quasiStaticStats: staticResult.stats,
        quasiStaticCPUResidualN: staticCPU.residual,
        adapter: engine.adapterInfo,
        errors: engine.errors,
      };
    },
    { pointerPath, expectedBuild },
  );
  fs.writeFileSync(
    "output/verification/cpu-gpu-parity.json",
    JSON.stringify(result, null, 2),
  );
  console.log(JSON.stringify(result));
  if (
    result.maxPositionErrorM > 1e-6 ||
    result.quasiStaticPositionErrorM > 1e-6 ||
    result.quasiStaticStats.staticResidualN >= 0.001 ||
    result.quasiStaticCPUResidualN >= 0.001 ||
    result.quasiStaticStats.integration !== "quasistatic" ||
    result.quasiStaticStats.simulatedSeconds !== 0 ||
    result.quasiStaticStats.maxNodeSpeedMps !== 0 ||
    result.errors.length
  )
    throw Error("CPU/GPU parity failed");
} catch (e) {
  process.exitCode = 1;
  console.log(e.message.slice(0, 5000));
}
console.log(JSON.stringify(errors));
await b.close();
