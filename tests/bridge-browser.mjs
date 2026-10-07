// Requires a parent-coordinated GPU slot. Does not run during CPU preparation.
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import fs from 'node:fs';
const origin = process.env.TISSUE_BASE_URL || 'http://127.0.0.1:8781';
const browser = await chromium.launch({ channel:'msedge', headless:true, args:['--enable-unsafe-webgpu'] });
try {
  const page = await browser.newPage(), errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(origin + '/tests/solver.html');
  const packing = await page.evaluate(async () => {
    const r = await (await fetch('/output/runtime-candidate.json')).json();
    const base = r.entry.slice(0, r.entry.lastIndexOf('/') + 1);
    const {loadModel} = await import(base + 'model.js');
    const {GPUHeadFEM} = await import(base + 'full_head_fem_gpu.js');
    const {CPUHeadFEM} = await import(base + 'tissue_cpu_backend.js');
    const {SynchronousGPUBridge} = await import(base + 'synchronous_gpu_bridge.js');
    const {model, arrays} = await loadModel(r.tissueBase);
    const engine = await GPUHeadFEM.create(model, arrays, null, {checkpointState:true});
    const bridge = new SynchronousGPUBridge(model); let cpu;
    try {
      const states = [];
      for (const quasiStatic of [false, true]) {
        const result = await engine.step({gravity:1, fatPercent:10, sagPercent:10},
          {dt:1/120, substeps:2, newton:2, cg:24, quasiStatic});
        bridge.accept(result);
        const raw = await engine.readState();
        let maxVelocityError = 0, maxPositionError = 0;
        for (let n=0;n<model.nodes;n++) for(let d=0;d<3;d++) {
          maxVelocityError = Math.max(maxVelocityError, Math.abs(result.velocities[n*3+d] - raw[n*64+12+d]));
          maxPositionError = Math.max(maxPositionError, Math.abs(result.packed[n*4+d] - (raw[n*64+d] + raw[n*64+4+d])));
        }
        states.push({quasiStatic,maxVelocityError,maxPositionError,stats:result.stats});
      }
      const checkpoint = bridge.lost();
      engine.device.destroy(); await engine.device.lost;
      cpu = await CPUHeadFEM.create(model, arrays, 'Injected diagnostic device loss');
      const receipt = await cpu.restore(checkpoint.positions, checkpoint.velocities, true);
      const restoredVersion = bridge.recovered(checkpoint, receipt);
      const resumed = await cpu.step({gravity:1}, {dt:1/120, substeps:1, newton:2, cg:24, checkpointState:true});
      bridge.accept(resumed);
      return {build:r.build, states, restoredVersion, resumed:resumed.stats, errors:engine.errors};
    } finally {bridge.dispose(); cpu?.dispose(); engine.dispose();}
  });
  for(const state of packing.states) { assert.equal(state.maxVelocityError,0); assert(state.maxPositionError<5e-8); }
  assert.deepEqual(packing.errors,[]);
  const runtime = await page.evaluate(async()=> (await (await fetch('/output/runtime-candidate.json')).json()).build);
  await page.goto(origin + '/neutral-tissue.html?runtime=' + runtime + '&bridge=versioned');
  await page.waitForFunction(()=>window.__fullHeadFEM?.ready, {}, {timeout:60000});
  await page.evaluate(()=>window.__fullHeadFEM.setParameters({gravity:1,fatPercent:5,sagPercent:5}));
  await page.waitForFunction(()=>window.__fullHeadFEM?.bridge?.version.sequence>=2, {}, {timeout:60000});
  const beforeLoss = await page.evaluate(()=>{
    const d=window.__fullHeadFEM;
    const node=d.grabPoint([0,.015,.095],[.001,0,0]); d.release();
    return {node,grabVersion:d.grabVersion,bridge:d.bridge,clock:d.clock};
  });
  assert(beforeLoss.node>=0); assert.equal(beforeLoss.grabVersion.epoch,beforeLoss.bridge.version.epoch);
  await page.evaluate(()=>{
    const d=window.__fullHeadFEM, engine=d.solver, nativeStep=engine.step.bind(engine);
    // Force loss after mapping succeeds but before the viewer consumes its result.
    engine.step=async(...args)=>{const result=await nativeStep(...args); engine.device.destroy(); await engine.device.lost; return result;};
    d.setParameters({gravity:1,fatPercent:6,sagPercent:5});
  });
  await page.waitForFunction(()=>window.__fullHeadFEM?.recovery?.actualSolverVelocity, {}, {timeout:60000});
  const recovery = await page.evaluate(()=>{const d=window.__fullHeadFEM;return {recovery:d.recovery,bridge:d.bridge,clock:d.clock,errors:d.errors};});
  assert.equal(recovery.bridge.version.epoch,2); assert.deepEqual(recovery.errors,[]);
  await page.locator('#resetShape').click();
  await page.waitForFunction(()=>window.__fullHeadFEM?.resetCount>=1, {}, {timeout:60000});
  assert.deepEqual(errors,[]);
  fs.mkdirSync('output/verification/bridge',{recursive:true});
  fs.writeFileSync('output/verification/bridge/results.json', JSON.stringify({packing,beforeLoss,recovery,
    limits:'Versioned synchronous bridge only; no resident production renderer or throughput claim.'},null,2));
  console.log(JSON.stringify({build:runtime,packing:packing.states.map(s=>({velocityError:s.maxVelocityError,positionError:s.maxPositionError})),recovery:recovery.recovery}));
} finally {await browser.close();}
