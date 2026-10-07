// Run ONLY in an exclusive parent-coordinated GPU slot. Hard browser budget: 420s.
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import fs from 'node:fs';
const origin = process.env.TISSUE_BASE_URL || 'http://127.0.0.1:8781';
const browser = await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']});
const watchdog = setTimeout(()=>{process.exitCode=1; void browser.close();},420000);
const receipt = {runs:[], limits:'GPU buffer bytes and sampled JS heap only; renderer/native/driver memory and physical displayed FPS are not measured. Input response ends at accepted render submission, not photons.'};
const directory = 'output/verification/pipeline';
fs.mkdirSync(directory,{recursive:true});
try {
  const context = await browser.newContext();
  const page = await context.newPage();
  await page.goto(origin+'/tests/solver.html');
  receipt.numerical = await page.evaluate(async()=>{
    const runtime=await (await fetch('/output/runtime-candidate.json')).json();
    const base=runtime.entry.slice(0,runtime.entry.lastIndexOf('/')+1);
    const {loadModel}=await import(base+'model.js');
    const {GPUHeadFEM}=await import(base+'full_head_fem_gpu.js');
    const {OrderedGPUPipeline}=await import(base+'ordered_gpu_pipeline.js');
    const {SynchronousGPUBridge}=await import(base+'synchronous_gpu_bridge.js');
    const {CPUHeadFEM}=await import(base+'tissue_cpu_backend.js');
    const {model,arrays}=await loadModel(runtime.tissueBase);
    const parameters={gravity:1,fatPercent:10,sagPercent:10};
    const options={dt:1/120,substeps:1,newton:2,cg:24,quasiStatic:false,checkpointState:true};
    const expected=[]; let engine,bridge,pipeline,cpu;
    try {
      engine=await GPUHeadFEM.create(model,arrays,null,{checkpointState:true});
      if(engine.boundedSubmissions) throw Error('Adapter uses bounded dispatch; pipeline experiment unavailable');
      for(let i=0;i<30;i++) {
        const r=await engine.step(parameters,options);
        expected.push({packed:r.packed.slice(),velocities:r.velocities.slice()});
      }
      engine.dispose();
      engine=await GPUHeadFEM.create(model,arrays,null,{checkpointState:true});
      bridge=new SynchronousGPUBridge(model);
      pipeline=new OrderedGPUPipeline({backend:engine,validate:r=>bridge.validate(r)});
      let positionError=0,velocityError=0;
      pipeline.enqueue(parameters,options); pipeline.enqueue(parameters,options);
      for(let i=0;i<30;i++) {
        const lease=await pipeline.takeNext();
        for(let n=0;n<lease.result.packed.length;n++) positionError=Math.max(positionError,Math.abs(lease.result.packed[n]-expected[i].packed[n]));
        for(let n=0;n<lease.result.velocities.length;n++) velocityError=Math.max(velocityError,Math.abs(lease.result.velocities[n]-expected[i].velocities[n]));
        bridge.accept(lease.result); pipeline.acknowledge(lease);
        if(i<28) pipeline.enqueue(parameters,options);
      }
      const memory=engine.bufferMemory;
      // Two outstanding real solves are discarded on loss; CPU recovery uses
      // the last fully accepted checkpoint, including actual solver velocities.
      pipeline.enqueue(parameters,options); pipeline.enqueue(parameters,options);
      await pipeline.submitChain;
      const checkpoint=bridge.lost();
      pipeline.stop(Error('Injected actual device loss')); engine.device.destroy(); await engine.device.lost;
      cpu=await CPUHeadFEM.create(model,arrays,'Pipeline diagnostic device loss');
      const restored=await cpu.restore(checkpoint.positions,checkpoint.velocities,true);
      const recovered=bridge.recovered(checkpoint,restored);
      return {build:runtime.build,steps:30,positionError,velocityError,memory,recovered,
        acceptedSimulationSeconds:checkpoint.version.simulationSeconds,errors:engine.errors};
    } finally {pipeline?.dispose();bridge?.dispose();cpu?.dispose();engine?.dispose();}
  });
  assert(receipt.numerical.positionError<2e-6,'Full packed state parity');
  assert(receipt.numerical.velocityError<2e-6,'Full actual velocity parity');
  assert.equal(receipt.numerical.recovered.epoch,2);
  await context.close();
  // ABBA ordering reduces a single warmup-order bias; fresh contexts, one at a time.
  for(const mode of ['versioned','pipeline','pipeline','versioned']) {
    const ctx=await browser.newContext({viewport:{width:1280,height:900},deviceScaleFactor:1});
    try {
      const p=await ctx.newPage(), errors=[];
      p.on('pageerror',e=>errors.push(e.message));
      const cdp=await ctx.newCDPSession(p);
      await cdp.send('Network.enable'); await cdp.send('Network.setCacheDisabled',{cacheDisabled:true});
      await cdp.send('Performance.enable');
      const start=performance.now(),build=receipt.numerical.build;
      await p.goto(origin+`/neutral-tissue.html?candidate=1&expectedBuild=${build}&bridge=${mode}`);
      await p.waitForFunction(()=>window.__fullHeadFEM?.ready,{}, {timeout:45000});
      const coldReadyMs=performance.now()-start;
      await p.locator('#quality').selectOption('reference');
      await p.evaluate(()=>{const d=window.__fullHeadFEM;d.pause();d.setParameters({gravity:1,fatPercent:10,sagPercent:10});});
      await p.waitForFunction(()=>window.__fullHeadFEM?.settlement?.converged && !window.__fullHeadFEM.solveInFlight,{}, {timeout:45000});
      await p.evaluate(()=>window.__fullHeadFEM.resume());
      await p.waitForTimeout(2000);
      const begin=await p.evaluate(()=>{const d=window.__fullHeadFEM;return {time:performance.now(),sim:d.clock.simulationSeconds,frame:d.solver.frames};});
      let peakJSHeapUsedBytes=0;
      for(let second=0;second<12;second++) {
        const metrics=await cdp.send('Performance.getMetrics');
        peakJSHeapUsedBytes=Math.max(peakJSHeapUsedBytes,metrics.metrics.find(x=>x.name==='JSHeapUsedSize')?.value??0);
        // Matched input pulse; report first host-accepted result with new gravity.
        if(second===5) await p.evaluate(()=>{
          const d=window.__fullHeadFEM;d.benchmarkInput={at:performance.now(),startFrame:d.solver.frames};d.setParameters({gravity:0,fatPercent:10,sagPercent:10});
        });
        await p.evaluate(()=>{
          const d=window.__fullHeadFEM;
          if(d.benchmarkInput && !d.benchmarkInput.responseMs && d.samples.at(-1)?.applied.gravity===0)
            d.benchmarkInput.responseMs=performance.now()-d.benchmarkInput.at;
        });
        await p.waitForTimeout(1000);
      }
      await p.evaluate(()=>window.__fullHeadFEM.pause());
      await p.waitForFunction(()=>{const d=window.__fullHeadFEM;return !d.solveInFlight && (!d.pipeline || d.pipeline.outstanding===0);},{},{timeout:20000});
      const run=await p.evaluate(begin=>{
        const d=window.__fullHeadFEM,samples=d.samples.filter(s=>s.frame>begin.frame);
        const summary=key=>{const values=samples.map(s=>s[key]).filter(Number.isFinite).sort((a,b)=>a-b);return values.length?{count:values.length,p50:values[Math.floor(values.length*.5)],p95:values[Math.floor(values.length*.95)],max:values.at(-1)}:null;};
        return {wallSecondsIncludingDrain:(performance.now()-begin.time)/1000,simulatedSeconds:d.clock.simulationSeconds-begin.sim,updates:d.solver.frames-begin.frame,
          mapMs:summary('readbackWaitMs'),gpuMs:summary('gpuMs'),frameWorkMs:summary('frameWorkMs'),inputAgeAtRenderSubmissionMs:summary('inputAgeAtRenderSubmissionMs'),
          memory:d.solver.bufferMemory,pipeline:d.pipeline,inputLatency:d.inputLatency,inputPulse:d.benchmarkInput,measurement:d.measurement(),clock:d.clock,errors:d.errors,gpuErrors:d.solver.errors};
      },begin);
      assert.deepEqual(errors,[]); assert.deepEqual(run.errors,[]); assert.deepEqual(run.gpuErrors,[]);
      if(mode==='pipeline') assert(run.pipeline?.peakOutstanding===2,'Actual two-slot overlap exercised');
      await p.screenshot({path:`${directory}/${receipt.runs.length}-${mode}.png`});
      receipt.runs.push({mode,coldReadyMs,peakJSHeapUsedBytes,...run});
      fs.writeFileSync(`${directory}/results.json`,JSON.stringify(receipt,null,2));
    } finally {await ctx.close();}
  }
  console.log(JSON.stringify({build:receipt.numerical.build,numerical:receipt.numerical,runs:receipt.runs.map(r=>({mode:r.mode,sim:r.simulatedSeconds,wall:r.wallSecondsIncludingDrain,map:r.mapMs,memory:r.memory}))}));
} finally {
  clearTimeout(watchdog);
  await browser.close();
  fs.writeFileSync(`${directory}/results.json`,JSON.stringify(receipt,null,2));
}
