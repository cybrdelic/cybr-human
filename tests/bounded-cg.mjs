import fs from 'node:fs';
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {appURL} from './runtime.mjs';
const b=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']});
try {
  const p=await b.newPage(), errors=[];
  p.on('pageerror',e=>errors.push(e.message));
  await p.goto(appURL);
  await p.waitForFunction(()=>window.__fullHeadFEM?.settlement?.converged,{}, {timeout:60000});
  const result=await p.evaluate(async()=>{
    const engine=window.__fullHeadFEM.solver;
    engine.boundedSubmissions=true;engine.cgBatchSize=16;
    const cases=[{parameters:{fatPercent:50,sagPercent:50},settings:{quasiStatic:true,newton:4,cg:256,substeps:1}},{parameters:{fatPercent:20,sagPercent:20},settings:{quasiStatic:false,newton:2,cg:64,substeps:2}}];
    const results=[];
    for(const test of cases){
      const runs=[];
      for(const enabled of [false,true]){
        engine.reset();engine.boundedEarlyExit=enabled;
        let last;const samples=[];
        for(let i=0;i<4;i++){last=await engine.step(test.parameters,test.settings);samples.push(last.stats);}
        runs.push({enabled,packed:last.packed,samples});
      }
      let maxError=0;
      for(let i=0;i<runs[0].packed.length;i++)maxError=Math.max(maxError,Math.abs(runs[0].packed[i]-runs[1].packed[i]));
      results.push({mode:test.settings.quasiStatic?'static':'dynamic',maxPackedError:maxError,runs:runs.map(({packed,...r})=>r)});
    }
    return {build:window.__tissueRuntime.build,results,errors:engine.errors};
  });
  assert.deepEqual(errors,[]);assert.deepEqual(result.errors,[]);
  for(const r of result.results){assert(r.maxPackedError<1e-6,JSON.stringify(r));assert(r.runs[1].samples.some(s=>s.cgIterationsSkipped>0));for(let i=0;i<4;i++)assert.equal(r.runs[0].samples[i].cgIterationsExecuted,r.runs[1].samples[i].cgIterationsExecuted);}
  fs.writeFileSync('output/verification/bounded-cg.json',JSON.stringify(result,null,2));
  console.log(JSON.stringify(result.results.map(r=>({mode:r.mode,maxPackedError:r.maxPackedError,submissions:r.runs.map(x=>x.samples.map(s=>s.submissionChunks)),scheduled:r.runs.map(x=>x.samples.map(s=>s.cgIterationsScheduled))}))));
}finally{await b.close();}
