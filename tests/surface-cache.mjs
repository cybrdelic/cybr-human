import fs from 'node:fs';
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { appURL } from './runtime.mjs';

const output='output/verification/surface-cache';fs.mkdirSync(output,{recursive:true});
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']});
const results=[];
try {
  for(const enabled of [false,true]) {
    const page=await browser.newPage({viewport:{width:1280,height:900}}),errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    const url = new URL(appURL); url.searchParams.set('surfaceCache', enabled?'1':'0'); await page.goto(url.href);
    await page.waitForFunction(()=>window.__fullHeadFEM?.settlement?.converged,{}, {timeout:60000});
    await page.locator('[data-view="three-quarter"]').click();
    await page.screenshot({path:output+'/'+(enabled?'cached':'legacy')+'-neutral.png'});
    const neutral=enabled?await page.evaluate(()=>{
      const d=window.__fullHeadFEM,cache=d.readCachedSurface();
      const ids=Array.from({length:1200},(_,i)=>Math.floor(i*(cache.length/4-1)/1199));
      const samples=d.sampleSurface(ids);let maxError=0;
      for(const s of samples) maxError=Math.max(maxError,Math.hypot(...s.position.map((v,k)=>v-cache[s.id*4+k])));
      return {maxError,evaluations:d.surfaceCache.parts.map(p=>p.evaluations)};
    }):null;
    await page.locator('#fat').fill('50');
    await page.locator('#sag').fill('50');
    await page.waitForFunction(()=>window.__fullHeadFEM?.settlement?.converged && window.__fullHeadFEM.samples.at(-1).applied.sagPercent===50,{}, {timeout:60000});
    await page.screenshot({path:output+'/'+(enabled?'cached':'legacy')+'-deformed.png'});
    const result=await page.evaluate(()=>({build:window.__tissueRuntime.build,last:window.__fullHeadFEM.samples.at(-1),settlement:window.__fullHeadFEM.settlement,performance:window.__fullHeadFEM.performance,cache:window.__fullHeadFEM.surfaceCache,errors:window.__fullHeadFEM.errors}));
    const deformed=enabled?await page.evaluate(()=>{
      const d=window.__fullHeadFEM,cache=d.readCachedSurface();
      const ids=Array.from({length:1200},(_,i)=>Math.floor(i*(cache.length/4-1)/1199));
      let maxError=0;
      for(const s of d.sampleSurface(ids)) maxError=Math.max(maxError,Math.hypot(...s.position.map((v,k)=>v-cache[s.id*4+k])));
      return {maxError};
    }):null;
    assert.deepEqual(errors,[]);assert.deepEqual(result.errors,[]);
    if(enabled) {assert(neutral.maxError<2e-6,JSON.stringify(neutral));assert(deformed.maxError<2e-6,JSON.stringify(deformed));}
    results.push({enabled,neutral,deformed,...result});
    await page.close();
  }
  fs.writeFileSync(output+'/results.json',JSON.stringify(results,null,2));
  console.log(JSON.stringify(results.map(r=>({enabled:r.enabled,neutral:r.neutral,deformed:r.deformed,performance:r.performance}))));
} finally {await browser.close();}
