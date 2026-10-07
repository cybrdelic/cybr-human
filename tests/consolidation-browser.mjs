// Serialized baseline/staged protocol. Fresh Playwright temporary profiles only.
import {chromium} from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
const strict=process.env.TISSUE_STRICT==='1', demoOnly=process.env.TISSUE_DEMO_ONLY==='1';
const root=process.cwd(), output=path.join(root,'output/verification/consolidation',demoOnly?'post-comparison-demo':strict?'strict':'');fs.mkdirSync(output,{recursive:true});
const selected=JSON.parse(fs.readFileSync('output/runtime-current.json','utf8')).build;
const source=JSON.parse(fs.readFileSync('provenance/public-source.json','utf8')).source_sha256;
const software=process.env.TISSUE_SOFTWARE==='1';
const viewport=software?{width:960,height:720}:{width:1280,height:900};
const browser=await chromium.launch({channel:'msedge',headless:true,args:software
  ? ['--use-angle=swiftshader','--enable-unsafe-swiftshader','--disable-webgpu','--disable-features=WebGPU','--num-raster-threads=2','--renderer-process-limit=1']
  : ['--enable-unsafe-webgpu']});
const results=[];
let captureStarted=0;
async function snapshot(page,label,notes){
  const state=await page.evaluate(()=>{
    const d=window.__fullHeadFEM, ids=Array.from({length:188},(_,i)=>Math.floor(i*(d.coverage.mainSkinVertices-1)/187));
    const gl=document.querySelector('canvas').getContext('webgl2'), info=gl.getExtension('WEBGL_debug_renderer_info');
    return {build:window.__tissueRuntime.build,ready:d.ready,engine:d.engine,adapter:d.adapter,webglRenderer:info?gl.getParameter(info.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER),errors:d.errors,coverage:d.coverage,settlement:d.settlement,last:d.samples.at(-1),resetCount:d.resetCount||0,sampleCount:d.samples.length,surface:d.sampleSurface(ids),clock:d.clock,performance:d.performance,status:document.querySelector('#tissueStatus').textContent};
  });
  const pixels=await page.screenshot({path:path.join(output,label+'.png')});
  return {label,notes,elapsedSeconds:(Date.now()-captureStarted)/1000,pixelSha256:createHash('sha256').update(pixels).digest('hex'),state};
}
async function sampleAdvance(page,prior,count,timeout=45000){
  await page.waitForFunction(({prior,count})=>window.__fullHeadFEM.errors.length || window.__fullHeadFEM.samples.length>=prior+count,{prior,count},{timeout});
}
try{
  const specs=software && (strict||demoOnly) ? [
    {name:'staged-cpu-software',url:'http://127.0.0.1:8841/neutral-tissue.html?backend=cpu',record:demoOnly}
  ] : software ? [
    {name:'baseline-cpu-software',url:'http://127.0.0.1:8840/neutral-tissue.html?backend=cpu',record:false},
    {name:'staged-cpu-software',url:'http://127.0.0.1:8841/neutral-tissue.html?backend=cpu',record:true}
  ] : [
    {name:'baseline-gpu',url:'http://127.0.0.1:8840/neutral-tissue.html',record:false},
    {name:'staged-gpu',url:'http://127.0.0.1:8841/neutral-tissue.html',record:true},
    {name:'staged-cpu',url:'http://127.0.0.1:8841/neutral-tissue.html?backend=cpu',record:true}
  ];
  for(const spec of specs){
    const context=await browser.newContext({viewport,deviceScaleFactor:1,recordVideo:spec.record?{dir:path.join(output,'recordings'),size:viewport}:undefined});
    const page=await context.newPage(), errors=[], failures=[], requests=[];
    page.on('pageerror',e=>errors.push(e.message));page.on('requestfailed',r=>failures.push({url:r.url(),error:r.failure()}));page.on('response',r=>{if(r.status()>=400 && !r.url().endsWith('favicon.ico'))failures.push({url:r.url(),status:r.status()})});
    await context.route('**/*',route=>{const u=new URL(route.request().url());requests.push(u.pathname);if(u.hostname!=='127.0.0.1')return route.abort();if(u.pathname==='/favicon.ico')return route.fulfill({status:204});return route.continue();});
    const started=Date.now();
    captureStarted=started;
    const url=new URL(spec.url);url.searchParams.set('expectedBuild',selected);
    const result={name:spec.name,url:url.href,viewport,deviceScaleFactor:1,selectedRuntime:selected,stagedSourceSha256:source,cases:[],errors,failures,requests};
    try{
      await page.goto(url.href);
      await page.waitForFunction(()=>window.__fullHeadFEM?.ready || window.__fullHeadFEM?.errors?.length,{}, {timeout:45000});
      assert.equal(await page.evaluate(()=>window.__tissueRuntime.build),selected);
      assert.equal(await page.evaluate(()=>window.__fullHeadFEM.errors.length),0);
      if(software){
        const renderer=await page.evaluate(()=>{const gl=document.querySelector('canvas').getContext('webgl2'),ext=gl.getExtension('WEBGL_debug_renderer_info');return ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER)});
        result.softwareRenderer=renderer;
        assert(/SwiftShader/i.test(renderer),'Software-rendering guard failed: '+renderer);
        result.softwareRendererVerified=true;
        console.log(JSON.stringify({name:spec.name,softwareRendererVerified:true,renderer}));
      }
      await page.locator('[data-view="three-quarter"]').click();
      await page.waitForTimeout(250);
      result.cases.push(await snapshot(page,spec.name+'-neutral','Rest state; no solve requested, static initial viewer.'));
      if(spec.record){
        await page.evaluate(({build,source,mode})=>{const badge=document.createElement('div');badge.textContent='LOCAL CONSOLIDATED Â· '+mode+' Â· runtime '+build+' Â· source '+source.slice(0,12);badge.style.cssText='position:fixed;top:10px;right:10px;max-width:calc(100vw - 320px);background:#10242ce8;color:#d8f0eb;padding:10px;font:11px system-ui;z-index:100;pointer-events:none';document.body.append(badge)},{build:selected,source,mode:software?'CPU solver + software WebGL':spec.name.endsWith('cpu')?'CPU fallback':'WebGPU'});
        result.videoStartSeconds=(Date.now()-started)/1000;
      }
      // Same authored controls/settings, checked by actual solver samples.
      let prior=await page.evaluate(()=>window.__fullHeadFEM.samples.length);
      await page.locator('#fat').fill('25');await page.locator('#sag').fill('25');
      await sampleAdvance(page,prior,1);
      await page.waitForFunction(()=>window.__fullHeadFEM.errors.length || (window.__fullHeadFEM.samples.at(-1)?.applied.fatPercent===25 && window.__fullHeadFEM.samples.at(-1)?.applied.sagPercent===25),{}, {timeout:45000});
      result.cases.push(await snapshot(page,spec.name+'-controls','Fat 25%, softness 25%, gravity on; actual applied parameters, settlement/cap reported.'));
      if(demoOnly)await page.waitForTimeout(3000);
      prior=await page.evaluate(()=>window.__fullHeadFEM.samples.length);
      await page.evaluate(()=>{window.__fullHeadFEM.grabPoint([0,0.06,0.06],[0,-0.003,0.002]);window.__fullHeadFEM.resume()});
      await page.waitForFunction(p=>window.__fullHeadFEM.errors.length || (window.__fullHeadFEM.samples.length>p && window.__fullHeadFEM.samples.at(-1)?.grabbed && window.__fullHeadFEM.samples.at(-1)?.integration==='dynamic'),prior,{timeout:45000});
      await page.evaluate(()=>window.__fullHeadFEM.pause());
      result.cases.push(await snapshot(page,spec.name+'-pull','Actual dynamic grab flag after 3 mm downward/2 mm outward skin pull.'));
      if(demoOnly)await page.waitForTimeout(3000);
      await page.locator('#release').click();prior=await page.evaluate(()=>window.__fullHeadFEM.samples.length);
      await page.waitForFunction(p=>window.__fullHeadFEM.errors.length || (window.__fullHeadFEM.samples.length>p && window.__fullHeadFEM.samples.at(-1)?.grabbed===false),prior,{timeout:45000});
      result.cases.push(await snapshot(page,spec.name+'-release','Release applied; check actual grabbed flag.'));
      await page.locator('#anatomy').check();await page.locator('[data-view="profile"]').click();await page.waitForTimeout(300);
      result.cases.push(await snapshot(page,spec.name+'-anatomy','Licensed static anatomy in shared coordinates.'));
      if(demoOnly)await page.waitForTimeout(8000);
      await page.locator('#anatomy').uncheck();const resets=await page.evaluate(()=>window.__fullHeadFEM.resetCount||0);prior=await page.evaluate(()=>window.__fullHeadFEM.samples.length);await page.locator('#resetShape').click();
      await page.waitForFunction(({prior,resets})=>window.__fullHeadFEM.errors.length || (window.__fullHeadFEM.samples.length>prior && (window.__fullHeadFEM.resetCount||0)>resets && window.__fullHeadFEM.samples.at(-1)?.applied.fatPercent===0 && window.__fullHeadFEM.samples.at(-1)?.applied.sagPercent===0 && !window.__fullHeadFEM.samples.at(-1)?.grabbed),{prior,resets},{timeout:45000});
      result.cases.push(await snapshot(page,spec.name+'-reset','Reset clears parameters and drag; actual solved state follows.'));
      prior=await page.evaluate(()=>window.__fullHeadFEM.samples.length);await page.locator('#running').check();
      await page.waitForFunction(p=>window.__fullHeadFEM.errors.length || (window.__fullHeadFEM.samples.length>p && window.__fullHeadFEM.samples.at(-1)?.integration==='dynamic' && window.__fullHeadFEM.samples.at(-1)?.simulatedSeconds>0),prior,{timeout:45000});
      await page.locator('#running').uncheck();await page.locator('[data-view="front"]').click();await page.waitForTimeout(200);
      result.cases.push(await snapshot(page,spec.name+'-dynamic','Actual fixed-time dynamic integration with gravity after completed reset.'));
      for(const c of result.cases){assert.deepEqual(c.state.errors,[]);if(c.state.last)assert(c.state.last.minJ>0.2 && Number.isFinite(c.state.last.residualN));}
      assert.deepEqual(errors,[]);assert.deepEqual(failures,[]);
      result.passed=true;
    }catch(e){result.passed=false;result.failure=String(e.stack||e);console.log(JSON.stringify({name:spec.name,error:result.failure}));}
    finally{
      result.videoEndSeconds=(Date.now()-started)/1000;
      const video=spec.record?page.video():null;
      await context.close();if(video)result.recording=await video.path();
      results.push(result);fs.writeFileSync(path.join(output,'browser-results.json'),JSON.stringify(results,null,2));
      console.log(JSON.stringify({name:spec.name,passed:result.passed,seconds:result.videoEndSeconds,engine:result.cases[0]?.state.engine,cases:result.cases.map(c=>({label:c.label,minJ:c.state.last?.minJ,residual:c.state.last?.residualN,pixelSha256:c.pixelSha256})),recording:result.recording}));
    }
    if(software && !result.softwareRendererVerified)break;
  }
}finally{await browser.close();}
const a=results.find(r=>r.name=== (software?'baseline-cpu-software':'baseline-gpu')),b=results.find(r=>r.name===(software?'staged-cpu-software':'staged-gpu'));
const compare={selectedRuntime:selected,mode:software?'CPU solver with SwiftShader software rendering':'hardware GPU/browser',phase:demoOnly?'post-comparison staged recording':strict?'strict staged transition check':'initial comparison',neutralPixelIdentical:a&&b?a.cases[0]?.pixelSha256===b.cases[0]?.pixelSha256:null,baselinePassed:a?.passed,stagedPassed:b?.passed,hardwareGPUVerified:!software && b?.passed,stagedCPUPassed:results.find(r=>r.name==='staged-cpu')?.passed};
fs.writeFileSync(path.join(output,'baseline-comparison.json'),JSON.stringify(compare,null,2));console.log(JSON.stringify(compare));
if(results.some(r=>!r.passed))process.exitCode=1;
