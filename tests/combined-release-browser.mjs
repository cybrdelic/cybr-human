// Run only in a coordinated GPU slot against dist-candidate HTTP serving.
import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import fs from 'node:fs';
const origin=process.env.TISSUE_BASE_URL||'http://127.0.0.1:8783';
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']});
const watchdog=setTimeout(()=>{process.exitCode=1;void browser.close();},360000);
const receipt={contexts:[],limits:'Real desktop/mobile viewport interactions, not physical Android or displayed-FPS validation.'};
const dir='output/verification/combined-release';fs.mkdirSync(dir,{recursive:true});
try {
 for(const viewport of [{width:1280,height:900},{width:390,height:844}]) {
  const context=await browser.newContext({viewport,deviceScaleFactor:1});
  try {
   const page=await context.newPage(),errors=[],requests=[];
   page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
   await page.goto(origin+'/');
   await page.waitForFunction(()=>window.__fullHeadFEM?.ready,{},{timeout:45000});
   const state=()=>page.evaluate(()=>{const d=window.__fullHeadFEM;return {build:window.__tissueRuntime.build,release:d.releaseConfiguration,diffusion:d.skinDiffusion,
     clock:d.clock,bridge:d.bridge,pipeline:d.pipeline,last:d.samples.at(-1),recovery:d.recovery,resetCount:d.resetCount,errors:d.errors,solver:d.engine,solverFrames:d.solver.frames};});
   const row={viewport,startup:await state()};receipt.contexts.push(row);
   assert.equal(row.startup.release.regional,true);assert.equal(row.startup.release.refractiveEyes,true);assert.equal(row.startup.release.bridge,'pipeline');
   assert.equal(row.startup.diffusion.requestedCoverageSamples,4);
   assert(!requests.some(s=>s.endsWith('internal-anatomy.glb')),'Hidden anatomy stays lazy');
   for(const values of [{gravity:1,fatPercent:10,sagPercent:10},{gravity:0,fatPercent:0,sagPercent:0}]) {
    await page.evaluate(p=>window.__fullHeadFEM.setParameters(p),values);
    await page.waitForFunction(()=>window.__fullHeadFEM?.settlement?.converged,{},{timeout:40000});
    assert((await state()).last.minJ>0);
   }
   await page.locator('[data-view="front"]').click();await page.locator('#grab').check();
   const point=await page.evaluate(()=>window.__fullHeadFEM.projectPoint([.018,.018,.09]));
   await page.mouse.move(point.x,point.y);await page.mouse.down();await page.mouse.move(point.x+5,point.y+3,{steps:4});
   await page.waitForFunction(()=>window.__fullHeadFEM?.grabVersion,{},{timeout:3000});
   await page.waitForTimeout(350);await page.mouse.up();await page.locator('#release').click();
   await page.evaluate(()=>window.__fullHeadFEM.resume());
   if(row.startup.release.pipelineEligible)
    await page.waitForFunction(()=>window.__fullHeadFEM?.pipeline?.peakOutstanding===2,{},{timeout:15000});
   await page.locator('#resetShape').click();
   await page.waitForFunction(()=>window.__fullHeadFEM?.resetCount>=1 && window.__fullHeadFEM.samples.at(-1)?.minJ>0,{},{timeout:25000});
   row.afterReset=await state();
   // Actual loss during submission; late completion must not advance the new epoch.
   await page.evaluate(()=>{
    const d=window.__fullHeadFEM,e=d.solver;
    if(!e.device) throw Error('GPU loss review requires actual GPU');
    const method=d.releaseConfiguration.pipelineEligible?'submitStep':'step',native=e[method].bind(e);let calls=0;
    e[method]=async(...args)=>{const r=await native(...args);if(++calls===2){e.device.destroy();await e.device.lost;}return r;};
    d.resume();d.setParameters({gravity:1,fatPercent:5,sagPercent:5});
   });
   await page.waitForFunction(()=>window.__fullHeadFEM?.recovery?.actualSolverVelocity,{},{timeout:50000});
   await page.evaluate(()=>window.__fullHeadFEM.pause());row.recovered=await state();
   assert.equal(row.recovered.solver,'CPU worker implicit FEM');assert.equal(row.recovered.bridge.version.epoch,2);
   assert(row.recovered.recovery.positionsPreserved);assert.deepEqual(row.recovered.errors,[]);
   const frame=row.recovered.solverFrames;
   await page.evaluate(()=>window.__fullHeadFEM.setParameters({gravity:0,fatPercent:0,sagPercent:0}));
   await page.waitForFunction(frame=>window.__fullHeadFEM.solver.frames>frame && window.__fullHeadFEM.samples.at(-1)?.applied.gravity===0,frame,{timeout:40000});
   row.postRecoveryControl=await state();assert(row.postRecoveryControl.last.minJ>0);
   await page.screenshot({path:`${dir}/${viewport.width}-recovered.png`});
   // Choices must work through the public UI, preserving the selected build.
   await page.locator('#renderProfile').selectOption('classic');
   await page.waitForFunction(()=>window.__fullHeadFEM?.ready && window.__fullHeadFEM?.releaseConfiguration?.regional===false,{},{timeout:45000});
   await page.locator('#computeProfile').selectOption('serial');
   await page.waitForFunction(()=>window.__fullHeadFEM?.ready && window.__fullHeadFEM?.releaseConfiguration?.bridge==='versioned',{},{timeout:45000});
   row.classicSerial=await state();assert.equal(row.classicSerial.build,row.startup.build);assert.equal(row.classicSerial.diffusion.coverageSamples,0);
   await page.locator('a[href="rollback.html?rollback=1"]').click();
   await page.waitForFunction(()=>window.__fullHeadFEM?.ready && window.__tissueRuntime?.build==='4bc8b53a8f15b8b62cf6a86e',{},{timeout:45000});
   row.rollback=await state();assert.deepEqual(row.rollback.errors,[]);assert.deepEqual(errors,[]);
   fs.writeFileSync(`${dir}/results.json`,JSON.stringify(receipt,null,2));
  } finally {await context.close();}
 }
 console.log(JSON.stringify(receipt));
} finally {clearTimeout(watchdog);await browser.close();fs.writeFileSync(`${dir}/results.json`,JSON.stringify(receipt,null,2));}
