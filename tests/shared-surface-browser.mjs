// Diagnostic only. Run with an explicitly allocated GPU/browser slot.
import { chromium } from "playwright";
import fs from "node:fs";
import assert from "node:assert/strict";
const output="output/verification/shared-surface";fs.mkdirSync(output,{recursive:true});
const origin=process.env.TISSUE_BASE_URL||"http://127.0.0.1:8779";
const browser=await chromium.launch({channel:"msedge",headless:true,args:["--enable-unsafe-webgpu"]});
try {
 const page=await browser.newPage({viewport:{width:960,height:720}}),errors=[];
 page.on("pageerror",error=>errors.push(error.message));await page.goto(origin+"/tests/solver.html");
 const result=await page.evaluate(async()=>{
  const runtime=await(await fetch("/output/runtime-candidate.json")).json();
  const moduleBase=runtime.entry.slice(0,runtime.entry.lastIndexOf("/")+1);
  const THREE=await import("three");
  const {GPUHeadFEM}=await import(moduleBase+"full_head_fem_gpu.js");
  const {loadModel,loadSkin,readJSON,verifiedBytes}=await import(moduleBase+"model.js");
  const {prepareSkinNormals,rebuildSkinNormals}=await import(moduleBase+"skin_rest_normals.js");
  const {createSharedSurfaceProbe}=await import("/src/shared_surface_probe.js");
  const {model,arrays,bundle}=await loadModel(runtime.tissueBase);
  const skin=await loadSkin(model);skin.scene.updateMatrixWorld(true);
  const mesh=skin.scene.getObjectByName(model.skin_main_node);prepareSkinNormals(mesh);
  const shape=await readJSON(model.skin_shape_url),part=shape.parts[0];
  const delta=new Float32Array(await verifiedBytes(part.url,part.sha256,"probe shape"));
  const linear=new THREE.Matrix3().setFromMatrix4(mesh.matrixWorld.clone().invert()),point=new THREE.Vector3();
  const position=mesh.geometry.attributes.position;
  for(let i=0;i<delta.length;i+=3){point.fromArray(delta,i).applyMatrix3(linear);for(let d=0;d<3;d++)position.array[i+d]+=point.getComponent(d);}
  rebuildSkinNormals(mesh);
  const count=position.count,positions=new Float32Array(count*3),normals=positions.slice();
  const normalWorld=new THREE.Matrix3().getNormalMatrix(mesh.matrixWorld),normal=mesh.geometry.attributes.normal;
  for(let i=0;i<count;i++){
   point.fromBufferAttribute(position,i).applyMatrix4(mesh.matrixWorld).toArray(positions,i*3);
   point.fromBufferAttribute(normal,i).applyMatrix3(normalWorld).normalize().toArray(normals,i*3);
  }
  const smooth=bundle.smoothData.subarray(0,count*16);
  const engine=await GPUHeadFEM.create(model,arrays);let probe,depth,readback;
  const canvas=document.createElement("canvas");canvas.width=960;canvas.height=720;document.body.replaceChildren(canvas);
  const context=canvas.getContext("webgpu"),format=navigator.gpu.getPreferredCanvasFormat();context.configure({device:engine.device,format,alphaMode:"opaque"});
  const report={build:runtime.build,vertices:count,adapter:engine.adapterInfo,cases:[]};
  const cleanup=()=>{readback?.destroy();depth?.destroy();probe?.dispose();context.unconfigure();engine.dispose();skin.scene.traverse(item=>{item.geometry?.dispose();for(const m of [item.material].flat())m?.dispose();});};
  try {
   probe=await createSharedSurfaceProbe(engine,{positions,normals,smooth,indices:mesh.geometry.index.array},format);
   report.ownedBufferBytes=probe.ownedBufferBytes;
   depth=engine.device.createTexture({size:[960,720],format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT});
   readback=engine.device.createBuffer({size:count*32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   const camera=new THREE.PerspectiveCamera(35,960/720,.001,10);camera.position.set(.38,.04,.49);camera.lookAt(0,0,-.025);camera.updateMatrixWorld();
   const vp=new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix,camera.matrixWorldInverse);
   // Default gravity provides a checked initial preload; the solver's untouched
   // zero-force startup scalar buffer is not an independently evaluated J.
   for(const parameters of [{gravity:1},{gravity:1,fatPercent:25,sagPercent:25}]){
    const step=await engine.step(parameters,{dt:1/60,substeps:1,newton:2,cg:24});
    const encoder=engine.device.createCommandEncoder();probe.encodeTransfer(encoder);
    probe.encodeRender(encoder,context.getCurrentTexture().createView(),depth.createView(),vp.elements);
    encoder.copyBufferToBuffer(probe.outputBuffer,0,readback,0,count*32);engine.device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);const output=new Float32Array(readback.getMappedRange()).slice();readback.unmap();
    let maxPositionErrorM=0,maxNormalError=0;const packed=step.packed;
    for(let j=0;j<188;j++){
     const id=Math.floor(j*(count-1)/187),q=new THREE.Vector3().fromArray(positions,id*3),expected=new THREE.Vector3(),f=new Float64Array(9);
     for(let k=0;k<8;k++){
      const node=smooth[id*16+k],w=smooth[id*16+8+k];
      const nf=new THREE.Matrix3();for(let c=0;c<3;c++)for(let d=0;d<3;d++)nf.elements[c*3+d]=packed[(engine.N+c*engine.S+node)*4+d];
      const offset=q.clone().sub(new THREE.Vector3().fromArray(arrays.rest,node*3)).applyMatrix3(nf).add(new THREE.Vector3().fromArray(packed,node*4));expected.addScaledVector(offset,w);
      for(let d=0;d<9;d++)f[d]+=w*nf.elements[d];
     }
     const frame=new THREE.Matrix3().fromArray(f),n=new THREE.Vector3().fromArray(normals,id*3).applyMatrix3(frame.invert().transpose()).normalize();
     maxPositionErrorM=Math.max(maxPositionErrorM,expected.distanceTo(new THREE.Vector3().fromArray(output,id*8)));
     maxNormalError=Math.max(maxNormalError,n.distanceTo(new THREE.Vector3().fromArray(output,id*8+4)));
     if(output[id*8+3]!==1)throw Error("GPU validity gate rejected valid state");
    }
    report.cases.push({parameters,maxPositionErrorM,maxNormalError,stats:step.stats});
   }
   report.errors=engine.errors.slice();window.disposeProbe=cleanup;return report;
  }catch(error){cleanup();throw error;}
 });
 assert.deepEqual(errors,[]);assert.deepEqual(result.errors,[]);
 for(const sample of result.cases){assert(sample.maxPositionErrorM<2e-6);assert(sample.maxNormalError<2e-4);}
 // Capture proves the diagnostic device path rendered, not skin-material parity.
 await page.screenshot({path:output+"/diagnostic.png"});
 await page.evaluate(()=>window.disposeProbe());
 fs.writeFileSync(output+"/results.json",JSON.stringify({...result,limits:"Diagnostic Lambert renderer without eyes/diffusion/fine wrinkles; solver still performs full correctness/recovery readback. No speedup claimed."},null,2));
 console.log(JSON.stringify(result));
}finally{await browser.close();}
