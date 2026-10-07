// Exercise the actual selected Web Worker module without a browser or network.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const root=path.resolve(new URL('..',import.meta.url).pathname.replace(/^\/([A-Za-z]:)/,'$1'));
const pointer=process.env.TISSUE_CANDIDATE==='1'?'runtime-candidate.json':'runtime-current.json';
const runtime=JSON.parse(fs.readFileSync(path.join(root,'output',pointer),'utf8'));
assert.equal(runtime.build,process.env.TISSUE_BUILD||runtime.build);
const moduleBase=path.join(root,path.dirname(runtime.entry));
const model=JSON.parse(fs.readFileSync(path.join(root,runtime.tissueBase,'model.json'),'utf8'));
const arrays={};
for(const [name,asset] of Object.entries(model.assets)){
  const raw=fs.readFileSync(path.join(root,asset.url));
  const Type=['nodes','faces','layers','groups'].includes(name)?Uint32Array:Float64Array;
  arrays[name]=new Type(raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength));
}
let reply;
globalThis.self={postMessage:value=>{reply=value}};
await import(pathToFileURL(path.join(moduleBase,'tissue_cpu_worker.js')));
function send(data){reply=null;self.onmessage({data:{id:1,...data}});assert(reply,'Worker did not reply');assert(!reply.error,reply.error);return reply;}
function init(){return send({type:'init',model,arrays:Object.fromEntries(Object.entries(arrays).map(([k,v])=>[k,v.slice()]))});}
const results=[];
init();
const options={dt:1/120,newton:2,cg:24};
const params={gravity:1,fatPercent:25,sagPercent:30};
const together=send({type:'step',parameters:params,options:{...options,substeps:2}});
init();send({type:'step',parameters:params,options:{...options,substeps:1}});
const separate=send({type:'step',parameters:params,options:{...options,substeps:1}});
let substepError=0;for(let i=0;i<together.packed.length;i++)substepError=Math.max(substepError,Math.abs(together.packed[i]-separate.packed[i]));
assert.equal(substepError,0);assert.equal(together.stats.simulatedSeconds,1/60);
assert(together.stats.minJ>0.2);assert(together.packed.every(Number.isFinite));
results.push({mode:'dynamic',substepError,stats:together.stats});
init();const staticResult=send({type:'step',parameters:{gravity:1,fatPercent:10,sagPercent:10},options:{quasiStatic:true,newton:2,cg:48,substeps:1}});
assert.equal(staticResult.stats.simulatedSeconds,0);assert.equal(staticResult.stats.maxNodeSpeedMps,0);
assert(staticResult.stats.minJ>0.2);assert(staticResult.packed.every(Number.isFinite));
results.push({mode:'static',stats:staticResult.stats});
send({type:'reset'});const reset=send({type:'step',parameters:{gravity:0},options:{...options,substeps:1}});
assert(reset.packed.every(Number.isFinite));
const report={build:runtime.build,nodes:model.nodes,tetrahedra:model.tetrahedra,results,reset:reset.stats,limitations:'Bounded worker contract checks; these do not establish settlement, visual acceptance or realtime performance.'};
fs.mkdirSync(path.join(root,'output/verification'),{recursive:true});
fs.writeFileSync(path.join(root,'output/verification/offline-worker.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify(report));
