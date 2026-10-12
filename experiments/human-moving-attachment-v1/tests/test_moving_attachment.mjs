import assert from 'node:assert/strict';
import {verifyParameterizedTriangleContact} from '../../../procedural/mechanics/regional-reference/workspace/human-preferred-v53-anatomical-assembly-20261011/resumed-source/parametric_triangle_contact.mjs';
import {certifyExactMovingAttachment as certify,sourceRepresentationWithinBound} from '../src/exact_moving_attachment_contact.mjs';
import {LayeredFEM} from '../../../procedural/mechanics/regional-reference/workspace/human-mechanics-20261010/reference-source/src/layered_fem.js';
import {StaticBoneSweptBVH,extractCompleteFEMBoundary} from '../src/static_bone_swept_bvh.mjs';
import {createVerifiedMovingBondedPatch as create,verifyMovingBondedState,verifyMovingBondedTrajectory} from '../src/moving_bonded_patch.mjs';
import {createVerifiedDistributedBondedPatch} from '../../../procedural/mechanics/regional-reference/workspace/human-preferred-v53-anatomical-assembly-20261011/resumed-source/parameterized_bonded_patch.mjs';
import {GuardedStaticBoneLayeredFEM} from '../src/guarded_static_bone_fem.mjs';
const checks=[];
function test(name,f){f();checks.push(name);}
const bone=[[0,0,0],[2,0,0],[0,2,0]],root=i=>({vertices:[bone[i],bone[i],bone[i]],weights:[1,0,0]}),spec={0:root(0)};
const start=[[0,0,0],[.25,.25,1],[.5,.25,1]],end=[[0,0,0],[.375,.25,1.5],[.5,.375,1.5]];
test('single-root safe linear motion',()=>assert.equal(certify(start,end,bone,spec).certified,true));
test('shared-root edge safe linear motion',()=>assert.equal(certify([bone[0],bone[1],[.5,.5,1]],[bone[0],bone[1],[.75,.5,2]],bone,{0:root(0),1:root(1)}).certified,true));
test('coplanar outward motion requires sum of weak planes',()=>{const q=certify([bone[0],[-1,0,0],[0,-1,0]],[bone[0],[-2,0,0],[0,-2,0]],bone,spec);assert.equal(q.certified,true);assert.ok(q.combinedAxes>=2);});
test('free vertex crosses into bone',()=>assert.equal(certify(start,[bone[0],[.25,.25,-1],[.5,.25,1]],bone,spec).certified,false));
test('endpoints clear but intermediate root-face crossing',()=>{const a=[bone[0],[3,-1,1],[3,-1,-1]],b=[bone[0],[-1,3,1],[-1,3,-1]];assert.equal(certify(a,a,bone,spec).certified,true);assert.equal(certify(b,b,bone,spec).certified,true);assert.equal(certify(a,b,bone,spec).certified,false);});
test('root-root chord cannot cross unrelated triangle',()=>{const a=[[-1,.5,0],[3,.5,0],[0,0,1]],s={0:{vertices:[a[0],a[0],a[0]],weights:[1,0,0]},1:{vertices:[a[1],a[1],a[1]],weights:[1,0,0]}};assert.equal(certify(a,a,bone,s).certified,false);});
test('exact positive gap below binary64 normal range',()=>{const b=[[-Number.MIN_VALUE,0,0],[-Number.MIN_VALUE,1,0],[-Number.MIN_VALUE,0,1]],q=certify(start,end,b,spec);assert.equal(q.certified,true);assert.equal(q.positiveSeparation,true);});
test('zero gap unregistered contact rejected',()=>{const b=[[0,0,0],[0,1,0],[0,0,1]];assert.equal(certify(start,end,b,{}).certified,false);});
const bary={vertices:bone,weights:[1,1,1]},p=[2/3,2/3,0],bs={0:bary},ba=[p,[.75,.5,1],[.5,.75,1]],bz=[p,[.75,.5,2],[.5,.75,2]];
test('normalized rational barycentric root',()=>{assert.equal(sourceRepresentationWithinBound(p,bary),true);assert.equal(certify(ba,bz,bone,bs).certified,true);});
test('nonunit and huge finite weight sums',()=>{assert.equal(certify(ba,bz,bone,{0:{vertices:bone,weights:[1e308,1e308,1e308]}}).certified,true);});
test('bad root representation rejected',()=>{const a=structuredClone(ba);a[0][2]=3e-15;assert.equal(certify(a,a,bone,bs).certified,false);});
test('moving registered root rejected',()=>{const a=structuredClone(end);a[0][0]=Number.MIN_VALUE;assert.equal(certify(start,a,bone,spec).certified,false);});
test('invalid negative zero-sum and indexed sources rejected',()=>{for(const weights of [[-1,1,1],[0,0,0],[NaN,1,1]])assert.throws(()=>certify(start,end,bone,{0:{vertices:bone,weights}}));assert.throws(()=>certify(start,end,bone,{3:root(0)}));});
test('nonfinite malformed inputs rejected',()=>{const a=structuredClone(start);a[1][2]=Infinity;assert.throws(()=>certify(a,end,bone,spec));assert.throws(()=>certify(start.slice(1),end,bone,spec));});
test('degenerate endpoint or bone rejected',()=>{assert.equal(certify([bone[0],[0,0,1],[0,0,2]],end,bone,spec).certified,false);assert.equal(certify(start,end,[[0,0,0],[1,0,0],[2,0,0]],spec).certified,false);});
function setup(extraBones=[]){const collider=new StaticBoneSweptBVH({bones:[{id:'b',vertices:bone,triangles:[[0,1,2]]},...extraBones]}),x=Float64Array.from(start.flat()),z=Float64Array.from(end.flat()),fixed=Uint8Array.from([1,0,0]),targets=x.slice(),faces=[[0,1,2]],args={collider,positions:x,boundaryFaces:faces,fixed,targets,rootRecords:[{node:0,boneId:'b',boneVertex:0}],rootPatchFaces:[],minimumDistance:1e-5};return{...args,x,z,faces,token:create(args)};}
const s=setup(),options={minimumDistance:1e-5,movingBondedPatch:s.token,maxDepth:0,maxIntervalsPerPair:1};
test('opt-in BVH certifies every attachment candidate',()=>{const q=s.collider.certifyStep(s.x,s.z,s.faces,options);assert.equal(q.certified,true);assert.equal(q.movingAttachmentPairs,1);assert.equal(q.fullUniformMarginClaim,false);assert.equal(q.intervals,0);});
test('old default guard still rejects contact and old collar rule remains',()=>{assert.equal(s.collider.certifyStep(s.x,s.z,s.faces,{minimumDistance:1e-5,maxDepth:0}).certified,false);assert.throws(()=>createVerifiedDistributedBondedPatch({...s,positions:s.x,boundaryFaces:s.faces}));});
test('forged or cloned tokens rejected',()=>{for(const token of [{},structuredClone(s.token)])assert.equal(s.collider.certifyStep(s.x,s.z,s.faces,{...options,movingBondedPatch:token}).certified,false);});
test('topology reordering and collider changes rejected',()=>{assert.equal(s.collider.certifyStep(s.x,s.z,[[0,2,1]],options).certified,false);assert.equal(setup().collider.certifyStep(s.x,s.z,s.faces,options).certified,false);});
test('changed margin rejected',()=>assert.equal(s.collider.certifyStep(s.x,s.z,s.faces,{...options,minimumDistance:2e-5}).certified,false));
test('altered pins targets and roots rejected',()=>{s.fixed[1]=1;assert.equal(s.collider.certifyStep(s.x,s.z,s.faces,options).certified,false);s.fixed[1]=0;s.targets[0]=1e-16;assert.equal(s.collider.certifyStep(s.x,s.z,s.faces,options).certified,false);s.targets[0]=0;const x=s.x.slice();x[0]=1e-16;assert.equal(s.collider.certifyStep(x,s.z,s.faces,options).certified,false);assert.equal(verifyMovingBondedState(s.token,{positions:s.x,fixed:Uint8Array.of(0,0,0),targets:s.targets}).valid,false);});
test('public summary mutations grant no authority',()=>{assert.equal(Object.isFrozen(s.token),true);assert.equal(s.collider.certifyStep(s.x,s.z,s.faces,{...options,movingBondedPatch:{summary:{...s.token.summary,movingPairsBlanketExempted:100}}}).certified,false);});
test('collider method tampering invalidates context',()=>{s.collider.getStaticTriangle=()=>bone;assert.equal(s.collider.certifyStep(s.x,s.z,s.faces,options).certified,false);delete s.collider.getStaticTriangle;assert.equal(s.collider.certifyStep(s.x,s.z,s.faces,options).certified,true);});
test('unrelated incident-face collision is individually rejected',()=>{const u=setup([{id:'obstacle',vertices:[[0,0,2],[2,0,2],[0,2,2]],triangles:[[0,1,2]]}]),z=u.z.slice();z[5]=3;z[8]=3;const q=u.collider.certifyStep(u.x,z,u.faces,{...options,movingBondedPatch:u.token});assert.equal(q.certified,false);assert.equal(q.pair.boneId,'obstacle');});
test('root-incident tiny-gap pair receives exact proof',()=>{const u=setup([{id:'tiny',vertices:[[-Number.MIN_VALUE,0,0],[-Number.MIN_VALUE,1,0],[-Number.MIN_VALUE,0,1]],triangles:[[0,1,2]]}]),q=u.collider.certifyStep(u.x,u.z,u.faces,{...options,movingBondedPatch:u.token});assert.equal(q.certified,true);assert.equal(q.movingAttachmentPositiveSeparationPairs,1);assert.equal(q.movingAttachmentPairs,2);});
test('all non-root moving faces retain full margin',()=>{const x=Float64Array.from([...start.flat(),0,0,5e-6,.1,0,5e-6,0,.1,5e-6]),fixed=Uint8Array.of(1,0,0,0,0,0),faces=[[0,1,2],[3,4,5]],token=create({...s,positions:x,boundaryFaces:faces,fixed,targets:x.slice()}),q=s.collider.certifyStep(x,x,faces,{...options,movingBondedPatch:token});assert.equal(q.certified,false);assert.equal(q.pair.tissueFace,1);assert.equal(q.reason,'uncertified-triangle-segment');});
test('source declarations snapshots resist caller mutation',()=>{const roots=[{node:0,boneId:'b',boneVertex:0}],u=setup(),token=create({...u,positions:u.x,boundaryFaces:u.faces,rootRecords:roots});roots[0].boneVertex=2;assert.equal(u.collider.certifyStep(u.x,u.z,u.faces,{...options,movingBondedPatch:token}).certified,true);});
test('literal bone root mismatch and invalid pin context rejected',()=>{const x=s.x.slice();x[0]=1e-16;assert.throws(()=>create({...s,positions:x,boundaryFaces:s.faces,targets:x}));assert.throws(()=>create({...s,positions:s.x,boundaryFaces:s.faces,fixed:Uint8Array.of(0,0,0)}));});
test('FEM wrapper loads and retains inherited original step restriction',()=>assert.equal(typeof GuardedStaticBoneLayeredFEM,'function'));
test('opt-in FEM wrapper checks pin topology and initialGuess paths',()=>{
 const model={schema:'cybr-layered-fem-v1',nodes:4,tetrahedra:1},arrays={rest:Float64Array.from([0,0,0,1,0,1,0,1,1,0,0,1]),nodes:Uint32Array.of(0,1,2,3),gradients:Float64Array.from([0,0,-1,1,0,0,0,1,0,-1,-1,1]),material:Float64Array.of(1/6,15,120),mass:Float64Array.of(1,1,1,1)},g=new GuardedStaticBoneLayeredFEM(model,arrays),base=new LayeredFEM(model,arrays);
 for(const v of[g,base])v.pin(0,[0,0,0]);
 const faces=extractCompleteFEMBoundary(g.nodes,4),token=create({collider:s.collider,positions:g.x,boundaryFaces:faces,fixed:g.fixed,targets:g.target,rootRecords:[{node:0,boneId:'b',boneVertex:0}],minimumDistance:1e-5});
 g.setStaticBoneGuard(s.collider,{...options,movingBondedPatch:token});
 g.fixed[1]=1;assert.throws(()=>g.step(Infinity,{iterations:0}),/pin-context-changed/);g.fixed[1]=0;
 g.target[0]=1e-16;assert.throws(()=>g.step(Infinity,{iterations:0}),/fixed-node-moved/);g.target[0]=0;
 [g.nodes[1],g.nodes[2]]=[g.nodes[2],g.nodes[1]];assert.throws(()=>g.step(Infinity,{iterations:0}),/topology-changed/);[g.nodes[1],g.nodes[2]]=[g.nodes[2],g.nodes[1]];
 const guess=g.x.slice();for(let i=5;i<guess.length;i+=3)guess[i]+=.1;
 const a=g.step(Infinity,{iterations:0,initialGuess:guess}),b=base.step(Infinity,{iterations:0,initialGuess:guess});
 assert.equal(a.boundaryFeasibility.certified,true);assert.deepEqual(g.x,base.x);assert.deepEqual(g.g,base.g);assert.equal(a.energy,b.energy);assert.equal(g.pcg,base.pcg);
});
test('zero requested margin still bounds source-coordinate broad phase error',()=>{
 const x=Float64Array.from([[2/3,2/3,1e-15],[.75,.5,1],[.5,.75,1]].flat()),fixed=Uint8Array.of(1,0,0),faces=[[0,1,2]],token=create({collider:s.collider,positions:x,boundaryFaces:faces,fixed,targets:x.slice(),rootRecords:[{node:0,boneId:'b',boneFace:0,weights:[1,1,1]}],minimumDistance:0});
 assert.equal(s.collider.findCandidates(x,x,faces,{minimumDistance:0}).pairs.length,0);
 const q=s.collider.certifyStep(x,x,faces,{movingBondedPatch:token,minimumDistance:0});assert.equal(q.certified,true);assert.equal(q.movingAttachmentPairs,1);
});
test('duplicate roots invalid bone features and out-of-bound roots rejected',()=>{
 for(const rootRecords of [[{node:0,boneId:'b',boneVertex:0},{node:0,boneId:'b',boneVertex:0}],[{node:0,boneId:'missing',boneVertex:0}],[{node:0,boneId:'b',boneVertex:99}],[{node:0,boneId:'b',boneVertex:0,boneFace:0}],[{node:0,boneId:'b',boneFace:0,weights:[0,0,0]}]])assert.throws(()=>create({...s,positions:s.x,boundaryFaces:s.faces,rootRecords}));
 assert.equal(sourceRepresentationWithinBound([1.5e-15,1.5e-15,0],root(0)),false);
});
test('moving-pair traversal budgets fail closed',()=>{
 assert.equal(s.collider.certifyStep(s.x,s.z,s.faces,{...options,maxBoxTests:1}).certified,false);
});
let sampledCertified=0,sampledChecks=0;
test('deterministic adversarial corroboration at nine exact dyadic times',()=>{
 let seed=587341;const rnd=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return(seed%9)-4;};
 for(let k=0;k<1000;k++){
  const a=[bone[0],[rnd(),rnd(),rnd()],[rnd(),rnd(),rnd()]],b=[bone[0],[rnd(),rnd(),rnd()],[rnd(),rnd(),rnd()]],q=certify(a,b,bone,spec);if(!q.certified)continue;sampledCertified++;
  for(let t=0;t<=8;t++){const v=a.map((p,i)=>p.map((x,d)=>(x*(8-t)+b[i][d]*t)/8)),r=verifyParameterizedTriangleContact(v,bone,spec);if(r.reason==='degenerate-triangle')continue;assert.equal(r.valid,true,JSON.stringify({a,b,t,q,r}));sampledChecks++;}
 }
 assert.ok(sampledCertified>25);assert.ok(sampledChecks>200);
});
console.log(JSON.stringify({passed:checks.length,checks,corroboration:{generatedMotions:1000,certifiedMotions:sampledCertified,exactDyadicSampleChecks:sampledChecks,proofUsesSamples:false}},null,2));
