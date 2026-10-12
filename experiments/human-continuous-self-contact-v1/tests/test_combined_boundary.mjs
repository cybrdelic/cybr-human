import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {LayeredFEM} from '../../../procedural/mechanics/regional-reference/workspace/human-mechanics-20261010/reference-source/src/layered_fem.js';
import {ActiveFibreFEM} from '../../../procedural/mechanics/regional-reference/workspace/human-mechanics-20261010/src/active_fibre_fem.mjs';
import {StaticBoneSweptBVH,extractCompleteFEMBoundary} from '../../human-moving-attachment-v1/src/static_bone_swept_bvh.mjs';
import {certifyTissueSelfStep} from '../src/tissue_self_bvh.mjs';
import {createVerifiedMovingBondedPatch} from '../../human-moving-attachment-v1/src/moving_bonded_patch.mjs';
import {createCombinedBoundaryAuthority,CombinedBoundaryLayeredFEM as Combined,withCombinedBoundaryFeasibility} from '../src/combined_boundary_fem.mjs';

function fixture(count=1,spacing=3) {
  const rest=[],nodes=[],gradients=[],material=[],mass=[];
  for(let k=0;k<count;k++) {
    rest.push(k*spacing,0,0,k*spacing+1,0,0,k*spacing,1,0,k*spacing,0,1);
    nodes.push(k*4,k*4+1,k*4+2,k*4+3);
    gradients.push(-1,-1,-1,1,0,0,0,1,0,0,0,1);material.push(1/6,15,120);mass.push(1,1,1,1);
  }
  return [{schema:'cybr-layered-fem-v1',nodes:4*count,surface_nodes:4*count,tetrahedra:count},
    {rest:Float64Array.from(rest),nodes:Uint32Array.from(nodes),gradients:Float64Array.from(gradients),material:Float64Array.from(material),mass:Float64Array.from(mass)}];
}
const farBone={id:'far',vertices:[[20,20,20],[21,20,20],[20,21,20]],triangles:[[0,1,2]]};
const collider=(self={},bones=[farBone])=>createCombinedBoundaryAuthority(new StaticBoneSweptBVH({bones}),{self});
const make=(Class=Combined,count=1,spacing=3)=>new Class(...fixture(count,spacing));
const snapshot=s=>Object.fromEntries(Object.entries(s).filter(([,v])=>ArrayBuffer.isView(v)).map(([k,v])=>[k,v.slice()]));
const unchanged=(s,q)=>{for(const [k,v] of Object.entries(q))assert.deepEqual(s[k],v,k);};
const springs=(s,delta)=>{for(let n=0;n<s.mass.length;n++){const d=typeof delta==='function'?delta(n):delta;s.pointSprings.push({node:n,target:Array.from(s.rest.slice(3*n,3*n+3),(x,k)=>x+d[k]),k:1000});}};
const swapped=s=>Float64Array.from(s.x,(v,i)=>v+(i%3===0?(i<12?3:-3):0));

test('frozen physical source unchanged and PCG/multiply remain inherited',()=>{
  const hash=crypto.createHash('sha256').update(fs.readFileSync(new URL('../../../procedural/mechanics/regional-reference/workspace/human-mechanics-20261010/reference-source/src/layered_fem.js',import.meta.url))).digest('hex');
  assert.equal(hash,'8618d5517de196dd3dae905d598a69eaf14555682895c80dcfc994e2501dbc71');
  assert.equal(Combined.prototype.pcg,LayeredFEM.prototype.pcg);assert.equal(Combined.prototype.multiply,LayeredFEM.prototype.multiply);
  assert.throws(()=>withCombinedBoundaryFeasibility(class extends LayeredFEM {step(){}}),/exact recovered/);
});
test('combined class fails closed if absent, cleared, or replaced with bone-only authority',()=>{
  const s=make();assert.throws(()=>s.step(Infinity),/not-installed/);
  assert.throws(()=>s.setStaticBoneGuard(new StaticBoneSweptBVH({bones:[farBone]})),/verified combined authority/);
  s.setCombinedBoundaryGuard(collider());s.clearStaticBoneGuard();assert.throws(()=>s.step(Infinity),/not-installed/);
});
test('admission covers complete FEM faces and both proof reports',()=>{
  const s=make();s.setCombinedBoundaryGuard(collider());const r=s.lastBoundaryFeasibility;
  assert.equal(r.certified,true);assert.equal(r.completeFEMBoundary,true);assert.equal(r.solverIntegrated,true);
  assert.equal(r.bone.certified,true);assert.equal(r.tissueSelf.certified,true);assert.equal(r.tissueSelf.boundaryFaces,4);assert.equal(r.tissueSelf.checkedPairs,6);
});
test('clear motion retains unchanged original energy/gradient/positions/velocity',()=>{
  const a=make(),b=make(LayeredFEM);a.setCombinedBoundaryGuard(collider());
  for(const s of [a,b])springs(s,[.01,.02,.03]);
  const o={iterations:30,cgIterations:60,cgTolerance:1e-12,forceTolerance:1e-8},r=a.step(Infinity,o),q=b.step(Infinity,o);
  for(const k of ['x','velocity','g','diagonal'])assert.deepEqual(a[k],b[k],k);
  for(const k of Object.keys(q))if(k!=='elapsedMs')assert.deepEqual(r[k],q[k],k);
  assert(r.boundaryFeasibility.continuousTissueSelfContact);assert.equal(r.boundaryFeasibility.rejectedTrials,0);
});
test('initial guesses that exchange two disconnected tets reject before any state write',()=>{
  const s=make(Combined,2);s.setCombinedBoundaryGuard(collider());const q=snapshot(s);
  assert.throws(()=>s.step(Infinity,{iterations:0,initialGuess:swapped(s)}),/tissue-self-/);unchanged(s,q);
  assert(s.step(Infinity,{iterations:0}).boundaryFeasibility.certified);
  s.setCombinedBoundaryGuard(collider());assert(s.step(Infinity,{iterations:0}).boundaryFeasibility.certified);
});
test('fixed target writes also reject a full two-component exchange before state mutation',()=>{
  const s=make(Combined,2);s.setCombinedBoundaryGuard(collider());const x=swapped(s);
  for(let n=0;n<s.mass.length;n++)s.pin(n,Array.from(x.slice(3*n,3*n+3)));
  const q=snapshot(s);assert.throws(()=>s.step(Infinity,{iterations:0}),/tissue-self-/);unchanged(s,q);
});
test('fixed targets retain original precedence over the free initial guess',()=>{
  const a=make(),b=make(LayeredFEM);a.setCombinedBoundaryGuard(collider());
  for(const s of [a,b])s.pin(0,[.01,0,0]);const x=a.x.slice();x[0]=-100;
  const o={iterations:0,initialGuess:x},r=a.step(Infinity,o),q=b.step(Infinity,o);
  assert.deepEqual(a.x,b.x);assert.equal(r.energy,q.energy);assert(a.x[0]===.01);
});
test('every inherited backtracking candidate gets both filters, including clear-endpoint tunneling',()=>{
  const a=make(Combined,2),b=make(LayeredFEM,2);a.setCombinedBoundaryGuard(collider());
  for(const s of [a,b])springs(s,n=>[n<4?3:-3,0,0]);const old=a.x.slice();
  const o={iterations:1,cgIterations:80,cgTolerance:1e-12,forceTolerance:1e-9},r=a.step(Infinity,o);b.step(Infinity,o);
  assert(r.boundaryFeasibility.rejectedTrials>0);assert.equal(r.converged,false);
  const f=extractCompleteFEMBoundary(a.nodes,a.mass.length);
  assert(certifyTissueSelfStep(old,a.x,f).certified);assert(!certifyTissueSelfStep(old,b.x,f).certified);
  assert.equal(r.boundaryFeasibility.acceptedSegments,r.accepted);
});
test('bone filter still rejects its own tunneling even when tissue remains self-disjoint',()=>{
  const s=make(),bone={vertices:[[.18,.18,-.1],[.25,.18,-.1],[.18,.25,-.1]],triangles:[[0,1,2]]};
  s.setCombinedBoundaryGuard(collider({},[bone]));const x=Float64Array.from(s.x,(v,i)=>v-(i%3===2?2:0));
  assert.throws(()=>s.step(Infinity,{initialGuess:x,iterations:0}),/bone-/);
});
test('manual positions and diagnostic evaluate calls cannot create admission',()=>{
  const s=make();s.setCombinedBoundaryGuard(collider());s.x[0]+=.01;s.evaluate(s.x,Infinity,true);
  assert.throws(()=>s.step(Infinity,{iterations:0}),/positions-changed-outside/);
});
test('topology, rest and mass edits invalidate the bound context',()=>{
  for(const edit of [s=>{[s.nodes[1],s.nodes[2]]=[s.nodes[2],s.nodes[1]];},s=>{s.rest[0]+=.01;},s=>{s.mass[0]+=1;}]){
    const s=make();s.setCombinedBoundaryGuard(collider());edit(s);assert.throws(()=>s.step(Infinity,{iterations:0}),/changed-after-admission/);
  }
});
test('constraint mutation inside inherited evaluation rejects and rolls physical positions back',()=>{
  class Mutating extends LayeredFEM {evaluate(x,dt,blocks=true){const r=super.evaluate(x,dt,blocks);if(this.mutate)this.target[0]+=.1;return r;}}
  const s=make(withCombinedBoundaryFeasibility(Mutating));s.setCombinedBoundaryGuard(collider());const x=s.x.slice(),v=s.velocity.slice(),p=s.predict.slice();s.mutate=true;
  assert.throws(()=>s.step(Infinity,{iterations:0}),/fixed-mask-or-targets/);assert.deepEqual(s.x,x);assert.deepEqual(s.velocity,v);assert.deepEqual(s.predict,p);
  assert.throws(()=>s.step(Infinity,{iterations:0}),/readmission-required/);
});
test('inherited trial coordinate mutation is rejected rather than accepted under an earlier proof',()=>{
  class Mutating extends LayeredFEM {evaluate(x,dt,blocks=true){const r=super.evaluate(x,dt,blocks);if(!blocks)x[0]+=.01;return r;}}
  const s=make(withCombinedBoundaryFeasibility(Mutating));s.setCombinedBoundaryGuard(collider());springs(s,[.01,0,0]);const x=s.x.slice();
  assert.throws(()=>s.step(Infinity,{iterations:1}),/coordinates-or-topology-mutated/);assert.deepEqual(s.x,x);
});
test('independent coincident vertices do not acquire an adjacency exemption',()=>{
  const s=make(Combined,2,0),q=snapshot(s);assert.throws(()=>s.setCombinedBoundaryGuard(collider()),/tissue-self-/);unchanged(s,q);
});
test('proof budgets and unsupported topology relaxations fail closed',()=>{
  const s=make();assert.throws(()=>s.setCombinedBoundaryGuard(collider({maxPairs:1})),/tissue-self-pair-budget/);
  assert.throws(()=>collider({requireClosedEdges:false}),/Unsupported/);assert.throws(()=>s.setCombinedBoundaryGuard(collider({maxPairs:NaN})),/budget/);
});
test('caller mutation cannot change the copied self budget options',()=>{
  const o={maxPairs:6},c=collider(o);o.maxPairs=1;const s=make();s.setCombinedBoundaryGuard(c);assert(s.step(Infinity,{iterations:0}).boundaryFeasibility.certified);
});
test('active-fibre mechanics and metric remain inherited for standalone diagnostic evaluation',()=>{
  const C=withCombinedBoundaryFeasibility(ActiveFibreFEM),a=new C(...fixture(),[{a:0,b:1,stiffness:3,activation:.1}]),b=new ActiveFibreFEM(...fixture(),[{a:0,b:1,stiffness:3,activation:.1}]);
  a.setCombinedBoundaryGuard(collider());const x=Float64Array.from(a.x,(v,i)=>v+.001*Math.sin(i));
  assert.deepEqual(a.evaluate(x,Infinity,true),b.evaluate(x,Infinity,true));assert.deepEqual(a.g,b.g);assert.equal(C.prototype.pcg,ActiveFibreFEM.prototype.pcg);
});
test('forged public summaries, subclasses and changed bone methods cannot grant authority',()=>{
  const s=make();
  for(const token of [{kind:'combined-boundary-authority-v1'},structuredClone(collider())])assert.throws(()=>s.setCombinedBoundaryGuard(token),/verified combined authority/);
  class Forged extends StaticBoneSweptBVH {certifyStep(){return {certified:true,boxTests:0,candidatePairs:0,intervals:0};}}
  assert.throws(()=>createCombinedBoundaryAuthority(new Forged({bones:[farBone]})),/unmodified/);
  const b=new StaticBoneSweptBVH({bones:[farBone]}),token=createCombinedBoundaryAuthority(b);s.setCombinedBoundaryGuard(token);
  b.certifyStep=()=>({certified:true});assert.throws(()=>s.step(Infinity,{iterations:0}),/implementation changed/);
});
test('real moving-root capability stays bound to the exact original bone collider',()=>{
  const model={schema:'cybr-layered-fem-v1',nodes:4,tetrahedra:1},arrays={rest:Float64Array.from([0,0,0,1,0,1,0,1,1,0,0,1]),nodes:Uint32Array.of(0,1,2,3),gradients:Float64Array.from([0,0,-1,1,0,0,0,1,0,-1,-1,1]),material:Float64Array.of(1/6,15,120),mass:Float64Array.of(1,1,1,1)},s=new Combined(model,arrays),base=new LayeredFEM(model,arrays);
  for(const q of [s,base])q.pin(0,[0,0,0]);
  const b=new StaticBoneSweptBVH({bones:[{id:'b',vertices:[[0,0,0],[2,0,0],[0,2,0]],triangles:[[0,1,2]]}]}),faces=extractCompleteFEMBoundary(s.nodes,4);
  const movingBondedPatch=createVerifiedMovingBondedPatch({collider:b,positions:s.x,boundaryFaces:faces,fixed:s.fixed,targets:s.target,rootRecords:[{node:0,boneId:'b',boneVertex:0}],minimumDistance:1e-5});
  s.setCombinedBoundaryGuard(createCombinedBoundaryAuthority(b),{movingBondedPatch,minimumDistance:1e-5});
  const initialGuess=s.x.slice();for(let i=5;i<initialGuess.length;i+=3)initialGuess[i]+=.1;
  const result=s.step(Infinity,{iterations:0,initialGuess}),r=base.step(Infinity,{iterations:0,initialGuess});
  assert(result.boundaryFeasibility.certified);assert.deepEqual(s.x,base.x);assert.equal(result.energy,r.energy);
  assert(result.boundaryFeasibility.initialAdmission.bone.exactMovingAttachmentContact);
});
test('nonprimitive solver options cannot run side effects after certification',()=>{
  const s=make();s.setCombinedBoundaryGuard(collider());const before=snapshot(s),last=s.last;
  const iterations={valueOf(){s.target[0]+=.01;return 0;}};
  assert.throws(()=>s.step(.01,{iterations,initialGuess:Float64Array.from(s.x,(v,i)=>v+(i%3===0?.1:0))}),/finite primitive/);
  unchanged(s,before);assert.equal(s.last,last);
});
test('replacing physical buffers inside evaluation restores their original references and values',()=>{
  for(const field of ['x','velocity','predict']){
    class Mutating extends LayeredFEM {evaluate(x,dt,blocks=true){const r=super.evaluate(x,dt,blocks);if(this.mutate){this[field]=new Float64Array(1);throw Error('inherited deliberate exception');}return r;}}
    const s=make(withCombinedBoundaryFeasibility(Mutating));s.setCombinedBoundaryGuard(collider());
    const refs={x:s.x,velocity:s.velocity,predict:s.predict},before={x:s.x.slice(),velocity:s.velocity.slice(),predict:s.predict.slice()},last=s.last;s.mutate=true;
    assert.throws(()=>s.step(Infinity,{iterations:0}),/inherited deliberate/);
    for(const k of Object.keys(refs)){assert.equal(s[k],refs[k]);assert.deepEqual(s[k],before[k]);}
    assert.equal(s.last,last);assert.equal(s.lastBoundaryFeasibility.certified,false);assert.throws(()=>s.step(Infinity,{iterations:0}),/readmission-required/);
  }
});
test('admission options cannot mutate topology between face extraction and binding',()=>{
  const s=make(Combined,2),before=s.nodes.slice();
  assert.throws(()=>s.setCombinedBoundaryGuard(collider(),{get minimumDistance(){s.nodes.set([0,1,2,3],4);return 0;}}),/accessors/);
  assert.deepEqual(s.nodes,before);assert.throws(()=>s.step(Infinity,{iterations:0}),/not-installed/);
});
test('time-step and step-option accessors cannot execute side effects in a transaction',()=>{
  const s=make();s.setCombinedBoundaryGuard(collider());const before=snapshot(s);
  assert.throws(()=>s.step({valueOf(){s.target[0]+=1;return .01;}},{iterations:0}),/time step/);
  assert.throws(()=>s.step(Infinity,{get iterations(){s.target[0]+=1;return 0;}}),/accessors/);unchanged(s,before);
});
test('dynamic collider accessors cannot swap validation and invocation implementations',()=>{
  const b=new StaticBoneSweptBVH({bones:[farBone]}),original=b.certifyStep;let reads=0;
  Object.defineProperty(b,'certifyStep',{get(){++reads;return reads<3||reads%2===0?original:()=>({certified:true,boxTests:0,candidatePairs:0,intervals:0});}});
  assert.throws(()=>createCombinedBoundaryAuthority(b),/unmodified/);assert.equal(reads,0);
  const clean=new StaticBoneSweptBVH({bones:[farBone]}),token=createCombinedBoundaryAuthority(clean),s=make();s.setCombinedBoundaryGuard(token);
  Object.defineProperty(clean,'certifyStep',{get(){throw Error('Accessor must never execute');}});
  assert.throws(()=>s.step(Infinity,{iterations:0}),/implementation changed/);
});
test('a proxy cannot borrow genuine private-field collider authority',()=>{
  const target=new StaticBoneSweptBVH({bones:[farBone]}),proxy=new Proxy(target,{}),s=make();
  assert.throws(()=>s.setCombinedBoundaryGuard(createCombinedBoundaryAuthority(proxy)),/private|receiver|member/i);
  assert.throws(()=>s.step(Infinity,{iterations:0}),/not-installed/);
});
