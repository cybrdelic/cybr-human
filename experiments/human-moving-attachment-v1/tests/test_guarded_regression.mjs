/** Bounded synthetic fixtures. No authored full-domain experiment. */
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {LayeredFEM} from '../../../procedural/mechanics/regional-reference/workspace/human-mechanics-20261010/reference-source/src/layered_fem.js';
import {ActiveFibreFEM} from '../../../procedural/mechanics/regional-reference/workspace/human-mechanics-20261010/src/active_fibre_fem.mjs';
import {StaticBoneSweptBVH,extractCompleteFEMBoundary} from '../src/static_bone_swept_bvh.mjs';
import {GuardedStaticBoneLayeredFEM,withStaticBoneFeasibility,BoundaryFeasibilityError} from '../src/guarded_static_bone_fem.mjs';
import {certifyTriangleStep} from '../../../procedural/mechanics/regional-reference/workspace/human-mechanics-20261010/src/prototypes/complete_boundary_contact.mjs';

const flat=a=>Float64Array.from(a.flat());
const shift=(a,s)=>a.map(p=>p.map((x,d)=>x+s[d]));
const points=v=>Array.from({length:v.length/3},(_,i)=>Array.from(v.slice(3*i,3*i+3)));
const bone=[[.18,.18,0],[.25,.18,0],[.18,.25,0]],face=[[0,1,2]];
const collider=(vertices=bone,extra=[])=>new StaticBoneSweptBVH({bones:[{id:'bone',vertices,triangles:face},...extra],leafSize:1});
function fixture() {
  const rest=flat([[0,0,.1],[1,0,.1],[0,1,.1],[0,0,1.1]]);
  return {model:{schema:'cybr-layered-fem-v1',nodes:4,surface_nodes:4,tetrahedra:1},arrays:{
    rest,nodes:Uint32Array.from([0,1,2,3]),gradients:Float64Array.from([-1,-1,-1,1,0,0,0,1,0,0,0,1]),
    material:Float64Array.from([1/6,15,120]),mass:Float64Array.from([1,1,1,1])}};
}
const make=(Class=GuardedStaticBoneLayeredFEM)=>{const f=fixture();return new Class(f.model,f.arrays);};
const snapshot=s=>Object.fromEntries(Object.entries(s).filter(([,v])=>ArrayBuffer.isView(v)).map(([k,v])=>[k,v.slice()]));
const assertSnapshot=(s,saved)=>{for(const [k,v] of Object.entries(saved))assert.deepEqual(s[k],v,k);};
function addTranslationSprings(s,delta=[0,0,-2]) {
  for(let n=0;n<4;n++)s.pointSprings.push({node:n,target:points(s.rest)[n].map((x,d)=>x+delta[d]),k:1000});
}

test('protected original solver remains byte-identical; PCG and constitutive methods are inherited',()=>{
  const hash=crypto.createHash('sha256').update(fs.readFileSync(new URL('../../../procedural/mechanics/regional-reference/workspace/human-mechanics-20261010/reference-source/src/layered_fem.js',import.meta.url))).digest('hex');
  assert.equal(hash,'8618d5517de196dd3dae905d598a69eaf14555682895c80dcfc994e2501dbc71');
  assert.equal(GuardedStaticBoneLayeredFEM.prototype.pcg,LayeredFEM.prototype.pcg);
  assert.equal(GuardedStaticBoneLayeredFEM.prototype.multiply,LayeredFEM.prototype.multiply);
  assert.throws(()=>withStaticBoneFeasibility(class extends LayeredFEM {step(){}}),/exact recovered/);
});

test('boundary extraction includes all exterior faces and removes only two-tet interfaces',()=>{
  const a=extractCompleteFEMBoundary(Uint32Array.from([0,1,2,3]),4);assert.equal(a.length,4);
  const b=extractCompleteFEMBoundary(Uint32Array.from([0,1,2,3,0,2,1,4]),5);assert.equal(b.length,6);
  assert.ok(!b.some(f=>f.toSorted().join(':')==='0:1:2'));
  assert.throws(()=>extractCompleteFEMBoundary(Uint32Array.from([0,1,2,3,3,2,1,0]),4),/Duplicate/);
  assert.throws(()=>extractCompleteFEMBoundary(Uint32Array.from([0,1,2,3,0,2,1,4,0,1,2,5]),6),/Nonmanifold/);
});

test('swept candidate discovery retains an interior crossing with disjoint endpoint boxes',()=>{
  const c=collider(),a=[[0,0,.1],[1,0,.1],[0,1,.1]],z=shift(a,[0,0,-.2]);
  assert.equal(c.findCandidates(a,a,face).pairs.length,0);assert.equal(c.findCandidates(z,z,face).pairs.length,0);
  const candidates=c.findCandidates(a,z,face);assert.equal(candidates.complete,true);assert.equal(candidates.pairs.length,1);
  assert.equal(c.certifyStep(a,z,face).certified,false);
});

test('clearance padding is outward and retains close but disjoint boxes',()=>{
  const c=collider(),a=[[0,0,.05],[1,0,.05],[0,1,.05]];
  assert.equal(c.findCandidates(a,a,face,{minimumDistance:.05}).pairs.length,1);
  assert.equal(c.certifyStep(a,a,face,{minimumDistance:.05}).certified,false);
  assert.equal(c.certifyStep(shift(a,[0,0,.001]),shift(a,[0,0,.001]),face,{minimumDistance:.05}).certified,true);
});

test('all separate bone meshes remain candidates, without normal or regional exclusions',()=>{
  const c=collider(shift(bone,[5,0,0]),[{id:'second',vertices:bone,triangles:[[0,2,1]]}]),a=[[0,0,.1],[1,0,.1],[0,1,.1]],z=shift(a,[0,0,-.2]);
  const r=c.findCandidates(a,z,face);assert.equal(r.complete,true);assert.deepEqual(r.pairs.map(p=>p.boneId),['second']);
  const q=c.certifyStep(a,z,face);assert.equal(q.certified,false);assert.equal(q.pair.boneId,'second');
});

test('input geometry is copied so later caller mutations cannot stale the static BVH',()=>{
  const b=bone.map(p=>p.slice()),f=[[0,1,2]],c=new StaticBoneSweptBVH({bones:[{vertices:b,triangles:f}]});
  b[0][2]=100;b[1][2]=100;b[2][2]=100;f[0][0]=2;
  const a=[[0,0,.1],[1,0,.1],[0,1,.1]];assert.equal(c.certifyStep(a,shift(a,[0,0,-.2]),face).certified,false);
});

test('box and candidate budgets fail closed with incomplete enumeration explicitly marked',()=>{
  const verts=[...bone,...shift(bone,[.01,0,0])],c=new StaticBoneSweptBVH({bones:[{vertices:verts,triangles:[[0,1,2],[3,4,5]]}],leafSize:1}),
    a=[[0,0,.1],[1,0,.1],[0,1,.1]],z=shift(a,[0,0,-.2]);
  const r=c.findCandidates(a,z,face,{maxCandidatePairs:1});assert.equal(r.complete,false);assert.equal(r.reason,'candidate-pair-budget');assert.equal(r.pairs.length,1);
  const q=c.certifyStep(a,a,face,{maxBoxTests:1,minimumDistance:.2});assert.equal(q.certified,false);assert.equal(q.reason,'box-test-budget');
});

test('global narrow-phase interval budget rejects rather than dropping a second candidate',()=>{
  const b=[[2,2,0],[3,2,0],[2,3,0]],c=new StaticBoneSweptBVH({bones:[{vertices:[...b,...shift(b,[.01,0,0])],triangles:[[0,1,2],[3,4,5]]}],leafSize:1}),
    a=[[0,0,0],[3,0,0],[0,3,0]],r=c.certifyStep(a,a,face,{maxTotalIntervals:1});
  assert.equal(r.certified,false);assert.equal(r.reason,'total-interval-budget');assert.equal(r.intervals,1);
});

test('BVH candidates include every exhaustive AABB overlap in a deterministic small mesh',()=>{
  let seed=9123;const random=()=>((seed=(1664525*seed+1013904223)>>>0)/2**32),v=()=>[4*random()-2,4*random()-2,4*random()-2],
    b=Array.from({length:60},v),fs=Array.from({length:20},(_,i)=>[3*i,3*i+1,3*i+2]),
    a=Array.from({length:18},v),z=Array.from({length:18},v),af=fs.slice(0,6),c=new StaticBoneSweptBVH({bones:[{id:'random',vertices:b,triangles:fs}],leafSize:3}),
    r=c.findCandidates(a,z,af,{minimumDistance:.07}),keys=new Set(r.pairs.map(p=>`${p.tissueFace}:${p.boneFace}`));
  assert.equal(r.complete,true);assert.equal(keys.size,r.pairs.length);
  for(let i=0;i<af.length;i++)for(let j=0;j<fs.length;j++) {
    const av=af[i].flatMap(k=>[a[k],z[k]]),bv=fs[j].map(k=>b[k]);
    const overlaps=[0,1,2].every(d=>Math.min(...av.map(p=>p[d]))<=Math.max(...bv.map(p=>p[d]))+.07&&Math.min(...bv.map(p=>p[d]))<=Math.max(...av.map(p=>p[d]))+.07);
    if(overlaps)assert.ok(keys.has(`${i}:${j}`),`missed pair ${i}:${j}`);
  }
});

test('initial intersecting boundary is rejected before the guard is installed',()=>{
  const s=make(),saved=snapshot(s);assert.throws(()=>s.setStaticBoneGuard(collider(shift(bone,[0,0,.1]))),e=>e instanceof BoundaryFeasibilityError&&e.phase==='initial-state');
  assertSnapshot(s,saved);
});

test('tunneling initialGuess is rejected transactionally before any original state writes',()=>{
  const s=make();s.setStaticBoneGuard(collider());s.velocity[0]=.7;s.predict[0]=.4;s.g[0]=99;
  const saved=snapshot(s),last=s.last,guess=Float64Array.from(s.x,(x,i)=>x-(i%3===2?2:0));
  assert.throws(()=>s.step(Infinity,{initialGuess:guess}),e=>e instanceof BoundaryFeasibilityError&&e.phase==='initial-segment');
  assertSnapshot(s,saved);assert.equal(s.last,last);
});

test('fixed target movements are guarded even with no free solve iterations',()=>{
  const s=make();s.setStaticBoneGuard(collider());
  for(let n=0;n<4;n++)s.pin(n,points(s.rest)[n].map((x,d)=>x-(d===2?2:0)));
  const saved=snapshot(s);assert.throws(()=>s.step(Infinity,{iterations:0}),e=>e instanceof BoundaryFeasibilityError&&e.phase==='initial-segment');
  assertSnapshot(s,saved);
});

test('fixed targets have the same precedence over initialGuess as the original solver',()=>{
  const s=make();s.setStaticBoneGuard(collider());s.pin(0,points(s.x)[0]);
  const guess=s.x.slice();guess[2]=-100;
  const r=s.step(Infinity,{iterations:0,initialGuess:guess});assert.equal(r.boundaryFeasibility.certified,true);assert.equal(s.x[2],.1);
});

test('admitted fixed target and initialGuess changes use unchanged original mechanics',()=>{
  const s=make(),base=make(LayeredFEM),far=collider(shift(bone,[10,0,0]));s.setStaticBoneGuard(far);
  for(const solver of [s,base])solver.pin(0,[0,0,.12]);
  const guess=s.x.map((x,i)=>x+(i%3===0?.001:0)),r=s.step(Infinity,{iterations:0,initialGuess:guess}),b=base.step(Infinity,{iterations:0,initialGuess:guess});
  assert.deepEqual(s.x,base.x);assert.equal(r.energy,b.energy);assert.deepEqual(s.g,base.g);
});

test('inactive guard gives an unchanged no-contact equilibrium solve',()=>{
  const s=make(),base=make(LayeredFEM);for(const solver of [s,base])addTranslationSprings(solver,[.01,.02,.03]);
  const r=s.step(Infinity,{iterations:40,cgIterations:80,cgTolerance:1e-12,forceTolerance:1e-9}),b=base.step(Infinity,{iterations:40,cgIterations:80,cgTolerance:1e-12,forceTolerance:1e-9});
  assert.equal(r.converged,true);assert.deepEqual(s.x,base.x);assert.deepEqual(s.velocity,base.velocity);assert.deepEqual(s.g,base.g);
  for(const key of Object.keys(b))if(key!=='elapsedMs')assert.deepEqual(r[key],b[key],key);
});

test('enabled but distant guard preserves the complete no-contact solve bit for bit',()=>{
  const s=make(),base=make(LayeredFEM);s.setStaticBoneGuard(collider(shift(bone,[10,0,0])));
  for(const solver of [s,base])addTranslationSprings(solver,[.01,.02,.03]);
  const opts={iterations:40,cgIterations:80,cgTolerance:1e-12,forceTolerance:1e-9},r=s.step(Infinity,opts),b=base.step(Infinity,opts);
  assert.equal(r.converged,true);assert.deepEqual(s.x,base.x);assert.deepEqual(s.velocity,base.velocity);assert.deepEqual(s.g,base.g);
  for(const key of Object.keys(b))if(key!=='elapsedMs')assert.deepEqual(r[key],b[key],key);
  assert.equal(r.boundaryFeasibility.acceptedSegments,r.accepted);assert.equal(r.boundaryFeasibility.rejectedTrials,0);
});

test('every line-search segment is guarded; existing forces may stall against the hard filter',()=>{
  const s=make(),base=make(LayeredFEM),c=collider();s.setStaticBoneGuard(c,{minimumDistance:.02});
  for(const solver of [s,base])addTranslationSprings(solver);
  const opts={iterations:1,cgIterations:80,cgTolerance:1e-12,forceTolerance:1e-8},old=s.x.slice(),r=s.step(Infinity,opts),b=base.step(Infinity,opts),faces=extractCompleteFEMBoundary(s.nodes,4);
  assert.ok(r.boundaryFeasibility.rejectedTrials>0);assert.equal(r.converged,false);
  assert.equal(r.boundaryFeasibility.termination,'not-converged-with-guard-rejections');assert.equal(r.boundaryFeasibility.forceModelComplete,false);
  assert.equal(c.certifyStep(old,s.x,faces,{minimumDistance:.02}).certified,true);
  assert.equal(c.certifyStep(old,base.x,faces,{minimumDistance:.02}).certified,false);
  assert.ok(r.energy>b.energy);assert.equal(r.boundaryFeasibility.acceptedSegments,r.accepted);
});

test('an exhausted initial traversal budget is rejected without partial admission',()=>{
  const s=make(),c=collider(shift(bone,[10,.6,.6]));s.setStaticBoneGuard(c,{maxBoxTests:4});
  const saved=snapshot(s);
  // Every boundary face visits one root during setup. A swept motion adds leaf
  // work and must not be accepted when that exceeds the same per-call bound.
  const guess=Float64Array.from(s.x,(x,i)=>x+(i%3===0?20:0));
  assert.throws(()=>s.step(Infinity,{initialGuess:guess,iterations:0}),e=>e instanceof BoundaryFeasibilityError&&e.report.reason==='box-test-budget');assertSnapshot(s,saved);
});

test('failed inherited initial determinant admission restores physical state',()=>{
  const s=make();s.setStaticBoneGuard(collider(shift(bone,[10,0,0])));s.velocity[0]=.4;s.predict[0]=.7;
  const old=s.x.slice(),velocity=s.velocity.slice(),predict=s.predict.slice(),guess=s.x.slice();guess[5]=2;
  // Lift node 1 beyond node 3 while swapping two vertices to force negative J.
  for(let d=0;d<3;d++){guess[3+d]=s.x[6+d];guess[6+d]=s.x[3+d];}
  assert.throws(()=>s.step(Infinity,{initialGuess:guess,iterations:0}),/invert|compress/);
  assert.deepEqual(s.x,old);assert.deepEqual(s.velocity,velocity);assert.deepEqual(s.predict,predict);
});

test('manual position edits cannot silently bypass the stored admitted trajectory',()=>{
  const s=make();s.setStaticBoneGuard(collider());s.x[2]+=.01;
  assert.throws(()=>s.step(Infinity),e=>e instanceof BoundaryFeasibilityError&&e.phase==='unadmitted-position-edit');
});

test('mixin preserves an existing active-fibre energy/gradient and its original PCG',()=>{
  const C=withStaticBoneFeasibility(ActiveFibreFEM),f=fixture(),records=[{a:0,b:1,stiffness:3,activation:.1}],a=new C(f.model,f.arrays,records),b=new ActiveFibreFEM(f.model,f.arrays,records);
  a.setStaticBoneGuard(collider(shift(bone,[10,0,0])));const x=a.x.map((v,i)=>v+.001*Math.sin(i)),ra=a.evaluate(x,Infinity,true),rb=b.evaluate(x,Infinity,true);
  assert.deepEqual(ra,rb);assert.deepEqual(a.g,b.g);assert.deepEqual(a.diagonal,b.diagonal);assert.equal(C.prototype.pcg,ActiveFibreFEM.prototype.pcg);
});

test('topology edits cannot leave a stale incomplete boundary silently active',()=>{
  const s=make();s.setStaticBoneGuard(collider());[s.nodes[1],s.nodes[2]]=[s.nodes[2],s.nodes[1]];
  assert.throws(()=>s.step(Infinity,{iterations:0}),e=>e instanceof BoundaryFeasibilityError&&e.phase==='topology-edit');
});

test('mutating inherited initial evaluation is detected and physical positions roll back',()=>{
  class Mutating extends LayeredFEM {
    evaluate(x,dt,blocks=true){const r=super.evaluate(x,dt,blocks);if(this.mutate)for(let i=2;i<x.length;i+=3)x[i]-=2;return r;}
  }
  const C=withStaticBoneFeasibility(Mutating),s=make(C);s.setStaticBoneGuard(collider());s.mutate=true;const old=s.x.slice();
  assert.throws(()=>s.step(Infinity,{iterations:0}),e=>e instanceof BoundaryFeasibilityError&&e.phase==='mutating-evaluation');assert.deepEqual(s.x,old);
});

test('mutating inherited trial evaluation cannot substitute an unverified accepted trial',()=>{
  class Mutating extends LayeredFEM {
    evaluate(x,dt,blocks=true){const r=super.evaluate(x,dt,blocks);if(!blocks)for(let i=2;i<x.length;i+=3)x[i]-=2;return r;}
  }
  const C=withStaticBoneFeasibility(Mutating),s=make(C);s.setStaticBoneGuard(collider());addTranslationSprings(s,[0,0,.01]);const old=s.x.slice();
  assert.throws(()=>s.step(Infinity,{iterations:1}),e=>e instanceof BoundaryFeasibilityError&&e.phase==='mutating-evaluation');assert.deepEqual(s.x,old);
});

test('oversized geometry and inconsistent narrow-phase budgets reject before work',()=>{
  const oversized={length:900003,0:0,[Symbol.iterator](){throw Error('must not copy oversized input');}};
  assert.throws(()=>new StaticBoneSweptBVH({bones:[{vertices:oversized,triangles:face}]}),/vertex budget/);
  assert.throws(()=>collider().findCandidates(bone,bone,face,{maxIntervalsPerPair:1000001}),/maxIntervalsPerPair/);
});
