/** Tiny fixtures only; run with node --test tests/complete_boundary_contact.test.mjs. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {nextUp,nextDown,trianglePairFeatures,certifyTriangleStep,certifyBoundaryStep,StaticTrianglePairBarrier}
  from '../src/prototypes/complete_boundary_contact.mjs';

const clone=a=>a.map(p=>p.slice());
const shift=(a,s)=>a.map(p=>p.map((x,d)=>x+s[d]));
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const bone=[[-.1,-.1,0],[.1,-.1,0],[0,.1,0]],
  large=[[-2,-2,.2],[2,-2,.22],[0,2,.3]];
const potential=(a=large,b=bone,options={})=>new StaticTrianglePairBarrier({restTissue:a,bone:b,activationDistance:1,...options});

function gradientError(p,a,h=1e-6) {
  const r=p.evaluate(a);assert.equal(r.valid,true);let absolute=0,scaled=0;
  for(let i=0;i<3;i++)for(let d=0;d<3;d++) {
    const plus=clone(a),minus=clone(a);plus[i][d]+=h;minus[i][d]-=h;
    const ep=p.evaluate(plus),em=p.evaluate(minus);assert.equal(ep.valid,true);assert.equal(em.valid,true);
    const fd=(ep.energy-em.energy)/(2*h),e=Math.abs(fd-r.gradient[i][d]);
    absolute=Math.max(absolute,e);scaled=Math.max(scaled,e/Math.max(1,Math.abs(fd),Math.abs(r.gradient[i][d])));
  }
  return {absolute,scaled};
}

test('adjacent binary64 values cover signs, zero, subnormal and infinity boundaries',()=>{
  assert.equal(nextUp(0),Number.MIN_VALUE);assert.equal(nextDown(0),-Number.MIN_VALUE);
  assert.equal(nextDown(Number.MIN_VALUE),0);assert.equal(nextUp(-Number.MIN_VALUE),-0);
  assert.equal(nextDown(1),1-Number.EPSILON/2);assert.equal(nextUp(1),1+Number.EPSILON);
  assert.equal(nextUp(-1),-1+Number.EPSILON/2);assert.equal(nextDown(-1),-1-Number.EPSILON);
  assert.equal(nextDown(Infinity),Number.MAX_VALUE);assert.equal(nextUp(-Infinity),-Number.MAX_VALUE);
  assert.equal(nextUp(Number.MAX_VALUE),Infinity);assert.equal(nextDown(-Number.MAX_VALUE),-Infinity);
});

test('all 6 bidirectional vertex-face and 9 edge-edge terms are present',()=>{
  const f=trianglePairFeatures(large,bone);assert.equal(f.length,15);
  for(const kind of ['tissue-vertex/bone-face','bone-vertex/tissue-face'])assert.equal(f.filter(x=>x.kind===kind).length,3);
  assert.equal(f.filter(x=>x.kind==='edge/edge').length,9);
  for(const q of f){assert.ok(q.d2>=0);assert.ok(Math.abs(q.wa.reduce((a,b)=>a+b,0)-1)<1e-15);assert.ok(Math.abs(q.wb.reduce((a,b)=>a+b,0)-1)<1e-15);}
});

test('reverse vertex-face catches a bone tip under a large face with all tissue vertices remote',()=>{
  const a=[[-4,-4,.1],[4,-4,.1],[0,4,.1]],p=potential(a,bone,{activationDistance:.3}),r=p.evaluate(a);
  assert.equal(r.valid,true);assert.equal(r.activeByKind['tissue-vertex/bone-face'],0);
  assert.equal(r.activeByKind['bone-vertex/tissue-face'],3);assert.ok(r.energy>0);
  assert.ok(r.gradient.flat().some(x=>x!==0));
});

test('edge-edge contacts detect interleaved triangle projections with no close vertex-face contacts',()=>{
  const a=[[-2,-1,.1],[2,-1,.1],[0,2,.1]],b=[[-2,1,0],[2,1,0],[0,-2,0]],
    f=trianglePairFeatures(a,b),vf=f.filter(q=>q.kind!=='edge/edge'),ee=f.filter(q=>q.kind==='edge/edge');
  assert.ok(Math.min(...vf.map(q=>q.d2))>.1);assert.ok(Math.abs(Math.min(...ee.map(q=>q.d2))-.01)<1e-14);
  const r=potential(a,b,{activationDistance:.2}).evaluate(a);
  assert.equal(r.activeByKind['tissue-vertex/bone-face'],0);assert.equal(r.activeByKind['bone-vertex/tissue-face'],0);
  assert.ok(r.activeByKind['edge/edge']>0);assert.ok(r.energy>0);
  assert.equal(certifyTriangleStep(a,shift(a,[0,0,-.2]),b).certified,false);
});

test('already intersecting triangles are rejected although all 15 feature distances are positive',()=>{
  const a=[[-2,-2,0],[2,-2,0],[0,2,0]],b=[[0,-1,-1],[0,-1,1],[0,1,.5]];
  assert.equal(Math.min(...trianglePairFeatures(a,b).map(q=>q.d2)),.25);
  assert.equal(certifyTriangleStep(a,a,b,{maxDepth:0}).certified,false);
  const r=potential(a,b).evaluate(a);assert.equal(r.valid,false);assert.equal(r.energy,Infinity);
});

test('separated endpoints do not authorize tunneling through the static bone',()=>{
  const a=[[-4,-4,.1],[4,-4,.1],[0,4,.1]],end=shift(a,[0,0,-.2]);
  assert.equal(certifyTriangleStep(a,a,bone).certified,true);assert.equal(certifyTriangleStep(end,end,bone).certified,true);
  const r=certifyTriangleStep(a,end,bone);assert.equal(r.certified,false);assert.equal(r.reason,'unresolved');
  assert.ok(r.unresolved[0]<=.5&&r.unresolved[1]>=.5);
});

test('clearance is strict across the complete path and cannot be replaced by endpoint checks',()=>{
  const a=[[-2,-2,.2],[2,-2,.2],[0,2,.2]],safe=shift(a,[.4,.1,-.149]),touch=shift(a,[.4,.1,-.15]);
  assert.equal(certifyTriangleStep(a,safe,bone,{minimumDistance:.05}).certified,true);
  assert.equal(certifyTriangleStep(a,touch,bone,{minimumDistance:.05,maxDepth:10}).certified,false);
  assert.equal(certifyTriangleStep(a,shift(a,[0,0,-.151]),bone,{minimumDistance:.05}).certified,false);
});

test('coplanar disjoint triangles may pass; coplanar overlap, touching and grazing reject',()=>{
  const a=[[0,0,0],[1,0,0],[0,1,0]],b=[[2,2,0],[3,2,0],[2,3,0]];
  assert.equal(certifyTriangleStep(a,a,b,{minimumDistance:.5}).certified,true);
  assert.equal(certifyTriangleStep(a,a,shift(a,[.1,.1,0]),{maxDepth:0}).certified,false);
  assert.equal(certifyTriangleStep(a,a,shift(a,[1,0,0]),{maxDepth:0}).certified,false);
  const start=shift(a,[-2,1,0]),end=shift(a,[2,1,0]);
  assert.equal(certifyTriangleStep(start,end,a,{maxDepth:16}).certified,false);
});

test('work-budget exhaustion is an explicit rejection and never collision-free',()=>{
  const a=[[-2,-2,.2],[2,-2,.2],[0,2,.2]],end=shift(a,[0,0,-.4]);
  const r=certifyTriangleStep(a,end,bone,{maxIntervals:1});assert.equal(r.certified,false);assert.equal(r.reason,'interval-budget');
  const mesh={startVertices:a,endVertices:a,tissueTriangles:[[0,1,2]],boneVertices:bone,boneTriangles:[[0,1,2],[0,2,1]]};
  const q=certifyBoundaryStep(mesh,{maxTrianglePairs:1});assert.equal(q.certified,false);assert.equal(q.reason,'pair-budget');assert.equal(q.checkedPairs,0);
});

test('tiny complete-boundary harness checks the second bone face and reports the offending pair',()=>{
  const a=[[-2,-2,.2],[2,-2,.2],[0,2,.2]],end=shift(a,[0,0,-.4]),b=[...shift(bone,[20,0,0]),...bone];
  const mesh={startVertices:a,endVertices:end,tissueTriangles:[[0,1,2]],boneVertices:b,boneTriangles:[[0,1,2],[3,4,5]]};
  const r=certifyBoundaryStep(mesh);assert.equal(r.certified,false);assert.equal(r.checkedPairs,2);assert.equal(r.boneTriangle,1);
  assert.equal(certifyBoundaryStep({...mesh,endVertices:a}).certified,true);
});

test('analytic total energy gradient includes reverse VF and general edge features',()=>{
  const e=gradientError(potential(),large);assert.ok(e.scaled<2e-7,JSON.stringify(e));
});

test('near-parallel EE mollifier derivative participates in the analytic gradient',()=>{
  const a=[[-2,-1,.15],[2,-.98,.154],[0,2,.16]],b=[[-2,-1.1,0],[2,-1.1,0],[.1,2.1,0]],
    p=potential(a,b,{activationDistance:.6,mollifierScale:.05}),e=gradientError(p,a,2e-7);
  assert.ok(e.scaled<3e-6,JSON.stringify(e));
});

test('minimum-distance offset also has a consistent energy derivative',()=>{
  const e=gradientError(potential(large,bone,{minimumDistance:.1}),large);assert.ok(e.scaled<2e-7,JSON.stringify(e));
});

test('equal/opposite forces and net torque balance include fixed-bone reactions',()=>{
  const p=potential(),r=p.evaluate(large),sum=[0,0,0],torque=[0,0,0];
  for(let i=0;i<3;i++)for(const [x,g] of [[large[i],r.gradient[i]],[bone[i],r.boneGradient[i]]]) {
    const t=cross(x,g);for(let d=0;d<3;d++){sum[d]+=g[d];torque[d]+=t[d];}
  }
  assert.ok(Math.hypot(...sum)<1e-11);assert.ok(Math.hypot(...torque)<1e-10);
});

test('positive search blocks are only a PSD metric, with no exact Hessian claim',()=>{
  const r=potential().evaluate(large),v=[[.2,-.3,.5],[-.4,.2,.7],[.3,.6,-.1]];
  assert.equal(r.metricIsExactHessian,false);assert.equal(r.meshDiscretizationNormalized,false);
  const quadratic=r.metricBlocks.reduce((s,b)=>s+b.coefficient*dot(b.tissueDistanceSquaredGradient.flat(),v.flat())**2,0);
  assert.ok(quadratic>=0);assert.ok(r.metricBlocks.every(b=>b.coefficient>=0));
});

test('barrier is inactive beyond activation and rises toward the strict clearance',()=>{
  const a=[[-2,-2,.2],[2,-2,.2],[0,2,.2]],p=potential(a,bone,{minimumDistance:.02,activationDistance:.3});
  const far=p.evaluate(shift(a,[0,0,1]));assert.equal(far.energy,0);assert.ok(far.gradient.flat().every(x=>x===0));
  const e1=p.evaluate(shift(a,[0,0,-.1])).energy,e2=p.evaluate(shift(a,[0,0,-.17])).energy,e3=p.evaluate(shift(a,[0,0,-.179])).energy;
  assert.ok(e1<e2&&e2<e3);
});

test('degenerate/nonfinite geometry and unrepresentable arithmetic fail closed',()=>{
  assert.throws(()=>certifyTriangleStep([[NaN,0,0],[1,0,0],[0,1,0]],large,bone));
  const p=potential(),r=p.evaluate([[0,0,.1],[1,0,.1],[2,0,.1]]);assert.equal(r.valid,false);assert.equal(r.energy,Infinity);
  const huge=[[1e308,0,0],[1e308,1,0],[1e308,0,1]],negative=[[-1e308,0,0],[-1e308,1,0],[-1e308,0,1]];
  assert.equal(certifyTriangleStep(huge,huge,negative,{maxDepth:0}).certified,false);
});

test('very small represented separation is rejected when projection roundoff cannot prove it',()=>{
  const b=[[0,0,0],[1,0,0],[0,1,0]],a=shift(b,[0,0,Number.MIN_VALUE]);
  assert.equal(certifyTriangleStep(a,a,b,{maxDepth:0}).certified,false);
  assert.equal(certifyTriangleStep(shift(b,[0,0,1e-9]),shift(b,[.1,.1,1e-9]),b,{maxDepth:0}).certified,true);
});

test('96 known interior crossing paths are rejected after integer shears and translations',()=>{
  for(let i=0;i<96;i++) {
    const z=(1+i%7)/8,base=[[-4,-4,z],[4,-4,z],[0,4,z]],end=shift(base,[0,0,-1]),
      shear=p=>[p[0]+(i%5-2)*p[2]+i*8,p[1]+(i%3-1)*p[0]-i*4,p[2]];
    const r=certifyTriangleStep(base.map(shear),end.map(shear),bone.map(shear),{maxDepth:14,maxIntervals:256});
    assert.equal(r.certified,false,`crossing ${i}`);
  }
});

test('96 known separated affine paths pass under integer shears and translations',()=>{
  for(let i=0;i<96;i++) {
    const a=[[-4,-4,.25],[4,-4,.25],[0,4,.25]],end=shift(a,[.2,.3,.125]),
      shear=p=>[p[0]+(i%5-2)*p[2]+i*8,p[1]+(i%3-1)*p[0]-i*4,p[2]];
    assert.equal(certifyTriangleStep(a.map(shear),end.map(shear),bone.map(shear),{minimumDistance:.1,maxDepth:0}).certified,true,`separated ${i}`);
  }
});

test('bone reaction is also the exact derivative with rest mollifier thresholds held fixed',()=>{
  const a=[[-2,-1,.15],[2,-.98,.154],[0,2,.16]],b=[[-2,-1.1,0],[2,-1.1,0],[.1,2.1,0]],
    p=potential(a,b,{activationDistance:.6,mollifierScale:.05}),r=p.evaluate(a),h=2e-7;
  for(let i=0;i<3;i++)for(let d=0;d<3;d++) {
    p.bone[i][d]=b[i][d]+h;const ep=p.evaluate(a).energy;
    p.bone[i][d]=b[i][d]-h;const em=p.evaluate(a).energy;p.bone[i][d]=b[i][d];
    const fd=(ep-em)/(2*h),e=Math.abs(fd-r.boneGradient[i][d])/Math.max(1,Math.abs(fd));
    assert.ok(e<3e-6,`${i},${d}: ${e}`);
  }
});

test('rigid objectivity holds for energy and rotated gradients',()=>{
  const r=p=>[-p[1],p[2],-p[0]],translation=[6,-8,3],transform=p=>r(p).map((x,d)=>x+translation[d]),
    p=potential(),q=potential(large.map(transform),bone.map(transform)),a=p.evaluate(large),b=q.evaluate(large.map(transform));
  assert.ok(Math.abs(a.energy-b.energy)<1e-11);
  for(let i=0;i<3;i++)for(let d=0;d<3;d++)assert.ok(Math.abs(r(a.gradient[i])[d]-b.gradient[i][d])<1e-10);
});

// Independent exact-dyadic replay: represent each finite binary64 as n * 2^e.
// This does not use the prototype's interval helpers or closest-point formulas.
const exactBits=new DataView(new ArrayBuffer(8));
function exact(x) {
  assert.ok(Number.isFinite(x));if(x===0)return [0n,0];exactBits.setFloat64(0,x);
  const b=exactBits.getBigUint64(0),sign=(b>>63n)?-1n:1n,exp=Number((b>>52n)&2047n),fraction=b&((1n<<52n)-1n);
  return [sign*(exp===0?fraction:((1n<<52n)|fraction)),exp===0?-1074:exp-1023-52];
}
const eadd=(a,b)=>{const e=Math.min(a[1],b[1]);return [(a[0]<<BigInt(a[1]-e))+(b[0]<<BigInt(b[1]-e)),e];};
const eneg=a=>[-a[0],a[1]],esub=(a,b)=>eadd(a,eneg(b)),emul=(a,b)=>[a[0]*b[0],a[1]+b[1]],
  ecmp=(a,b)=>{const d=esub(a,b)[0];return d<0n?-1:d>0n?1:0;},
  emin=a=>a.reduce((a,b)=>ecmp(a,b)<0?a:b),emax=a=>a.reduce((a,b)=>ecmp(a,b)>0?a:b),
  edot=(p,n)=>p.reduce((s,x,i)=>eadd(s,emul(x,n[i])),exact(0));
function replayProof(a0,a1,b,clearance,proof) {
  const ordered=proof.toSorted((a,b)=>a.lo-b.lo);assert.equal(ordered[0].lo,0);assert.equal(ordered.at(-1).hi,1);
  for(let i=1;i<ordered.length;i++)assert.equal(ordered[i-1].hi,ordered[i].lo);
  for(const leaf of proof) {
    const n=leaf.axis.map(exact),bp=b.map(p=>edot(p.map(exact),n)),ap=[];
    for(let i=0;i<3;i++)for(const time of [leaf.lo,leaf.hi]) {
      const t=exact(time),oneMinus=esub(exact(1),t),p=a0[i].map((x,d)=>eadd(emul(exact(x),oneMinus),emul(exact(a1[i][d]),t)));
      ap.push(edot(p,n));
    }
    const gap=emax([esub(emin(ap),emax(bp)),esub(emin(bp),emax(ap))]);assert.ok(ecmp(gap,exact(0))>0);
    const required=emul(emul(exact(clearance),exact(clearance)),edot(n,n));
    assert.ok(ecmp(emul(gap,gap),required)>0,'exact rational replay did not establish claimed separation');
  }
}

test('subdivided swept-triangle cover replays using independent exact BigInt arithmetic',()=>{
  const a0=[[1.3692622780799866,.2376929810270667,-1.1464840304106474],[1.6135526085272431,-.4000192992389202,-1.1797937722876668],[.715454762801528,1.2833241047337651,1.999703861773014]],
    a1=[[-1.9852103823795915,.6325415018945932,-1.9122870108112693],[-.5923987366259098,-1.5628103511407971,.044539256021380424],[1.6494008796289563,1.9434362798929214,-.7819393435493112]],
    b=[[-.12,-.1,0],[.12,-.1,0],[0,.15,0]],r=certifyTriangleStep(a0,a1,b,{minimumDistance:.02,collectProof:true});
  assert.equal(certifyTriangleStep(a0,a1,b,{minimumDistance:.02,maxDepth:0}).certified,false);
  assert.equal(r.certified,true);assert.ok(r.leaves>1);replayProof(a0,a1,b,.02,r.proof);
});

test('exact replay validates 48 nonzero-clearance certificates after large translations',()=>{
  for(let i=0;i<48;i++) {
    const offset=[2**(i%24),-(2**(i%20)),2**(i%16)],a=shift(large,offset),b=shift(bone,offset),end=shift(a,[.1,.2,.1]),
      r=certifyTriangleStep(a,end,b,{minimumDistance:.1,collectProof:true});
    assert.equal(r.certified,true);replayProof(a,end,b,.1,r.proof);
  }
});
