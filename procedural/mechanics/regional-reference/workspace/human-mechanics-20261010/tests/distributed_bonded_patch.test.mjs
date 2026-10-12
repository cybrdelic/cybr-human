/** Bounded exact-source multi-triangle patch fixtures, without a full solve. */
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {StaticBoneSweptBVH,extractCompleteFEMBoundary} from '../src/prototypes/static_bone_swept_bvh_distributed.mjs';
import {createVerifiedDistributedBondedPatch,DistributedBondedPatchError,verifyDistributedBondedState} from '../src/prototypes/distributed_bonded_patch.mjs';
import {GuardedStaticBoneLayeredFEM} from '../src/prototypes/guarded_static_bone_fem_distributed.mjs';

function fixture() {
  const boneVertices=Array.from({length:16},(_,i)=>[i%4,Math.floor(i/4),0]),boneFaces=[];
  for(let y=0;y<3;y++)for(let x=0;x<3;x++){const a=4*y+x;boneFaces.push([a,a+1,a+5],[a,a+5,a+4]);}
  const sourceIds=[5,6,9,10],cap=[[0,1,3],[0,3,2]],vertices=[0,.2,.5].flatMap(z=>sourceIds.map(i=>[...boneVertices[i].slice(0,2),z])),tets=[];
  for(let l=0;l<2;l++)for(const f of cap){const [a,b,c]=f.toSorted((a,b)=>a-b).map(i=>i+4*l);tets.push([a,b,c,c+4],[a,b,b+4,c+4],[a,a+4,b+4,c+4]);}
  const positions=Float64Array.from(vertices.flat()),nodes=Uint32Array.from(tets.flat()),boundaryFaces=extractCompleteFEMBoundary(nodes,vertices.length),
    staticFaces=boundaryFaces.filter(f=>f.some(i=>i<4)),fixed=Uint8Array.from({length:12},(_,i)=>i<8?1:0),targets=positions.slice(),
    collider=new StaticBoneSweptBVH({bones:[{id:'bone',vertices:boneVertices,triangles:boneFaces}],leafSize:2}),
    args={collider,positions,boundaryFaces,fixed,targets,minimumDistance:.05,rootVertexMap:sourceIds.map((boneVertex,node)=>({node,boneId:'bone',boneVertex})),
      rootPatchFaces:[{nodes:cap[0],boneId:'bone',boneFace:8},{nodes:cap[1],boneId:'bone',boneFace:9}],staticFaces,fixedCollarNodes:[0,1,2,3,4,5,6,7]};
  return {args,nodes,boneVertices,boneFaces};
}
test('multi-triangle source patch admits exact incident face, edge and vertex contacts',()=>{
  const {args}=fixture(),token=createVerifiedDistributedBondedPatch(args),s=token.summary;
  assert.equal(s.originalRootFaces,2);assert.equal(s.rootVertices,4);assert.equal(s.featureCounts.face,2);
  assert.ok(s.featureCounts.edge>0&&s.featureCounts.vertex>0);assert.ok(s.adjacentBonePairsOutsidePatch>0);
  assert.equal(s.movingFacesExempted,0);assert.equal(s.regionExclusions,0);assert.equal(s.exactSourceVertexIdentity,true);
  assert.equal(args.collider.certifyStep(args.positions,args.positions,args.boundaryFaces,{minimumDistance:.05}).certified,false);
  const r=args.collider.certifyStep(args.positions,args.positions,args.boundaryFaces,{minimumDistance:.05,distributedBondedPatch:token});assert.equal(r.certified,true);assert.equal(r.distributedBondedPairs,s.admittedPairs);
});
test('bit-level source identity rejects barycentric approximations and displaced roots',()=>{
  const {args}=fixture();args.positions[0]+=Number.EPSILON;args.targets.set(args.positions);
  assert.throws(()=>createVerifiedDistributedBondedPatch(args),e=>e instanceof DistributedBondedPatchError&&e.reason==='root-is-not-exact-source-vertex');
});
test('root cap source triangle identities cannot be replaced by patch labels',()=>{
  const {args}=fixture();args.rootPatchFaces[0].boneFace=0;assert.throws(()=>createVerifiedDistributedBondedPatch(args),/root-cap-does-not-preserve/);
});
test('every root-touching boundary face must be explicitly declared and fixed',()=>{
  const {args}=fixture();args.staticFaces=args.staticFaces.slice(0,-1);assert.throws(()=>createVerifiedDistributedBondedPatch(args),/root-contact-seam/);
  const f=fixture();f.args.fixed[4]=0;assert.throws(()=>createVerifiedDistributedBondedPatch(f.args),/not-pinned/);
});
test('neighboring bone triangles outside selected patch IDs are verified, not excluded',()=>{
  const f=fixture();
  // Add a bone triangle sharing only source vertex 5, but overlapping the
  // interior of the first root cap. Its extra intersection must be rejected.
  f.boneVertices.push([1.8,1.1,0],[1.8,1.3,0]);f.boneFaces.push([5,16,17]);
  f.args.collider=new StaticBoneSweptBVH({bones:[{id:'bone',vertices:f.boneVertices,triangles:f.boneFaces}]});
  assert.throws(()=>createVerifiedDistributedBondedPatch(f.args),e=>e.reason==='intersection-outside-declared-source-feature');
});
test('unexpected contact without any shared source identity still fails strict clearance',()=>{
  const f=fixture();f.boneVertices.push([1.3,1.2,0],[1.8,1.2,0],[1.3,1.4,0]);f.boneFaces.push([16,17,18]);
  f.args.collider=new StaticBoneSweptBVH({bones:[{id:'bone',vertices:f.boneVertices,triangles:f.boneFaces}]});
  assert.throws(()=>createVerifiedDistributedBondedPatch(f.args),e=>e.reason==='undeclared-static-pair-does-not-meet-strict-clearance');
});
test('truncated candidate discovery cannot create a partial contact token',()=>{
  const {args}=fixture();assert.throws(()=>createVerifiedDistributedBondedPatch({...args,maxCandidatePairs:1}),e=>e.reason==='incomplete-static-bone-candidate-enumeration');
});
test('entire explicit collar is immutable, including a node absent from contact faces',()=>{
  const {args}=fixture();args.fixedCollarNodes.push(8);args.fixed[8]=1;const token=createVerifiedDistributedBondedPatch(args),end=args.positions.slice();end[3*8+2]+=.1;
  assert.equal(args.collider.certifyStep(args.positions,end,args.boundaryFaces,{minimumDistance:.05,distributedBondedPatch:token}).certified,false);
  args.fixed[8]=0;assert.equal(verifyDistributedBondedState(token,{positions:args.positions,fixed:args.fixed,targets:args.targets}).valid,false);
});
test('moving faces past the fixed collar retain continuous collision filtering',()=>{
  const {args}=fixture(),token=createVerifiedDistributedBondedPatch(args),end=args.positions.slice();for(let i=8;i<12;i++)end[3*i+2]=-1;
  const r=args.collider.certifyStep(args.positions,end,args.boundaryFaces,{minimumDistance:.05,distributedBondedPatch:token});assert.equal(r.certified,false);assert.equal(r.reason,'uncertified-triangle-segment');
  assert.ok(args.boundaryFaces[r.pair.tissueFace].every(i=>i>=4));
});
test('token identity, topology, and clearance are bound to verified geometry',()=>{
  const {args}=fixture(),token=createVerifiedDistributedBondedPatch(args),other=fixture().args.collider;
  assert.equal(other.certifyStep(args.positions,args.positions,args.boundaryFaces,{minimumDistance:.05,distributedBondedPatch:token}).certified,false);
  assert.equal(args.collider.certifyStep(args.positions,args.positions,args.boundaryFaces,{minimumDistance:.06,distributedBondedPatch:token}).certified,false);
  assert.equal(args.collider.certifyStep(args.positions,args.positions,args.boundaryFaces.toReversed(),{minimumDistance:.05,distributedBondedPatch:token}).certified,false);
});
test('complete-separation baseline remains exactly pinned',()=>{
  for(const [file,expected] of [['static_bone_swept_bvh.mjs','4269a5c5fe6b5420cfbc427e5f518a9c772bd3ad18c700e29e6bc33a652255a8'],['guarded_static_bone_fem.mjs','2decfa027a7239dd2e8dab6a729a9cdf8b620fd4aa3548ce2a8cc84f2ca6decd']])
    assert.equal(crypto.createHash('sha256').update(fs.readFileSync(new URL(`../src/prototypes/${file}`,import.meta.url))).digest('hex'),expected);
});

test('distributed wrapper checks collar pins and targets around the unchanged original step',()=>{
  const f=fixture(),rest=f.args.positions,nodes=f.nodes.slice(),gradients=[],material=[],mass=new Float64Array(12);
  const inverse=m=>{const [a,b,c,d,e,g,h,i,j]=m,det=a*(e*j-g*i)-b*(d*j-g*h)+c*(d*i-e*h);return {det,inv:[e*j-g*i,c*i-b*j,b*g-c*e,g*h-d*j,a*j-c*h,c*d-a*g,d*i-e*h,b*h-a*i,a*e-b*d].map(x=>x/det)};};
  for(let t=0;t<nodes.length;t+=4) {
    const matrix=()=>[0,1,2].flatMap(d=>[1,2,3].map(a=>rest[3*nodes[t+a]+d]-rest[3*nodes[t]+d]));
    if(inverse(matrix()).det<0)[nodes[t+1],nodes[t+2]]=[nodes[t+2],nodes[t+1]];
    const {det,inv}=inverse(matrix()),v=det/6;gradients.push(...[0,1,2].map(d=>-inv[d]-inv[3+d]-inv[6+d]),...inv);material.push(v,15,120);for(let a=0;a<4;a++)mass[nodes[t+a]]+=v/4;
  }
  const solver=new GuardedStaticBoneLayeredFEM({schema:'cybr-layered-fem-v1',nodes:12,tetrahedra:nodes.length/4,surface_nodes:12},
    {rest,nodes,gradients:Float64Array.from(gradients),material:Float64Array.from(material),mass});
  for(let i=0;i<8;i++)solver.pin(i,Array.from(rest.slice(3*i,3*i+3)));
  const boundaryFaces=extractCompleteFEMBoundary(nodes,12),token=createVerifiedDistributedBondedPatch({...f.args,positions:solver.x,fixed:solver.fixed,targets:solver.target,
    boundaryFaces,staticFaces:boundaryFaces.filter(t=>t.some(i=>i<4))});
  solver.setStaticBoneGuard(f.args.collider,{minimumDistance:.05,distributedBondedPatch:token});
  const old=solver.x.slice(),r=solver.step(Infinity,{iterations:0});assert.equal(r.boundaryFeasibility.certified,true);assert.deepEqual(solver.x,old);
  solver.unpin(4);assert.throws(()=>solver.step(Infinity,{iterations:0}),e=>e.phase==='distributed-bonded-invariant');assert.deepEqual(solver.x,old);
  solver.pin(4,Array.from(old.slice(12,15)));solver.target[0]+=.001;assert.throws(()=>solver.step(Infinity,{iterations:0}),e=>e.phase==='distributed-bonded-invariant');
});
