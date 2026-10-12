import assert from 'node:assert/strict';
import {enumerateSweptBoundaryPairs as enumerate,certifyTissueSelfStep as certify} from '../src/tissue_self_bvh.mjs';
const vertices=[0,0,0,1,0,0,0,1,0,0,0,1],faces=[[0,2,1],[0,1,3],[0,3,2],[1,2,3]];
let count=0;const tests=[];function test(name,fn){fn();count++;tests.push(name);}
const moved=(v,d)=>v.map((x,i)=>x+d[i%3]);
test('complete closed tetrahedral boundary',()=>{const r=certify(vertices,vertices,faces);assert(r.certified,JSON.stringify(r));assert.equal(r.checkedPairs,6);assert.equal(r.byShared[2],6);});
test('shared topology survives arbitrary face winding',()=>{const r=certify(vertices,moved(vertices,[2,3,4]),faces.map((f,i)=>i%2?f.toReversed():f));assert(r.certified,JSON.stringify(r));assert(!r.orientedEdgeIncidence);});
test('two disjoint components with complete sweep',()=>{const v=vertices.concat(moved(vertices,[0,0,3])),f=faces.concat(faces.map(t=>t.map(i=>i+4)));const r=certify(v,moved(v,[1,1,0]),f);assert(r.certified,JSON.stringify(r));});
test('mutually moving components tunnel despite disjoint endpoints',()=>{const a=vertices.concat(moved(vertices,[0,0,3])),b=moved(vertices,[0,0,3]).concat(vertices),f=faces.concat(faces.map(t=>t.map(i=>i+4)));const r=certify(a,b,f);assert(!r.certified);assert(r.completeEnumeration);});
test('duplicate faces rejected',()=>assert.throws(()=>certify(vertices,vertices,[...faces,faces[0].toReversed()]),/Duplicate/));
test('open boundary rejected by default',()=>assert.equal(certify(vertices,vertices,faces.slice(0,3)).reason,'not-closed-edge-incidence'));
test('global proof budget fails closed',()=>{const r=certify(vertices,vertices,faces,{maxIntervals:1});assert(!r.certified);});
test('broad phase budgets fail closed',()=>{assert(!enumerate(vertices,vertices,faces,{maxBoxTests:1}).complete);assert(!enumerate(vertices,vertices,faces,{maxPairs:1}).complete);});
test('nonfinite and oversized inputs rejected',()=>{assert.throws(()=>certify([Infinity,...vertices.slice(1)],vertices,faces),/Nonfinite/);assert.throws(()=>enumerate(vertices,vertices,faces,{maxFaces:1}),/oversized/);});
let seed=125483;const num=()=>{seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return(seed>>>0)%33-16;};
test('all swept overlaps equal exhaustive enumeration, no duplicates',()=>{
 for(let repeat=0;repeat<200;repeat++){
  const a=Array.from({length:135},num),b=Array.from({length:135},num),f=Array.from({length:15},(_,i)=>[3*i,3*i+1,3*i+2]);
  const r=enumerate(a,b,f);assert(r.complete);const found=new Set();for(let k=0;k<r.pairs.length;k+=2){const key=`${r.pairs[k]}:${r.pairs[k+1]}`;assert(!found.has(key));found.add(key);}
  const expected=new Set();for(let i=0;i<f.length;i++)for(let j=i+1;j<f.length;j++)if(![0,1,2].some(d=>r.boxes[i][d]>r.boxes[j][3+d]||r.boxes[j][d]>r.boxes[i][3+d]))expected.add(`${i}:${j}`);
  assert.deepEqual([...found].sort(),[...expected].sort());
 }
});
test('unrelated coincident vertices are not topology aliases',()=>{const a=vertices.concat(vertices),f=faces.concat(faces.map(t=>t.map(i=>i+4)));assert(!certify(a,a,f).certified);});
test('stationary feature route still checks every pair',()=>{const a=certify(vertices,vertices,faces,{stationaryFeatureFastPath:false}),b=certify(vertices,vertices,faces);assert(a.certified&&b.certified);assert.equal(a.checkedPairs,b.checkedPairs);assert.equal(a.stationaryFeaturePairs,0);assert.equal(b.stationaryFeaturePairs,6);});
test('no moving pair gets stationary proof from endpoint coincidence alone',()=>{const b=moved(vertices,[0,0,1]),r=certify(vertices,b,faces);assert(r.certified);assert.equal(r.stationaryFeaturePairs,0);});
console.log(JSON.stringify({passed:count,tests,exhaustiveCoverageCases:200},null,2));
