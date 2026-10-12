/** Experimental complete swept-AABB enumeration plus exact pair certificates.
 * Zero uniform margin. No region, normal or adjacency exemption. Inputs are
 * copied before use. This stateless check is not a solver trajectory authority.
 */
import {compileExactVertexTrajectory,certifyIndexedMovingTrianglePair} from './exact_moving_triangle_pair.mjs';
import {verifyTriangleIntersectionFeature} from '../../../procedural/mechanics/regional-reference/workspace/human-mechanics-20261010/src/prototypes/exact_triangle_contact_feature.mjs';
const positiveInt=(x,max)=>Number.isInteger(x)&&x>0&&x<=max;
function snapshot(start,end,faces,maxFaces,maxVertices){
 if(!start||!end||start.length!==end.length||start.length%3||start.length/3>maxVertices||!Array.isArray(faces)||!faces.length||faces.length>maxFaces)throw Error('Invalid/oversized boundary input');
 const a=Float64Array.from(start),b=Float64Array.from(end),n=a.length/3;
 for(let i=0;i<a.length;i++)if(!Number.isFinite(a[i])||!Number.isFinite(b[i]))throw Error('Nonfinite boundary input');
 const seen=new Set(),f=faces.map(t=>{if(!t||t.length!==3)throw Error('Invalid boundary triangle');const v=Array.from(t);if(v.some(x=>!Number.isInteger(x)||x<0||x>=n)||new Set(v).size!==3)throw Error('Invalid boundary index');const k=v.slice().sort((a,b)=>a-b).join(':');if(seen.has(k))throw Error('Duplicate boundary face');seen.add(k);return v;});
 return{a,b,f,n};
}
const disjoint=(a,b)=>a[0]>b[3]||b[0]>a[3]||a[1]>b[4]||b[1]>a[4]||a[2]>b[5]||b[2]>a[5];
function boxesFor(a,b,f){
 return f.map(t=>{const box=[Infinity,Infinity,Infinity,-Infinity,-Infinity,-Infinity];for(const i of t)for(let d=0;d<3;d++){box[d]=Math.min(box[d],a[3*i+d],b[3*i+d]);box[3+d]=Math.max(box[3+d],a[3*i+d],b[3*i+d]);}return box;});
}
function build(boxes,ids){
 const box=[Infinity,Infinity,Infinity,-Infinity,-Infinity,-Infinity];for(const i of ids)for(let d=0;d<3;d++){box[d]=Math.min(box[d],boxes[i][d]);box[3+d]=Math.max(box[3+d],boxes[i][3+d]);}
 if(ids.length<=8)return{box,ids};
 let axis=0;for(let d=1;d<3;d++)if(box[d+3]-box[d]>box[axis+3]-box[axis])axis=d;
 // The ordering heuristic may overflow; it never participates in exclusion.
 // Stable ID fallback also handles coincident or overflowed centres.
 ids.sort((i,j)=>{const q=(boxes[i][axis]/2+boxes[i][axis+3]/2)-(boxes[j][axis]/2+boxes[j][axis+3]/2);return Number.isNaN(q)||q===0?i-j:q;});
 const mid=ids.length>>1;return{box,left:build(boxes,ids.slice(0,mid)),right:build(boxes,ids.slice(mid))};
}
function enumerate(boxes,maxBoxTests,maxPairs){
 const root=build(boxes,Array.from({length:boxes.length},(_,i)=>i)),stack=[[root,root]],pairs=[];let boxTests=0;
 while(stack.length){
  const[a,b]=stack.pop();if(++boxTests>maxBoxTests)return{complete:false,reason:'box-budget',boxTests:boxTests-1,pairs};
  if(disjoint(a.box,b.box))continue;
  if(a.ids&&b.ids){
   for(const i of a.ids)for(const j of b.ids){
    if(a===b&&i>=j)continue;
    if(++boxTests>maxBoxTests)return{complete:false,reason:'box-budget',boxTests:boxTests-1,pairs};
    if(disjoint(boxes[i],boxes[j]))continue;
    if(pairs.length/2>=maxPairs)return{complete:false,reason:'pair-budget',boxTests,pairs};
    pairs.push(Math.min(i,j),Math.max(i,j));
   }
  }else if(a===b){stack.push([a.right,a.right],[a.left,a.right],[a.left,a.left]);}
  else if(!a.ids){stack.push([a.left,b],[a.right,b]);}
  else{stack.push([a,b.left],[a,b.right]);}
 }
 return{complete:true,reason:'complete-swept-box-enumeration',boxTests,pairs};
}
function closedEdges(f){
 const edges=new Map();for(const t of f)for(let j=0;j<3;j++){const a=t[j],b=t[(j+1)%3],k=a<b?`${a}:${b}`:`${b}:${a}`,p=edges.get(k)??[0,0];p[0]++;p[1]+=a<b?1:-1;edges.set(k,p);}
 let closed=true,oriented=true;for(const[count,balance]of edges.values()){if(count!==2)closed=false;if(balance!==0)oriented=false;}return{closed,oriented};
}
export function enumerateSweptBoundaryPairs(start,end,faces,{maxFaces=100000,maxVertices=100000,maxBoxTests=5000000,maxPairs=500000}={}){
 for(const[x,m]of[[maxFaces,100000],[maxVertices,100000],[maxBoxTests,10000000],[maxPairs,1000000]])if(!positiveInt(x,m))throw Error('Invalid enumeration budget');
 const{a,b,f}=snapshot(start,end,faces,maxFaces,maxVertices),boxes=boxesFor(a,b,f),r=enumerate(boxes,maxBoxTests,maxPairs);
 return{...r,pairs:Int32Array.from(r.pairs),faces:f,start:a,end:b,boxes};
}
export function certifyTissueSelfStep(start,end,faces,{requireClosedEdges=true,maxFaces=100000,maxVertices=100000,maxBoxTests=5000000,maxPairs=500000,maxIntervals=1000000,maxAxisTests=10000000,pairMaxDepth=12,pairMaxIntervals=255,pairMaxAxisTests=100000,collectPairs=false,stationaryFeatureFastPath=true}={}){
 if(typeof stationaryFeatureFastPath!=='boolean'||typeof requireClosedEdges!=='boolean'||typeof collectPairs!=='boolean'||!positiveInt(maxIntervals,10000000)||!positiveInt(maxAxisTests,100000000)||!positiveInt(pairMaxIntervals,1000000)||!positiveInt(pairMaxAxisTests,1000000)||!Number.isInteger(pairMaxDepth)||pairMaxDepth<0||pairMaxDepth>30)throw Error('Invalid certificate options');
 const e=enumerateSweptBoundaryPairs(start,end,faces,{maxFaces,maxVertices,maxBoxTests,maxPairs});
 const base={completeEnumeration:e.complete,boxTests:e.boxTests,candidatePairs:e.pairs.length/2,boundaryFaces:e.faces.length,uniformMargin:false,topologySource:'Caller-supplied global vertex indices; complete FEM extraction is a consuming-wrapper requirement',solverIntegrated:false};
 if(!e.complete)return{...base,certified:false,reason:e.reason,checkedPairs:0};
 const incidence=closedEdges(e.faces);if(requireClosedEdges&&!incidence.closed)return{...base,certified:false,reason:'not-closed-edge-incidence',checkedPairs:0};
 const trajectory=compileExactVertexTrajectory(e.start,e.end);
 const tri=f=>f.map(i=>[e.start[3*i],e.start[3*i+1],e.start[3*i+2]]);
 const stationary=f=>f.every(i=>[0,1,2].every(d=>e.start[3*i+d]===e.end[3*i+d]));
 let checkedPairs=0,intervals=0,axisTests=0,stationaryFeaturePairs=0;const byShared={0:0,1:0,2:0};
 for(let k=0;k<e.pairs.length;k+=2){
  if(intervals>=maxIntervals||axisTests>=maxAxisTests)return{...base,certified:false,reason:'global-proof-budget',checkedPairs,intervals,axisTests};
  const i=e.pairs[k],j=e.pairs[k+1],f=e.faces[i],g=e.faces[j],shared=[];
  for(let a=0;a<3;a++)for(let b=0;b<3;b++)if(f[a]===g[b])shared.push([a,b]);
  let r;
  if(stationaryFeatureFastPath&&stationary(f)&&stationary(g)){
   // Both full trajectories are constant, exactly, so the exact static feature
   // intersection is the entire-time result. This is not a sampled exemption.
   const feature=verifyTriangleIntersectionFeature(tri(f),tri(g),shared.map(([a,b])=>({a,b})));
   r={certified:feature.valid,reason:'exact-stationary-feature-'+feature.reason,intervals:1,axisTests:0,feature};stationaryFeaturePairs++;
  }else r=certifyIndexedMovingTrianglePair(trajectory,f,g,{maxDepth:pairMaxDepth,maxIntervals:Math.min(pairMaxIntervals,maxIntervals-intervals),maxAxisTests:Math.min(pairMaxAxisTests,maxAxisTests-axisTests)});
  checkedPairs++;intervals+=r.intervals;axisTests+=r.axisTests;byShared[shared.length]++;
  if(!r.certified)return{...base,certified:false,reason:'pair-'+r.reason,pair:[i,j],shared,checkedPairs,intervals,axisTests,pairReport:r};
 }
 return{...base,certified:true,reason:'complete-exact-moving-pair-cover',checkedPairs,intervals,axisTests,byShared,stationaryFeaturePairs,closedEdgeIncidence:incidence.closed,orientedEdgeIncidence:incidence.oriented,...(collectPairs?{pairs:Array.from(e.pairs)}:{})};
}
