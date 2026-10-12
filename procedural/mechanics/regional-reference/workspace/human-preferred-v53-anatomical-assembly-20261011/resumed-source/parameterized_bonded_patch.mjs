/** Static source-parametric bone bonds. Root coordinates are represented by
 * their declared affine source features; the numerical representation error
 * is measured explicitly. Every moving face retains the full clearance guard.
 * Rigid noncontact pairs require exact disjointness AND positive separation.
 */
import {verifyParameterizedTriangleContact} from './parametric_triangle_contact.mjs';
import {verifyTriangleIntersectionFeature} from '../../human-mechanics-20261010/src/prototypes/exact_triangle_contact_feature.mjs';
import {certifyTriangleStep,trianglePairFeatures} from '../../human-mechanics-20261010/src/prototypes/complete_boundary_contact.mjs';
const tokens=new WeakMap(),at=(v,i)=>Array.from(v.slice(3*i,3*i+3)),same=(a,b)=>a.length===b.length&&a.every((x,i)=>Object.is(x,b[i])),key=f=>Array.from(f).toSorted((a,b)=>a-b).join(':');
export function createVerifiedDistributedBondedPatch({collider,positions,boundaryFaces,fixed,targets,rootRecords,rootPatchFaces,minimumDistance=.00001}){
 const roots=new Map(),stationary=new Map(),pairs=new Set(),failures=[],gapPairs=[];
 let maxNumericalRootError=0;
 for(const r of rootRecords){
  if(roots.has(r.node)||fixed[r.node]!==1||!same(at(positions,r.node),at(targets,r.node)))throw Error('Root is not unique and stationary');
  const source=r.boneVertex!==undefined?{vertices:[0,1,2].map(()=>collider.getStaticVertex(r.boneId,r.boneVertex)),weights:[1,0,0]}:{vertices:collider.getStaticTriangle(r.boneId,r.boneFace),weights:r.weights};
  const w=source.weights,sum=w.reduce((a,b)=>a+b,0),reconstructed=[0,1,2].map(k=>source.vertices.reduce((a,p,j)=>a+p[k]*w[j]/sum,0)),err=Math.hypot(...at(positions,r.node).map((x,k)=>x-reconstructed[k]));
  maxNumericalRootError=Math.max(maxNumericalRootError,err);if(err>2e-15)throw Error('Root differs from source parameter beyond binary64 reconstruction bound');
  roots.set(r.node,source);
 }
 const faceMap=new Map(boundaryFaces.map((f,i)=>[key(f),i]));
 for(const p of rootPatchFaces){
  if(!faceMap.has(key(p.nodes))||!p.nodes.every(n=>roots.has(n)))throw Error('Missing declared root face');
  const specs=Object.fromEntries(p.nodes.map((n,i)=>[i,roots.get(n)])),proof=verifyParameterizedTriangleContact(p.nodes.map(n=>at(positions,n)),collider.getStaticTriangle(p.boneId,p.boneFace),specs);
  if(!proof.valid||proof.allowedRootPoints!==3)throw Error('Root cap not contained on its source bone triangle: '+JSON.stringify({p,proof}));
 }
 const fixedFaces=[] ,globalIds=[];
 boundaryFaces.forEach((f,i)=>{if(f.some(n=>roots.has(n))&&!f.every(n=>fixed[n]===1))throw Error('Root seam outside fixed collar');if(f.every(n=>fixed[n]===1)){fixedFaces.push(f);globalIds.push(i);for(const n of f)stationary.set(n,at(positions,n));}});
 const found=collider.findCandidates(positions,positions,fixedFaces,{minimumDistance,maxBoxTests:3000000,maxCandidatePairs:120000});if(!found.complete)throw Error('Incomplete rigid-pair enumeration');
 let features=0,strict=0;
 for(const p of found.pairs){
  const f=fixedFaces[p.tissueFace],a=f.map(n=>at(positions,n)),b=collider.getStaticTriangle(p.boneId,p.boneFace),specs=Object.fromEntries(f.flatMap((n,i)=>roots.has(n)?[[i,roots.get(n)]]:[]));
  if(Object.keys(specs).length){const proof=verifyParameterizedTriangleContact(a,b,specs);if(proof.valid&&proof.allowedRootPoints>0){pairs.add(`${globalIds[p.tissueFace]}:${p.boneId}:${p.boneFace}`);features++;continue;}if(!proof.valid){failures.push({pair:p,face:f,proof});continue;}}
  const proof=certifyTriangleStep(a,a,b,{minimumDistance,maxDepth:0,maxIntervals:1});if(proof.certified){strict++;continue;}
  const exact=verifyTriangleIntersectionFeature(a,b,[]);let gap=Math.sqrt(Math.min(...trianglePairFeatures(a,b).map(q=>q.d2)))*.5,cert=null;
  for(let i=0;i<20&&gap>0;i++,gap*=.5){cert=certifyTriangleStep(a,a,b,{minimumDistance:gap,maxDepth:0,maxIntervals:1});if(cert.certified)break;}
  if(exact.valid&&cert?.certified&&gap>0){pairs.add(`${globalIds[p.tissueFace]}:${p.boneId}:${p.boneFace}`);gapPairs.push({tissueFace:globalIds[p.tissueFace],boneId:p.boneId,boneFace:p.boneFace,certifiedPositiveGap:gap,exactDisjoint:true});}
  else failures.push({pair:p,face:f,exact,proof});
 }
 const summary={rootNodes:roots.size,rootFaces:rootPatchFaces.length,rigidFaces:fixedFaces.length,stationaryNodes:stationary.size,parameterizedFeaturePairs:features,strictRigidPairs:strict,stationarySubMarginPairs:gapPairs.length,stationaryGapPairs:gapPairs,maxNumericalRootErrorSolverUnits:maxNumericalRootError,minimumDistanceMoving:minimumDistance,allPairsChecked:found.pairs.length,failures,fullUniformMarginClaim:false,movingPairsExempted:0};
 if(failures.length){const e=new Error('Rigid attachment audit failed '+failures.length+' pairs');e.report=summary;throw e;}
 const token=Object.freeze({summary:structuredClone(summary)});tokens.set(token,{collider,faces:boundaryFaces.map(f=>Array.from(f)),stationary,pairs,minimumDistance});return token;
}
export function verifyDistributedBondedState(token,{positions,fixed,targets}){const c=tokens.get(token);if(!c)return{valid:false,reason:'unverified-token'};for(const[n,p]of c.stationary)if(fixed[n]!==1||!same(at(positions,n),p)||!same(at(targets,n),p))return{valid:false,reason:'rigid-node-moved'};return{valid:true};}
export function verifyDistributedBondedTrajectory(token,{collider,start,end,faces,minimumDistance}){const c=tokens.get(token);if(!c||c.collider!==collider||c.minimumDistance!==minimumDistance||faces.length!==c.faces.length||!faces.every((f,i)=>same(f,c.faces[i])))return{valid:false,reason:'bond-collider-clearance-topology-changed'};for(const[n,p]of c.stationary)if(!same(at(start,n),p)||!same(at(end,n),p))return{valid:false,reason:'rigid-node-moved'};return{valid:true};}
export function permitsVerifiedDistributedBondedPair(token,pair){const c=tokens.get(token);return !!c&&c.pairs.has(`${pair.tissueFace}:${pair.boneId}:${pair.boneFace}`);}
