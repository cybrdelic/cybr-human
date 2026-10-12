/** Opt-in authority for exact moving attachment pairs. Tokens are private
 * WeakMap capabilities; their public summaries do not grant any exemption.
 * No force, energy, constitutive, PCG, quadrature, or line-search law changes. */
import {certifyExactMovingAttachment,sourceRepresentationWithinBound} from './exact_moving_attachment_contact.mjs';
import {verifyParameterizedTriangleContact} from '../../../procedural/mechanics/regional-reference/workspace/human-preferred-v53-anatomical-assembly-20261011/resumed-source/parametric_triangle_contact.mjs';
import {verifyTriangleIntersectionFeature} from '../../../procedural/mechanics/regional-reference/workspace/human-mechanics-20261010/src/prototypes/exact_triangle_contact_feature.mjs';
import {certifyTriangleStep,trianglePairFeatures} from '../../../procedural/mechanics/regional-reference/workspace/human-mechanics-20261010/src/prototypes/complete_boundary_contact.mjs';
import {isUnmodifiedStaticBoneCollider} from './static_bone_swept_bvh.mjs';
const tokens=new WeakMap(),BOUND=2e-15;
const at=(v,i)=>Array.from(v.slice(3*i,3*i+3));
const same=(a,b)=>a&&b&&a.length===b.length&&Array.from(a).every((x,i)=>Object.is(x,b[i]));
const faceKey=f=>Array.from(f).toSorted((a,b)=>a-b).join(':');
const pairKey=p=>JSON.stringify([p.tissueFace,p.boneId,p.boneFace]);
function fail(reason,report){const e=new Error(reason);e.report=report;throw e;}
function validateArrays(positions,fixed,targets){
 if(!positions||!Number.isInteger(positions.length)||positions.length<9||positions.length%3||positions.length>900000||!Array.from(positions).every(Number.isFinite))throw Error('Invalid finite positions');
 if(!fixed||fixed.length!==positions.length/3||!Array.from(fixed).every(x=>x===0||x===1)||!targets||targets.length!==positions.length||!Array.from(targets).every(Number.isFinite))throw Error('Invalid fixed/target context');
}
function specsFor(c,face){return Object.fromEntries(face.flatMap((n,i)=>c.roots.has(n)?[[i,c.roots.get(n)]]:[]));}
export function createVerifiedMovingBondedPatch({collider,positions,boundaryFaces,fixed,targets,rootRecords,rootPatchFaces=[],minimumDistance=.00001}){
 if(!isUnmodifiedStaticBoneCollider(collider))throw Error('Expected unmodified immutable StaticBoneSweptBVH');
 validateArrays(positions,fixed,targets);
 if(!Number.isFinite(minimumDistance)||minimumDistance<0)throw Error('Invalid minimumDistance');
 if(!Array.isArray(boundaryFaces)||!boundaryFaces.length||boundaryFaces.length>50000)throw Error('Invalid boundary topology');
 const faces=boundaryFaces.map(f=>{if(!f||f.length!==3||new Set(f).size!==3||!Array.from(f).every(n=>Number.isInteger(n)&&n>=0&&n<fixed.length))throw Error('Invalid boundary face');return Array.from(f);});
 const faceKeys=new Set(faces.map(faceKey));if(faceKeys.size!==faces.length)throw Error('Duplicate boundary face');
 if(!Array.isArray(rootRecords)||!rootRecords.length||rootRecords.length>fixed.length||!Array.isArray(rootPatchFaces)||rootPatchFaces.length>faces.length)throw Error('Invalid root declarations');
 const roots=new Map(),stationary=new Map(),fixedSnapshot=Array.from(fixed);
 for(let n=0;n<fixed.length;n++)if(fixed[n]===1){const p=at(positions,n);if(!same(p,at(targets,n)))throw Error('Fixed target differs from admitted coordinate');stationary.set(n,p);}
 const boundaryNodes=new Set(faces.flat());
 for(const r of rootRecords){
  if(!r||!Number.isInteger(r.node)||!boundaryNodes.has(r.node)||roots.has(r.node)||fixed[r.node]!==1)throw Error('Root is not unique, fixed, and on boundary');
  const vertex=r.boneVertex!==undefined;if(vertex===(r.boneFace!==undefined)||typeof r.boneId!=='string')throw Error('Ambiguous or missing source feature');
  const source=vertex?{vertices:[0,1,2].map(()=>collider.getStaticVertex(r.boneId,r.boneVertex)),weights:[1,0,0]}:{vertices:collider.getStaticTriangle(r.boneId,r.boneFace),weights:Array.from(r.weights??[])};
  if(vertex&&!same(at(positions,r.node),source.vertices[0]))throw Error('Literal bone vertex root identity mismatch');
  if(!sourceRepresentationWithinBound(at(positions,r.node),source,BOUND))throw Error('Root differs from exact normalized source beyond 2e-15 bound');
  roots.set(r.node,source);
 }
 const patchKeys=new Set();
 for(const p of rootPatchFaces){
  if(!p||!p.nodes||p.nodes.length!==3||new Set(p.nodes).size!==3||!faceKeys.has(faceKey(p.nodes))||!Array.from(p.nodes).every(n=>roots.has(n)))throw Error('Missing declared root face');
  const k=faceKey(p.nodes);if(patchKeys.has(k))throw Error('Duplicate declared root face');patchKeys.add(k);
  const specs=Object.fromEntries(Array.from(p.nodes,(n,i)=>[i,roots.get(n)])),proof=verifyParameterizedTriangleContact(Array.from(p.nodes,n=>at(positions,n)),collider.getStaticTriangle(p.boneId,p.boneFace),specs);
  if(!proof.valid||proof.allowedRootPoints!==3)fail('Root cap not contained on source triangle',{p,proof});
 }
 const c={collider,faces,roots,stationary,fixedSnapshot,fixedReference:fixed,targetReference:targets,minimumDistance,stationaryPairs:new Set(),movingFaces:new Set(),padding:BOUND};
 const fixedFaces=[],fixedGlobal=[],movingFaces=[],movingGlobal=[];
 faces.forEach((f,i)=>{if(f.every(n=>fixed[n]===1)){fixedFaces.push(f);fixedGlobal.push(i);}else if(f.some(n=>roots.has(n))){c.movingFaces.add(i);movingFaces.push(f);movingGlobal.push(i);}});
 const enumerate=fs=>fs.length?collider.findCandidates(positions,positions,fs,{minimumDistance:Math.max(minimumDistance,BOUND),maxBoxTests:3000000,maxCandidatePairs:120000}):{complete:true,pairs:[]};
 const found=enumerate(fixedFaces);if(!found.complete)fail('Incomplete rigid-pair enumeration',found);
 let features=0,strict=0;const gapPairs=[],failures=[];
 // Retain the previous stationary policy: exact source contact feature, or
 // strict requested margin, or exact disjointness plus a certified positive gap.
 for(const p of found.pairs){
  const f=fixedFaces[p.tissueFace],global={...p,tissueFace:fixedGlobal[p.tissueFace]},a=f.map(n=>at(positions,n)),b=collider.getStaticTriangle(p.boneId,p.boneFace),specs=specsFor(c,f);
  if(Object.keys(specs).length){const proof=verifyParameterizedTriangleContact(a,b,specs);if(proof.valid&&proof.allowedRootPoints>0){c.stationaryPairs.add(pairKey(global));features++;continue;}if(!proof.valid){failures.push({pair:global,proof});continue;}}
  const proof=certifyTriangleStep(a,a,b,{minimumDistance,maxDepth:0,maxIntervals:1});if(proof.certified){strict++;continue;}
  const exact=verifyTriangleIntersectionFeature(a,b,[]);let gap=Math.sqrt(Math.min(...trianglePairFeatures(a,b).map(q=>q.d2)))*.5,cert=null;
  for(let i=0;i<20&&gap>0;i++,gap*=.5){cert=certifyTriangleStep(a,a,b,{minimumDistance:gap,maxDepth:0,maxIntervals:1});if(cert.certified)break;}
  if(exact.valid&&cert?.certified&&gap>0){c.stationaryPairs.add(pairKey(global));gapPairs.push({...global,certifiedPositiveGap:gap,exactDisjoint:true});}else failures.push({pair:global,proof,exact});
 }
 if(failures.length)fail('Rigid attachment audit failed',{failures});
 const movingFound=enumerate(movingFaces);if(!movingFound.complete)fail('Incomplete moving attachment enumeration',movingFound);
 let movingFeaturePairs=0,movingPositiveSeparationPairs=0;
 for(const p of movingFound.pairs){const f=movingFaces[p.tissueFace],a=f.map(n=>at(positions,n)),proof=certifyExactMovingAttachment(a,a,collider.getStaticTriangle(p.boneId,p.boneFace),specsFor(c,f));if(!proof.certified)fail('Initial moving attachment pair unresolved',{pair:{...p,tissueFace:movingGlobal[p.tissueFace]},proof});if(proof.positiveSeparation)movingPositiveSeparationPairs++;else movingFeaturePairs++;}
 const summary={scheme:'opt-in-exact-moving-root-hull-v1',rootNodes:roots.size,rootFaces:rootPatchFaces.length,rigidFaces:fixedFaces.length,stationaryNodes:stationary.size,movingRootIncidentFaces:movingFaces.length,parameterizedFeaturePairs:features,strictRigidPairs:strict,stationarySubMarginPairs:gapPairs.length,stationaryGapPairs:gapPairs,initialMovingAttachmentPairs:movingFound.pairs.length,initialMovingFeaturePairs:movingFeaturePairs,initialMovingPositiveSeparationPairs:movingPositiveSeparationPairs,minimumDistanceOtherPairs:minimumDistance,rootNumericalRepresentationBoundSolverUnits:BOUND,rootRepresentation:'exact normalized binary64 source parameters; rounded solver coordinates within stated bound',fullUniformMarginClaim:false,movingPairsBlanketExempted:0};
 const token=Object.freeze({summary:Object.freeze(structuredClone(summary))});tokens.set(token,c);return token;
}
export function verifyMovingBondedState(token,{positions,fixed,targets}){
 const c=tokens.get(token);if(!c)return{valid:false,reason:'unverified-moving-attachment-token'};
 if(!isUnmodifiedStaticBoneCollider(c.collider))return{valid:false,reason:'bone-collider-implementation-changed'};
 if(!same(fixed,c.fixedSnapshot)||!positions||positions.length!==c.fixedSnapshot.length*3||!targets||targets.length!==positions.length)return{valid:false,reason:'moving-attachment-pin-context-changed'};
 for(const[n,p]of c.stationary)if(!same(at(positions,n),p)||!same(at(targets,n),p))return{valid:false,reason:'moving-attachment-fixed-node-moved'};
 return{valid:true};
}
export function verifyMovingBondedTrajectory(token,{collider,start,end,faces,minimumDistance}){
 const c=tokens.get(token);if(!c)return{valid:false,reason:'unverified-moving-attachment-token'};
 if(c.collider!==collider||c.minimumDistance!==minimumDistance||!faces||faces.length!==c.faces.length||!faces.every((f,i)=>same(f,c.faces[i])))return{valid:false,reason:'moving-attachment-collider-clearance-topology-changed'};
 for(const positions of[start,end]){const check=verifyMovingBondedState(token,{positions,fixed:c.fixedReference,targets:c.targetReference});if(!check.valid)return check;}
 return{valid:true};
}
export function movingBondedBroadPhasePadding(token){return tokens.get(token)?.padding??0;}
/** Caller must first validate the complete trajectory; BVH does so before any
 * pair can be skipped. Each moving pair gets a fresh exact continuous proof. */
export function certifyVerifiedMovingBondedPair(token,pair,{startTriangle,endTriangle}){
 const c=tokens.get(token);if(!c)return{applicable:true,certified:false,reason:'unverified-moving-attachment-token'};
 if(c.stationaryPairs.has(pairKey(pair)))return{applicable:true,certified:true,stationary:true,reason:'verified-stationary-attachment-pair'};
 if(!c.movingFaces.has(pair.tissueFace))return{applicable:false};
 const f=c.faces[pair.tissueFace],proof=certifyExactMovingAttachment(startTriangle,endTriangle,c.collider.getStaticTriangle(pair.boneId,pair.boneFace),specsFor(c,f));
 return{applicable:true,stationary:false,...proof};
}
