/**
 * Original exact-source-vertex distributed STATIC bone patch admission.
 * No region/patch-face exclusion: every nearby bone triangle is discovered,
 * including incident edge/vertex neighbors outside the selected root patch.
 * Only exact verified shared source features become immutable pair exceptions.
 * All other pairs stay on the strict positive-clearance narrow phase.
 * This verifies surface intersection features, NOT outside-solid containment.
 */
import {verifyTriangleIntersectionFeature} from './exact_triangle_contact_feature.mjs';
import {certifyTriangleStep} from './complete_boundary_contact.mjs';

const tokens=new WeakMap(),key=f=>Array.from(f).toSorted((a,b)=>a-b).join(':'),
  same=(a,b)=>a.length===b.length&&a.every((v,i)=>Object.is(v,b[i])),
  at=(v,i)=>typeof v[0]==='number'?[v[3*i],v[3*i+1],v[3*i+2]]:Array.from(v[i]||[]),
  count=v=>typeof v[0]==='number'?v.length/3:v.length,
  finite3=p=>p&&p.length===3&&p.every(Number.isFinite),fail=reason=>({valid:false,reason});
export class DistributedBondedPatchError extends Error {
  constructor(reason,details={}){super(`Distributed bonded patch rejected: ${reason}`);this.name='DistributedBondedPatchError';this.reason=reason;this.details=details;}
}
const reject=(reason,details)=>{throw new DistributedBondedPatchError(reason,details);};

export function createVerifiedDistributedBondedPatch({collider,positions,boundaryFaces,fixed,targets,rootVertexMap,rootPatchFaces,
  staticFaces,fixedCollarNodes,minimumDistance=0,maxBoxTests=2000000,maxCandidatePairs=50000,maxTotalIntervals=100000,maxDepth=24,maxIntervalsPerPair=2048}={}) {
  if(!collider||typeof collider.getStaticVertex!=='function'||typeof collider.getStaticTriangleRecord!=='function'||typeof collider.findCandidates!=='function')reject('requires-distributed-static-bvh');
  if(!positions||!targets||!fixed||!Array.isArray(boundaryFaces)||!boundaryFaces.length||boundaryFaces.length>50000)reject('invalid-boundary-input');
  const n=count(positions);if(!Number.isInteger(n)||n<3||n>300000||count(targets)!==n||fixed.length!==n)reject('invalid-coordinate-layout');
  if(!Number.isFinite(minimumDistance)||minimumDistance<0)reject('invalid-clearance');
  for(const [name,list,max] of [['roots',rootVertexMap,4096],['root-patch-faces',rootPatchFaces,8192],['static-faces',staticFaces,8192],['fixed-collar',fixedCollarNodes,8192]])if(!Array.isArray(list)||!list.length||list.length>max)reject(`invalid-${name}`);
  const faceMap=new Map();boundaryFaces.forEach((f,i)=>{if(f.length!==3||new Set(f).size!==3||!f.every(j=>Number.isInteger(j)&&j>=0&&j<n)||faceMap.has(key(f)))reject('invalid-or-duplicate-boundary-face');faceMap.set(key(f),i);});
  const roots=new Map(),sourceOwners=new Set(),stationary=new Map();
  for(const r of rootVertexMap) {
    if(!r||!Number.isInteger(r.node)||r.node<0||r.node>=n||roots.has(r.node)||typeof r.boneId!=='string'||!Number.isInteger(r.boneVertex)||r.boneVertex<0)reject('invalid-root-map');
    const sourceKey=`${r.boneId}:${r.boneVertex}`;if(sourceOwners.has(sourceKey))reject('source-vertex-used-by-multiple-root-nodes');sourceOwners.add(sourceKey);
    const source=collider.getStaticVertex(r.boneId,r.boneVertex),p=at(positions,r.node);
    if(!finite3(p)||!same(p,source))reject('root-is-not-exact-source-vertex',{node:r.node,boneId:r.boneId,boneVertex:r.boneVertex});
    roots.set(r.node,{boneId:r.boneId,boneVertex:r.boneVertex});
  }
  for(const node of fixedCollarNodes) {
    if(!Number.isInteger(node)||node<0||node>=n||stationary.has(node))reject('invalid-or-duplicate-fixed-collar-node');
    const p=at(positions,node);if(fixed[node]!==1||!finite3(p)||!same(p,at(targets,node)))reject('fixed-collar-node-is-not-pinned-at-current-position',{node});stationary.set(node,p);
  }
  for(const node of roots.keys())if(!stationary.has(node))reject('root-outside-declared-fixed-collar',{node});
  const patchKeys=new Set(),patchBoneFaces=new Set(),usedRoots=new Set();
  for(const p of rootPatchFaces) {
    const k=key(p.nodes||[]);if(!faceMap.has(k)||patchKeys.has(k)||p.nodes.length!==3||!p.nodes.every(i=>roots.has(i)))reject('invalid-or-duplicate-root-cap');
    const source=collider.getStaticTriangleRecord(p.boneId,p.boneFace),mapped=p.nodes.map(i=>roots.get(i));
    if(mapped.some(r=>r.boneId!==p.boneId)||key(mapped.map(r=>r.boneVertex))!==key(source.vertexIds))reject('root-cap-does-not-preserve-original-bone-triangle',{nodes:p.nodes,boneId:p.boneId,boneFace:p.boneFace});
    const bk=`${p.boneId}:${p.boneFace}`;if(patchBoneFaces.has(bk))reject('duplicate-source-root-triangle');
    patchKeys.add(k);patchBoneFaces.add(bk);for(const node of p.nodes)usedRoots.add(node);
  }
  if(usedRoots.size!==roots.size)reject('root-map-contains-unused-source-vertex');
  for(const face of boundaryFaces)if(face.every(i=>roots.has(i))&&!patchKeys.has(key(face)))reject('undeclared-all-root-boundary-face',{face});
  const declared=new Set(),globalFaceIds=[];
  for(const face of staticFaces) {
    const k=key(face),index=faceMap.get(k);if(index===undefined||declared.has(k))reject('invalid-or-duplicate-static-face');
    if(!face.some(i=>roots.has(i)))reject('arbitrary-static-adjacency-exclusion');
    if(!face.every(i=>stationary.has(i)))reject('static-contact-face-outside-fixed-collar',{face});
    declared.add(k);globalFaceIds.push(index);
  }
  for(const face of boundaryFaces)if(face.some(i=>roots.has(i))&&!declared.has(key(face)))reject('root-contact-seam-is-not-explicitly-fixed',{face});
  const work={minimumDistance,maxBoxTests,maxCandidatePairs,maxTotalIntervals,maxDepth,maxIntervalsPerPair};
  if(!Number.isInteger(maxTotalIntervals)||maxTotalIntervals<1||maxTotalIntervals>5000000)reject('invalid-interval-budget');
  const candidates=collider.findCandidates(positions,positions,staticFaces,work);
  if(!candidates.complete)reject('incomplete-static-bone-candidate-enumeration',candidates);
  const pairs=new Set(),features={vertex:0,edge:0,face:0};let intervals=0,strictStaticPairs=0,adjacentBonePairsOutsidePatch=0;
  for(const pair of candidates.pairs) {
    const face=staticFaces[pair.tissueFace],source=collider.getStaticTriangleRecord(pair.boneId,pair.boneFace),a=face.map(i=>at(positions,i)),shared=[];
    for(let i=0;i<3;i++){const root=roots.get(face[i]);if(root&&root.boneId===pair.boneId){const j=source.vertexIds.indexOf(root.boneVertex);if(j>=0)shared.push({a:i,b:j});}}
    if(shared.length) {
      const proof=verifyTriangleIntersectionFeature(a,source.vertices,shared);
      if(!proof.valid)reject('intersection-outside-declared-source-feature',{pair,face,shared,proof});
      pairs.add(`${globalFaceIds[pair.tissueFace]}:${pair.boneId}:${pair.boneFace}`);features[['vertex','edge','face'][shared.length-1]]++;
      if(!patchBoneFaces.has(`${pair.boneId}:${pair.boneFace}`))adjacentBonePairsOutsidePatch++;
    }else {
      strictStaticPairs++;const remaining=maxTotalIntervals-intervals;if(remaining<1)reject('static-verification-interval-budget');
      const proof=certifyTriangleStep(a,a,source.vertices,{minimumDistance,maxDepth,maxIntervals:Math.min(maxIntervalsPerPair,remaining)});intervals+=proof.intervals;
      if(!proof.certified)reject('undeclared-static-pair-does-not-meet-strict-clearance',{pair,face,proof});
    }
  }
  if(features.face!==rootPatchFaces.length)reject('source-root-cap-contact-count-mismatch',{verified:features.face,declared:rootPatchFaces.length});
  const summary={rootVertices:roots.size,originalRootFaces:rootPatchFaces.length,staticFaces:staticFaces.length,stationaryNodes:stationary.size,admittedPairs:pairs.size,
    featureCounts:features,adjacentBonePairsOutsidePatch,strictStaticPairs,boxTests:candidates.boxTests,candidatePairs:candidates.candidatePairs,intervals,
    minimumDistance,exactSourceVertexIdentity:true,exactIntersectionFeatures:true,movingFacesExempted:0,regionExclusions:0,solidContainmentChecked:false};
  const token=Object.freeze({summary:structuredClone(summary)});tokens.set(token,{collider,faces:boundaryFaces.map(f=>Array.from(f)),stationary,pairs,minimumDistance});return token;
}
export function verifyDistributedBondedState(token,{positions,fixed,targets}) {
  const c=tokens.get(token);if(!c)return fail('unverified-distributed-bonded-token');
  for(const [node,p] of c.stationary)if(fixed[node]!==1||!same(at(positions,node),p)||!same(at(targets,node),p))return fail('distributed-fixed-collar-invariant-changed');return {valid:true};
}
export function verifyDistributedBondedTrajectory(token,{collider,start,end,faces,minimumDistance}) {
  const c=tokens.get(token);if(!c||c.collider!==collider)return fail('distributed-bonded-collider-mismatch');
  if(c.minimumDistance!==minimumDistance||faces.length!==c.faces.length||!faces.every((f,i)=>same(Array.from(f),c.faces[i])))return fail('distributed-bonded-clearance-or-topology-mismatch');
  for(const [node,p] of c.stationary)if(!same(at(start,node),p)||!same(at(end,node),p))return fail('distributed-bonded-collar-is-not-stationary');return {valid:true};
}
export function permitsVerifiedDistributedBondedPair(token,pair){const c=tokens.get(token);return !!c&&c.pairs.has(`${pair.tissueFace}:${pair.boneId}:${pair.boneFace}`);}
