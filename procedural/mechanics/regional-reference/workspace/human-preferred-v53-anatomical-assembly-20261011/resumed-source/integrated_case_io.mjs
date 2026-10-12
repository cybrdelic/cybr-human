import {loadBilateralCase} from '../../human-mechanics-20261010/src/bilateral_case_io.mjs';
import {InterdigitatedMuscleFEM} from '../../human-mechanics-20261010/src/interdigitated_muscle_fem.mjs';
import {createVerifiedDistributedBondedPatch} from './parameterized_bonded_patch.mjs';
import {extractCompleteFEMBoundary} from './static_bone_swept_bvh.mjs';
export function loadIntegratedCase(dir){
 const a=loadBilateralCase(dir),roots=new Set(a.d.immutable_bonded_root_nodes),restContacts=[],distances=[];
 const original=a.create;
 function create(FEMClass=InterdigitatedMuscleFEM){
  const s=original(FEMClass);s.solidMeshContacts=s.solidMeshContacts.filter(q=>!roots.has(q.node));
  for(const q of s.solidMeshContacts){
   const p=Array.from(s.rest.slice(q.node*3,q.node*3+3)),base=a.collider.closestPoint(p),restDistance=base.signedDistance;
   if(!(restDistance>0))throw Error('Nonpositive rest contact distance at '+q.node+': '+restDistance);
   const clearance=Math.min(.0003,restDistance),k=q.collider.stiffness;
   q.collider={stiffness:k,evaluate(p){const r=a.collider.closestPoint(p),violation=Math.max(clearance-r.signedDistance,0);return{energy:.5*k*violation*violation,gradient:r.normal.map(n=>-k*violation*n),normal:r.normal,violation,signedDistance:r.signedDistance,closestPoint:r.point,triangle:r.triangle,inside:r.inside};}};
   distances.push({node:q.node,restDistance,clearance,stiffness:k});
  }
  return s;
 }
 function makeBondedToken(collider,s,minimumDistance){return createVerifiedDistributedBondedPatch({collider,positions:s.x,boundaryFaces:extractCompleteFEMBoundary(s.nodes,s.mass.length),fixed:s.fixed,targets:s.target,rootRecords:a.d.root_records,rootPatchFaces:a.d.root_patch_faces,minimumDistance});}
 return{...a,create,makeBondedToken,contactReference:distances,contactSummary:{method:'Area-lumped unilateral quadratic energy with c_n=min(.0003,positive rest gap). Stress-free authored reference; positive stiffness retained.',density:1e8,rootsOmitted:roots.size,scope:'Root bonds carry constitutive reaction; moving bone feasibility remains separate.',units:'Uncalibrated solver units, design coordinates scaled .001'}};
}
