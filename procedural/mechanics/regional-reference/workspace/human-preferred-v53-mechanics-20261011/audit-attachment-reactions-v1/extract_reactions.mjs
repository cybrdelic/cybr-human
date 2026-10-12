/** Read-only state audit. Does not step, optimize, move, or rewrite accepted states. */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {loadDistributedCase} from '../src/regional_case_io.mjs';
const B=path.resolve(path.dirname(new URL(import.meta.url).pathname),'..'),O=path.join(B,'audit-attachment-reactions-v1');
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const files=['domain.json','arrays.bin','bones.bin'].map(n=>path.join(B,'cheek-case-v3',n));
const phases=['neutral-0','right-02','right-08','right-16','released-0'];
const stateFiles=phases.map(n=>path.join(B,'cheek-equilibrium-v3',`${n}-state.json`));
const before=Object.fromEntries([...files,...stateFiles].map(p=>[p,sha(p)]));
const a=loadDistributedCase(path.join(B,'cheek-case-v3')),s=a.create();
const fixed=Array.from(s.fixed),fixedNodes=fixed.flatMap((v,i)=>v?[i]:[]),roots=a.d.immutable_bonded_root_nodes,rootSet=new Set(roots),collar=fixedNodes.filter(n=>!rootSet.has(n));
const contacts=s.solidMeshContacts;
const vector=(v,n)=>Array.from(v.subarray(n*3,n*3+3));
const norm=v=>Math.hypot(...v);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const centroid=roots.reduce((p,n)=>p.map((v,k)=>v+s.rest[3*n+k]/roots.length),[0,0,0]);
function stats(g,nodes,sign=1){
 const net=[0,0,0],torque=[0,0,0];let l2=0,sumNorm=0,maxNorm=0,maxNode=null;
 for(const n of nodes){const v=vector(g,n).map(x=>sign*x),r=vector(s.x,n).map((x,k)=>x-centroid[k]),t=cross(r,v),q=norm(v);for(let k=0;k<3;k++){net[k]+=v[k];torque[k]+=t[k];} l2+=q*q;sumNorm+=q;if(q>maxNorm){maxNorm=q;maxNode=n;}}
 return {nodes:nodes.length,net,netNorm:norm(net),torqueAboutRootCentroid:torque,torqueNorm:norm(torque),nodalL2:Math.sqrt(l2),sumNodalNorms:sumNorm,maxNodalNorm:maxNorm,maxNode};
}
let allFixedRootAdjacent=true,rootAdjacent=0,activeTouchesFixed=0;
for(let t=0;t<s.nodes.length/4;t++){const ns=Array.from(s.nodes.subarray(t*4,t*4+4));if(ns.some(n=>rootSet.has(n))){rootAdjacent++;if(!ns.every(n=>fixed[n]))allFixedRootAdjacent=false;}}
for(const r of s.muscleRegions)for(const e of r.elements)if(e.nodes.some(n=>fixed[n]))activeTouchesFixed++;
const reports=[],nodeRecords=[];
for(const [i,phase] of phases.entries()){
 const previous=JSON.parse(fs.readFileSync(stateFiles[i],'utf8'));
 if(previous.sourceDomainSha256!==before[files[0]]||previous.sourceArraysSha256!==before[files[1]]||previous.sourceBonesSha256!==before[files[2]])throw Error('Source identity mismatch');
 s.x.set(previous.positions);s.fixed.set(fixed);for(const r of s.muscleRegions)s.setMuscleActivation(0,r.id);s.setMuscleActivation(previous.activation.rightIncoming,'right-zygomatic-like');
 let maxFixedMotion=0,exactFixed=true;for(const n of fixedNodes)for(let k=0;k<3;k++){maxFixedMotion=Math.max(maxFixedMotion,Math.abs(s.x[3*n+k]-s.rest[3*n+k]));exactFixed&&=Object.is(s.x[3*n+k],s.rest[3*n+k]);}
 const projected=s.evaluate(s.x,Infinity,false),projectedG=s.g.slice();
 // Disabling the projection ONLY in this ephemeral audit instance exposes dE/dx.
 // Evaluating never enforces target positions or changes s.x; no solve is run.
 s.fixed.fill(0);const full=s.evaluate(s.x,Infinity,false),fullG=s.g.slice();
 const muscle=s.muscleForces.slice();s.solidMeshContacts=[];const internal=s.evaluate(s.x,Infinity,false),internalG=s.g.slice();s.solidMeshContacts=contacts;
 const contactG=fullG.map((v,j)=>v-internalG[j]),free=Array.from({length:s.mass.length},(_,n)=>n).filter(n=>!fixed[n]);
 let projectionDifference=0;for(const n of free)for(let k=0;k<3;k++)projectionDifference=Math.max(projectionDifference,Math.abs(fullG[3*n+k]-projectedG[3*n+k]));
 const actualContacts=contacts.map(c=>({node:c.node,...c.collider.evaluate(vector(s.x,c.node))})).filter(q=>q.violation>0);
 const report={phase,activation:previous.activation.rightIncoming,sourceStateSha256:before[stateFiles[i]],storedConverged:previous.converged,storedResidual:previous.residual,recomputedResidual:projected.residual,minJ:full.minJ,exactFixedCoordinates:exactFixed,maxFixedCoordinateMotion:maxFixedMotion,projectionDifference,energyMatches:full.energy===projected.energy,energy:full.energy,muscleEnergy:full.muscleEnergy,contactEnergy:full.energy-internal.energy,activeNodalContacts:actualContacts.length,maximumPenaltyViolation:Math.max(0,...actualContacts.map(q=>q.violation)),rootSupportOnTissue:stats(fullG,roots),collarSupportOnTissue:stats(fullG,collar),combinedSupportOnTissue:stats(fullG,fixedNodes),rootConstitutiveTissueOnSupport:stats(internalG,roots,-1),collarConstitutiveTissueOnSupport:stats(internalG,collar,-1),nodalContactBoneReaction:stats(contactG,Array.from({length:s.mass.length},(_,n)=>n)),freeResidual:stats(fullG,free),muscleForcesAtFixed:stats(muscle,fixedNodes),muscleDiagnostics:s.muscleDiagnostics};
 if(phase==='right-16'){
  const n=report.collarSupportOnTissue.maxNode,fd=[];
  for(const h of [1e-8,1e-9])for(let k=0;k<3;k++){
   const j=3*n+k,old=s.x[j];s.x[j]=old+h;const plus=s.evaluate(s.x,Infinity,false).energy;s.x[j]=old-h;const minus=s.evaluate(s.x,Infinity,false).energy;s.x[j]=old;
   const value=(plus-minus)/(2*h),analytic=fullG[j];fd.push({node:n,axis:k,h,analytic,centralDifference:value,absoluteError:Math.abs(value-analytic)});
  }
  report.finiteDifferenceSpotCheck=fd;
 }
 reports.push(report);
 for(const n of fixedNodes)nodeRecords.push({phase,node:n,group:rootSet.has(n)?'bone_root':'fixed_collar',position:vector(s.x,n),supportOnTissue:vector(fullG,n),tissueOnSupport:vector(fullG,n).map(v=>-v),constitutiveTissueForce:vector(internalG,n).map(v=>-v),muscleOnlyForce:vector(muscle,n),contactGradient:vector(contactG,n)});
 console.log(JSON.stringify({phase,rootNet:report.rootSupportOnTissue.netNorm,rootMax:report.rootSupportOnTissue.maxNodalNorm,collarNet:report.collarSupportOnTissue.netNorm,collarMax:report.collarSupportOnTissue.maxNodalNorm,collarL2:report.collarSupportOnTissue.nodalL2,contacts:report.activeNodalContacts,residual:report.recomputedResidual}));
}
const unchanged=Object.keys(before).every(p=>before[p]===sha(p));if(!unchanged)throw Error('Input changed while audit ran');
const report={schema:'preferred-v53-read-only-attachment-reactions-audit-v1',created:new Date().toISOString(),sourceHashes:before,inputsUnchanged:unchanged,nodes:s.mass.length,tetrahedra:s.nodes.length/4,fixedNodes:fixedNodes.length,rootNodes:roots.length,additionalFixedCollarNodes:collar.length,rootAdjacentTetrahedra:rootAdjacent,allRootAdjacentTetrahedraEntirelyFixed:allFixedRootAdjacent,activeTetrahedraTouchingFixedNodes:activeTouchesFixed,rootCentroid:centroid,method:'Evaluate saved positions and activation with exact original force law, first with the original fixed mask, then with only the in-memory projection mask cleared. No step, optimization or position update. Remove contacts only in a second diagnostic evaluation to decompose force sources. Inputs hashed before and after.',signConvention:'supportOnTissue = dE_total/dx at constrained nodes; tissueOnSupport is its negative. Contact bone reaction = dE_contact/dx at all nodal contacts. Constitutive tissue force = -dE_internal/dx. Torque is about declared root centroid.',units:'Uncalibrated solver force and torque units. Design length is scaled by 0.001, but this authored geometry/material has no anatomical scale/force calibration; do not label magnitudes measured human forces.',limitations:['Exact bone roots and an offset first collar are independently fixed in world coordinates. The collar is a rigid attachment approximation; its loads are not resolved stresses propagated through the all-fixed root-to-collar tissue.','These diagnostics expose reaction loads but do not feed them back to dynamic bones.','Contact is area-lumped finite nodal penalty. Continuous geometric admissibility is separately verified by the saved solver state; this audit does not rerun collision certification.','No skin-transfer, anatomical segmentation, physiological activation, tendon law, bilateral pose, skull or jaw-dynamics claim.'],states:reports};
fs.writeFileSync(path.join(O,'reactions.json'),JSON.stringify(report,null,2));fs.writeFileSync(path.join(O,'fixed_node_reactions.json'),JSON.stringify(nodeRecords));
console.log(JSON.stringify({inputsUnchanged:unchanged,rootAdjacent,allFixedRootAdjacent,activeTouchesFixed,report:path.join(O,'reactions.json')}));
