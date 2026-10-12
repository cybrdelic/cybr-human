import fs from 'node:fs';import path from 'node:path';import crypto from 'node:crypto';
import {loadIntegratedCase} from './integrated_case_io.mjs';
import {StaticBoneSweptBVH,extractCompleteFEMBoundary} from './static_bone_swept_bvh.mjs';
import {certifyTriangleStep,trianglePairFeatures} from '../../human-mechanics-20261010/src/prototypes/complete_boundary_contact.mjs';
import {permitsVerifiedDistributedBondedPair} from './parameterized_bonded_patch.mjs';
const[dir,out]=process.argv.slice(2);if(!dir||!out||fs.existsSync(out))throw Error('New output required');fs.mkdirSync(out,{recursive:true});
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
let report={domain_sha256:sha(path.join(dir,'domain.json')),admitted:false};
try{
 const a=loadIntegratedCase(dir),s=a.create(),bvh=new StaticBoneSweptBVH({bones:[{id:'combined',vertices:a.boneVertices,triangles:a.boneTriangles}],leafSize:8});
 const bond=a.makeBondedToken(bvh,s,.00001);report.bond=bond.summary;
 const faces=extractCompleteFEMBoundary(s.nodes,s.mass.length),found=bvh.findCandidates(s.x,s.x,faces,{minimumDistance:.00001,maxBoxTests:3000000,maxCandidatePairs:120000});if(!found.complete)throw Error('Incomplete full rest enumeration');
 const failures=[];let admitted=0,exempt=0;
 for(const p of found.pairs){
  if(permitsVerifiedDistributedBondedPair(bond,p)){exempt++;continue;}
  const face=faces[p.tissueFace],x=face.map(n=>Array.from(s.x.slice(3*n,3*n+3))),b=bvh.getStaticTriangle(p.boneId,p.boneFace),q=certifyTriangleStep(x,x,b,{minimumDistance:.00001,maxDepth:0,maxIntervals:1});
  if(q.certified)admitted++;else failures.push({pair:p,nodes:face,positions:x,minimumFeatureDistance:Math.sqrt(Math.min(...trianglePairFeatures(x,b).map(r=>r.d2))),allFixed:face.every(n=>s.fixed[n]===1)});
 }
 const eval0=s.evaluate(s.x,Infinity,true),rootSet=new Set(a.d.immutable_bonded_root_nodes),farSet=new Set(a.d.far_support_nodes);let contactE=0;const forces={root:[0,0,0],collar:[0,0,0],far:[0,0,0],free:[0,0,0]};
 for(let n=0;n<s.mass.length;n++){const name=rootSet.has(n)?'root':farSet.has(n)?'far':s.fixed[n]?'collar':'free';for(let k=0;k<3;k++)forces[name][k]+=s.g[3*n+k];}
 for(const q of s.solidMeshContacts)contactE+=q.collider.evaluate(Array.from(s.x.slice(3*q.node,3*q.node+3))).energy;
 fs.writeFileSync(path.join(out,'contact_reference.json'),JSON.stringify(a.contactReference));
 report={...report,full_boundary_faces:faces.length,enumerated_pairs:found.pairs.length,strict_margin_pairs:admitted,rigid_pair_exceptions:exempt,moving_margin_failures:failures,neutral:{...eval0,contactEnergy:contactE,signedGradientSums:forces,positionsExact:Array.from(s.x).every((v,i)=>Object.is(v,s.rest[i]))},admitted:!failures.length&&eval0.residual<1e-7&&contactE===0};
}catch(e){report.error={message:e.message,details:e.report||null};}
fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));if(!report.admitted)process.exitCode=1;
