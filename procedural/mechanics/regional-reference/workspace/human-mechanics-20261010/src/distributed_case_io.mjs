/** Exact-source fixed root binding for the authored distributed attachment.
 * All other contact nodes retain the inherited finite nodal penalty. The
 * immutable root is a Dirichlet bond, not a positive-clearance contact sample.
 * Complete trajectory certification is separately mandatory in the runner.
 */
import {loadBilateralCase} from './bilateral_case_io.mjs';
import {InterdigitatedMuscleFEM} from './interdigitated_muscle_fem.mjs';
import {createVerifiedDistributedBondedPatch} from './prototypes/distributed_bonded_patch.mjs';
import {extractCompleteFEMBoundary} from './prototypes/static_bone_swept_bvh_distributed.mjs';

export function loadDistributedCase(dir){
 const a=loadBilateralCase(dir),attachments=a.d.distributed_origin_attachments;
 if(!Array.isArray(attachments)||attachments.length!==2)throw Error('Expected both explicit distributed origins');
 const roots=new Set(),collar=new Set(),rootVertexMap=[],rootPatchFaces=[];
 for(const q of attachments){
  if(q.root_nodes.length!==q.root_combined_bone_vertex_ids.length||q.root_face_nodes.length!==q.root_combined_bone_triangle_ids.length)throw Error('Attachment metadata layout mismatch');
  q.root_nodes.forEach((node,i)=>{
   if(roots.has(node))throw Error('Repeated root');roots.add(node);
   const boneVertex=q.root_combined_bone_vertex_ids[i];
   for(let k=0;k<3;k++)if(!Object.is(a.arrays.rest[3*node+k],a.boneVertices[3*boneVertex+k]))throw Error('Root is not exact bone coordinate');
   rootVertexMap.push({node,boneId:'combined',boneVertex});
  });
  for(const n of q.fixed_collar_nodes)collar.add(n);
  q.root_face_nodes.forEach((nodes,i)=>rootPatchFaces.push({nodes,boneId:'combined',boneFace:q.root_combined_bone_triangle_ids[i]}));
 }
 if(roots.size!==a.d.immutable_bonded_root_nodes.length||a.d.immutable_bonded_root_nodes.some(n=>!roots.has(n)))throw Error('Unverified penalty omission');
 const originalCreate=a.create;
 function create(FEMClass=InterdigitatedMuscleFEM){
  const s=originalCreate(FEMClass);
  for(const n of roots){if(s.fixed[n]!==1||!collar.has(n))throw Error('Bonded root not fixed');for(let k=0;k<3;k++)if(!Object.is(s.target[3*n+k],s.rest[3*n+k]))throw Error('Bonded root target moved');}
  const before=s.solidMeshContacts.length;
  s.solidMeshContacts=s.solidMeshContacts.filter(q=>!roots.has(q.node));
  if(before-s.solidMeshContacts.length!==roots.size)throw Error('Every root must be an actual boundary contact sample');
  return s;
 }
 function makeBondedToken(collider,s,minimumDistance){
  const faces=extractCompleteFEMBoundary(s.nodes,s.mass.length);
  const token=createVerifiedDistributedBondedPatch({collider,positions:s.x,boundaryFaces:faces,
   fixed:s.fixed,targets:s.target,rootVertexMap,rootPatchFaces,
   staticFaces:faces.filter(f=>f.some(n=>roots.has(n))),fixedCollarNodes:[...collar],minimumDistance});
  return token;
 }
 return {...a,create,makeBondedToken,contactSummary:{...a.contactSummary,
  allBoundaryNodes:a.contactSummary.nodes,nodes:a.contactSummary.nodes-roots.size,
  immutableBondedRootNodes:roots.size,omittedPenaltyNodes:[...roots],
  method:'Area-lumped finite nodal penalty on all unbonded boundary nodes; exact immutable bone roots use Dirichlet bonds',
  bondedRootReactionSource:'Full constitutive gradient before Dirichlet projection; no artificial clearance penalty on bonded root'}};
}
