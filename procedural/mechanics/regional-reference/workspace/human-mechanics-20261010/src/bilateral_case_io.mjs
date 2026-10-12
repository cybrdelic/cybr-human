/** Original connected regional mechanics loader. No source geometry is altered. */
import fs from 'node:fs';import path from 'node:path';
import {VolumetricMuscleFEM} from './volumetric_muscle_fem.mjs';
import {BoneMeshCollider} from './bone_mesh_contact.mjs';
import {nodalBlockJacobiPCG} from './nodal_block_jacobi_fem.mjs';
const ownedBuffer=p=>{const b=fs.readFileSync(p);return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)};
export function loadBilateralCase(dir){
 const d=JSON.parse(fs.readFileSync(path.join(dir,'domain.json'),'utf8'));if(d.schema!=='connected-bilateral-native-mechanics-v1')throw Error('Wrong domain schema');
 const bytes=ownedBuffer(path.join(dir,d.binary)),arrays={};for(const [k,q] of Object.entries(d.arrays)){const Type=q.dtype==='<f8'?Float64Array:q.dtype==='<u4'?Uint32Array:null;if(!Type)throw Error('Unknown binary array type');const n=q.shape.reduce((x,y)=>x*y,1);if(n*Type.BYTES_PER_ELEMENT!==q.bytes)throw Error('Wrong byte count');arrays[k]=new Type(bytes,q.byte_offset,n);}
 const bb=ownedBuffer(path.join(dir,d.bones.binary)),boneVertices=new Float64Array(bb,0,d.bones.vertex_count*3),boneTriangles=new Uint32Array(bb,d.bones.vertex_count*24,d.bones.triangle_count*3);
 const collider=new BoneMeshCollider({vertices:boneVertices,triangles:boneTriangles,stiffness:1,clearance:.0003});
 const areas=new Float64Array(d.model.nodes);for(const f of d.boundary_triangles){const x=f.map(n=>Array.from(arrays.rest.subarray(n*3,n*3+3))),u=x[1].map((v,k)=>v-x[0][k]),v=x[2].map((v,k)=>v-x[0][k]);const area=.5*Math.hypot(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]);for(const n of f)areas[n]+=area/3;}
 const density=1e8,contacts=[];for(let node=0;node<areas.length;node++)if(areas[node]>0){const scale=areas[node]*density;const weighted={stiffness:scale,evaluate(p){const q=collider.evaluate(p);return {...q,energy:q.energy*scale,gradient:q.gradient.map(v=>v*scale)};}};contacts.push({node,collider:weighted});}
 function create(FEMClass=VolumetricMuscleFEM){const s=new FEMClass(d.model,arrays,[],[],d.muscles);for(const n of d.fixed_nodes)s.pin(n,Array.from(s.rest.subarray(n*3,n*3+3)));s.solidMeshContacts=contacts;s.pointSprings=d.omitted_deep_tissue_spring_nodes.map(node=>({node,target:Array.from(s.rest.subarray(node*3,node*3+3)),k:d.omitted_deep_tissue_spring_stiffness}));s.pcg=nodalBlockJacobiPCG;s.gravity=[0,0,0];return s;}
 return {d,arrays,boneVertices,boneTriangles,collider,create,contactSummary:{method:'Area-lumped finite nodal positive-clearance penalty; separate continuous feasibility guard required',nodes:contacts.length,density,clearanceSolverUnits:.0003,clearanceDesignUnits:.3,calibrated:false}};
}
