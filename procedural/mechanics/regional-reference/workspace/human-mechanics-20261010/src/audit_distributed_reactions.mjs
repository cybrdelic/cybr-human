/** Read-only signed force accounting for the actual fixed finite root/collar.
 * Separates constitutive attachment load from any constant clearance penalty
 * on immutable collar nodes. No state, force law or constraint is changed.
 */
import fs from 'node:fs';import path from 'node:path';import crypto from 'node:crypto';
import {loadDistributedCase} from './distributed_case_io.mjs';
const [dir,statePath,out]=process.argv.slice(2),a=loadDistributedCase(dir),d=JSON.parse(fs.readFileSync(statePath,'utf8')),s=a.create();
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
if(!d.converged||d.sourceDomainSha256!==sha(path.join(dir,'domain.json')))throw Error('Mismatched or unconverged input');
s.x.set(d.positions);for(const r of s.muscleRegions)s.setMuscleActivation(0,r.id);
s.setMuscleActivation(d.activation.rightIncoming,'right-zygomatic-like');
const fixed=Array.from(s.fixed);s.fixed.fill(0);s.evaluate(s.x,Infinity,false);
const vec=()=>[0,0,0],sum=(nodes,g=s.g)=>nodes.reduce((v,n)=>v.map((q,k)=>q+g[n*3+k]),vec()),mag=v=>Math.hypot(...v),subtract=(x,y)=>x.map((v,i)=>v-y[i]);
const cg=new Float64Array(s.x.length);let contactEnergy=0,fixedContactEnergy=0;
for(const q of s.solidMeshContacts){const c=q.collider.evaluate(Array.from(s.x.subarray(q.node*3,q.node*3+3)));contactEnergy+=c.energy;if(fixed[q.node])fixedContactEnergy+=c.energy;for(let k=0;k<3;k++)cg[q.node*3+k]+=c.gradient[k];}
const sg=new Float64Array(s.x.length);for(const q of s.pointSprings)for(let k=0;k<3;k++)sg[q.node*3+k]+=q.k*(s.x[q.node*3+k]-q.target[k]);
const attachments=a.d.distributed_origin_attachments,collar=new Set(attachments.flatMap(q=>q.fixed_collar_nodes)),all=Array.from({length:fixed.length},(_,i)=>i),free=all.filter(n=>!fixed[n]),clamps=a.d.fixed_nodes.filter(n=>!collar.has(n));
const regions={};
for(const q of attachments){const label=q.side===1?'right':'left';regions[`${label}_bone_bound_collar_constraint_reaction`]=sum(q.fixed_collar_nodes);
 regions[`${label}_bone_bound_collar_constitutive_reaction`]=subtract(subtract(sum(q.fixed_collar_nodes),sum(q.fixed_collar_nodes,cg)),sum(q.fixed_collar_nodes,sg));
 regions[`${label}_root_only_constitutive_reaction`]=subtract(sum(q.root_nodes),sum(q.root_nodes,cg));
 regions[`${label}_fixed_collar_clearance_force`]=sum(q.fixed_collar_nodes,cg).map(v=>-v);}
regions.omitted_interface_clamp_reaction=sum(clamps);regions.omitted_deep_spring_force=sum(all,sg).map(v=>-v);
regions.bone_contact_force=sum(all,cg).map(v=>-v);regions.free_gradient_sum=sum(free);
const support=sum(a.d.fixed_nodes),balance=support.map((v,k)=>v+regions.omitted_deep_spring_force[k]+regions.bone_contact_force[k]+regions.free_gradient_sum[k]);
const report={schema:'distributed-bilateral-read-only-reaction-accounting-v1',source_state_sha256:sha(statePath),source_domain_sha256:sha(path.join(dir,'domain.json')),
 coordinate_convention:'X lateral, Y superior, Z anterior',unit_note:'Authored uncalibrated solver force units',vectors_xyz:regions,
 vector_magnitudes:Object.fromEntries(Object.entries(regions).map(([k,v])=>[k,mag(v)])),
 contact_energy:contactEnergy,immutable_contact_energy:fixedContactEnergy,free_contact_energy:contactEnergy-fixedContactEnergy,
 balance_plus_free_gradient_error:mag(balance),
 limits:['The finite first collar is rigidly bone-bound; total attachment load includes its first-layer support, not only zero-gap root nodes',
 'Positive-clearance penalties on fixed collar nodes are separated from constitutive load; the exact root itself has no penalty',
 'Resultant magnitudes are not additive load fractions','Read-only evaluation at exact solved positions; no constraint or force-law edit']};
if(report.balance_plus_free_gradient_error>1e-9)throw Error('Force balance failed');fs.writeFileSync(out,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
