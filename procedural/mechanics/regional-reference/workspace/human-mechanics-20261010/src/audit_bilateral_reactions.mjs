/** Read-only force-path accounting. Unmasking fixed gradients is evaluation only. */
import fs from 'node:fs';import path from 'node:path';import crypto from 'node:crypto';
import {loadBilateralCase} from './bilateral_case_io.mjs';
const [dir,statePath,out]=process.argv.slice(2),a=loadBilateralCase(dir),d=JSON.parse(fs.readFileSync(statePath,'utf8')),s=a.create();
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
if(!d.converged||d.sourceDomainSha256!==sha(path.join(dir,'domain.json')))throw Error('Mismatched or unconverged input');
s.x.set(d.positions);s.setMuscleActivation(d.activation.rightIncoming,'right-zygomatic-like');s.setMuscleActivation(d.activation.leftIncoming,'left-zygomatic-like');
const fixed=Array.from(s.fixed);s.fixed.fill(0);s.evaluate(s.x,Infinity,false);
const vec=()=>[0,0,0],sum=nodes=>nodes.reduce((v,n)=>v.map((q,k)=>q+s.g[n*3+k]),vec()),mag=v=>Math.hypot(...v);
const rootR=a.d.bone_origin_supports.find(q=>q.side===1).nodes,rootL=a.d.bone_origin_supports.find(q=>q.side===-1).nodes,root=new Set([...rootR,...rootL]),clamps=a.d.fixed_nodes.filter(n=>!root.has(n));
const spring=vec();for(const q of s.pointSprings)for(let k=0;k<3;k++)spring[k]-=q.k*(s.x[q.node*3+k]-q.target[k]);
const contact=vec();for(const q of s.solidMeshContacts){const c=q.collider.evaluate(Array.from(s.x.subarray(q.node*3,q.node*3+3)));for(let k=0;k<3;k++)contact[k]-=c.gradient[k];}
const free=[];for(let n=0;n<fixed.length;n++)if(!fixed[n])free.push(n);const freeG=sum(free),support=sum(a.d.fixed_nodes),balance=support.map((q,k)=>q+spring[k]+contact[k]+freeG[k]);
const regions={right_origin_support_reaction:sum(rootR),left_origin_support_reaction:sum(rootL),omitted_interface_clamp_reaction:sum(clamps),omitted_deep_spring_force:spring,bone_contact_force:contact,free_gradient_sum:freeG};
const result={schema:'bilateral-read-only-reaction-accounting-v1',source_state_sha256:sha(statePath),source_domain_sha256:sha(path.join(dir,'domain.json')),coordinate_convention:'X lateral, Y superior, Z anterior',unit_note:'Solver force units from authored uncalibrated material/length parameters, not validated physiological forces',vectors_xyz:regions,vector_magnitudes:Object.fromEntries(Object.entries(regions).map(([k,v])=>[k,mag(v)])),balance_vector:support.map((q,k)=>q+spring[k]+contact[k]),balance_plus_free_gradient_error:mag(balance),scope:'Evaluated exact same positions with fixed-gradient masking disabled, without a solve or position change. Vector resultants can cancel; their magnitudes are not additive load fractions.'};
if(result.balance_plus_free_gradient_error>1e-9)throw Error('Reaction accounting balance failed');fs.writeFileSync(out,JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
