/** Original bounded whole-midface incoming-muscle equilibrium experiment.
 * Continuous static-bone feasibility is mandatory. Force sampling and anatomical
 * attachment offsets remain separate, explicit approximations.
 */
import fs from 'node:fs';import path from 'node:path';import crypto from 'node:crypto';
import {loadDistributedCase} from './regional_case_io.mjs';
import {InterdigitatedMuscleFEM} from '../../human-mechanics-20261010/src/interdigitated_muscle_fem.mjs';
import {StaticBoneSweptBVH} from '../../human-mechanics-20261010/src/prototypes/static_bone_swept_bvh_distributed.mjs';
import {withStaticBoneFeasibility} from '../../human-mechanics-20261010/src/prototypes/guarded_static_bone_fem_distributed.mjs';
const [dir,out,phase,commandText,resumeFile]=process.argv.slice(2),command=Number(commandText);
if(!dir||!out||!phase||!Number.isFinite(command)||command<0||command>.16)throw Error('case output phase activation[0,.16] [resume] required');
fs.mkdirSync(out,{recursive:true});const target=path.join(out,`${phase}-state.json`);if(fs.existsSync(target))throw Error('Refusing existing state identity');
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const begin=performance.now();console.log(JSON.stringify({phase:'loading',utc:new Date().toISOString()}));
const a=loadDistributedCase(dir),Guarded=withStaticBoneFeasibility(InterdigitatedMuscleFEM),s=a.create(Guarded);s.useSearchSubspace=true;
if(a.d.material_family_model!=='interdigitated')throw Error('Expected the explicit interdigitated passive-family case');
for(const r of s.muscleRegions)s.setMuscleActivation(0,r.id);
s.setMuscleActivation(command,'right-zygomatic-like');
console.log(JSON.stringify({phase:'building-static-bone-bvh',seconds:(performance.now()-begin)/1000,rssMiB:process.memoryUsage().rss/2**20}));
const bvh=new StaticBoneSweptBVH({bones:[{id:'combined',vertices:a.boneVertices,triangles:a.boneTriangles}],leafSize:8});
// The existing geometry has been independently checked for external containment.
// Resume is separately admitted as a new STATIC state before any next trajectory.
if(resumeFile){const previous=JSON.parse(fs.readFileSync(resumeFile,'utf8'));if(previous.sourceDomainSha256!==sha(path.join(dir,'domain.json')))throw Error('Resume domain changed');s.x.set(previous.positions);s.target.set(s.rest);}
const bond=a.makeBondedToken(bvh,s,.00001);
console.log(JSON.stringify({phase:'verified-distributed-bond',summary:bond.summary}));
s.setStaticBoneGuard(bvh,{distributedBondedPatch:bond,minimumDistance:.00001,maxBoxTests:2000000,maxCandidatePairs:50000,maxTotalIntervals:100000,maxDepth:24,maxIntervalsPerPair:2048});
console.log(JSON.stringify({phase:'admitted-rest-or-resume',seconds:(performance.now()-begin)/1000,guard:s.lastBoundaryFeasibility,rssMiB:process.memoryUsage().rss/2**20}));
const history=[];let last=null,error=null;const batches=Number(process.env.CYBR_BILATERAL_BATCHES||12);
try{
 for(let batch=0;batch<batches;batch++){
  last=s.step(Infinity,{iterations:4,cgIterations:1000,cgTolerance:.0002,forceTolerance:1e-7,minimumJ:.25});
  const record={batch,seconds:(performance.now()-begin)/1000,residual:last.residual,minJ:last.minJ,energy:last.energy,accepted:last.accepted,totalCG:last.totalCG,converged:last.converged,boundaryFeasibility:last.boundaryFeasibility,rssMiB:process.memoryUsage().rss/2**20};history.push(record);console.log(JSON.stringify(record));
  fs.writeFileSync(path.join(out,`${phase}-checkpoint.json`),JSON.stringify({phase,activation:command,positions:Array.from(s.x),last:record,sourceDomainSha256:sha(path.join(dir,'domain.json'))}));
  if(record.rssMiB>850)throw Error('Owned 850MiB soft stop; external900MiB guard');
  if(record.seconds>65)throw Error('Owned 65-second solve phase budget');
  if(last.converged)break;
  if(last.accepted===0)throw Error('No accepted progress; refusing convergence claim');
 }
}catch(e){error={name:e.name,message:e.message,report:e.report||null};console.log(JSON.stringify({phase:'stopped',error}));}
const final=s.evaluate(s.x,Infinity,true);const geometryGuard=s.lastBoundaryFeasibility;
const maxMotion=Math.max(...Array.from({length:s.mass.length},(_,n)=>Math.hypot(...[0,1,2].map(k=>s.x[n*3+k]-s.rest[n*3+k]))))*1000;
const result={schema:'preferred-v53-loaded-insertion-equilibrium-state-v1',phase,activation:{rightIncoming:command,leftIncoming:0,oral:0,allOtherFamilies:0},converged:!error&&Boolean(last?.converged),positions:Array.from(s.x),sourceDomainSha256:sha(path.join(dir,'domain.json')),sourceArraysSha256:sha(path.join(dir,'arrays.bin')),sourceBonesSha256:sha(path.join(dir,'bones.bin')),sourceGeometrySha256:a.d.source_geometry_sha256,sourceBindingSha256:a.d.native_binding_sha256,minJ:final.minJ,residual:final.residual,energy:final.energy,maximumFEMMotionDesignUnits:maxMotion,forceContactModel:a.contactSummary,boundaryFeasibility:geometryGuard,distributedBond:bond.summary,muscleDiagnostics:s.muscleDiagnostics,history,error,seconds:(performance.now()-begin)/1000,limitations:a.d.limits.concat(['Static-bone continuous boundary feasibility is distinct from the finite nodal force model','No moving-bone CCD, continuous Jacobian proof, tissue self-contact or dense rendered-skin clearance claim','No anatomical calibration or whole-human fidelity'])};
fs.writeFileSync(target,JSON.stringify(result));console.log(JSON.stringify({phase:'terminal',converged:result.converged,residual:result.residual,minJ:result.minJ,maximumFEMMotionDesignUnits:maxMotion,seconds:result.seconds,error}));if(!result.converged)process.exitCode=1;
