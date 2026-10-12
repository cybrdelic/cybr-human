/** Experimental combined continuous boundary authority, GPL-2.0.
 * This separate module retains the frozen PR6 transaction flow and the exact
 * original FEM step, energy, gradient, PCG and determinant gates. Three former
 * bone-only certificate calls now pass a private bone AND tissue-self filter.
 * No partial face mask, adjacency exemption, metric margin, contact force law,
 * solid containment, continuous tetrahedral determinant, or equilibrium claim.
 * Failed inherited transactions roll x/velocity/predict/last back and require
 * explicit readmission; arbitrary subclass caches are not transactionally restored.
 * Preserve recovered reference-source/LICENSING.md and LICENSES/GPL-2.0.txt.
 */
import {LayeredFEM} from '../../../procedural/mechanics/regional-reference/workspace/human-mechanics-20261010/reference-source/src/layered_fem.js';
import {StaticBoneSweptBVH,extractCompleteFEMBoundary,isUnmodifiedStaticBoneCollider} from '../../human-moving-attachment-v1/src/static_bone_swept_bvh.mjs';
import {certifyTissueSelfStep} from '../src/tissue_self_bvh.mjs';
import {verifyDistributedBondedState} from '../../../procedural/mechanics/regional-reference/workspace/human-preferred-v53-anatomical-assembly-20261011/resumed-source/parameterized_bonded_patch.mjs';
import {verifyMovingBondedState} from '../../human-moving-attachment-v1/src/moving_bonded_patch.mjs';

const same=(a,b)=>a.length===b.length&&a.every((v,i)=>v===b[i]);
function checkBondedPins(solver,options) {
  if(options.movingBondedPatch){
    if(options.distributedBondedPatch)throw Error('Choose one attachment policy');
    const check=verifyMovingBondedState(options.movingBondedPatch,{positions:solver.x,fixed:solver.fixed,targets:solver.target});
    if(!check.valid)throw new BoundaryFeasibilityError('moving-bonded-invariant',check);
  }
  if(!options.distributedBondedPatch)return;
  const check=verifyDistributedBondedState(options.distributedBondedPatch,{positions:solver.x,fixed:solver.fixed,targets:solver.target});
  if(!check.valid)throw new BoundaryFeasibilityError('distributed-bonded-invariant',check);
}
export class BoundaryFeasibilityError extends Error {
  constructor(phase,report){super(`Combined boundary feasibility rejected ${phase}: ${report.reason}`);this.name='BoundaryFeasibilityError';this.phase=phase;this.report=report;}
}


const authorities=new WeakMap();
const boneMethodNames=['getStaticVertex','getStaticTriangle','getStaticTriangleRecord','findCandidates','certifyStep'];
const genuineBoneMethods=Object.fromEntries(boneMethodNames.map(key=>[key,StaticBoneSweptBVH.prototype[key]]));
function genuineBoneCollider(collider){
  if(!collider||Object.getPrototypeOf(collider)!==StaticBoneSweptBVH.prototype)return false;
  for(const key of boneMethodNames){
    if(Object.getOwnPropertyDescriptor(collider,key))return false;
    const d=Object.getOwnPropertyDescriptor(StaticBoneSweptBVH.prototype,key);
    if(!d||!('value' in d)||d.value!==genuineBoneMethods[key])return false;
  }
  return isUnmodifiedStaticBoneCollider(collider);
}
const SELF_KEYS=new Set(['maxFaces','maxVertices','maxBoxTests','maxPairs','maxIntervals','maxAxisTests','pairMaxDepth','pairMaxIntervals','pairMaxAxisTests','stationaryFeatureFastPath']);
const BONE_KEYS=new Set(['minimumDistance','attachmentCoordinatePadding','maxBoxTests','maxCandidatePairs','maxTotalIntervals','maxDepth','maxIntervalsPerPair','movingBondedPatch','distributedBondedPatch']);
function copyOptions(input,keys){
  if(!input||typeof input!=='object'||Array.isArray(input))throw Error('Expected options object');
  const out={};for(const [k,d] of Object.entries(Object.getOwnPropertyDescriptors(input))){if(!('value' in d))throw Error('Certificate option accessors are unsupported');if(!keys.has(k))throw Error('Unsupported certificate option: '+k);out[k]=d.value;}return Object.freeze(out);
}
/** Opaque copied configuration, bound to the genuine unchanged bone collider.
 * Root tokens must be constructed with this SAME genuine collider. The public
 * summary is informational; only the private WeakMap capability grants use. */
export function createCombinedBoundaryAuthority(collider,{self={}}={}){
  if(!genuineBoneCollider(collider))throw Error('Expected unmodified StaticBoneSweptBVH');
  const options=copyOptions(self,SELF_KEYS),token=Object.freeze({kind:'combined-boundary-authority-v1'});
  authorities.set(token,{collider,self:options});return token;
}
function certifyCombined(token,start,end,faces,boneOptions){
  const a=authorities.get(token);
  if(!a||!genuineBoneCollider(a.collider))throw Error('Combined bone authority is absent or implementation changed');
  const bone=Reflect.apply(genuineBoneMethods.certifyStep,a.collider,[start,end,faces,boneOptions]);
  if(!bone.certified)return {...bone,reason:'bone-'+bone.reason,failedGuard:'bone',bone,tissueSelf:null,combined:true};
  const tissueSelf=certifyTissueSelfStep(start,end,faces,{...a.self,requireClosedEdges:true,collectPairs:false});
  return {certified:tissueSelf.certified,combined:true,failedGuard:tissueSelf.certified?null:'tissue-self',
    reason:tissueSelf.certified?'complete-bone-and-tissue-self-cover':'tissue-self-'+tissueSelf.reason,
    boxTests:(bone.boxTests??0)+(tissueSelf.boxTests??0),candidatePairs:(bone.candidatePairs??0)+(tissueSelf.candidatePairs??0),
    intervals:(bone.intervals??0)+(tissueSelf.intervals??0),bone,tissueSelf,
    uniformTissueMargin:false,forceModelComplete:false,solidContainmentChecked:false};
}
function validateDomain(s,g,c){
  if(g.invalid)throw new BoundaryFeasibilityError('authority',{reason:'combined-readmission-required'});
  if(!same(s.rest,g.rest)||!same(s.mass,g.mass))throw new BoundaryFeasibilityError('material-domain-edit',{reason:'rest-or-mass-changed-after-admission'});
  if(c&&(!same(s.fixed,c.fixed)||!same(s.target,c.target)))throw new BoundaryFeasibilityError('constraint-edit',{reason:'fixed-mask-or-targets-mutated-during-step'});
}
function stepOptions(input,n){
  if(!input||typeof input!=='object'||Array.isArray(input))throw Error('Invalid step options');
  const out={},allowed=new Set(['iterations','cgIterations','cgTolerance','forceTolerance','minimumJ','initialGuess']);
  for(const [key,descriptor] of Object.entries(Object.getOwnPropertyDescriptors(input))){
    if(!('value' in descriptor))throw Error('Step option accessors are unsupported');
    const value=descriptor.value;
    if(!allowed.has(key))throw Error('Unsupported step option: '+key);
    if(key==='initialGuess'){
      if(value===null){out[key]=null;continue;}
      if(!value||value.length!==n)throw Error('Invalid FEM initial guess');
      out[key]=Float64Array.from(value);if(!out[key].every(Number.isFinite))throw Error('Invalid FEM initial guess');
    }else{
      if(typeof value!=='number'||!Number.isFinite(value))throw Error('Step options must be finite primitive numbers');
      if(['iterations','cgIterations'].includes(key)?!Number.isInteger(value)||value<0:value<=0)throw Error('Invalid step option: '+key);
      out[key]=value;
    }
  }
  return out;
}

/** Make a derived class without replacing any constitutive/PCG implementation. */
export function withCombinedBoundaryFeasibility(Base=LayeredFEM) {
  if(Base.prototype.step!==LayeredFEM.prototype.step)throw Error('Guard requires the exact recovered original FEM step');
  const privateState=new WeakMap();
  return class CombinedBoundaryFEM extends Base {
    constructor(...args){super(...args);privateState.set(this,{guard:null,context:null});this.lastBoundaryFeasibility=null;}
    setCombinedBoundaryGuard(authority,input={}) {
      const s=privateState.get(this);if(s.context)throw Error('Cannot change guard during a step');
      this.lastBoundaryFeasibility={certified:false,phase:'admission-pending'};
      if(!authorities.has(authority))throw Error('Expected a verified combined authority');
      const options=copyOptions(input,BONE_KEYS);
      checkBondedPins(this,options);
      const pinned={x:this.x.slice(),nodes:this.nodes.slice(),rest:this.rest.slice(),mass:this.mass.slice(),fixed:this.fixed.slice(),target:this.target.slice()};
      const faces=extractCompleteFEMBoundary(pinned.nodes,pinned.mass.length),admission=certifyCombined(authority,pinned.x,pinned.x,faces,options);
      if(Object.entries(pinned).some(([key,value])=>!same(this[key],value)))throw new BoundaryFeasibilityError('admission-context',{reason:'domain-mutated-during-admission'});
      if(!admission.certified){this.lastBoundaryFeasibility={phase:'initial-state',...admission};throw new BoundaryFeasibilityError('initial-state',admission);}
      s.guard={authority,faces,options,trustedX:pinned.x,topology:pinned.nodes,rest:pinned.rest,mass:pinned.mass,invalid:false};this.lastBoundaryFeasibility={phase:'initial-state',...admission,solverIntegrated:true,completeFEMBoundary:true};return this;
    }
    setStaticBoneGuard(authority,input={}){return this.setCombinedBoundaryGuard(authority,input);}
    clearStaticBoneGuard(){const s=privateState.get(this);if(s.context)throw Error('Cannot change guard during a step');s.guard=null;return this;}
    evaluate(x,dt,blocks=true) {
      const s=privateState.get(this),c=s?.context,g=s?.guard;
      if(!c||!g)return super.evaluate(x,dt,blocks);
      validateDomain(this,g,c);
      checkBondedPins(this,g.options);
      if(!same(this.nodes,g.topology))throw new BoundaryFeasibilityError('topology-edit',{reason:'tetrahedral-topology-changed-after-admission'});
      if(blocks) {
        if(c.firstEvaluation) {
          if(!same(x,c.initial))throw new BoundaryFeasibilityError('unexpected-initial-write',{reason:'original-step-contract-changed'});
          c.firstEvaluation=false;
        }else {
          if(!c.lastTrialCertified||!c.pendingTrial||!same(x,c.pendingTrial))throw new BoundaryFeasibilityError('unexpected-accepted-write',{reason:'original-step-contract-changed'});
          c.acceptedSegments++;c.acceptedCurrent=c.pendingTrial.slice();
        }
      }else {
        if(!same(this.x,c.acceptedCurrent))throw new BoundaryFeasibilityError('unexpected-current-write',{reason:'positions-mutated-during-search'});
        c.trialChecks++;
        const report=certifyCombined(g.authority,this.x,x,g.faces,g.options);c.lastTrialCertified=report.certified;
        c.boxTests+=report.boxTests;c.candidatePairs+=report.candidatePairs;c.intervals+=report.intervals;
        if(!report.certified) {
          c.rejectedTrials++;c.lastRejected=report;c.pendingTrial=null;
          // Fail before inherited evaluation can mutate gradient/metric buffers.
          return {energy:Infinity,residual:Infinity,minJ:NaN,maxJ:NaN,boundaryFeasible:false,boundaryFeasibility:report};
        }
        c.pendingTrial=x.slice();
      }
      const evaluatedCoordinates=x.slice(),result=super.evaluate(x,dt,blocks);
      if(!same(x,evaluatedCoordinates)||!same(this.x,c.acceptedCurrent)||!same(this.nodes,g.topology))
        throw new BoundaryFeasibilityError('mutating-evaluation',{reason:'coordinates-or-topology-mutated-during-evaluation'});
      checkBondedPins(this,g.options);
      validateDomain(this,g,c);
      return result;
    }
    step(dt,input={}) {
      const s=privateState.get(this),g=s.guard;
      this.lastBoundaryFeasibility={certified:false,phase:'step-pending'};
      if(!g)throw new BoundaryFeasibilityError('authority',{reason:'combined-guard-not-installed'});
      validateDomain(this,g,null);
      input=stepOptions(input,this.x.length);
      validateDomain(this,g,null);
      if(s.context)throw Error('Reentrant guarded step');
      checkBondedPins(this,g.options);
      if(typeof dt!=='number'||!(dt===Infinity||(dt>0&&dt<=.05)))throw Error('Invalid FEM time step');
      const initialGuess=input.initialGuess??null;
      if(initialGuess&&(initialGuess.length!==this.x.length||!initialGuess.every(Number.isFinite)))throw Error('Invalid FEM initial guess');
      if(!same(this.x,g.trustedX))throw new BoundaryFeasibilityError('unadmitted-position-edit',{reason:'positions-changed-outside-guarded-step'});
      if(!same(this.nodes,g.topology))throw new BoundaryFeasibilityError('topology-edit',{reason:'tetrahedral-topology-changed-after-admission'});
      const oldXReference=this.x,oldVelocityReference=this.velocity,oldPredictReference=this.predict;
      const old=this.x.slice(),oldVelocity=this.velocity.slice(),oldPredict=this.predict.slice(),oldLast=this.last,initial=this.x.slice();
      for(let n=0;n<this.mass.length;n++)for(let d=0;d<3;d++) {
        const j=3*n+d;if(this.fixed[n])initial[j]=this.target[j];else if(initialGuess)initial[j]=initialGuess[j];
      }
      // No source position/predict/velocity/metric write has occurred yet.
      const admission=certifyCombined(g.authority,old,initial,g.faces,g.options);
      if(!admission.certified){this.lastBoundaryFeasibility={phase:'initial-segment',...admission};throw new BoundaryFeasibilityError('initial-segment',admission);}
      const c={fixed:this.fixed.slice(),target:this.target.slice(),initial,acceptedCurrent:initial.slice(),pendingTrial:null,firstEvaluation:true,trialChecks:0,rejectedTrials:0,acceptedSegments:0,lastTrialCertified:false,lastRejected:null,
        boxTests:admission.boxTests,candidatePairs:admission.candidatePairs,intervals:admission.intervals};s.context=c;
      try {
        const result=super.step(dt,input);
        if(c.firstEvaluation||result.accepted!==c.acceptedSegments||!same(this.x,c.acceptedCurrent)||!same(this.nodes,g.topology))throw new BoundaryFeasibilityError('step-contract',{reason:'original-step-contract-changed'});
        checkBondedPins(this,g.options);
        validateDomain(this,g,c);
        g.trustedX=this.x.slice();
        const report={certified:true,solverIntegrated:true,completeFEMBoundary:true,continuousTissueSelfContact:true,continuousStaticBoneContact:true,uniformTissueMargin:false,physicalCellGate:'Inherited endpoint determinant checks unchanged; no continuous volume certificate',phase:'step',initialAdmission:admission,trialChecks:c.trialChecks,rejectedTrials:c.rejectedTrials,
          acceptedSegments:c.acceptedSegments,boxTests:c.boxTests,candidatePairs:c.candidatePairs,intervals:c.intervals,lastRejected:c.lastRejected,
          forceModelComplete:false,energyAndGradientUnchanged:true,solidContainmentChecked:false,
          termination:result.converged?'converged':c.rejectedTrials?'not-converged-with-guard-rejections':'not-converged'};
        this.lastBoundaryFeasibility=report;this.last={...result,boundaryFeasibility:report};return this.last;
      }catch(error) {
        // Restore physical state if an inherited initial-J check or unexpected
        // failure occurs. Recompute metric/diagnostics on the next successful
        // evaluation; no claim that arbitrary subclass caches are transactional.
        g.invalid=true;this.lastBoundaryFeasibility={certified:false,phase:'step-error',reason:error.message,readmissionRequired:true};
        this.x=oldXReference;this.velocity=oldVelocityReference;this.predict=oldPredictReference;
        this.x.set(old);this.velocity.set(oldVelocity);this.predict.set(oldPredict);this.last=oldLast;throw error;
      }finally{s.context=null;}
    }
  };
}

export const CombinedBoundaryLayeredFEM=withCombinedBoundaryFeasibility(LayeredFEM);
