/**
 * Opt-in continuous STATIC-bone feasibility around the recovered original FEM.
 * Original CPU FEM: cybrdelic/cybr-human@c0e6d626dfa48c0b05657bd87b880cb7a6bc92a9.
 * Retain reference-source/LICENSES/GPL-2.0.txt and LICENSING.md (GPL-2.0).
 * No production source changes. Original energy/gradient, metric, PCG, energy
 * line search, determinant tests and iteration budget are inherited unchanged.
 *
 * This is a rejection filter, NOT a replacement complete contact force model.
 * Existing finite-sample penalties may fail to balance at the guard. Such a
 * result stays unconverged; uncertified candidates are never accepted.
 * Guards all original step's fixed-target/initialGuess writes, and every trial
 * before inherited evaluation. It only supports a Base inheriting the exact
 * recovered LayeredFEM.step method. Raw external writes to x are not authorized
 * trajectory changes; reconfigure/admit explicitly, or use initialGuess.
 * Solid containment and a valid exterior initial volume remain prerequisites.
 */
import {LayeredFEM} from '../../reference-source/src/layered_fem.js';
import {StaticBoneSweptBVH,extractCompleteFEMBoundary} from './static_bone_swept_bvh_distributed.mjs';
import {verifyDistributedBondedState} from './distributed_bonded_patch.mjs';

const same=(a,b)=>a.length===b.length&&a.every((v,i)=>v===b[i]);
function checkBondedPins(solver,options) {
  if(!options.distributedBondedPatch)return;
  const check=verifyDistributedBondedState(options.distributedBondedPatch,{positions:solver.x,fixed:solver.fixed,targets:solver.target});
  if(!check.valid)throw new BoundaryFeasibilityError('distributed-bonded-invariant',check);
}
export class BoundaryFeasibilityError extends Error {
  constructor(phase,report){super(`Static-bone feasibility rejected ${phase}: ${report.reason}`);this.name='BoundaryFeasibilityError';this.phase=phase;this.report=report;}
}

/** Make a derived class without replacing any constitutive/PCG implementation. */
export function withStaticBoneFeasibility(Base=LayeredFEM) {
  if(Base.prototype.step!==LayeredFEM.prototype.step)throw Error('Guard requires the exact recovered original FEM step');
  const privateState=new WeakMap();
  return class StaticBoneFeasibilityFEM extends Base {
    constructor(...args){super(...args);privateState.set(this,{guard:null,context:null});this.lastBoundaryFeasibility=null;}
    setStaticBoneGuard(collider,input={}) {
      const s=privateState.get(this);if(s.context)throw Error('Cannot change guard during a step');
      if(!(collider instanceof StaticBoneSweptBVH))throw Error('Expected an immutable StaticBoneSweptBVH');
      checkBondedPins(this,input);
      const faces=extractCompleteFEMBoundary(this.nodes,this.mass.length),options=Object.freeze({...input}),admission=collider.certifyStep(this.x,this.x,faces,options);
      if(!admission.certified){this.lastBoundaryFeasibility={phase:'initial-state',...admission};throw new BoundaryFeasibilityError('initial-state',admission);}
      s.guard={collider,faces,options,trustedX:this.x.slice(),topology:this.nodes.slice()};this.lastBoundaryFeasibility={phase:'initial-state',...admission};return this;
    }
    clearStaticBoneGuard(){const s=privateState.get(this);if(s.context)throw Error('Cannot change guard during a step');s.guard=null;return this;}
    evaluate(x,dt,blocks=true) {
      const s=privateState.get(this),c=s?.context,g=s?.guard;
      if(!c||!g)return super.evaluate(x,dt,blocks);
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
        const report=g.collider.certifyStep(this.x,x,g.faces,g.options);c.lastTrialCertified=report.certified;
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
      return result;
    }
    step(dt,input={}) {
      const s=privateState.get(this),g=s.guard;if(!g)return super.step(dt,input);
      if(s.context)throw Error('Reentrant guarded step');
      checkBondedPins(this,g.options);
      if(!(dt===Infinity||(dt>0&&dt<=.05)))throw Error('Invalid FEM time step');
      const initialGuess=input.initialGuess??null;
      if(initialGuess&&(initialGuess.length!==this.x.length||!initialGuess.every(Number.isFinite)))throw Error('Invalid FEM initial guess');
      if(!same(this.x,g.trustedX))throw new BoundaryFeasibilityError('unadmitted-position-edit',{reason:'positions-changed-outside-guarded-step'});
      if(!same(this.nodes,g.topology))throw new BoundaryFeasibilityError('topology-edit',{reason:'tetrahedral-topology-changed-after-admission'});
      const old=this.x.slice(),oldVelocity=this.velocity.slice(),oldPredict=this.predict.slice(),oldLast=this.last,initial=this.x.slice();
      for(let n=0;n<this.mass.length;n++)for(let d=0;d<3;d++) {
        const j=3*n+d;if(this.fixed[n])initial[j]=this.target[j];else if(initialGuess)initial[j]=initialGuess[j];
      }
      // No source position/predict/velocity/metric write has occurred yet.
      const admission=g.collider.certifyStep(old,initial,g.faces,g.options);
      if(!admission.certified){this.lastBoundaryFeasibility={phase:'initial-segment',...admission};throw new BoundaryFeasibilityError('initial-segment',admission);}
      const c={initial,acceptedCurrent:initial.slice(),pendingTrial:null,firstEvaluation:true,trialChecks:0,rejectedTrials:0,acceptedSegments:0,lastTrialCertified:false,lastRejected:null,
        boxTests:admission.boxTests,candidatePairs:admission.candidatePairs,intervals:admission.intervals};s.context=c;
      try {
        const result=super.step(dt,input);
        if(c.firstEvaluation||result.accepted!==c.acceptedSegments||!same(this.x,c.acceptedCurrent)||!same(this.nodes,g.topology))throw new BoundaryFeasibilityError('step-contract',{reason:'original-step-contract-changed'});
        checkBondedPins(this,g.options);
        g.trustedX=this.x.slice();
        const report={certified:true,phase:'step',initialAdmission:admission,trialChecks:c.trialChecks,rejectedTrials:c.rejectedTrials,
          acceptedSegments:c.acceptedSegments,boxTests:c.boxTests,candidatePairs:c.candidatePairs,intervals:c.intervals,lastRejected:c.lastRejected,
          forceModelComplete:false,energyAndGradientUnchanged:true,solidContainmentChecked:false,
          termination:result.converged?'converged':c.rejectedTrials?'not-converged-with-guard-rejections':'not-converged'};
        this.lastBoundaryFeasibility=report;this.last={...result,boundaryFeasibility:report};return this.last;
      }catch(error) {
        // Restore physical state if an inherited initial-J check or unexpected
        // failure occurs. Recompute metric/diagnostics on the next successful
        // evaluation; no claim that arbitrary subclass caches are transactional.
        this.x.set(old);this.velocity.set(oldVelocity);this.predict.set(oldPredict);this.last=oldLast;
        this.lastBoundaryFeasibility={certified:false,phase:'step-error',reason:error.message};throw error;
      }finally{s.context=null;}
    }
  };
}

export const GuardedStaticBoneLayeredFEM=withStaticBoneFeasibility(LayeredFEM);
