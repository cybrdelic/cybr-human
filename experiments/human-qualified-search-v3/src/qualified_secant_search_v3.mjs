/** Search-only inverse-BFGS qualification; no physical term changes.
 * The caller's fingerprint MUST cover mesh/arrays, materials and physical law,
 * constraints/fixed targets, contact law/obstacles, activation, and any dynamic
 * metric configuration. dt is also tracked explicitly below. The contact
 * callback MUST cover all active free-node contact features (not only normals).
 * Restored secants/endpoints must preserve fixed DOFs exactly. Selected and
 * fallback directions are explicitly projected to exact fixed zeros because
 * the underlying trial construction does not reproject fixed targets.
 * Each nonzero-gradient search calls underlying PCG exactly once. Stationary
 * gradients return a finite zero direction without PCG. No legacy migration.
 */
const dot=(a,b)=>{let z=0;for(let i=0;i<a.length;i++)z+=a[i]*b[i];return z;};
const finite=a=>{if(!a||typeof a.length!=='number')return false;for(let i=0;i<a.length;i++)if(!Number.isFinite(a[i]))return false;return true;};
const positiveFinite=a=>{if(!finite(a))return false;for(let i=0;i<a.length;i++)if(!(a[i]>0))return false;return true;};
const curvature=(dx,dy)=>{const sy=dot(dx,dy),ss=dot(dx,dx),yy=dot(dy,dy);return Number.isFinite(sy)&&Number.isFinite(ss)&&Number.isFinite(yy)&&sy>1e-28&&sy>1e-10*Math.sqrt(ss)*Math.sqrt(yy)?sy:null;};
const metricKey=dt=>{if(dt!==Infinity&&!(Number.isFinite(dt)&&dt>0))throw Error('Invalid search time step');return dt===Infinity?'static:Infinity':'dynamic:'+dt;};
export function installQualifiedSecantSearch(s,{history=12,restore=null,contextFingerprint,contactSignature=()=>'',restoreLegacy=false}={}){
 if(typeof contextFingerprint!=='function')throw Error('A physical-context fingerprint callback is required');
 if(!Number.isInteger(history)||history<1||history>64)throw Error('History size must be an integer from 1 to 64');
 if(restoreLegacy)throw Error('Unqualified legacy history cannot be implicitly trusted');
 const original=s.pcg.bind(s),records=[];let previousX=null,previousG=null,previousContact=null,previousMetric=null,context=String(contextFingerprint());
 const report=s.qualifiedSecantReport={restoredPairs:0,restoreRejected:null,acceptedPairs:0,skippedPairs:0,contextInvalidations:0,contactInvalidations:0,metricInvalidations:0,qualifiedCandidates:0,rejectedCandidates:0,innerDirectionFallbacks:0,jacobiFallbacks:0,nonfiniteHistoryInvalidations:0,stationaryReturns:0,pcgCalls:0,last:null};
 const validArray=a=>finite(a)&&a.length===s.x.length;
 const fixedIndices=[];if(s.fixed){if(s.fixed.length*3!==s.x.length)throw Error('Invalid fixed-node mask');for(let n=0;n<s.fixed.length;n++){if(s.fixed[n]!==0&&s.fixed[n]!==1)throw Error('Invalid fixed-node mask');if(s.fixed[n])for(let d=0;d<3;d++)fixedIndices.push(3*n+d);}}
 if(fixedIndices.length&&!validArray(s.target))throw Error('Fixed targets must be finite and correctly sized');
 const fixedZero=a=>{for(const i of fixedIndices)if(a[i]!==0)return false;return true;};
 const fixedTarget=a=>{for(const i of fixedIndices)if(a[i]!==s.target[i])return false;return true;};
 const project=a=>{for(const i of fixedIndices)a[i]=0;return a;};
 const clear=()=>{records.length=0;previousX=previousG=null;previousContact=previousMetric=null;};
 if(restore){
  try{
   if(restore.schema!=='qualified-fem-secant-v3')throw Error('unqualified legacy checkpoint');
   if(restore.contextFingerprint!==context)throw Error('physical context fingerprint changed');
   if(typeof restore.previousContactSignature!=='string')throw Error('missing contact signature');
   if(!Array.isArray(restore.records)||restore.records.length>history)throw Error('invalid or oversized qualified history');
   if(restore.previousMetricSignature!==null&&typeof restore.previousMetricSignature!=='string')throw Error('missing metric signature');
   const px=restore.previousX,pg=restore.previousG;
   if((px!==null&&!validArray(px))||(pg!==null&&!validArray(pg))||((px===null)!==(pg===null)))throw Error('invalid or unpaired history endpoints');
   if(px&&(!fixedTarget(px)||!fixedZero(pg)))throw Error('Restored endpoint violates fixed targets or projected gradient');
   if(restore.records.length&&(!px||!pg||!restore.previousMetricSignature))throw Error('nonempty history requires endpoints and metric signature');
   for(const r of restore.records){
    if(!validArray(r.dx)||!validArray(r.dy)||!(Number.isFinite(r.rho)&&r.rho>0))throw Error('invalid qualified secant record');
    if(!fixedZero(r.dx)||!fixedZero(r.dy))throw Error('Restored secant has nonzero fixed-DOF components');
    const sy=curvature(r.dx,r.dy);
    if(sy===null||!Number.isFinite(r.rho*sy)||Math.abs(r.rho*sy-1)>1e-12)throw Error('invalid secant curvature or reciprocal rho');
    records.push({dx:Float64Array.from(r.dx),dy:Float64Array.from(r.dy),rho:1/sy});
   }
   previousX=px?Float64Array.from(px):null;previousG=pg?Float64Array.from(pg):null;previousContact=restore.previousContactSignature;previousMetric=restore.previousMetricSignature;report.restoredPairs=records.length;
  }catch(e){clear();report.restoreRejected=e.message;}
 }
 s.exportQualifiedSecantCheckpoint=()=>({schema:'qualified-fem-secant-v3',contextFingerprint:context,previousContactSignature:previousContact??'',previousMetricSignature:previousMetric,records:records.map(r=>({dx:Array.from(r.dx),dy:Array.from(r.dy),rho:r.rho})),previousX:previousX?Array.from(previousX):null,previousG:previousG?Array.from(previousG):null});
 const metrics=(d,g,dt)=>{
  if(!validArray(d))return{finite:false};const md=new Float64Array(d.length);s.multiply(d,md,dt);if(!finite(md))return{finite:false};
  let rr=0;for(let i=0;i<g.length;i++)rr+=(g[i]+md[i])**2;const slope=dot(g,d),c=dot(d,md),gg=dot(g,g),m=slope+.5*c,r=gg?Math.sqrt(rr/gg):Math.sqrt(rr);
  return{finite:[slope,c,m,r].every(Number.isFinite),slope,curvature:c,quadraticModelChange:m,metricResidualRatio:r};
 };
 const safe=m=>m.finite&&m.slope<0&&m.curvature>=0&&m.quadraticModelChange<0;
 s.pcg=function(dt,limit,tolerance){
  const g=this.g.slice(),x=this.x.slice(),nowContext=String(contextFingerprint()),signature=String(contactSignature()),metric=metricKey(dt);
  if(!validArray(g)||!validArray(x)||!validArray(this.diagonal)||!positiveFinite(this.diagonal))throw Error('Search requires finite positions/gradient and positive finite diagonals');
  if(!fixedTarget(x)||!fixedZero(g))throw Error('Current search violates fixed targets or projected gradient');
  const gg=dot(g,g);if(!Number.isFinite(gg))throw Error('Nonfinite gradient norm');
  if(nowContext!==context){clear();context=nowContext;report.contextInvalidations++;}
  if(previousContact!==null&&signature!==previousContact){clear();report.contactInvalidations++;}
  if(previousMetric!==null&&metric!==previousMetric){clear();report.metricInvalidations++;}
  if(previousX){const dx=Float64Array.from(x,(v,i)=>v-previousX[i]),dy=Float64Array.from(g,(v,i)=>v-previousG[i]),sy=curvature(dx,dy);if(sy!==null){records.push({dx,dy,rho:1/sy});if(records.length>history)records.shift();report.acceptedPairs++;}else report.skippedPairs++;}
  previousX=x;previousG=g;previousContact=signature;previousMetric=metric;
  if(gg===0){this.direction.fill(0);report.stationaryReturns++;report.last={usedPairs:records.length,selected:'stationary zero direction',iterations:0,selectedMetrics:metrics(this.direction,g,dt)};return 0;}
  const q=g.slice();let alpha=new Float64Array(records.length),usedPairs=records.length;
  for(let j=records.length-1;j>=0;j--){const r=records[j];alpha[j]=r.rho*dot(r.dx,q);for(let i=0;i<q.length;i++)q[i]-=alpha[j]*r.dy[i];}
  if(!finite(q)||!finite(alpha)){records.length=0;usedPairs=0;alpha=new Float64Array(0);q.set(g);report.nonfiniteHistoryInvalidations++;}
  this.g.set(q);let iterations;try{report.pcgCalls++;iterations=original(dt,limit,tolerance);}finally{this.g.set(g);}
  const inner=project(this.direction.slice()),innerMetrics=metrics(inner,g,dt),candidate=Float64Array.from(inner,v=>-v);
  for(let j=0;j<records.length;j++){const r=records[j],beta=r.rho*dot(r.dy,candidate);for(let i=0;i<candidate.length;i++)candidate[i]+=r.dx[i]*(alpha[j]-beta);}
  for(let i=0;i<candidate.length;i++)candidate[i]=-candidate[i];project(candidate);const candidateMetrics=metrics(candidate,g,dt);
  let selected;
  if(usedPairs&&safe(candidateMetrics)){this.direction.set(candidate);report.qualifiedCandidates++;selected='qualified inverse-BFGS';}
  else if(!usedPairs&&safe(innerMetrics)){this.direction.set(inner);selected='plain PCG; no qualified history';}
  else{
   records.length=0;report.rejectedCandidates++;
   if(safe(innerMetrics)){this.direction.set(inner);report.innerDirectionFallbacks++;selected='inner PCG direction; history invalidated';}
   else{
    for(let i=0;i<g.length;i++)this.direction[i]=-g[i]/this.diagonal[i];project(this.direction);const jm=metrics(this.direction,g,dt);
    if(!(jm.finite&&jm.slope<0&&jm.curvature>0))throw Error('No finite admissible metric fallback');
    const scale=-jm.slope/jm.curvature;if(!(Number.isFinite(scale)&&scale>0))throw Error('Nonfinite fallback scaling');
    for(let i=0;i<g.length;i++)this.direction[i]*=scale;project(this.direction);if(!safe(metrics(this.direction,g,dt)))throw Error('No finite qualified Jacobi fallback');
    report.jacobiFallbacks++;selected='scaled Jacobi direction; history invalidated';
   }
  }
  project(this.direction);if(!validArray(this.direction)||!fixedZero(this.direction)||!safe(metrics(this.direction,g,dt)))throw Error('Selected direction failed final finite/descent qualification');
  report.last={usedPairs,selected,inner:innerMetrics,candidate:candidateMetrics,selectedMetrics:metrics(this.direction,g,dt),iterations};return iterations;
 };
 return s;
}
