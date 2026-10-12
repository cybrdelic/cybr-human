/**
 * BOUNDED PROTOTYPE: triangle-complete static-bone contact building blocks.
 * Original implementation; no collision library or production solver edits.
 *
 * 1. A variational pair energy includes all 3 tissue-point/bone-face, all
 *    3 bone-point/tissue-face, and all 9 edge/edge primitives. Exact analytic
 *    first derivatives include closest-point envelope weights and the
 *    near-parallel edge mollifier derivative. Derivatives are floating-point
 *    evaluations of analytic formulas, NOT interval-certified gradients.
 * 2. A separate fail-closed segment filter proves whole triangles stay farther
 *    than minimumDistance for every time on a LINEAR vertex trajectory. It
 *    uses outward-rounded separating-plane projections of swept triangles,
 *    not time samples or spatial quadrature. Unresolved means REJECT.
 *
 * This is not an IPC implementation or a production collision engine.
 * Triangle-pair energies overcount shared mesh primitives; a mesh integration
 * must deduplicate VF/EE primitives and choose stiffness/area scaling. The PSD
 * metric below omits geometric/mollifier Hessian terms and is NOT a Hessian.
 * No friction, self-contact, solid-containment test, rest repair, BVH, rigid
 * rotation trajectory, or determinant-path guarantee is supplied here.
 * A valid exterior initial state and complete boundary pair coverage are
 * prerequisites. An unsigned boundary distance cannot establish containment.
 *
 * Certification is conditional on conforming IEEE-754 binary64 JS arithmetic
 * with gradual underflow. Each primitive +,-,* is enclosed with nextDown/up.
 * Nonfinite enclosure/axis calculations cannot certify. No Math.sqrt/log or
 * unverified closest-distance value participates in a separation proof.
 * These assumptions and prototype tests are not an independent formal audit.
 */

const sub = (a,b) => a.map((x,i) => x-b[i]);
const dot = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const scale = (a,s) => a.map(x => x*s);
const mix = (a,b,t) => a.map((x,i) => x+t*(b[i]-x));
const finite3 = p => p && p.length===3 && Array.from(p).every(Number.isFinite);
const edges = [[0,1],[1,2],[2,0]];
const zero3 = () => [0,0,0];
const zeros = () => [zero3(),zero3(),zero3()];
function triangle(t,name) {
  if(!t || t.length!==3 || !Array.from(t).every(finite3)) throw Error(`${name} requires three finite 3D vertices`);
  return Array.from(t,p=>Array.from(p));
}
function regularTriangle(t) {
  const u=sub(t[1],t[0]),v=sub(t[2],t[0]),s=Math.max(dot(u,u),dot(v,v),dot(sub(t[2],t[1]),sub(t[2],t[1])));
  const a2=dot(cross(u,v),cross(u,v));
  return Number.isFinite(s) && Number.isFinite(a2) && s>0 && a2>1e-28*s*s;
}
const unitWeights = i => [0,1,2].map(j=>i===j?1:0);
function weightedPoint(t,w) {
  return t[0].map((x,d)=>x+w[1]*(t[1][d]-x)+w[2]*(t[2][d]-x));
}
function segmentPoint(p,a,b) {
  const u=sub(b,a),uu=dot(u,u);
  const t=uu>0?Math.max(0,Math.min(1,dot(sub(p,a),u)/uu)):0;
  const q=mix(a,b,t),delta=sub(p,q);
  return {t,q,delta,d2:dot(delta,delta)};
}
function pointTriangle(p,t) {
  const u=sub(t[1],t[0]),v=sub(t[2],t[0]),r=sub(p,t[0]),n=cross(u,v),nn=dot(n,n);
  let best=null;
  if(nn>0 && Number.isFinite(nn)) {
    // Cross-product form avoids subtracting two nearly equal Gram products.
    const w1=dot(cross(r,v),n)/nn,w2=dot(cross(u,r),n)/nn,w0=1-w1-w2;
    if([w0,w1,w2].every(w=>Number.isFinite(w)&&w>=0)) {
      const weights=[w0,w1,w2],q=weightedPoint(t,weights),delta=sub(p,q);
      best={weights,q,delta,d2:dot(delta,delta),region:'face'};
    }
  }
  for(const [i,j] of edges) {
    const r=segmentPoint(p,t[i],t[j]),weights=[0,0,0];weights[i]=1-r.t;weights[j]=r.t;
    if(!best || r.d2<best.d2)best={...r,weights,region:r.t===0||r.t===1?'vertex':'edge'};
  }
  return best;
}
function segmentSegment(a0,a1,b0,b1) {
  const u=sub(a1,a0),v=sub(b1,b0),w=sub(a0,b0),n=cross(u,v),nn=dot(n,n);
  let best=null;
  const add=(s,t)=>{
    const pa=mix(a0,a1,s),pb=mix(b0,b1,t),delta=sub(pa,pb),d2=dot(delta,delta);
    if(Number.isFinite(d2)&&(!best||d2<best.d2))best={s,t,pa,pb,delta,d2};
  };
  if(nn>0 && Number.isFinite(nn)) {
    const s=dot(cross(v,w),n)/nn,t=dot(cross(u,w),n)/nn;
    if(s>=0&&s<=1&&t>=0&&t<=1)add(s,t);
  }
  add(0,segmentPoint(a0,b0,b1).t);add(1,segmentPoint(a1,b0,b1).t);
  add(segmentPoint(b0,a0,a1).t,0);add(segmentPoint(b1,a0,a1).t,1);
  return best;
}

/** Closest-feature list. It is NOT an intersection test or a certificate. */
export function trianglePairFeatures(tissue,bone) {
  const a=triangle(tissue,'tissue'),b=triangle(bone,'bone'),out=[];
  for(let i=0;i<3;i++) {
    const q=pointTriangle(a[i],b);
    out.push({kind:'tissue-vertex/bone-face',ids:[i],wa:unitWeights(i),wb:q.weights,delta:q.delta,d2:q.d2});
  }
  for(let i=0;i<3;i++) {
    const q=pointTriangle(b[i],a);
    out.push({kind:'bone-vertex/tissue-face',ids:[i],wa:q.weights,wb:unitWeights(i),delta:scale(q.delta,-1),d2:q.d2});
  }
  for(let i=0;i<3;i++)for(let j=0;j<3;j++) {
    const ea=edges[i],eb=edges[j],q=segmentSegment(...ea.map(k=>a[k]),...eb.map(k=>b[k]));
    if(!q)throw Error('Nonfinite edge geometry');
    const wa=[0,0,0],wb=[0,0,0];wa[ea[0]]=1-q.s;wa[ea[1]]=q.s;wb[eb[0]]=1-q.t;wb[eb[1]]=q.t;
    out.push({kind:'edge/edge',ids:[i,j],wa,wb,delta:q.delta,d2:q.d2});
  }
  return out;
}

const bits=new DataView(new ArrayBuffer(8));
export function nextUp(x) {
  if(Number.isNaN(x)||x===Infinity)return x;
  if(x===0)return Number.MIN_VALUE;
  bits.setFloat64(0,x);let u=bits.getBigUint64(0);u+=x>0?1n:-1n;bits.setBigUint64(0,u);return bits.getFloat64(0);
}
export const nextDown=x=>-nextUp(-x);
const iv=x=>[x,x];
const addI=(a,b)=>[nextDown(a[0]+b[0]),nextUp(a[1]+b[1])];
const subI=(a,b)=>[nextDown(a[0]-b[1]),nextUp(a[1]-b[0])];
function mulI(a,b) {
  const p=[a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]];
  return [nextDown(Math.min(...p)),nextUp(Math.max(...p))];
}
function projectionI(p,n,origin) {
  let r=iv(0);for(let d=0;d<3;d++)r=addI(r,mulI(subI(iv(p[d]),iv(origin[d])),iv(n[d])));return r;
}
function lerpI(a,b,t) {
  if(t===0)return a;if(t===1)return b;
  return addI(mulI(subI(iv(1),iv(t)),a),mulI(iv(t),b));
}

// This proof only needs an arbitrary finite, nonzero axis. The method used to
// find it can be inaccurate or incomplete; that can cause rejection, not pass.
function axisCertifies(a0,a1,b,n,lo,hi,minimumDistance) {
  if(!finite3(n)||n.every(x=>x===0))return false;
  const origin=b[0],bp=b.map(p=>projectionI(p,n,origin));
  const ap=[];
  for(let i=0;i<3;i++) {
    const p=projectionI(a0[i],n,origin),q=projectionI(a1[i],n,origin);
    ap.push(lerpI(p,q,lo),lerpI(p,q,hi));
  }
  const vals=[...bp.flat(),...ap.flat()];if(!vals.every(Number.isFinite))return false;
  const alo=Math.min(...ap.map(x=>x[0])),ahi=Math.max(...ap.map(x=>x[1])),
    blo=Math.min(...bp.map(x=>x[0])),bhi=Math.max(...bp.map(x=>x[1])),
    gap=Math.max(nextDown(alo-bhi),nextDown(blo-ahi));
  if(!(gap>0))return false;
  let n2=iv(0);for(const c of n)n2=addI(n2,mulI(iv(c),iv(c)));
  const gap2Lower=nextDown(gap*gap),thresholdUpper=mulI(mulI(iv(minimumDistance),iv(minimumDistance)),n2)[1];
  return Number.isFinite(gap2Lower)&&Number.isFinite(thresholdUpper)&&gap2Lower>thresholdUpper;
}
function candidateAxes(a,b) {
  const ae=edges.map(([i,j])=>sub(a[j],a[i])),be=edges.map(([i,j])=>sub(b[j],b[i])),
    an=cross(ae[0],sub(a[2],a[0])),bn=cross(be[0],sub(b[2],b[0]));
  const axes=[[1,0,0],[0,1,0],[0,0,1],an,bn];
  for(const e of ae)for(const f of be)axes.push(cross(e,f));
  // In-plane axes help coplanar but disjoint triangles.
  for(const e of ae)axes.push(cross(an,e));for(const e of be)axes.push(cross(bn,e));
  try {for(const f of trianglePairFeatures(a,b))axes.push(f.delta);}catch {/* Heuristic failure cannot certify. */}
  return axes;
}

/**
 * Whole-triangle filter for x(t)=(1-t)*start+t*end, t in [0,1].
 * A true result certifies separation under the arithmetic assumptions above.
 * A false result means collision OR insufficient proof; never accept it.
 * It is a decision filter, not a time-of-impact estimator.
 */
export function certifyTriangleStep(start,end,bone,{minimumDistance=0,maxDepth=24,maxIntervals=2048,collectProof=false}={}) {
  const a0=triangle(start,'start'),a1=triangle(end,'end'),b=triangle(bone,'bone');
  if(!(Number.isFinite(minimumDistance)&&minimumDistance>=0))throw Error('Invalid minimumDistance');
  if(!Number.isInteger(maxDepth)||maxDepth<0||maxDepth>48||!Number.isInteger(maxIntervals)||maxIntervals<1||maxIntervals>1e6)
    throw Error('Invalid certificate work budget');
  if(typeof collectProof!=='boolean')throw Error('Invalid collectProof');
  const stack=[[0,1,0]],proof=[];let intervals=0,leaves=0,deepest=0;
  while(stack.length) {
    const [lo,hi,depth]=stack.pop();
    if(++intervals>maxIntervals)return {certified:false,reason:'interval-budget',intervals:intervals-1,leaves,unresolved:[lo,hi]};
    deepest=Math.max(deepest,depth);const mid=(lo+hi)/2,a=a0.map((p,i)=>mix(p,a1[i],mid));
    const axes=candidateAxes(a,b);
    const axis=axes.find(n=>axisCertifies(a0,a1,b,n,lo,hi,minimumDistance));
    if(axis){leaves++;if(collectProof)proof.push({lo,hi,axis:axis.slice()});continue;}
    if(depth>=maxDepth||mid===lo||mid===hi)return {certified:false,reason:'unresolved',intervals,leaves,deepest,unresolved:[lo,hi]};
    stack.push([mid,hi,depth+1],[lo,mid,depth+1]);
  }
  return {certified:true,reason:'separating-plane-cover',intervals,leaves,deepest,minimumDistance,...(collectProof?{proof}:{})};
}

/** Exhaustive tiny-mesh harness. Refuses large jobs before allocating pairs. */
export function certifyBoundaryStep({startVertices,endVertices,tissueTriangles,boneVertices,boneTriangles},options={}) {
  if(!Array.isArray(startVertices)||!Array.isArray(endVertices)||!Array.isArray(boneVertices)||
    startVertices.length!==endVertices.length||![...startVertices,...endVertices,...boneVertices].every(finite3))throw Error('Invalid boundary vertices');
  for(const [faces,count] of [[tissueTriangles,startVertices.length],[boneTriangles,boneVertices.length]])
    if(!Array.isArray(faces)||!faces.length||!faces.every(f=>f.length===3&&new Set(f).size===3&&f.every(i=>Number.isInteger(i)&&i>=0&&i<count)))throw Error('Invalid boundary triangles');
  const {maxTrianglePairs=4096,...pairOptions}=options,totalPairs=tissueTriangles.length*boneTriangles.length;
  if(!Number.isInteger(maxTrianglePairs)||maxTrianglePairs<1)throw Error('Invalid pair budget');
  if(totalPairs>maxTrianglePairs)return {certified:false,reason:'pair-budget',totalPairs,checkedPairs:0};
  let checkedPairs=0,intervals=0;
  for(let i=0;i<tissueTriangles.length;i++)for(let j=0;j<boneTriangles.length;j++) {
    const f=tissueTriangles[i],g=boneTriangles[j],r=certifyTriangleStep(f.map(k=>startVertices[k]),f.map(k=>endVertices[k]),g.map(k=>boneVertices[k]),pairOptions);
    checkedPairs++;intervals+=r.intervals;
    if(!r.certified)return {...r,reason:`pair-${r.reason}`,tissueTriangle:i,boneTriangle:j,totalPairs,checkedPairs,intervals};
  }
  return {certified:true,reason:'all-boundary-pairs',totalPairs,checkedPairs,intervals};
}

function barrier(d2,minimumDistance,activationDistance) {
  const s=d2-minimumDistance**2,h=activationDistance**2-minimumDistance**2;
  if(!(s>0))return {value:Infinity,first:NaN,second:NaN};
  if(s>=h)return {value:0,first:0,second:0};
  const q=s-h,l=Math.log(s/h);
  return {value:-q*q*l,first:-2*q*l-q*q/s,second:-2*l-4*q/s+q*q/(s*s)};
}
function mollifier(a,b,ia,ib,epsilon) {
  const [i,j]=edges[ia],[k,l]=edges[ib],u=sub(a[j],a[i]),v=sub(b[l],b[k]),w=cross(u,v),c=dot(w,w);
  const ga=zeros(),gb=zeros();
  if(c>=epsilon)return {value:1,ga,gb};
  const r=c/epsilon,derivative=2*(1-r)/epsilon,
    gu=scale(cross(v,w),2*derivative),gv=scale(cross(w,u),2*derivative);
  ga[j]=gu;ga[i]=scale(gu,-1);gb[l]=gv;gb[k]=scale(gv,-1);
  return {value:r*(2-r),ga,gb};
}

/** Small, fixed-topology triangle-pair potential with fixed rest mollifiers. */
export class StaticTrianglePairBarrier {
  constructor({restTissue,bone,minimumDistance=0,activationDistance,stiffness=1,mollifierScale=1e-3}={}) {
    this.restTissue=triangle(restTissue,'restTissue');this.bone=triangle(bone,'bone');
    if(!regularTriangle(this.restTissue)||!regularTriangle(this.bone))throw Error('Degenerate or ill-conditioned rest triangle');
    if(!(Number.isFinite(minimumDistance)&&minimumDistance>=0&&Number.isFinite(activationDistance)&&activationDistance>minimumDistance&&
      Number.isFinite(stiffness)&&stiffness>0&&Number.isFinite(mollifierScale)&&mollifierScale>0&&mollifierScale<=1))throw Error('Invalid barrier parameters');
    this.minimumDistance=minimumDistance;this.activationDistance=activationDistance;this.stiffness=stiffness;
    this.epsilons=edges.map(([i,j])=>edges.map(([k,l])=>mollifierScale*dot(sub(this.restTissue[j],this.restTissue[i]),sub(this.restTissue[j],this.restTissue[i]))*dot(sub(this.bone[l],this.bone[k]),sub(this.bone[l],this.bone[k]))));
    if(!this.epsilons.flat().every(x=>Number.isFinite(x)&&x>0))throw Error('Unrepresentable mollifier threshold');
  }
  evaluate(tissue) {
    const a=triangle(tissue,'tissue'),b=this.bone;
    if(!regularTriangle(a))return {valid:false,energy:Infinity,reason:'degenerate-or-ill-conditioned-tissue'};
    // The feature-distance list alone can miss an already-intersecting state.
    const certificate=certifyTriangleStep(a,a,b,{minimumDistance:this.minimumDistance,maxDepth:0,maxIntervals:1});
    if(!certificate.certified)return {valid:false,energy:Infinity,reason:'uncertified-current-separation',certificate};
    const gradient=zeros(),boneGradient=zeros(),metricBlocks=[],features=trianglePairFeatures(a,b),
      activeByKind={'tissue-vertex/bone-face':0,'bone-vertex/tissue-face':0,'edge/edge':0};let energy=0,activeTerms=0;
    for(const f of features) {
      const q=barrier(f.d2,this.minimumDistance,this.activationDistance);
      if(!Number.isFinite(q.value))return {valid:false,energy:Infinity,reason:'invalid-feature-distance'};
      if(q.value===0)continue;
      const m=f.kind==='edge/edge'?mollifier(a,b,...f.ids,this.epsilons[f.ids[0]][f.ids[1]]):{value:1,ga:zeros(),gb:zeros()},
        qa=f.wa.map(w=>scale(f.delta,2*w)),qb=f.wb.map(w=>scale(f.delta,-2*w)),k=this.stiffness;
      energy+=k*m.value*q.value;activeTerms++;activeByKind[f.kind]++;
      for(let i=0;i<3;i++)for(let d=0;d<3;d++) {
        gradient[i][d]+=k*(m.value*q.first*qa[i][d]+q.value*m.ga[i][d]);
        boneGradient[i][d]+=k*(m.value*q.first*qb[i][d]+q.value*m.gb[i][d]);
      }
      metricBlocks.push({coefficient:k*m.value*q.second,tissueDistanceSquaredGradient:qa});
    }
    if(!Number.isFinite(energy)||![...gradient.flat(),...boneGradient.flat(),...metricBlocks.map(q=>q.coefficient)].every(Number.isFinite))
      return {valid:false,energy:Infinity,reason:'nonfinite-energy-or-derivative'};
    return {valid:true,energy,gradient,boneGradient,metricBlocks,activeTerms,activeByKind,primitiveTerms:15,certificate,
      minimumFeatureDistance:Math.sqrt(Math.min(...features.map(f=>f.d2))),
      metricIsExactHessian:false,meshDiscretizationNormalized:false};
  }
}
