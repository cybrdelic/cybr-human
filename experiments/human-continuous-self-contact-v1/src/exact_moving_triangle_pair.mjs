/** Experimental sufficient continuous two-moving-triangle certificate.
 * Binary64 endpoints are lifted exactly to one integer lattice. The path is
 * linear in time. Topologically shared vertex/edge pairs are declared by the
 * caller and must have exactly identical endpoint coordinates. No tolerance,
 * samples, centroid trim or blanket incident-face exemption proves acceptance.
 * This pair primitive is NOT complete mesh enumeration or solver integration.
 */
const bits=new DataView(new ArrayBuffer(8));
const abs=x=>x<0n?-x:x;
const gcd=(a,b)=>{a=abs(a);b=abs(b);while(b){const r=a%b;a=b;b=r;}return a;};
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0n);
const sub=(a,b)=>a.map((x,i)=>x-b[i]);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const zero=p=>p.every(x=>x===0n);
function triangle(t,name){
 if(!Array.isArray(t)||t.length!==3)throw Error(`${name}: expected finite 3-by-3 array`);
 const copy=t.map(p=>{if(!Array.isArray(p)||p.length!==3)throw Error(`${name}: expected finite 3-by-3 array`);return Array.from(p);});
 if(!copy.every(p=>p.every(Number.isFinite)))throw Error(`${name}: expected finite 3-by-3 array`);return copy;
}
function dyadic(x){
 if(x===0)return[0n,0];bits.setFloat64(0,x);const b=bits.getBigUint64(0),e=Number((b>>52n)&2047n),f=b&((1n<<52n)-1n);
 return[(b>>63n?-1n:1n)*(e?(1n<<52n)|f:f),e?e-1075:-1074];
}
function lift(points){const r=points.map(p=>p.map(dyadic));let e=0;for(const p of r)for(const[n,k]of p)if(n&&k<e)e=k;return r.map(p=>p.map(([n,k])=>n<<BigInt(k-e)));}
function canonical(n){if(zero(n))return null;let g=0n;for(const x of n)g=gcd(g,x);n=n.map(x=>x/g);return n.find(x=>x!==0n)<0n?n.map(x=>-x):n;}
const addP=(a,b)=>Array.from({length:Math.max(a.length,b.length)},(_,i)=>(a[i]??0n)+(b[i]??0n));
const negP=a=>a.map(x=>-x);
const subP=(a,b)=>addP(a,negP(b));
function mulP(a,b){const c=Array(a.length+b.length-1).fill(0n);for(let i=0;i<a.length;i++)for(let j=0;j<b.length;j++)c[i+j]+=a[i]*b[j];return c;}
const dotP=(a,b)=>a.reduce((s,p,i)=>addP(s,mulP(p,b[i])),[0n]);
const crossP=(a,b)=>[subP(mulP(a[1],b[2]),mulP(a[2],b[1])),subP(mulP(a[2],b[0]),mulP(a[0],b[2])),subP(mulP(a[0],b[1]),mulP(a[1],b[0]))];
const subVP=(a,b)=>a.map((p,i)=>subP(p,b[i]));
const mixVP=(a,b,k,l)=>a.map((p,i)=>subP(p.map(x=>x*k),b[i].map(x=>x*l)));
const trajectory=(a,b)=>a.map((x,i)=>[x,b[i]-x]);
function choose(n,k){let x=1n;for(let i=1;i<=k;i++)x=x*BigInt(n-i+1)/BigInt(i);return x;}
// b_j = sum_{k<=j} a_k C(j,k)/C(n,k). Common positive denominator
// is retained, so every returned integer has the exact Bernstein sign.
function bernstein(p){
 let n=p.length-1;while(n&&p[n]===0n)n--;let den=1n;
 for(let k=0;k<=n;k++){const c=choose(n,k);den=den/gcd(den,c)*c;}
 return Array.from({length:n+1},(_,j)=>{let x=0n;for(let k=0;k<=j;k++)x+=p[k]*choose(j,k)*(den/choose(n,k));return x;});
}
function signs(p){const b=bernstein(p);return{weak:b.every(x=>x>=0n),strict:b.every(x=>x>0n)};}
function regular(a){const n=crossP(subVP(a[1],a[0]),subVP(a[2],a[0]));return signs(dotP(n,n)).strict;}
function* hullAxes(a,b){
 const seen=new Set();function* emit(n){n=canonical(n);if(!n)return;const key=n.join(':');if(!seen.has(key)){seen.add(key);yield n;}}
 yield* emit([1n,0n,0n]);yield* emit([0n,1n,0n]);yield* emit([0n,0n,1n]);
 const edgeSets=[],normalSets=[];
 for(const h of [a,b]){
  const edges=[],normals=[];
  for(let i=0;i<h.length;i++)for(let j=i+1;j<h.length;j++){
   const e=sub(h[j],h[i]);edges.push(e);
   for(let k=j+1;k<h.length;k++){const n=cross(e,sub(h[k],h[i]));if(!zero(n)){normals.push(n);yield* emit(n);}}
  }
  edgeSets.push(edges);normalSets.push(normals);
 }
 for(const e of edgeSets[0])for(const f of edgeSets[1])yield* emit(cross(e,f));
 for(const normals of normalSets)for(const n of normals)for(const edges of edgeSets)for(const e of edges)yield* emit(cross(n,e));
}
function separated(a,b,n){const ap=a.map(p=>dot(p,n)),bp=b.map(p=>dot(p,n));return ap.reduce((x,y)=>x>y?x:y)<bp.reduce((x,y)=>x<y?x:y)||bp.reduce((x,y)=>x>y?x:y)<ap.reduce((x,y)=>x<y?x:y);}
function isqrt(n){if(n<0n)throw Error('Negative squared magnitude');if(n<2n)return n;let x=1n<<BigInt(Math.ceil(n.toString(2).length/2));while(true){const y=(x+n/x)>>1n;if(y>=x)return x;x=y;}}
function* incidentAxes(a,b,shared){
 const origin=a[shared[0][0]],ar=a.map(p=>subVP(p,origin)),br=b.map(p=>subVP(p,origin));
 if(shared.length===2){
  const edge=ar[shared[1][0]],p=ar.find((_,i)=>!shared.some(([j])=>i===j)),q=br.find((_,i)=>!shared.some(([,j])=>i===j));
  // Every such polynomial normal is exactly orthogonal to the moving edge.
  // Positive scale choices affect search completeness, never proof validity.
  yield crossP(edge,crossP(mixVP(p,q,1n,1n),edge));
  const mid=v=>v.map(p=>2n*p[0]+(p[1]??0n)),em=mid(edge),pm=cross(em,cross(mid(p),em)),qm=cross(em,cross(mid(q),em));
  // A rational approximation of the projected angular bisector is only a
  // candidate. The subsequent exact polynomial sign test remains mandatory.
  const k=isqrt(dot(qm,qm)<<128n),l=isqrt(dot(pm,pm)<<128n);
  if(k&&l)yield crossP(edge,crossP(mixVP(p,q,k,l),edge));
  for(const[k,l]of[[1n,0n],[0n,1n],...[1,2,4,8,12,16].flatMap(i=>[[1n<<BigInt(i),1n],[1n,1n<<BigInt(i)]])])yield crossP(edge,crossP(mixVP(p,q,k,l),edge));
 }else{
  const af=ar.filter((_,i)=>i!==shared[0][0]),bf=br.filter((_,i)=>i!==shared[0][1]);
  for(const n of [[1n,0n,0n],[0n,1n,0n],[0n,0n,1n]])yield n.map(x=>[x]);
  yield subVP(af[0].map((p,i)=>addP(p,af[1][i])),bf[0].map((p,i)=>addP(p,bf[1][i])));
  for(const p of af)for(const q of bf){yield subVP(p,q);yield crossP(p,q);}
  yield crossP(af[0],af[1]);yield crossP(bf[0],bf[1]);
  // Mid-time cone normals are arbitrary exact constant candidates. Their
  // continuous signs are checked against the original linear trajectories.
  const mid=v=>v.map(p=>2n*p[0]+(p[1]??0n));
  const rays=[...af,...bf].map(mid);
  const normals=[];
  for(let i=0;i<rays.length;i++)for(let j=i+1;j<rays.length;j++){const n=cross(rays[i],rays[j]);normals.push(n);yield n.map(x=>[x]);}
  // Coplanar incident wedges require in-plane separators. Merely testing the
  // common normal would leave every signed projection zero and reject safely.
  for(const n of normals)for(const r of rays)yield cross(n,r).map(x=>[x]);
 }
}

/** A true result covers the entire linear path, with intersection confined to
 * the declared shared feature at every time. A false result includes safe but
 * unresolved paths. Shared topology is caller-supplied, NOT exemption authority.
 */
export function certifyMovingTrianglePair(startA,endA,startB,endB,{shared=[],maxDepth=12,maxIntervals=255,maxAxisTests=100000}={}){
 const aa=triangle(startA,'startA'),az=triangle(endA,'endA'),ba=triangle(startB,'startB'),bz=triangle(endB,'endB');
 if(!Array.isArray(shared))throw Error('Invalid shared topology');
 shared=shared.map(p=>{if(!Array.isArray(p))throw Error('Invalid shared topology');return Array.from(p);});
 if(!Array.isArray(shared)||shared.length>2||shared.some(p=>!Array.isArray(p)||p.length!==2||p.some(i=>!Number.isInteger(i)||i<0||i>2))||new Set(shared.map(p=>p[0])).size!==shared.length||new Set(shared.map(p=>p[1])).size!==shared.length)throw Error('Invalid shared topology');
 if(!Number.isInteger(maxDepth)||maxDepth<0||maxDepth>30||!Number.isInteger(maxIntervals)||maxIntervals<1||maxIntervals>1000000||!Number.isInteger(maxAxisTests)||maxAxisTests<1||maxAxisTests>1000000)throw Error('Invalid proof budget');
 for(const[i,j]of shared)if(!aa[i].every((x,k)=>x===ba[j][k])||!az[i].every((x,k)=>x===bz[j][k]))throw Error('Shared endpoint trajectory mismatch');
 return certifyLiftedPair(lift([...aa,...az,...ba,...bz]),shared,{maxDepth,maxIntervals,maxAxisTests});
}
function certifyLiftedPair(z,shared,{maxDepth,maxIntervals,maxAxisTests}){
 const a0=z.slice(0,3),a1=z.slice(3,6),b0=z.slice(6,9),b1=z.slice(9,12);
 for(const t of[a0,a1,b0,b1])if(zero(cross(sub(t[1],t[0]),sub(t[2],t[0]))))return{certified:false,reason:'degenerate-endpoint',intervals:0,axisTests:0};
 const stack=[[0n,1n,0]],leaves=[];let intervals=0,axisTests=0;
 const fail=(reason,lo,hi,depth)=>({certified:false,reason,intervals,axisTests,unresolvedDyadic:{lo:String(lo),hi:String(hi),depth}});
 while(stack.length){
  const[lo,hi,depth]=stack.pop(),den=1n<<BigInt(depth);if(++intervals>maxIntervals)return fail('interval-budget',lo,hi,depth);
  const at=(start,end,n)=>start.map((p,i)=>p.map((x,k)=>x*(den-n)+end[i][k]*n));
  const al=at(a0,a1,lo),ah=at(a0,a1,hi),bl=at(b0,b1,lo),bh=at(b0,b1,hi),ap=al.map((p,i)=>trajectory(p,ah[i])),bp=bl.map((p,i)=>trajectory(p,bh[i]));
  let proved=false;
  if(regular(ap)&&regular(bp)){
   if(!shared.length){
    for(const n of hullAxes([...al,...ah],[...bl,...bh])){if(++axisTests>maxAxisTests)return fail('axis-budget',lo,hi,depth);if(separated([...al,...ah],[...bl,...bh],n)){proved=true;break;}}
   }else{
    const origin=ap[shared[0][0]],freeA=ap.filter((_,i)=>!shared.some(([j])=>i===j)).map(p=>subVP(p,origin)),freeB=bp.filter((_,i)=>!shared.some(([,j])=>i===j)).map(p=>subVP(p,origin));
    const covered=new Set();
    axisLoop:for(const raw of incidentAxes(ap,bp,shared))for(const sign of[1n,-1n]){
     if(++axisTests>maxAxisTests)return fail('axis-budget',lo,hi,depth);
     const n=raw.map(p=>p.map(x=>x*sign));
     const polys=[...freeA.map(p=>dotP(n,p)),...freeB.map(p=>negP(dotP(n,p)))],ss=polys.map(signs);
     if(ss.every(s=>s.weak)){ss.forEach((s,i)=>{if(s.strict)covered.add(i);});if(covered.size===ss.length){proved=true;break axisLoop;}}
    }
   }
  }
  if(proved){leaves.push({lo:String(lo),hi:String(hi),depth});continue;}
  if(depth===maxDepth)return fail('unresolved',lo,hi,depth);
  stack.push([lo+hi,2n*hi,depth+1],[2n*lo,lo+hi,depth+1]);
 }
 return{certified:true,reason:shared.length?'exact-moving-feature-plane-cover':'exact-disjoint-swept-hull-cover',intervals,axisTests,leaves,exactBinary64:true,continuousNondegeneracy:true,sharedFeatureSize:shared.length,uniformMargin:false,completeMeshEnumeration:false};
}


// Memoize only exact binary64 decoding, not a collision decision. Every pair
// still uses the same local minimum exponent, lifted integers and proof search.
const lattices=new WeakMap();
export function compileExactVertexTrajectory(start,end){
 if(!start||!end||start.length!==end.length||start.length%3||!start.length||start.length>300000)throw Error('Invalid indexed trajectory');
 const a=Float64Array.from(start),b=Float64Array.from(end),n=a.length/3;
 for(let i=0;i<a.length;i++)if(!Number.isFinite(a[i])||!Number.isFinite(b[i]))throw Error('Nonfinite indexed trajectory');
 const encode=v=>Array.from({length:n},(_,i)=>[dyadic(v[3*i]),dyadic(v[3*i+1]),dyadic(v[3*i+2])]);
 const token=Object.freeze({vertices:n,scheme:'exact-binary64-trajectory-cache-v1'});
 lattices.set(token,{a:encode(a),b:encode(b),n});return token;
}
export function certifyIndexedMovingTrianglePair(token,faceA,faceB,{maxDepth=12,maxIntervals=255,maxAxisTests=100000}={}){
 const c=lattices.get(token);if(!c)throw Error('Unrecognized exact trajectory token');
 const check=f=>{if(!f||f.length!==3)throw Error('Invalid indexed face');const p=Array.from(f);if(p.some(i=>!Number.isInteger(i)||i<0||i>=c.n)||new Set(p).size!==3)throw Error('Invalid indexed face');return p;};
 const a=check(faceA),b=check(faceB),shared=[];for(let i=0;i<3;i++)for(let j=0;j<3;j++)if(a[i]===b[j])shared.push([i,j]);
 if(shared.length>2)throw Error('Duplicate indexed face');
 if(!Number.isInteger(maxDepth)||maxDepth<0||maxDepth>30||!Number.isInteger(maxIntervals)||maxIntervals<1||maxIntervals>1000000||!Number.isInteger(maxAxisTests)||maxAxisTests<1||maxAxisTests>1000000)throw Error('Invalid proof budget');
 const r=[...a.map(i=>c.a[i]),...a.map(i=>c.b[i]),...b.map(i=>c.a[i]),...b.map(i=>c.b[i])];let e=0;for(const p of r)for(const[n,k]of p)if(n&&k<e)e=k;
 const z=r.map(p=>p.map(([n,k])=>n<<BigInt(k-e)));
 return certifyLiftedPair(z,shared,{maxDepth,maxIntervals,maxAxisTests});
}
