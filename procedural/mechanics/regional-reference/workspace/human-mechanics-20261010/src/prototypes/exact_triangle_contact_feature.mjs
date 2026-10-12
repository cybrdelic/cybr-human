/**
 * Exact intersection-feature verification for two represented binary64 triangles.
 * Original bounded implementation: lift the six input vertices to one integer
 * lattice, section A by B's plane, then clip that section against B's three
 * inward edge half-spaces. Homogeneous rational points avoid all division and
 * floating predicates. A polygon has at most six vertices; work is bounded.
 * No geometric tolerance, region exclusion, signed-solid test or force law.
 */
const bits=new DataView(new ArrayBuffer(8));
const abs=a=>a<0n?-a:a;
function gcd(a,b){a=abs(a);b=abs(b);while(b){const r=a%b;a=b;b=r;}return a;}
function dyadic(x){if(!Number.isFinite(x))throw Error('Exact contact inputs must be finite');if(x===0)return {n:0n,e:0};bits.setFloat64(0,x);const b=bits.getBigUint64(0),e=Number((b>>52n)&2047n),m=b&((1n<<52n)-1n);return {n:(b>>63n?-1n:1n)*(e?((1n<<52n)|m):m),e:e?e-1075:-1074};}
const sub=(a,b)=>a.map((x,i)=>x-b[i]),dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0n),
  cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],zero=a=>a.every(x=>x===0n);
function normalize(p){if(p[3]===0n)throw Error('Invalid homogeneous point');if(p[3]<0n)p=p.map(x=>-x);let g=0n;for(const x of p)g=gcd(g,x);return p.map(x=>x/g);}
function unique(points){const seen=new Set();return points.filter(p=>{const k=p.join(':');if(seen.has(k))return false;seen.add(k);return true;});}
const evaluate=(plane,p)=>dot(plane,p);
function intersectEdge(p,q,dp,dq){return normalize(p.map((x,i)=>dp*q[i]-dq*x));}
function clip(points,plane) {
  if(!points.length)return points;const out=[];
  for(let i=0;i<points.length;i++) {
    const p=points[(i+points.length-1)%points.length],q=points[i],dp=evaluate(plane,p),dq=evaluate(plane,q);
    if(dq>=0n){if(dp<0n)out.push(intersectEdge(p,q,dp,dq));out.push(q);}
    else if(dp>=0n)out.push(intersectEdge(p,q,dp,dq));
  }
  return unique(out);
}
function section(triangle,plane) {
  const p=triangle.map(v=>[...v,1n]),d=p.map(v=>evaluate(plane,v));if(d.every(x=>x===0n))return p;
  const out=[];for(let i=0;i<3;i++){const j=(i+1)%3;if(d[i]===0n)out.push(p[i]);if((d[i]<0n&&d[j]>0n)||(d[i]>0n&&d[j]<0n))out.push(intersectEdge(p[i],p[j],d[i],d[j]));}return unique(out);
}
function dimension(points) {
  if(!points.length)return -1;if(points.length===1)return 0;
  const a=points[0],u=points[1].slice(0,3).map((x,d)=>x*a[3]-a[d]*points[1][3]);
  for(const p of points.slice(2)){const v=p.slice(0,3).map((x,d)=>x*a[3]-a[d]*p[3]);if(!zero(cross(u,v)))return 2;}return 1;
}
function onSegment(h,a,b) {
  const delta=sub(b,a),r=h.slice(0,3).map((x,d)=>x-a[d]*h[3]),projection=dot(r,delta);
  return zero(cross(r,delta))&&projection>=0n&&projection<=dot(delta,delta)*h[3];
}

/** sharedPairs are explicit local vertex-index identities [{a:0,b:2}, ...]. */
export function verifyTriangleIntersectionFeature(a,b,sharedPairs=[]) {
  for(const t of [a,b])if(!t||t.length!==3||!t.every(p=>p&&p.length===3&&Array.from(p).every(Number.isFinite)))throw Error('Expected two finite 3-by-3 triangles');
  if(!Array.isArray(sharedPairs)||sharedPairs.length>3||new Set(sharedPairs.map(p=>p.a)).size!==sharedPairs.length||new Set(sharedPairs.map(p=>p.b)).size!==sharedPairs.length||!sharedPairs.every(p=>Number.isInteger(p.a)&&p.a>=0&&p.a<3&&Number.isInteger(p.b)&&p.b>=0&&p.b<3))throw Error('Invalid expected shared vertex pairs');
  for(const p of sharedPairs)if(!a[p.a].every((v,d)=>Object.is(v,b[p.b][d])))return {valid:false,reason:'source-vertex-identity-mismatch'};
  const encoded=[...a,...b].map(p=>p.map(dyadic)),exponents=encoded.flat().filter(x=>x.n!==0n).map(x=>x.e),emin=exponents.length?Math.min(...exponents):0,
    integer=encoded.map(p=>p.map(x=>x.n===0n?0n:x.n<<BigInt(x.e-emin))),A=integer.slice(0,3),B=integer.slice(3),
    na=cross(sub(A[1],A[0]),sub(A[2],A[0])),nb=cross(sub(B[1],B[0]),sub(B[2],B[0]));
  if(zero(na)||zero(nb))return {valid:false,reason:'degenerate-triangle'};
  let polygon=section(A,[...nb,-dot(nb,B[0])]);
  for(let i=0;i<3&&polygon.length;i++){const edge=sub(B[(i+1)%3],B[i]),inward=cross(nb,edge);polygon=clip(polygon,[...inward,-dot(inward,B[i])]);}
  const actualDimension=dimension(polygon),allowedDimension=sharedPairs.length-1,
    result={intersectionDimension:actualDimension,allowedDimension,intersectionVertexCount:polygon.length,exact:true,arithmetic:'integer-homogeneous-binary64-input'};
  if(!sharedPairs.length)return {...result,valid:polygon.length===0,reason:polygon.length?'unexpected-intersection':'disjoint'};
  if(actualDimension!==allowedDimension)return {...result,valid:false,reason:'intersection-exceeds-or-misses-shared-feature'};
  const shared=sharedPairs.map(p=>A[p.a]);
  const contained=shared.length===3||polygon.every(h=>shared.length===1?h.slice(0,3).every((x,d)=>x===shared[0][d]*h[3]):onSegment(h,...shared));
  return {...result,valid:contained,reason:contained?'exact-shared-feature':'intersection-outside-shared-feature'};
}
