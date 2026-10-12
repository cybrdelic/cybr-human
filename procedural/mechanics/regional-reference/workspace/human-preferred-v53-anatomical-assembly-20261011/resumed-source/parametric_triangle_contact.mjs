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


const lcm=(a,b)=>abs(a/gcd(a,b)*b);
function rational(x){const q=dyadic(x);return q.e>=0?[q.n<<BigInt(q.e),1n]:[q.n,1n<<BigInt(-q.e)];}
function add(a,b){const d=lcm(a[1],b[1]),n=a[0]*(d/a[1])+b[0]*(d/b[1]),g=gcd(n,d);return[n/g,d/g];}
function mul(a,b){const n=a[0]*b[0],d=a[1]*b[1],g=gcd(n,d);return[n/g,d/g];}
function div(a,b){const n=a[0]*b[1],d=a[1]*b[0],g=gcd(n,d);return[d<0n?-n/g:n/g,abs(d/g)];}
function parametricPoint(record){
 if(!record || record.vertices?.length!==3 || record.weights?.length!==3 || record.weights.some(w=>!Number.isFinite(w)||w<0))throw Error('Invalid barycentric source');
 const w=record.weights.map(rational),sum=w.reduce(add,[0n,1n]);if(sum[0]===0n)throw Error('Zero barycentric sum');
 return [0,1,2].map(k=>div(w.map((q,j)=>mul(q,rational(record.vertices[j][k]))).reduce(add,[0n,1n]),sum));
}
function pointInside(h,B,nb){
 if(evaluate([...nb,-dot(nb,B[0])],h)!==0n)return false;
 for(let i=0;i<3;i++){const edge=sub(B[(i+1)%3],B[i]),inward=cross(nb,edge);if(evaluate([...inward,-dot(inward,B[i])],h)<0n)return false;}
 return true;
}
export function verifyParameterizedTriangleContact(a,b,sourceByLocalIndex={}){
 const ra=a.map((p,i)=>sourceByLocalIndex[i]?parametricPoint(sourceByLocalIndex[i]):p.map(rational)),rb=b.map(p=>p.map(rational));
 let den=1n;for(const p of [...ra,...rb])for(const q of p)den=lcm(den,q[1]);
 const ints=[...ra,...rb].map(p=>p.map(q=>q[0]*(den/q[1]))),A=ints.slice(0,3),B=ints.slice(3),na=cross(sub(A[1],A[0]),sub(A[2],A[0])),nb=cross(sub(B[1],B[0]),sub(B[2],B[0]));
 if(zero(na)||zero(nb))return{valid:false,reason:'degenerate-triangle'};
 let polygon=section(A,[...nb,-dot(nb,B[0])]);
 for(let i=0;i<3&&polygon.length;i++){const edge=sub(B[(i+1)%3],B[i]),inward=cross(nb,edge);polygon=clip(polygon,[...inward,-dot(inward,B[i])]);}
 const allowed=Object.keys(sourceByLocalIndex).map(Number).filter(i=>pointInside([...A[i],1n],B,nb)).map(i=>A[i]);
 const contained=allowed.length===3||polygon.every(h=>allowed.length===1?h.slice(0,3).every((x,d)=>x===allowed[0][d]*h[3]):allowed.length===2?onSegment(h,...allowed):false);
 return{valid:polygon.length===0||contained,reason:polygon.length===0?'parametric-disjoint':contained?'source-parametric-feature':'intersection-outside-source-feature',intersectionDimension:dimension(polygon),allowedRootPoints:allowed.length,intersectionVertexCount:polygon.length,arithmetic:'exact rational normalized-binary64 barycentric source',usesParametricRoots:Object.keys(sourceByLocalIndex).length};
}
