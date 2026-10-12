/**
 * Conservative, exact continuous contact certificate. A triangle with linear
 * vertex paths lies in the convex hull of its six endpoint vertices. Registered
 * stationary roots are interpreted as normalized binary64 barycentric source
 * points, not as their rounded solver coordinates. No sampling or tolerance is
 * used in the contact proof. Failure to find this sufficient proof is rejection.
 */
import {verifyParameterizedTriangleContact} from '../../../procedural/mechanics/regional-reference/workspace/human-preferred-v53-anatomical-assembly-20261011/resumed-source/parametric_triangle_contact.mjs';
const bits=new DataView(new ArrayBuffer(8));
const abs=x=>x<0n?-x:x;
const gcd=(a,b)=>{a=abs(a);b=abs(b);while(b){const r=a%b;a=b;b=r;}return a;};
const lcm=(a,b)=>a/gcd(a,b)*b;
const reduce=(n,d)=>{if(d===0n)throw Error('Zero rational denominator');if(d<0n){n=-n;d=-d;}const g=gcd(n,d);return[n/g,d/g];};
function rational(x){
 if(!Number.isFinite(x))throw Error('Expected finite binary64 number');if(x===0)return[0n,1n];
 bits.setFloat64(0,x);const b=bits.getBigUint64(0),e=Number((b>>52n)&2047n),m=b&((1n<<52n)-1n),n=(b>>63n?-1n:1n)*(e?((1n<<52n)|m):m),p=e?e-1075:-1074;
 return p>=0?[n<<BigInt(p),1n]:reduce(n,1n<<BigInt(-p));
}
const add=(a,b)=>reduce(a[0]*b[1]+b[0]*a[1],a[1]*b[1]);
const mul=(a,b)=>reduce(a[0]*b[0],a[1]*b[1]);
const div=(a,b)=>reduce(a[0]*b[1],a[1]*b[0]);
const minus=(a,b)=>add(a,[-b[0],b[1]]);
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0n);
const sub=(a,b)=>a.map((x,i)=>x-b[i]);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const zero=a=>a.every(x=>x===0n);
const finite3=p=>p&&p.length===3&&Array.from(p).every(Number.isFinite);
function triangle(t,name){if(!t||t.length!==3||!Array.from(t).every(finite3))throw Error(`${name} requires a finite 3-by-3 triangle`);return Array.from(t,p=>Array.from(p));}
function sourcePoint(r){
 if(!r||!r.weights||r.weights.length!==3||!Array.from(r.weights).every(x=>Number.isFinite(x)&&x>=0))throw Error('Invalid normalized barycentric source');
 const v=triangle(r.vertices,'source'),w=Array.from(r.weights,rational),sum=w.reduce(add,[0n,1n]);if(sum[0]===0n)throw Error('Zero barycentric source sum');
 return[0,1,2].map(k=>div(w.map((q,i)=>mul(q,rational(v[i][k]))).reduce(add,[0n,1n]),sum));
}
function validateSpecs(specs){
 if(!specs||typeof specs!=='object'||Array.isArray(specs))throw Error('Invalid source index map');
 const keys=Reflect.ownKeys(specs);
 if(keys.length>3||keys.some(k=>typeof k!=='string'||!['0','1','2'].includes(k)))throw Error('Invalid root local index');
 return keys.map(Number);
}
/** Exact Euclidean comparison with the inherited solver representation bound. */
export function sourceRepresentationWithinBound(point,source,bound=2e-15){
 if(!finite3(point)||!Number.isFinite(bound)||bound<0)throw Error('Invalid representation-bound input');
 const p=sourcePoint(source),d=p.map((q,k)=>minus(rational(point[k]),q));
 const d2=d.map(q=>mul(q,q)).reduce(add,[0n,1n]),b2=mul(rational(bound),rational(bound));
 return d2[0]*b2[1]<=b2[0]*d2[1];
}
function lift(a0,a1,b,specs){
 const roots=new Map(validateSpecs(specs).map(i=>[i,sourcePoint(specs[i])]));
 const r=[...a0.map((p,i)=>roots.get(i)??p.map(rational)),...a1.map((p,i)=>roots.get(i)??p.map(rational)),...b.map(p=>p.map(rational))];
 let den=1n;for(const p of r)for(const q of p)den=lcm(den,q[1]);
 const v=r.map(p=>p.map(q=>q[0]*(den/q[1])));return{h:v.slice(0,6),b:v.slice(6),roots:new Set(roots.keys()),den};
}
function canonical(n){if(zero(n))return null;let g=0n;for(const x of n)g=gcd(g,x);n=n.map(x=>x/g);return n.find(x=>x!==0n)<0n?n.map(x=>-x):n;}
function* candidates(h,b){
 const normals=[],edges=[],seen=new Set();
 function* push(n){const p=canonical(n);if(!p)return;const k=p.join(':');if(!seen.has(k)){seen.add(k);yield p;}}
 yield* push([1n,0n,0n]);yield* push([0n,1n,0n]);yield* push([0n,0n,1n]);
 const bn=cross(sub(b[1],b[0]),sub(b[2],b[0]));yield* push(bn);
 for(let i=0;i<h.length;i++)for(let j=i+1;j<h.length;j++){
  const edge=sub(h[j],h[i]);edges.push(edge);
  for(let k=j+1;k<h.length;k++){const n=cross(edge,sub(h[k],h[i]));if(!zero(n)){normals.push(n);yield* push(n);}}
 }
 const be=[sub(b[1],b[0]),sub(b[2],b[1]),sub(b[0],b[2])];
 for(const e of edges)for(const f of be)yield* push(cross(e,f));
 // Required for lower-dimensional coplanar hulls; candidate quality affects
 // completeness only. Every returned witness is checked with exact arithmetic.
 for(const e of be)yield* push(cross(bn,e));for(const e of edges)yield* push(cross(bn,e));
 for(const n of normals)for(const e of edges)yield* push(cross(n,e));
}
function projections(n,h,b){const bp=b.map(p=>dot(n,p)),c=bp.reduce((a,x)=>a>x?a:x),g=h.map(p=>dot(n,p)-c);return{n,c,g};}
/** Root records here are explicit pair data, not exemption authority. Use the
 * opaque token integration to ground them to immutable bone and pin context. */
export function certifyExactMovingAttachment(start,end,bone,sourceByLocalIndex={}){
 const a0=triangle(start,'start'),a1=triangle(end,'end'),b0=triangle(bone,'bone'),ids=validateSpecs(sourceByLocalIndex);
 for(const i of ids){
  if(!a0[i].every((x,k)=>Object.is(x,a1[i][k])))return{certified:false,reason:'registered-root-moved'};
  if(!sourceRepresentationWithinBound(a0[i],sourceByLocalIndex[i]))return{certified:false,reason:'root-source-representation-mismatch'};
 }
 const {h,b,roots,den}=lift(a0,a1,b0,sourceByLocalIndex);
 if(zero(cross(sub(b[1],b[0]),sub(b[2],b[0])))||[0,3].some(k=>zero(cross(sub(h[k+1],h[k]),sub(h[k+2],h[k])))))return{certified:false,reason:'degenerate-triangle'};
 const free=[0,1,2,3,4,5].filter(i=>!roots.has(i%3));
 // Exact endpoint contact-feature checks also constrain the fixed root hull:
 // it is a subset of either endpoint triangle. Thus a root-root chord cannot
 // cross an unrelated bone face and acquire unauthorized contact.
 const contacts=[verifyParameterizedTriangleContact(a0,b0,sourceByLocalIndex),verifyParameterizedTriangleContact(a1,b0,sourceByLocalIndex)];
 if(contacts.some(q=>!q.valid))return{certified:false,reason:'endpoint-contact-outside-registered-feature',contacts};
 if(!free.length)return{certified:true,reason:'fixed-source-feature',exact:true,contacts,positiveSeparation:false,uniformMargin:false,axesTested:0};
 const axes=candidates(h,b),weak=[],covered=new Set();let tested=0;
 for(const raw of axes)for(const sign of[1n,-1n]){
  const q=projections(raw.map(x=>sign*x),h,b);tested++;
  if(q.g.every(x=>x>0n))return{certified:true,reason:'exact-positive-swept-hull-separation',exact:true,positiveSeparation:true,uniformMargin:false,axesTested:tested,witness:{axis:q.n.map(String),offset:String(q.c),coordinateDenominator:String(den)}};
  if(q.g.every(x=>x>=0n)){
   const useful=free.filter(i=>q.g[i]>0n&&!covered.has(i));if(!useful.length)continue;
   weak.push(q);useful.forEach(i=>covered.add(i));
   if(covered.size===free.length){
    const n=[0n,0n,0n];let c=0n;for(const w of weak){w.n.forEach((x,k)=>n[k]+=x);c+=w.c;}
    // Recheck the combined witness itself. Its offset need not be the support
    // of B: max(sum n_k.B) <= sum(max n_k.B) is sufficient.
    if(!b.every(p=>dot(n,p)<=c)||!h.every(p=>dot(n,p)>=c)||!free.every(i=>dot(n,h[i])>c))throw Error('Internal combined-witness invariant failed');
    return{certified:true,reason:'exact-fixed-root-swept-hull-contact',exact:true,positiveSeparation:false,uniformMargin:false,axesTested:tested,combinedAxes:weak.length,contacts,witness:{axis:n.map(String),offset:String(c),coordinateDenominator:String(den)}};
   }
  }
 }
 return{certified:false,reason:'unresolved-swept-root-hull',exact:true,axesTested:tested,strictFreeEndpoints:covered.size,requiredFreeEndpoints:free.length};
}
