import assert from 'node:assert/strict';
import {certifyMovingTrianglePair as certify} from '../src/exact_moving_triangle_pair.mjs';
import {verifyTriangleIntersectionFeature as feature} from '../../../procedural/mechanics/regional-reference/workspace/human-mechanics-20261010/src/prototypes/exact_triangle_contact_feature.mjs';
let seed=0x61af3219;const integer=()=>{seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return(seed>>>0)%9-4;};
const tri=()=>Array.from({length:3},()=>Array.from({length:3},integer));
let accepted=0,sampleChecks=0,attempts=0;const byShared={0:0,1:0,2:0};
for(let i=0;i<900;i++){
 const a=tri(),az=tri(),b=tri(),bz=tri(),shared=Array.from({length:i%3},(_,k)=>[k,k]);
 for(const[x,y]of shared){b[y]=a[x].slice();bz[y]=az[x].slice();}
 const result=certify(a,az,b,bz,{shared,maxDepth:5,maxIntervals:63,maxAxisTests:5000});attempts++;
 if(!result.certified)continue;accepted++;byShared[shared.length]++;
 // Integer inputs and t=k/8 make these represented interpolation points exact.
 // This is independent exact endpoint-feature corroboration, not the proof.
 for(let k=0;k<=8;k++){
  const at=(x,y)=>x.map((p,i)=>p.map((v,j)=>(v*(8-k)+y[i][j]*k)/8));
  const r=feature(at(a,az),at(b,bz),shared.map(([a,b])=>({a,b})));
  assert(r.valid,JSON.stringify({i,k,shared,result,r}));sampleChecks++;
 }
}
assert(accepted>50&&Object.values(byShared).every(x=>x>5));
console.log(JSON.stringify({attempts,accepted,byShared,exactDyadicSampleChecks:sampleChecks,proofUsesSamples:false},null,2));
