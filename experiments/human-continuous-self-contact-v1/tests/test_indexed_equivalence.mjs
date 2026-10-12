import assert from 'node:assert/strict';
import {certifyMovingTrianglePair as before} from './fixtures/exact_moving_triangle_pair-hardened-v2.mjs';
import {compileExactVertexTrajectory,certifyIndexedMovingTrianglePair} from '../src/exact_moving_triangle_pair.mjs';
let seed=0x61af3219;const num=()=>{seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return(seed>>>0)%9-4;};
const tri=()=>Array.from({length:3},()=>Array.from({length:3},num));
let accepted=0;
for(let i=0;i<900;i++){
 const a=tri(),az=tri(),b=tri(),bz=tri(),shared=Array.from({length:i%3},(_,k)=>[k,k]);
 for(const[x,y]of shared){b[y]=a[x].slice();bz[y]=az[x].slice();}
 const options={maxDepth:5,maxIntervals:63,maxAxisTests:5000},first=before(a,az,b,bz,{shared,...options});
 const start=[...a.flat(),...b.flat()],end=[...az.flat(),...bz.flat()],token=compileExactVertexTrajectory(start,end),fa=[0,1,2],fb=[3,4,5];
 for(const[x,y]of shared)fb[y]=fa[x];
 const second=certifyIndexedMovingTrianglePair(token,fa,fb,options);
 assert.equal(JSON.stringify(second),JSON.stringify(first),`case ${i} differs`);if(second.certified)accepted++;
 // Caller-owned changes cannot alter an already compiled trajectory.
 start.fill(20);end.fill(-20);
 assert.equal(JSON.stringify(certifyIndexedMovingTrianglePair(token,fa,fb,options)),JSON.stringify(first));
}
assert.throws(()=>certifyIndexedMovingTrianglePair({vertices:6},[0,1,2],[3,4,5]),/Unrecognized/);
const t=compileExactVertexTrajectory([0,0,0,1,0,0,0,1,0],[0,0,0,1,0,0,0,1,0]);
assert.throws(()=>certifyIndexedMovingTrianglePair(t,[0,1,2],[0,2,1]),/Duplicate/);
assert.throws(()=>certifyIndexedMovingTrianglePair(t,[0,1,2],[0,3,1]),/Invalid/);
console.log(JSON.stringify({cases:900,accepted,byteIdenticalResults:true,callerMutationCannotChangeCompiledTrajectory:true,forgedAndInvalidTokensRejected:true},null,2));
