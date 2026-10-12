import assert from "node:assert/strict";
import { applyAffineTransfer, recoverSurfaceGradients } from "../src/affine_surface_transfer.mjs";
const rest=Float64Array.from([0,0,0, .01,0,0, 0,.01,0, 0,0,.01]),
  nodes=Uint32Array.from([0,1,2,3]),
  gradients=Float64Array.from([-100,-100,-100,100,0,0,0,100,0,0,0,100]),
  material=Float64Array.from([1e-6/6,1000,10000]),
  q=Float64Array.from([.002,.003,.001,.005,.002,-.0008]),
  indices=Uint32Array.from([0,1,2,0,1,2]),
  weights=Float64Array.from([.5,.2,.3,.3,.5,.2]);
const F0=recoverSurfaceGradients(rest,rest,nodes,gradients,material,4),
  zero=applyAffineTransfer(q,indices,weights,rest,rest,F0);
const restError=Math.max(...zero.map((v,j)=>Math.abs(v-q[j])));
assert(restError<1e-16,"Rest residual must be preserved exactly to roundoff");
const A=[.97,-.2,.01,.2,.99,.03,-.02,.01,1.03],t=[.001,-.002,.0005],
  transform=(data)=>Float64Array.from(data,(_,j)=>t[j%3]+[0,1,2].reduce((sum,k)=>sum+A[(j%3)*3+k]*data[Math.floor(j/3)*3+k],0)),
  x=transform(rest),truth=transform(q),F=recoverSurfaceGradients(x,rest,nodes,gradients,material,4),
  moved=applyAffineTransfer(q,indices,weights,x,rest,F),
  affineError=Math.max(...moved.map((v,j)=>Math.abs(v-truth[j])));
assert(affineError<1e-9,"Recovered Float32 gradients must reproduce an affine field");
console.log(JSON.stringify({passed:true,restError,affineError,formula:"sum w_i (x_i + F_i (q-X_i))",offSurfaceRestResidualsTested:true}));
