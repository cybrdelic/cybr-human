import assert from "node:assert/strict";
import { packProbeVertices } from "../src/shared_surface_probe.js";
const positions=new Float32Array([.1,.2,.3]),normals=new Float32Array([0,0,1]);
const smooth=new Float32Array([0,1,0,1,0,1,0,1,.25,.25,.25,.25,0,0,0,0]);
const packed=packProbeVertices(positions,normals,smooth,2),floats=new Float32Array(packed),ids=new Uint32Array(packed);
assert.equal(packed.byteLength,96);assert.equal(ids[9],1);
assert.deepEqual([...floats.subarray(0,3)],[...positions]);
assert.deepEqual([...floats.subarray(4,7)],[...normals]);
assert.equal(floats[16],.25);
for(const [label,p,n,s,nodes] of [
 ["bad length",positions,normals,new Float32Array(8),2],
 ["zero normal",positions,new Float32Array(3),smooth,2],
 ["missing support",positions,normals,smooth,1],
 ["bad weights",positions,normals,new Float32Array(16),2],
 ["nonfinite",new Float32Array([NaN,0,0]),normals,smooth,2],
])assert.throws(()=>packProbeVertices(p,n,s,nodes),undefined,label);
console.log("Shared-device probe input representation and rejection contracts passed; GPU execution pending");
