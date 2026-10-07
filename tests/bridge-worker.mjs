// Bounded actual-worker regression, no browser/GPU. Use the isolated candidate.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { SynchronousGPUBridge } from '../src/synchronous_gpu_bridge.js';
const root = fileURLToPath(new URL('..', import.meta.url));
const runtime = JSON.parse(fs.readFileSync(path.join(root, 'output/runtime-candidate.json')));
const model = JSON.parse(fs.readFileSync(path.join(root, runtime.tissueBase, 'model.json')));
const arrays = {};
for (const [name, asset] of Object.entries(model.assets)) {
  const raw = fs.readFileSync(path.join(root, asset.url));
  const Type = ['nodes','faces','layers','groups'].includes(name) ? Uint32Array : Float64Array;
  arrays[name] = new Type(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
}
let reply;
globalThis.self = { postMessage: value => { reply = value; } };
await import(pathToFileURL(path.join(root, path.dirname(runtime.entry), 'tissue_cpu_worker.js')));
function send(data) { reply = null; self.onmessage({ data: { id: 1, ...data } }); assert(reply); assert(!reply.error, reply.error); return reply; }
const init = () => send({ type:'init', model, arrays: Object.fromEntries(Object.entries(arrays).map(([k,v])=>[k,v.slice()])) });
const options = { dt: 1/120, newton: 2, cg: 24, checkpointState: true };
const parameters = { gravity: 1, fatPercent: 10, sagPercent: 10 };
init();
const together = send({ type:'step', parameters, options:{...options, substeps:2} });
init(); send({type:'step', parameters, options:{...options, substeps:1} });
const separate = send({type:'step', parameters, options:{...options, substeps:1} });
assert.deepEqual(together.velocities, separate.velocities);
assert.deepEqual(together.packed, separate.packed);
const bridge = new SynchronousGPUBridge(model);
bridge.accept(together);
const checkpoint = bridge.lost();
init();
const receipt = send({type:'restore', positions:checkpoint.positions, velocity:checkpoint.velocities, verifyCheckpoint:true});
bridge.recovered(checkpoint, receipt);
let restorePositionError = 0;
for (let i = 0; i < receipt.positions.length; i++) {
  const error = Math.abs(receipt.positions[i] - checkpoint.positions[i]);
  restorePositionError = Math.max(restorePositionError, error);
  if (arrays.fixedNodes?.[Math.floor(i / 3)] > 0.5) assert(error < 1e-7);
  else assert.equal(error, 0);
}
assert.deepEqual(receipt.velocities, checkpoint.velocities);
const resumed = send({type:'step', parameters, options:{...options, substeps:1} });
bridge.accept(resumed);
const staticResult = send({type:'step', parameters, options:{...options, substeps:1, quasiStatic:true} });
assert(staticResult.velocities.every(v=>v===0));
bridge.accept(staticResult);
const seconds = bridge.summary.version.simulationSeconds;
assert.equal(seconds, 1/40);
send({type:'reset'});
bridge.dispose();
const fresh = new SynchronousGPUBridge(model);
assert.equal(fresh.summary.version.simulationSeconds, 0);
assert.equal(fresh.picking(), null);
console.log(JSON.stringify({build:runtime.build, nodes:model.nodes, finalVelocitySubstepError:0,
  restorePositionError, restoreVelocityError:0, recoveredSimulationSeconds:seconds,
  minJ:resumed.stats.minJ, limits:'Actual CPU worker only; GPU velocity packing/shader/viewer/device loss require paired browser review.'}));
