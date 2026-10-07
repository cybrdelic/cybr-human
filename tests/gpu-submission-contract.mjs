import assert from 'node:assert/strict';
import { GPUHeadFEM } from '../src/full_head_fem_gpu.js';
globalThis.GPUBufferUsage = { UNIFORM: 1, COPY_DST: 2, MAP_READ: 4, QUERY_RESOLVE: 8, COPY_SRC: 16 };
globalThis.GPUMapMode = { READ: 1 };
function engine() {
  const resources = [], submits = [], bindings = [], dispatches = [];
  const device = {
    lost: new Promise(() => {}), addEventListener() {}, destroy() {},
    createBuffer({ size }) {
      const b = { size, data: new ArrayBuffer(size), destroys: 0,
        destroy() { this.destroys++; this.reject?.(Error('destroyed')); },
        unmap() { this.unmapped = true; this.unmaps = (this.unmaps ?? 0) + 1; }, getMappedRange() { return this.data; },
        mapAsync() { return new Promise((resolve, reject) => { this.resolve = resolve; this.reject = reject; }); } };
      resources.push(b); return b;
    },
    createQuerySet() { const q = { destroys: 0, destroy() { this.destroys++; } }; resources.push(q); return q; },
    createBindGroup(desc) { bindings.push(desc); return desc; },
    createCommandEncoder() {
      const copies = [];
      return {
        beginComputePass() { return { setBindGroup() {}, setPipeline(p) { this.p = p; },
          dispatchWorkgroups(n) { dispatches.push([this.p, n]); }, end() {} }; },
        copyBufferToBuffer(a, offset, b, dest, length) { copies.push(() => new Uint8Array(b.data, dest, length).set(new Uint8Array(a.data, offset, length))); },
        resolveQuerySet(q, start, count, dest) { copies.push(() => new BigUint64Array(dest.data).set([100n, 300n])); },
        finish() { return copies; },
      };
    },
    queue: {
      writeBuffer(b, offset, bytes) { new Uint8Array(b.data, offset, bytes.byteLength).set(new Uint8Array(bytes.buffer ?? bytes, bytes.byteOffset ?? 0, bytes.byteLength)); },
      submit(commands) { submits.push(commands); for (const cmd of commands) for (const copy of cmd) copy(); },
    },
  };
  const e = new GPUHeadFEM(device, { nodes: 2, surface_nodes: 1 }, {}, true);
  Object.assign(e, { N: 2, S: 1, T: 1, groups: 1, edges: 1, checkpointState: true,
    outputBytes: 96, velocityBytes: 32, statsOffset: 128,
    paramBytes: new ArrayBuffer(80), controlNames: new Set(['predict','cgStart','cgBeta','currentEnergy','acceptTrial']),
    pipelines: new Proxy({}, { get: (_, name) => name }), bindGroup: {}, controlBindGroup: {},
    submissionLayout: {}, submissionControlLayout: {} });
  e.paramF = new Float32Array(e.paramBytes); e.paramU = new Uint32Array(e.paramBytes);
  e.buffers = Array.from({ length: 8 }, () => device.createBuffer({ size: 256 }));
  e.params = device.createBuffer({ size: 80 }); e.readback = device.createBuffer({ size: 208 });
  e.queryBuffer = device.createBuffer({ size: 16 }); e.queries = device.createQuerySet();
  e.initialNodeData = new ArrayBuffer(256);
  const stats = new Float32Array(e.buffers[4].data); stats[9] = 1; stats[10] = 0.01;
  const packed = new Float32Array(e.buffers[5].data); packed[24] = 0.3; packed[25] = 0.4;
  return { e, resources, submits, bindings, dispatches };
}
const options = { indirect: false, cg: 1, substeps: 1, newton: 1 };
const h = engine();
const original = h.e.step({ fatPercent: 12 }, options);
h.e.readback.resolve(); const baseline = await original;
const defaultDispatch = h.dispatches.splice(0);
const a = await h.e.submitStep({ fatPercent: 12 }, options, { inputVersion: 1 });
const b = await h.e.submitStep({ fatPercent: 24 }, options, { inputVersion: 2 });
assert.deepEqual(h.dispatches.slice(0, defaultDispatch.length), defaultDispatch);
assert.deepEqual(h.dispatches.slice(defaultDispatch.length), defaultDispatch);
assert.notEqual(a.params, b.params); assert.notEqual(a.readback, b.readback);
assert.notEqual(a.queries, b.queries); assert.notEqual(a.queryBuffer, b.queryBuffer);
assert(Math.abs(new Float32Array(a.params.data)[6] - .12) < 1e-7);
assert(Math.abs(new Float32Array(b.params.data)[6] - .24) < 1e-7);
assert.equal(h.e.frames, 1);
await assert.rejects(h.e.submitStep({}, options), /backpressure/);
b.readback.resolve(); const rb = await b.completion;
assert.throws(() => h.e.acceptSubmittedStep(b, rb), /order/);
a.readback.resolve(); const ra = await a.completion;
assert.deepEqual(ra.packed, baseline.packed); assert.deepEqual(ra.velocities, baseline.velocities);
assert.equal(ra.stats.gpuMs, baseline.stats.gpuMs);
h.e.acceptSubmittedStep(a, ra); h.e.acceptSubmittedStep(b, rb);
assert.equal(h.e.frames, 3); assert.equal(h.e.submissionSlots.size, 0);
for (const s of [a, b]) for (const resource of [s.params, s.readback, s.queries, s.queryBuffer]) assert.equal(resource.destroys, 0);
const allocationCount = h.resources.length;
const c = await h.e.submitStep({}, options);
assert.equal(h.resources.length, allocationCount); assert.equal(c.bank, a.bank);
a.cancel(); assert.equal(c.readback.destroys, 0);
assert.equal(h.e.peakSubmissionSlotBytes, 2 * a.bytes);
assert.equal(h.e.allocatedReusableBankBytes, 2 * a.bytes);
assert.equal(h.e.bufferMemory.allocations.filter(a => a.bankIndex != null).length, 6);
assert.equal(h.e.bufferMemory.ownedBufferBytes, h.e.bufferMemory.baseBufferBytes + 2 * a.bytes);
assert.equal(h.e.bufferMemory.peakOwnedBufferBytes, h.e.bufferMemory.ownedBufferBytes);
// Previous owned CPU snapshots remain unchanged even when the GPU bank is reused.
assert.deepEqual(ra.packed, baseline.packed);
c.readback.resolve(); const rc = await c.completion;
assert.equal(c.readback.destroys, 0); assert.equal(c.bank.inUse, true);
h.e.acceptSubmittedStep(c, rc);
const d = await h.e.submitStep({}, options); h.e.cancelSubmittedSteps();
await assert.rejects(d.completion); await assert.rejects(h.e.step({}, options), /recovery/);
assert.equal(c.readback.destroys, 1); assert.equal(h.e.allocatedReusableBankBytes, 0); h.e.reset(); assert.equal(h.e.submissionNeedsRecovery, false);
h.e.boundedSubmissions = true; await assert.rejects(h.e.submitStep({}, options), /bounded/);
h.e.dispose(); h.e.dispose();
assert(h.resources.every(r => r.destroys === 1));
// A readback-copy exception unmaps in cleanup and destroys every bank once.
{
  const f = engine(), first = await f.e.submitStep({}, options), tail = await f.e.submitStep({}, options);
  first.readback.getMappedRange = () => { throw Error('copy failed'); };
  first.readback.resolve();
  await assert.rejects(first.completion, /copy failed/);
  await assert.rejects(tail.completion);
  assert.equal(first.readback.unmaps, 1);
  assert.equal(f.e.submissionBanks.length, 0); assert.equal(f.e.submissionNeedsRecovery, true);
  f.e.dispose(); f.e.dispose(); assert(f.resources.every(r => r.destroys === 1));
}
// A map failure never attempts to unmap an unmapped bank.
{
  const f = engine(), first = await f.e.submitStep({}, options);
  first.readback.reject(Error('map failed')); await assert.rejects(first.completion, /map failed/);
  assert.equal(first.readback.unmaps ?? 0, 0); assert.equal(first.readback.destroys, 1);
  f.e.dispose(); assert(f.resources.every(r => r.destroys === 1));
}
// Late old-generation rejection cannot poison/destroy banks created after reset.
{
  const f = engine(), old = await f.e.submitStep({}, options);
  f.e.reset();
  const fresh = await f.e.submitStep({}, options);
  await assert.rejects(old.completion);
  assert.equal(f.e.submissionNeedsRecovery, false); assert.equal(fresh.readback.destroys, 0);
  fresh.readback.resolve(); const result = await fresh.completion;
  f.e.acceptSubmittedStep(fresh, result); f.e.dispose();
  assert(f.resources.every(r => r.destroys === 1));
}
console.log('GPU submission contract: original dispatch parity, private params/timestamps/full readback, ordered acceptance, reset and exact-once resource cleanup passed');
