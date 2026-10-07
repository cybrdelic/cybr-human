import assert from 'node:assert/strict';
import { SynchronousGPUBridge } from '../src/synchronous_gpu_bridge.js';
const model = { nodes: 3, surface_nodes: 1 };
function result(seconds = 1 / 60) {
  const packed = new Float32Array(24), velocities = new Float64Array(9).fill(0.012);
  for (let n = 0; n < 3; n++) packed.set([0.001 * n, 0.002 * n, 0.003 * n, 0], n * 4);
  return { packed, velocities, stats: { minJ: 1, residualN: 0.001, staticResidualN: 0.001,
    maxNodeSpeedMps: 0.012, kineticEnergyJ: 0.01, simulatedSeconds: seconds } };
}
const bridge = new SynchronousGPUBridge(model), first = result();
assert.equal(bridge.picking(), null);
assert.equal(bridge.accept(first).sequence, 1);
const proxy = bridge.picking();
first.packed.fill(99); first.velocities.fill(99);
assert.equal(proxy.positions[0], 0);
assert.equal(bridge.picking().positions[0], 0);
const accepted = bridge.summary.version;
for (const mutate of [r => r.packed[7] = NaN, r => r.velocities[0] = Infinity,
  r => r.velocities = new Float64Array(3), r => r.stats.residualN = NaN,
  r => r.stats.kineticEnergyJ = Infinity, r => r.stats.simulatedSeconds = -1]) {
  const bad = result(); mutate(bad); assert.throws(() => bridge.accept(bad));
  assert.deepEqual(bridge.summary.version, accepted);
}
assert.equal(bridge.accept(result(0)).simulationSeconds, 1 / 60);
const checkpoint = bridge.lost();
assert.equal(bridge.picking(), null);
assert.equal(checkpoint.velocities[0], 0.012);
const receipt = { positions: checkpoint.positions.slice(), velocities: checkpoint.velocities.slice(), minJ: 1, energyJ: 0, residualN: 0 };
receipt.velocities[0] += 0.001;
assert.throws(() => bridge.recovered(checkpoint, receipt));
receipt.velocities.set(checkpoint.velocities);
const recovered = bridge.recovered(checkpoint, receipt);
assert.equal(recovered.epoch, 2); assert.equal(recovered.sequence, 0);
assert.equal(recovered.simulationSeconds, 1 / 60);
assert.equal(bridge.accept(result()).sequence, 1);
bridge.dispose(); assert.throws(() => bridge.accept(result()));
console.log('Synchronous bridge ownership, complete state gates, picking, actual velocity, recovery version/time and disposal passed');
