import assert from "node:assert/strict";
import { GPUBridgeState } from "../src/gpu_bridge_state.js";

const gate = { gpuValidityAccepted: true, forceEnergyPositionAccepted: true };
const make = (options = {}) => new GPUBridgeState({ nodes: 2, surfaceNodes: 1, ...options });
const accept = (state, seconds = state.version.simulationSeconds + 0.01) =>
  state.accept({ epoch: state.version.epoch, sequence: state.version.sequence + 1, simulationSeconds: seconds }, gate);
const positions = new Float32Array([1, 2, 3, 4, 5, 6]);
const velocities = new Float64Array([.1, .2, .3, .4, .5, .6]);
const state = make();
assert.equal(state.request("checkpoint"), null); // No unvalidated initial baseline.
const next = { epoch: 1, sequence: 1, simulationSeconds: .01 };
assert.equal(state.accept(next, { gpuValidityAccepted: true }), false);
assert.equal(state.accept(next, { forceEnergyPositionAccepted: true }), false);
assert.equal(state.accept({ ...next, simulationSeconds: NaN }, gate), false);
assert.equal(state.accept({ ...next, sequence: 2 }, gate), false);
assert.equal(state.accept(next, gate), true);
assert.equal(state.accept(next, gate), false);
assert.equal(state.accept({ epoch: 0, sequence: 2, simulationSeconds: .02 }, gate), false);
assert.throws(() => { state.version.sequence = 50; });
assert.equal(state.accept({ epoch: 1, sequence: 2, simulationSeconds: 0 }, gate), false);

// Encoding version, identity and queue ownership survive asynchronous completion.
const old = state.request("checkpoint");
assert.equal(state.request("picking", { ...state.version, sequence: 0 }), null);
assert.equal(accept(state), true);
const newer = state.request("checkpoint");
assert.equal(state.request("picking"), null);
assert.equal(state.complete({ ...newer }, { positions, velocities }), false);
assert.equal(state.summary.pending, 2);
const foreign = make();
accept(foreign);
assert.equal(state.complete(foreign.request("checkpoint"), { positions, velocities }), false);
assert.equal(state.complete(newer, { positions, velocities }), true);
assert.equal(state.complete(old, { positions, velocities }), false);
assert.equal(state.complete(newer, { positions, velocities }), false);
assert.equal(state.checkpoint().version.sequence, 2);
const savedFirst = state.checkpoint().positions[0];
positions[0] = 99;
assert.equal(state.checkpoint().positions[0], savedFirst);
const exposed = state.checkpoint();
exposed.positions[0] = -50;
exposed.velocities[0] = -50;
assert.equal(state.checkpoint().positions[0], savedFirst);
assert.equal(state.checkpoint().velocities[0], .1);

// Invalid shape/nonfinite data consumes its token without replacing the checkpoint.
for (const bad of [new Float32Array(3), new Float64Array([NaN, 0, 0, 0, 0, 0])]) {
  accept(state);
  assert.throws(() => state.complete(state.request("checkpoint"), { positions: bad, velocities }));
  assert.equal(state.summary.pending, 0);
  assert.equal(state.checkpoint().version.sequence, 2);
}
assert.equal(state.cancel(state.request("checkpoint")), true);
assert.equal(state.summary.pending, 0);
accept(state);
assert.throws(() => state.complete(state.request("checkpoint"), {
  positions, velocities: new Float64Array([0, 0, Infinity, 0, 0, 0])
}));
assert.equal(state.checkpoint().version.sequence, 2);

// Picking rejects late proxies and requires explicit maximum lag policy.
const pick = state.request("picking");
accept(state);
assert.equal(state.complete(pick, { positions: positions.subarray(0, 3) }), false);
assert.equal(state.picking(), null);
assert.equal(state.complete(state.request("picking"), { positions: positions.subarray(0, 3) }), true);
const captured = state.picking();
assert.equal(captured.version.sequence, state.version.sequence);
accept(state);
assert.equal(state.picking(), null);
assert.equal(captured.positions[0], 99); // The pointerdown snapshot remains owned by caller.
const tolerant = make({ maxPickingLag: 1 });
accept(tolerant);
const tolerantPick = tolerant.request("picking");
accept(tolerant);
assert.equal(tolerant.complete(tolerantPick, { positions: positions.subarray(0, 3) }), true);
assert.ok(tolerant.picking());
accept(tolerant);
assert.equal(tolerant.picking(), null);

// Loss during mapping retains only the previously completed accepted checkpoint.
const inFlight = state.request("checkpoint");
const recovery = state.checkpoint();
assert.equal(state.deviceLost(), true);
assert.equal(state.deviceLost(), false);
assert.equal(state.summary.pending, 0);
assert.equal(state.complete(inFlight, { positions, velocities }), false);
assert.equal(state.request("checkpoint"), null);
assert.equal(accept(state), false);
assert.equal(state.picking(), null);
assert.equal(state.finishRecovery({ restoredCheckpointVersion: recovery.version }), false);
assert.equal(state.finishRecovery({ restoredCheckpointVersion: { ...recovery.version, sequence: 99 }, forceEnergyPositionAccepted: true }), false);
assert.equal(state.finishRecovery({ restoredCheckpointVersion: recovery.version, forceEnergyPositionAccepted: true }), true);
assert.deepEqual(state.version, { epoch: 2, sequence: 0, simulationSeconds: .02 });
assert.equal(state.checkpoint().version.epoch, 2);
assert.equal(state.complete(inFlight, { positions, velocities }), false);
assert.equal(accept(state), true);
assert.equal(state.complete(state.request("picking"), { positions: positions.subarray(0, 3) }), true);
assert.equal(state.summary.ownedSnapshotBytes, state.summary.maxOwnedSnapshotBytes);
assert.equal(state.summary.maxOwnedSnapshotBytes, 120);

const disposeFlight = state.request("checkpoint");
state.dispose();
state.dispose();
assert.equal(state.complete(disposeFlight, { positions, velocities }), false);
assert.equal(state.checkpoint(), null);
assert.equal(state.picking(), null);
assert.equal(state.summary.ownedSnapshotBytes, 0);
assert.equal(state.request("checkpoint"), null);
assert.equal(state.finishRecovery({ restoredCheckpointVersion: recovery.version, forceEnergyPositionAccepted: true }), false);
const noCheckpoint = make();
noCheckpoint.deviceLost();
assert.equal(noCheckpoint.finishRecovery({ forceEnergyPositionAccepted: true }), false);
for (const options of [{ nodes: 0 }, { surfaceNodes: 3 }, { maxPending: 0 }, { maxPickingLag: -1 }, { maxSnapshotBytes: 119 }])
  assert.throws(() => make(options));
assert.throws(() => make().request("unknown"));
console.log("GPU bridge CPU contracts passed: numerical attestations, versions, bounded queue, stale results, copy ownership, picking, loss/recovery, disposal");
