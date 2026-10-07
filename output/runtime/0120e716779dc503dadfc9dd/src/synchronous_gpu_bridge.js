import { GPUBridgeState } from './gpu_bridge_state.js';

/** First production bridge stage: full synchronous readback, original numerical
 * solver gates and renderer remain. Owns recovery/picking versions, not GPU device.
 */
export class SynchronousGPUBridge {
  constructor(model) {
    this.nodes = model.nodes;
    this.surfaceNodes = model.surface_nodes;
    if (this.nodes > this.surfaceNodes * 6) throw Error('Model exceeds packed-state position capacity');
    this.state = new GPUBridgeState({ nodes: this.nodes, surfaceNodes: this.surfaceNodes });
  }
  validate(result) {
    const { packed, velocities, stats } = result;
    if (!(packed instanceof Float32Array) || packed.length !== this.surfaceNodes * 24 ||
        !(velocities instanceof Float32Array || velocities instanceof Float64Array) || velocities.length !== this.nodes * 3 ||
        !packed.every(Number.isFinite) || !velocities.every(Number.isFinite) ||
        !Number.isFinite(stats?.minJ) || stats.minJ <= 0 ||
        !Number.isFinite(stats.residualN) || !Number.isFinite(stats.staticResidualN) ||
        !Number.isFinite(stats.maxNodeSpeedMps) || !Number.isFinite(stats.kineticEnergyJ) ||
        !Number.isFinite(stats.simulatedSeconds) || stats.simulatedSeconds < 0)
      throw Error('Invalid full synchronous bridge state');
    return true;
  }
  accept(result) {
    this.validate(result);
    const { packed, velocities, stats } = result;
    const positions = new Float64Array(this.nodes * 3);
    for (let n = 0; n < this.nodes; n++) for (let d = 0; d < 3; d++) positions[n * 3 + d] = packed[n * 4 + d];
    const old = this.state.version;
    const version = { epoch: old.epoch, sequence: old.sequence + 1,
      simulationSeconds: old.simulationSeconds + stats.simulatedSeconds };
    // Only invoked after the existing solver's guarded Newton/validation/readback
    // and errors checks return. These attestations do not claim new biological gates.
    if (!this.state.accept(version, { gpuValidityAccepted: true, forceEnergyPositionAccepted: true }))
      throw Error('Bridge cannot accept this solver version');
    const checkpoint = this.state.request('checkpoint'), picking = this.state.request('picking');
    if (!checkpoint || !picking || !this.state.complete(checkpoint, { positions, velocities }) ||
        !this.state.complete(picking, { positions: positions.subarray(0, this.surfaceNodes * 3) }))
      throw Error('Synchronous bridge snapshot ownership failed');
    return this.state.version;
  }
  get summary() { return this.state.summary; }
  picking() { return this.state.picking(); }
  lost() { this.state.deviceLost(); return this.state.checkpoint(); }
  /** Restore acknowledgement includes evaluated finite/J state and actual copied
   * full vectors. Confirm backend restore before advancing the recovery epoch.
   */
  recovered(checkpoint, acknowledgement) {
    const a = acknowledgement;
    if (!checkpoint || !a || !Number.isFinite(a.minJ) || a.minJ <= 0.2 ||
        !Number.isFinite(a.energyJ) || !Number.isFinite(a.residualN))
      throw Error('Backend did not validate restored checkpoint');
    for (const key of ['positions', 'velocities']) {
      if (!(a[key] instanceof Float64Array) || a[key].length !== this.nodes * 3)
        throw Error('Incomplete restored checkpoint');
      for (let i = 0; i < a[key].length; i++)
        if (!Number.isFinite(a[key][i]) || Math.abs(a[key][i] - checkpoint[key][i]) > 1e-7)
          throw Error('Restored checkpoint differs from accepted state');
    }
    if (!this.state.finishRecovery({ restoredCheckpointVersion: checkpoint.version, forceEnergyPositionAccepted: true }))
      throw Error('Recovery checkpoint version rejected');
    return this.state.version;
  }
  dispose() { this.state.dispose(); }
}
