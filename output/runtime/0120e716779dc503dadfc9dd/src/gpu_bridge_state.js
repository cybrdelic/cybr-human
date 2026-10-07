/** CPU coordination contract for a future resident GPU bridge.
 * No solver validation, GPU submission, buffer lifetime or rendering occurs here.
 * The integrator must attest the complete numerical gate before accept().
 */
export class GPUBridgeState {
  #epoch = 1;
  #sequence = 0;
  #seconds = 0;
  #status = "active";
  #validated = false;
  #checkpoint = null;
  #picking = null;
  #pending = new Map();
  #nodes;
  #surfaceNodes;
  #maxPending;
  #maxPickingLag;

  constructor({ nodes, surfaceNodes, maxPending = 2, maxPickingLag = 0, maxSnapshotBytes = 8 * 1024 * 1024 }) {
    for (const [label, value] of Object.entries({ nodes, surfaceNodes, maxPending }))
      if (!Number.isSafeInteger(value) || value < 1) throw Error(`Invalid ${label}`);
    if (surfaceNodes > nodes || !Number.isSafeInteger(maxPickingLag) || maxPickingLag < 0)
      throw Error("Invalid surface count or picking lag");
    const bytes = nodes * 3 * 8 * 2 + surfaceNodes * 3 * 8;
    if (!Number.isSafeInteger(maxSnapshotBytes) || maxSnapshotBytes < 1 ||
        !Number.isSafeInteger(bytes) || bytes > maxSnapshotBytes) throw Error("Snapshot memory budget exceeded");
    this.#nodes = nodes;
    this.#surfaceNodes = surfaceNodes;
    this.#maxPending = maxPending;
    this.#maxPickingLag = maxPickingLag;
  }

  get version() {
    return Object.freeze({ epoch: this.#epoch, sequence: this.#sequence, simulationSeconds: this.#seconds });
  }

  get summary() {
    return Object.freeze({ status: this.#status, version: this.version,
      checkpointVersion: this.#checkpoint?.version ?? null,
      pickingVersion: this.#picking?.version ?? null, pending: this.#pending.size,
      // Owned snapshots use Float64 positions/velocity. Returned copies and GPU
      // staging allocation are caller-owned and must be budgeted separately.
      ownedSnapshotBytes: (this.#checkpoint ? this.#nodes * 3 * 8 * 2 : 0) +
        (this.#picking ? this.#surfaceNodes * 3 * 8 : 0),
      maxOwnedSnapshotBytes: this.#nodes * 3 * 8 * 2 + this.#surfaceNodes * 3 * 8 });
  }

  #matches(version) {
    return version?.epoch === this.#epoch && version?.sequence === this.#sequence &&
      version?.simulationSeconds === this.#seconds;
  }

  /** Serial monotonic acceptance, after complete force/energy/position checks.
   * Positive J/residual alone is insufficient. A rejected gate does not advance.
   */
  accept(version, { gpuValidityAccepted, forceEnergyPositionAccepted } = {}) {
    if (this.#status !== "active" || version?.epoch !== this.#epoch ||
        version?.sequence !== this.#sequence + 1 || !Number.isSafeInteger(version.sequence) ||
        !Number.isFinite(version.simulationSeconds) || version.simulationSeconds < this.#seconds ||
        gpuValidityAccepted !== true || forceEnergyPositionAccepted !== true) return false;
    this.#sequence = version.sequence;
    this.#seconds = version.simulationSeconds;
    this.#validated = true;
    return true;
  }

  /** Tag a copy encoded from the CURRENT accepted buffer version, never from
   * whichever mutable solver buffer happens to exist when mapping completes.
   * null means backpressure; the caller must not allocate/submit a staging copy.
   */
  request(kind, version = this.version) {
    if (!["checkpoint", "picking"].includes(kind)) throw Error("Unknown snapshot kind");
    if (this.#status !== "active" || !this.#validated || !this.#matches(version) || this.#pending.size >= this.#maxPending)
      return null;
    const token = Object.freeze({ kind, version: this.version });
    this.#pending.set(token, token);
    return token;
  }

  cancel(token) { return this.#pending.delete(token); }

  #copyVector(value, nodes) {
    if (!(value instanceof Float32Array || value instanceof Float64Array) || value.length !== nodes * 3)
      throw Error("Snapshot must contain exactly three components per node");
    const copy = new Float64Array(value.length);
    for (let i = 0; i < value.length; i++) {
      if (!Number.isFinite(value[i])) throw Error("Nonfinite snapshot component");
      copy[i] = value[i];
    }
    return copy;
  }

  /** Consumes a one-shot token even on failure. Never allows an older map
   * completion to overwrite a newer checkpoint or picking proxy.
   */
  complete(token, { positions, velocities } = {}) {
    if (!this.#pending.delete(token) || this.#status !== "active" || token.version.epoch !== this.#epoch)
      return false;
    const previous = token.kind === "checkpoint" ? this.#checkpoint : this.#picking;
    if (previous && token.version.sequence <= previous.version.sequence) return false;
    if (token.kind === "picking" && this.#sequence - token.version.sequence > this.#maxPickingLag)
      return false;
    const snapshot = { version: token.version,
      positions: this.#copyVector(positions, token.kind === "checkpoint" ? this.#nodes : this.#surfaceNodes) };
    if (token.kind === "checkpoint") {
      snapshot.velocities = this.#copyVector(velocities, this.#nodes);
      this.#checkpoint = snapshot;
    } else this.#picking = snapshot;
    return true;
  }

  #clone(snapshot) {
    if (!snapshot) return null;
    return { version: snapshot.version, positions: snapshot.positions.slice(),
      ...(snapshot.velocities ? { velocities: snapshot.velocities.slice() } : {}) };
  }

  checkpoint() { return this.#clone(this.#checkpoint); }

  /** Capture once per pointerdown, raycast this proxy, and retain its version
   * through target initialization. Default policy blocks picking if it is stale.
   */
  picking() {
    if (this.#status !== "active" || !this.#picking ||
        this.#sequence - this.#picking.version.sequence > this.#maxPickingLag) return null;
    return this.#clone(this.#picking);
  }

  /** Invalidate every pending completion immediately; retained CPU checkpoint
   * remains the sole recovery source. GPU resource destruction is caller-owned.
   */
  deviceLost() {
    if (this.#status !== "active") return false;
    this.#status = "lost";
    this.#epoch++;
    this.#pending.clear();
    this.#picking = null;
    return true;
  }

  /** Called only AFTER backend restore succeeds and its numerical gate passes.
   * New epoch sequence zero identifies the restored baseline; time rolls back
   * to the exact checkpoint rather than silently continuing the lost GPU clock.
   */
  finishRecovery({ restoredCheckpointVersion, forceEnergyPositionAccepted } = {}) {
    const saved = this.#checkpoint?.version;
    if (this.#status !== "lost" || !saved ||
        restoredCheckpointVersion?.epoch !== saved.epoch ||
        restoredCheckpointVersion?.sequence !== saved.sequence ||
        restoredCheckpointVersion?.simulationSeconds !== saved.simulationSeconds ||
        forceEnergyPositionAccepted !== true) return false;
    this.#sequence = 0;
    this.#seconds = saved.simulationSeconds;
    this.#status = "active";
    this.#validated = true;
    this.#checkpoint.version = this.version;
    return true;
  }

  dispose() {
    if (this.#status === "disposed") return;
    this.#status = "disposed";
    this.#epoch++;
    this.#pending.clear();
    this.#checkpoint = this.#picking = null;
  }
}
