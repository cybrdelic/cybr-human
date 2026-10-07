/** Same volume and hinge equations in a worker when WebGPU cannot boot. */
export class CPUHeadFEM {
  static async create(model, arrays, reason) {
    const e = new CPUHeadFEM(model, arrays, reason);
    await e.call({ type: "init", model, arrays });
    return e;
  }
  constructor(model, arrays, reason) {
    this.model = model;
    this.arrays = arrays;
    this.frames = 0;
    this.errors = [];
    this.S = model.surface_nodes;
    this.timestamp = false;
    this.adapterInfo = {
      vendor: "CPU worker",
      architecture: "full FEM",
      fallbackReason: reason,
    };
    this.worker = new Worker(
      new URL("./tissue_cpu_worker.js", import.meta.url),
      { type: "module" },
    );
    this.pending = new Map();
    this.next = 0;
    this.worker.onmessage = ({ data }) => {
      const p = this.pending.get(data.id);
      if (!p) return;
      this.pending.delete(data.id);
      clearTimeout(p.timer);
      data.error ? p.reject(Error(data.error)) : p.resolve(data);
    };
    this.worker.onerror = (e) =>
      this.fail(Error(e.message || "Tissue worker failed"));
  }
  fail(error) {
    if (this.failure) return;
    this.failure = error;
    this.errors.push(error.message);
    this.worker.terminate();
    for (const p of this.pending.values()) {
      clearTimeout(p.timer);
      p.reject(error);
    }
    this.pending.clear();
  }
  call(data) {
    if (this.failure) return Promise.reject(this.failure);
    return new Promise((resolve, reject) => {
      const id = ++this.next,
        timer = setTimeout(
          () =>
            this.fail(Error("Tissue worker did not respond within 60 seconds")),
          60000,
        );
      this.pending.set(id, { resolve, reject, timer });
      try {
        this.worker.postMessage({ ...data, id });
      } catch (error) {
        this.fail(error);
      }
    });
  }
  async step(parameters, options) {
    const r = await this.call({ type: "step", parameters, options });
    this.frames++;
    this.latestPacked = r.packed;
    return r;
  }
  reset() {
    return this.call({ type: "reset" });
  }
  restore(positions, velocity, verifyCheckpoint = false) {
    return this.call({ type: "restore", positions, velocity, verifyCheckpoint });
  }
  dispose() {
    this.fail(Error("Tissue worker disposed"));
  }
}
