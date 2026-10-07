/** Two outstanding solves, at most one ahead of the next host-accepted state.
 * The backend queue stays serial. Mapping may finish in any order; validation,
 * acceptance, display and checkpoint callbacks always run in submission order.
 * Cancellation poisons speculative GPU state until the caller restores/reset it.
 */
export class OrderedGPUPipeline {
  constructor({ backend, validate, onAccepted, onFailure = () => {} }) {
    if (!backend || typeof validate !== 'function' ||
        (onAccepted != null && typeof onAccepted !== 'function'))
      throw Error('Ordered pipeline requires backend and full validation/acceptance callbacks');
    this.backend = backend;
    this.validate = validate;
    this.onAccepted = onAccepted;
    this.onFailure = onFailure;
    this.queue = [];
    this.generation = 0;
    this.sequence = 0;
    this.stopped = false;
    this.submitChain = Promise.resolve();
    this.peakOutstanding = 0;
  }
  enqueue(parameters, options, metadata = {}) {
    if (this.stopped) return Promise.reject(Error('Ordered pipeline stopped'));
    if (this.queue.length >= 2) return Promise.reject(Error('Ordered pipeline backpressure'));
    // Snapshot mutable UI/grab arrays before an async submission boundary.
    const input = structuredClone(parameters), settings = structuredClone(options);
    const entry = { generation: this.generation, sequence: ++this.sequence,
      metadata: Object.freeze({ ...metadata }), ticket: null };
    entry.promise = new Promise((resolve, reject) => { entry.resolve = resolve; entry.reject = reject; });
    // Prevent cancellation of the speculative tail causing an unhandled rejection
    // before the viewer awaits it. The returned promise still rejects normally.
    entry.promise.catch(() => {});
    this.queue.push(entry);
    this.peakOutstanding = Math.max(this.peakOutstanding, this.queue.length);
    entry.ready = this.submitChain.then(async () => {
      if (entry.generation !== this.generation || this.stopped) throw Error('Stale queued GPU input');
      const ticket = await this.backend.submitStep(input, settings, entry.metadata);
      entry.ticket = ticket;
      if (entry.generation !== this.generation || this.stopped) {
        ticket.cancel();
        throw Error('Stale GPU submission');
      }
      return ticket;
    });
    this.submitChain = entry.ready.then(() => {}, () => {});
    entry.ready.catch(error => this.fail(error, entry.generation));
    if (this.onAccepted) this.drain();
    return entry.promise;
  }
  /** Manual viewer integration: hand off the oldest result, then explicitly
   * acknowledge only after full existing application/display/checkpoint work.
   * A ready mapped tail still occupies its outstanding submission slot.
   */
  async takeNext() {
    if (this.onAccepted) throw Error('Manual handoff unavailable in callback mode');
    if (this.stopped || !this.queue.length || this.lease) throw Error('No ordered GPU result available');
    const entry = this.queue[0], generation = this.generation;
    // Reserve the handoff before awaiting a map so two consumers cannot take it.
    const lease = { entry, generation, ready: false };
    this.lease = lease;
    try {
      const ticket = await entry.ready, result = await ticket.completion;
      if (this.stopped || generation !== this.generation) throw Error('Stale ordered GPU result');
      if (await this.validate(result, ticket, entry.metadata) !== true)
        throw Error('Ordered GPU result validation rejected');
      if (this.stopped || generation !== this.generation) throw Error('Stale ordered GPU validation');
      Object.assign(lease, { ticket, result, metadata: entry.metadata, ready: true });
      return lease;
    } catch (error) {
      this.fail(error, generation);
      throw error;
    }
  }
  acknowledge(lease) {
    if (this.stopped || lease !== this.lease || !lease.ready ||
        lease.generation !== this.generation || this.queue[0] !== lease.entry)
      throw Error('Ordered GPU acknowledgement rejected');
    try {
      if (lease.result.stats && Number.isFinite(lease.ticket.inputAt))
        lease.result.stats.inputAgeAtAcceptanceMs = performance.now() - lease.ticket.inputAt;
      this.backend.acceptSubmittedStep(lease.ticket, lease.result);
      this.queue.shift();
      this.lease = null;
      lease.entry.resolve(lease.result);
    } catch (error) {
      this.fail(error, lease.generation);
      throw error;
    }
  }
  async drain() {
    if (this.draining || this.stopped) return;
    this.draining = true;
    const generation = this.generation;
    try {
      while (this.queue.length && !this.stopped && generation === this.generation) {
        const entry = this.queue[0], ticket = await entry.ready;
        const result = await ticket.completion;
        if (this.stopped || generation !== this.generation) return;
        // Existing complete force/energy/position/J checks belong here; merely
        // receiving a GPU buffer never attests those checks.
        if (await this.validate(result, ticket, entry.metadata) !== true)
          throw Error('Ordered GPU result validation rejected');
        if (this.stopped || generation !== this.generation) return;
        if (result.stats && Number.isFinite(ticket.inputAt))
          result.stats.inputAgeAtAcceptanceMs = performance.now() - ticket.inputAt;
        this.backend.acceptSubmittedStep(ticket, result);
        await this.onAccepted(result, ticket, entry.metadata);
        if (this.stopped || generation !== this.generation) return;
        this.queue.shift();
        entry.resolve(result);
      }
    } catch (error) {
      this.fail(error, generation);
    } finally {
      this.draining = false;
    }
  }
  fail(error, generation) {
    if (this.stopped || generation !== this.generation) return;
    this.stop(error);
    // Caller owns validated-checkpoint restore and rollback of its simulation
    // clock. Never restart on the possibly poisoned resident buffers.
    try { const report = this.onFailure(error); report?.catch?.(() => {}); } catch {}
  }
  stop(reason = Error('Ordered GPU pipeline cancelled')) {
    if (this.stopped) return;
    this.stopped = true;
    this.generation++;
    this.backend.cancelSubmittedSteps();
    for (const entry of this.queue) {
      entry.ticket?.cancel();
      entry.reject(reason);
    }
    this.queue.length = 0;
  }
  dispose() { this.stop(Error('Ordered GPU pipeline disposed')); }
  get summary() {
    return { outstanding: this.queue.length, peakOutstanding: this.peakOutstanding,
      stopped: this.stopped, peakSubmissionSlotBytes: this.backend.peakSubmissionSlotBytes ?? null };
  }
}
