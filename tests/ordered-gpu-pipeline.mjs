import assert from 'node:assert/strict';
import { OrderedGPUPipeline } from '../src/ordered_gpu_pipeline.js';
const tick = () => new Promise(resolve => setImmediate(resolve));
function harness() {
  const tickets = [], accepted = [], presented = [], failures = [];
  const backend = {
    peakSubmissionSlotBytes: 128,
    async submitStep(input, options, metadata) {
      let resolve, reject;
      const ticket = { input, options, metadata, cleanup: 0 };
      ticket.release = () => { if (!ticket.released) { ticket.released = true; ticket.cleanup++; } };
      ticket.cancel = () => { ticket.cancelled = true; ticket.release(); reject(Error('map cancelled')); };
      ticket.completion = new Promise((a, b) => { resolve = a; reject = b; }).finally(ticket.release);
      ticket.completion.catch(() => {});
      ticket.resolve = resolve; ticket.reject = reject;
      tickets.push(ticket); return ticket;
    },
    acceptSubmittedStep(ticket, result) { assert(!ticket.cancelled); accepted.push(result.id); },
    cancelSubmittedSteps() { for (const ticket of tickets) ticket.cancel(); },
  };
  const pipeline = new OrderedGPUPipeline({ backend,
    validate: result => result.valid === true,
    onAccepted: result => presented.push(result.id),
    onFailure: error => failures.push(error.message) });
  return { tickets, accepted, presented, failures, pipeline };
}
// Out-of-order maps still accept, display and checkpoint in submission order.
{
  const h = harness(), input = { grab: { target: [1, 2, 3] } };
  const first = h.pipeline.enqueue(input, { dt: 1 }, { inputVersion: 1 });
  input.grab.target[0] = 99;
  const second = h.pipeline.enqueue({ value: 2 }, {}, { inputVersion: 2 });
  await assert.rejects(h.pipeline.enqueue({}, {}), /backpressure/);
  await tick(); assert.equal(h.tickets.length, 2);
  assert.equal(h.tickets[0].input.grab.target[0], 1);
  h.tickets[1].resolve({ id: 2, valid: true }); await tick();
  assert.deepEqual(h.accepted, []);
  h.tickets[0].resolve({ id: 1, valid: true });
  await Promise.all([first, second]);
  assert.deepEqual(h.accepted, [1, 2]); assert.deepEqual(h.presented, [1, 2]);
  assert.equal(h.pipeline.summary.peakOutstanding, 2);
  h.pipeline.dispose(); h.pipeline.dispose();
  assert.deepEqual(h.tickets.map(t => t.cleanup), [1, 1]);
}
// Invalid first result discards the completed speculative second result.
{
  const h = harness(), first = h.pipeline.enqueue({}, {}), second = h.pipeline.enqueue({}, {});
  await tick(); h.tickets[1].resolve({ id: 2, valid: true });
  h.tickets[0].resolve({ id: 1, valid: false });
  await assert.rejects(first, /validation/); await assert.rejects(second, /validation/);
  assert.deepEqual(h.accepted, []); assert.equal(h.failures.length, 1);
  assert.deepEqual(h.tickets.map(t => t.cleanup), [1, 1]);
}
// Device/map failure stops continuation and calls recovery once.
{
  const h = harness(), first = h.pipeline.enqueue({}, {}), second = h.pipeline.enqueue({}, {});
  await tick(); h.tickets[0].reject(Error('device lost'));
  await assert.rejects(first, /device lost/); await assert.rejects(second, /device lost/);
  await assert.rejects(h.pipeline.enqueue({}, {}), /stopped/);
  assert.deepEqual(h.presented, []); assert.equal(h.failures.length, 1);
  assert.deepEqual(h.tickets.map(t => t.cleanup), [1, 1]);
}
// Reset/disposal while maps or submission boundaries are pending never presents.
for (const beforeSubmit of [false, true]) {
  const h = harness(), first = h.pipeline.enqueue({}, {}), second = h.pipeline.enqueue({}, {});
  if (!beforeSubmit) await tick();
  h.pipeline.stop(Error('reset')); h.pipeline.dispose();
  await assert.rejects(first, /reset/); await assert.rejects(second, /reset/); await tick();
  assert.deepEqual(h.presented, []); assert.deepEqual(h.failures, []);
  assert(h.tickets.every(t => t.cleanup === 1));
}
// Validation has an async cancellation boundary as well as map completion.
{
  const h = harness(); let releaseValidation;
  h.pipeline.validate = () => new Promise(resolve => { releaseValidation = resolve; });
  const first = h.pipeline.enqueue({}, {}); await tick();
  h.tickets[0].resolve({ id: 1, valid: true }); await tick();
  h.pipeline.dispose(); releaseValidation(true);
  await assert.rejects(first, /disposed/); await tick(); assert.deepEqual(h.accepted, []);
}
// Manual handoff holds both slots until viewer display/checkpoint acknowledgment.
{
  const h = harness(); h.pipeline.onAccepted = null;
  const first = h.pipeline.enqueue({}, {}), second = h.pipeline.enqueue({}, {});
  await tick(); h.tickets[1].resolve({ id: 2, valid: true });
  h.tickets[0].resolve({ id: 1, valid: true });
  const lease = await h.pipeline.takeNext();
  assert.deepEqual(h.accepted, []); assert.equal(h.pipeline.summary.outstanding, 2);
  await assert.rejects(h.pipeline.takeNext(), /available/);
  await assert.rejects(h.pipeline.enqueue({}, {}), /backpressure/);
  // The viewer has now completed validation/application/display/checkpoint work.
  h.pipeline.acknowledge(lease); await first;
  assert.throws(() => h.pipeline.acknowledge(lease), /rejected/);
  assert.deepEqual(h.accepted, [1]);
  const tail = await h.pipeline.takeNext(); h.pipeline.acknowledge(tail); await second;
  assert.deepEqual(h.accepted, [1, 2]); h.pipeline.dispose();
}
// Cancellation after handoff rejects acknowledgment and discards speculative tail.
{
  const h = harness(); h.pipeline.onAccepted = null;
  const first = h.pipeline.enqueue({}, {}), second = h.pipeline.enqueue({}, {});
  await tick(); h.tickets[0].resolve({ id: 1, valid: true });
  const lease = await h.pipeline.takeNext(); h.pipeline.stop(Error('control changed'));
  assert.throws(() => h.pipeline.acknowledge(lease), /rejected/);
  await assert.rejects(first, /control changed/); await assert.rejects(second, /control changed/);
  assert.deepEqual(h.accepted, []);
}
console.log('Ordered GPU pipeline: delayed/out-of-order maps, input ownership, invalid result, loss, reset, disposal and backpressure passed');
