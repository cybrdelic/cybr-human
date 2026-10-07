import assert from "node:assert/strict";
import { createLazyAnatomy } from "../src/lazy_anatomy.js";
import { createFrameMetrics } from "../src/frame_metrics.js";

let resolve, loads = 0, attached = 0, disposed = false, discarded = 0, visible;
let failure = false;
const anatomy = createLazyAnatomy({
  load: () => { loads++; if (failure) throw Error("checksum failure"); return new Promise(r => { resolve = r; }); },
  attach: () => attached++, show: (_, show) => { visible = show; },
  discard: () => discarded++, isDisposed: () => disposed,
});
assert.equal(loads, 0, "Optional anatomy must not load during startup");
const a = anatomy.setVisible(true), b = anatomy.setVisible(true);
await Promise.resolve(); assert.equal(loads, 1);
await anatomy.setVisible(false); resolve({ scene: "verified" }); await Promise.all([a, b]);
assert.equal(attached, 1); assert.equal(visible, false, "Unchecking during fetch keeps skin visible");
await anatomy.setVisible(true); assert.equal(loads, 1); assert.equal(visible, true);

const failed = createLazyAnatomy({
  load: () => { if (failure) throw Error("checksum failure"); return Promise.resolve({}); },
  attach: () => attached++, show: (_, show) => { visible = show; },
  discard: () => discarded++, isDisposed: () => disposed,
});
failure = true; await failed.setVisible(true);
assert.equal(failed.state.phase, "failed"); assert.equal(visible, false);
failure = false; await failed.setVisible(true);
assert.equal(failed.state.phase, "ready"); assert.equal(failed.state.attempts, 2);

const late = createLazyAnatomy({ load: () => new Promise(r => { resolve = r; }),
  attach: () => { throw Error("must not attach after disposal"); }, show: () => {},
  discard: () => discarded++, isDisposed: () => disposed });
const pending = late.setVisible(true); await Promise.resolve(); disposed = true;
resolve({}); await pending; assert.equal(discarded, 1);

const metrics = createFrameMetrics(() => 0, 3);
metrics.submitted(0, 2); metrics.submitted(16, 19); metrics.submitted(216, 220);
let report = metrics.snapshot({ activeWallSeconds: 10, simulationSeconds: 6 });
assert.equal(report.submissionGapP95Ms, 200); assert.equal(report.worstRenderCpuMs, 4);
assert.equal(report.simulationToActiveWallRatio, .6);
metrics.suspend(); metrics.submitted(10000, 10001); metrics.submitted(10016, 10018);
report = metrics.snapshot(); assert.equal(report.windowSamples, 3);
assert.equal(report.worstSubmissionGapMs, 200, "Hidden interval must not inflate frame stalls");
assert.equal(report.simulationToActiveWallRatio, null);
console.log("Progressive startup races, retry, disposal and bounded timing checks passed");
