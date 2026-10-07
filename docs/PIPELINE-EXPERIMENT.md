# Ordered two-slot bridge experiment

`?bridge=pipeline` opts into two reusable GPU submission banks. Production remains
on the existing path. Each bank owns its uniform, complete checkpoint readback and
timestamp storage. GPU dispatch order is serial; mapping may overlap. A mapped
tail retains its bank until ordered host acceptance. No force, energy, position,
finite-state or positive-J gate has been removed. Full readback remains.

The viewer validates the oldest result, updates its checkpoint/picking version,
applies the original dense rendering and fine-detail path, submits rendering, and
then acknowledges the bank. It only queues a second dynamic solve when already
behind wall time. Inputs are copied when queued, so a new input can wait behind one
older step. Applied-state labels use that accepted input. Pausing drains the queued
dynamic tail; measured wall time includes that drain. Reset, cancellation, mapping
failure and device loss poison speculative GPU state until reset or restore from
the last fully validated CPU checkpoint. Late results cannot advance a new epoch.

CPU contracts cover dispatch equivalence to the original step, separate uniforms,
full readback, two-bank reuse, out-of-order mapping, acceptance backpressure,
validation failure, map/copy failures and cancellation across reset generations.
They are not driver or browser validation. An isolated coordinated GPU review
subsequently passed 30 matched full-state steps exactly and actual loss/restore.
One bounded ABBA workload measured about 1.67x aggregate simulation advancement;
both paths still lagged wall time and long stalls persisted. Combined-default
browser acceptance remains pending; see COMBINED-RELEASE.md for exact limits.

## Requested bounded GPU review

Reserve one exclusive slot of at most eight minutes. Serve this isolated checkout
on port 8781 and run `node tests/pipeline-browser.mjs`. Its fresh Edge browser has
a 420-second close watchdog. It first compares 30 complete packed/velocity states
at matched step indices against the synchronous solver, then discards two real
outstanding solves on injected device loss and validates actual CPU checkpoint
restore. It runs fresh viewer contexts serially in ABBA order, fixed reference
quality, 1280 by 900 at DPR 1, two-second warmup and twelve-second windows.

Report simulation-time advancement versus wall time (including tail drain),
update count, map/GPU/work p50/p95/max, cold ready time, input version age through
render submission, exact peak owned solver-buffer bytes and sampled JS heap.
The one-second polled input pulse is a coarse upper bound; fine pipeline input
latencies end at accepted render submission. Neither is photon latency. Rendering
submission metrics are not physical displayed frames per second. Solver buffer
counts exclude renderer, native allocations and driver memory; JS heap excludes
native/typed-array backing stores. Total application peak memory remains unknown.

Harness source is prepared, not GPU-verified. Stop on timeout or numerical error;
close the owned browser and HTTP server and immediately release the slot. Do not
encode media or edit documentation while holding the slot. No runtime selection,
merge or Site publication follows automatically. A speedup claim requires matched
numerical acceptance and repeated paired throughput/latency evidence.
