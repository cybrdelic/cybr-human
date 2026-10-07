# Resident compute/render bridge: isolated production plan

Status: CPU-verified first synchronous production integration, selectable by
`bridge=versioned`; browser/GPU acceptance remains pending. Default bridge is
unchanged. This does not introduce asynchronous readback or a new renderer.
The selected public runtime remains the baseline. The first experiment remains
`shared_surface_probe.js`, whose Lambert renderer does not establish production
eye, skin diffusion, fine-detail or shadow parity.

## Evidence and constraints

The measured candidate transfer evaluated 307,477 skin vertices on the solver's
device. Across 188 sampled vertices in two states its maximum position error
was 32.365 nm and normal-vector error was 2.316e-7, with 46,869,736 owned GPU
buffer bytes. These are sampled transfer results, not full-state validation.
The current solver still maps full packed state and statistics every step.
The matched dynamic windows lagged wall time and showed long map waits; neither
a production speedup nor the initiating driver/queue cause was established.
See [recorded measurements](UPGRADE-MEASUREMENTS.md).

`GPUHeadFEM.step()` completes evaluate/gradient/validation/pack work before
copying packed state and statistics into one mapped buffer. The viewer's coarse
position upload feeds picking. Its checkpoint velocity is reconstructed from
successive displayed positions with damping, rather than copied directly from
solver velocity. A future bridge must specify and validate this difference;
it cannot silently call a partial surface snapshot a recoverable tissue state.

## First synchronous production integration

`SynchronousGPUBridge` now sits between accepted solver readback and viewer
upload. It keeps the original guarded Newton, force/energy evaluation and GPU
validation passes, then additionally requires complete finite packed state,
actual full-node velocity and finite validation telemetry before replacing its
owned checkpoint and picking proxy. Gate attestations describe those original
solver checks; this module does not calculate a new energy criterion or prove
biological accuracy. Invalid output stops without updating the displayed skin.
The optional viewer performs one checked initial gravitational preload to
establish its first picking/recovery snapshot; this startup work must be counted
in the paired timings. Controls block picking until that snapshot completes.

Optional GPU checkpoint packing appends actual `nodes.velocity.xyz` after the
existing packed layout in the same final command encoder and map. It never
includes the contact scratch `.w`. Positions, recovered gradients and all
statistics still map every step. This adds 147,648 bytes to each output/readback
buffer at the current model size; it is a correctness stage, not a demonstrated
throughput optimization. Default mode keeps the prior buffer sizes and dispatch.
The CPU worker similarly returns its actual final velocity only when requested.

The viewer consumes one owned, current-version surface proxy at pointerdown and
initializes the target from it, excluding fixed nodes. Reset disposes the old
coordinator and clears its time; disposal invalidates all ownership. GPU loss
invalidates picking immediately. CPU fallback restores the full checkpoint and
returns actual copied arrays plus finite energy/residual and positive-J checks.
The bridge compares those arrays (maximum 100 nm component tolerance, including
fixed-node reapplication) before finishing recovery. Time returns to the exact
completed checkpoint; pre-loss drag is released. Failure before the first
validated checkpoint stops with the existing retry/reload flow.

CPU tests: `node tests/gpu-bridge-state.mjs`,
`node tests/synchronous-gpu-bridge.mjs`, `node tests/bridge-worker.mjs`.
The actual 9,228-node worker's two-substep versus two-separate-step final velocity
error was zero; recovered velocity error was zero; fixed-node position
reapplication changed at most 7.448 nm, while all free-node positions matched
exactly. A resumed dynamic step had J=0.87474 and a following static step returned
zero velocity without advancing simulation time. These are bounded numerical
fixtures, not equilibrium or biological validation. Source syntax passes.

`node tests/bridge-browser.mjs` is prepared but unrun. It checks GPU-packed
velocity against actual node storage, evaluates CPU restoration after injected
device loss, then exercises viewer picking/recovery/reset. Run only in an
allocated GPU slot. GPU shader compilation, actual rendering, complete retained
regressions and coordination overhead remain release gates.

## Implemented CPU contract

`src/gpu_bridge_state.js` tracks epoch, accepted sequence and simulation time.
The complete numerical gate must pass before `accept()` advances the version.
The two explicit gate attestations represent existing GPU validity checks and
full force/energy/position acceptance; the module does not calculate or replace
those checks. Unvalidated initial state cannot generate snapshot requests.
After baseline validation, accept sequence one at its actual simulation time
and complete a full checkpoint before permitting asynchronous advancement.

`request()` issues identity-bound, one-shot tokens for the current accepted
state. Default backpressure permits two total pending requests. The integrator
must encode the copy while its source still has that exact version. `complete()`
copies finite full-size vectors into owned Float64 storage, rejects older or
duplicate results, and consumes failed tokens without replacing good snapshots.
Only complete full-node position/velocity snapshots can become checkpoints.
Picking uses a separate surface proxy and defaults to zero accepted-step lag.
This conservative default may pause pointerdown until its proxy is current;
changing the permitted lag requires measured interaction error bounds.

The latest accepted GPU state, last completed recoverable checkpoint and
current picking proxy are distinct owners. Simulation time advances only with
accepted solves; rendering the same version twice must not advance time.
`picking()` returns an owned snapshot for a single pointerdown. Raycast and
initial grab node/target must derive from that same snapshot, not a later proxy.
Existing node IDs, rest-space support normalization and fixed-node exclusions
must remain unchanged. A drag begun on a permitted older proxy must revalidate
its node and target against current accepted state before forces are submitted.

`deviceLost()` invalidates pending tokens and picking immediately. The retained
checkpoint is the sole recovery source. `finishRecovery()` requires exact
restored-checkpoint identity and a passed restore acceptance gate. It creates
a new epoch at sequence zero and explicitly rolls simulation time back to the
checkpoint. No checkpoint means no valid recovery; show the existing failure
and retry flow. Disposal permanently invalidates requests and drops snapshots.

Default owned CPU snapshot ceiling is 8 MiB, checked before allocation. At
9,228 full nodes and 3,076 surface nodes, retained arrays occupy 516,768 bytes.
During replacement, temporary copies can add one complete checkpoint or proxy.
Returned snapshot copies, old pointerdown copies, full GPU staging buffers,
history and solver/render allocations are outside this ceiling and require a
separate total budget. Reject backpressure before allocating staging buffers.
Cancel rejected maps in `finally`, unmap/destroy staging buffers exactly once,
and consume queue errors; invalidating a token does not cancel a GPU command.

Run `node tests/gpu-bridge-state.mjs` for bounded CPU tests. They cover the
state-machine contract, not WebGPU scheduling, backend restoration accuracy,
mechanical validation or shader parity.

## Integration stages and ownership

1. Keep the existing bridge as a selectable reference. First add the coordinator
   around existing synchronous copies without reducing validation. Exercise
   reset, CPU fallback, device loss and drag behavior before asynchronous work.
   Reset/model replacement must dispose the old coordinator and create a fresh
   one, even if the GPU device survives; tokens must never cross model identity.
2. Use a single solver-owned device with explicit borrowed-device lifetime for
   rendering. Present buffer ownership and disposal order need a separate API:
   stop scheduling, invalidate versions, complete/cancel maps, release borrowed
   resources, then destroy the device once. Current solver `dispose()` destroys
   the device, so directly sharing it without refactoring ownership is unsafe.
3. Encode compute validation, transfer and rendering with versioned immutable
   accepted-state slots. At least one accepted presentation buffer must survive
   an invalid trial. The mutable packed solver buffer is not that retained slot.
   GPU-resident gate must inspect complete original acceptance criteria and
   prevent invalid data from replacing the accepted slot. Probe `minJ > .2`
   and finite residual flags alone are not the full production numerical gate.
4. Port production materials, eye/cornea transparency, lighting/shadows,
   tone/color handling, smooth normals, skin diffusion and fine-detail transfer
   to a common-device renderer. WebGL buffers cannot directly alias WebGPU
   buffers. A renderer migration or explicit bridge remains necessary; the
   diagnostic WGSL shader cannot replace the current production renderer.
5. Separate small validation telemetry, surface picking copies and full recovery
   snapshots. First preserve every-step reference acceptance. Reduce full copies
   only after resident acceptance is checked against the reference and a bounded
   checkpoint cadence is justified. Cadence must cap lost simulation time and
   wall-time checkpoint age; if exceeded, stop advancing until a full snapshot
   completes. Staging copies must capture full solver positions and actual
   velocities at one version, including fixed nodes and all mechanical layers.
   Check whether CPU restore needs any further integrator state. No history is
   inferred from intermittent display updates.
6. Keep the CPU reference and synchronous bridge for unsupported devices and
   failed acceptance. Fine forehead worker currently depends on coarse CPU
   state; update it at a versioned cadence and measure wrinkle error or port it
   before claiming resident full-quality parity. Lazy anatomy uploads must use
   the current accepted state; pending loads may not resurrect a disposed scene.

GPU resource leases and immutable slot rings remain unimplemented. This CPU
module cannot prove a copy came from the attested buffer, cancel mapAsync,
detect a dishonest gate attestation or validate biological fidelity.

## Required paired GPU acceptance before release

Compare the preserved runtime and candidate on identical cameras, lights,
materials, solver settings, load inputs and warmup in alternating run order.
Capture actual neutral, passive-load, pull and release frames under studio,
side and overhead lights. Include eyes, lids, lips, wrinkle state, shadows and
diffusion. Keep full quality and mechanical discretization identical initially.

Verify force/energy/position criteria, J and residual gates against existing
CPU/GPU fixtures, plus exact checkpoint-version and velocity provenance. Inject
loss/reset/disposal during compute, map and anatomy loading; require stale
results to be ignored and recovered state to match the recorded checkpoint.
Exercise rapid pointerdown/move/up, stale proxy backpressure and bounded-error
grab selection. Invalid solver output must preserve the last accepted display.

Measure displayed presentation frames separately from render submissions,
solver updates and simulation-seconds advancement. Report medians and worst
frame latency, map wait distribution, cold start, peak full-process/GPU memory,
checkpoint age/rollback and picking latency. Include coordination/telemetry
overhead. A numerical probe pass alone never certifies production performance
or material parity. Request a coordinated GPU slot only after the isolated
integration candidate and reference paths are ready; release the device before
encoding captures or assembling reports.
