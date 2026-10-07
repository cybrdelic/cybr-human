# HUMAN implementation stages

This is a new implementation effort. Its baseline is public tree
`11854a15dd3e88e4141d85d4ba0d7b1765106da8`, deployed source
`14228fd7b7449a611685d3e5a571fbd94ffdd766`, and public main merge
`f3b700ac281b998c0b8105a028288c2773b5ee79`. The selected immutable runtime
`4bc8b53a8f15b8b62cf6a86e` remains reproducible and unchanged.

Before implementation the 17 selected source modules were restored into the
editable source lane. A nonpublishing reconstruction reproduced the exact
selected runtime identity. This deliberately removes the unverified newer
cache, bounded-CG early-exit, lip tint and static scheduling changes from the
editable candidate; it does not remove them from Git history. Reviewing the
new candidate against the selected runtime is the relevant behavior comparison.

## First reviewable milestone

The candidate preserves smooth normals transported by the inverse transpose
of the recovered gradient. The prior fragment override replaced them with
triangle derivatives and also discarded fine-patch normal changes. Query
`normals=geometric` retains that old path for paired inspection. Skin diffusion,
fine patch, authored material response, geometry, solver equations, positivity,
energy/backtracking checks and recovery checkpoint readback are preserved.

Hidden registered anatomy loads only on selection. Loading verifies the same
asset hash, coalesces concurrent requests, keeps skin visible until ready,
respects an uncheck during loading, disposes a late result after page disposal,
and allows retry after failure. The existing compressed-model bootstrap and
visible initial-load retry remain unchanged.

Measurements separate final-submit CPU cost, map waiting and mapped-array copy
cost. `hostBeforeFinalSubmitMs` includes encoding and any awaited bounded queue
flushes; it is not a pure CPU-work measurement. GPU timestamps cover the timed
compute pass, not every host wait. Render metrics are bounded to 240 samples
and explicitly count CPU render submissions, not display presentation. Their
submission gaps can include on-demand inactivity; dynamic-window sample gaps
must be evaluated separately. `debug.measurement()` also returns simulation
advancement divided by active simulation wall time. Startup records first
prepared-face submission and solver readiness, not first browser paint.

`setInspectionLighting()` provides studio, side and overhead presets for QA.
Studio exactly matches the baseline's default lights. No inspection control
is added to the public UI. `tests/upgrade-browser.mjs` serializes baseline,
candidate geometric and candidate smooth paths, captures actual face images,
records 12-second dynamic windows, checks passive settled positions, and
exercises optional anatomy failure/retry. Its baseline/geometric comparison
separates added measurement/startup code from the normal change. The controlled
geometric/smooth comparison isolates normal appearance.

## Physical calibration lane

[TISSUE-CALIBRATION.md](TISSUE-CALIBRATION.md) describes regional material
candidates, published suction protocols, an authored displacement hold/release
fixture, strict SI/provenance imports, subject-separated holdout evaluation and
exact input hashes. Six bounded standard-library tests exercise this framework.
No material values are fitted, no human dataset is invented, and no additional
physiological fidelity is claimed. Actual raw observations and matched
geometry/preload remain necessary before integrating constitutive changes.

## Following milestones

1. Prototype dense transfer and rendering on the solver's actual GPU device.
   Measure full readback, reduced asynchronous telemetry and versioned recovery
   checkpoints separately. Retain GPU validity/acceptance checks and CPU
   reference parity. Do not promote a diagnostic renderer as skin-render parity.
2. Improve regional skin/eye/lip materials and silhouette under the controlled
   views, retaining existing diffuse transport. Establish approved asset
   provenance for any introduced geometry. Keep mechanical and visual resolution
   independent; additional polygons alone are not the appearance target.
3. Adapt passive fixtures to the actual contact/preload geometry, fit only on
   permitted training data, then evaluate held-out loads/rates/subjects before
   incrementally integrating anisotropy, relaxation, sliding or muscle/jaw work.
4. Add a neutral-face-first startup path and independent visual LOD, measuring
   cold start, memory, geometry/normal error and resolution-transition continuity.
   Assess multilevel preconditioning only on difficult measured loads.

Every public release needs an exact reviewed candidate identity, paired actual
appearance captures, force/position/energy/recovery checks, and honest timing
limits. Single-device short benchmarks cannot establish Android performance,
driver-level causes, physiological accuracy or broad realtime guarantees.

The conceptual shared-device path follows the
[WebGPU specification](https://www.w3.org/TR/webgpu/). Existing diffusion work
should be evaluated against [d'Eon et al., 2007](https://research.nvidia.com/publication/2007-06_efficient-rendering-human-skin)
before adding another scattering layer. Regional calibration context comes from
[Pensalfini et al., 2018](https://doi.org/10.1016/j.jmbbm.2017.10.021).
