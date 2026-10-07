# Measured validation and limitations

All behavior and visual checks used selected runtime **4bc8b53a8f15b8b62cf6a86e**. Original and consolidated runtime/entry/vendor bytes matched on 69 active paths. Public packaging changes documentation, licensing notices and privacy-safe verification metadata; it preserves the selected runtime and required asset bytes.

## Completed checks

| Check | Measured result |
| --- | --- |
| Runtime asset closure | 60 manifest-listed files and 58 references resolve; recorded native/runtime asset hashes pass. |
| Rest shape | Midline slope 0.000133724; minimum neck radius 54.8294 mm; crown-pole ratio variation 0.00006266. |
| Exported head GLB | 307,477 vertices, 613,746 triangles, six expected boundary loops, no folded frontal triangles; 193 registered internal parts and no sampled chart-outside vertices. |
| Selected CPU worker | Static, dynamic, reset and exact substep contract checks pass. Original/staged fixed-case packed states are bit-identical. |
| Actual CPU/software browser | Verified SwiftShader with CPU solver, 960x720/device scale 1. Neutral and 25% fat/25% softness images match exactly. Actual pull, release, anatomy, reset and dynamic transitions pass. |
| Hardware behavior/visual comparison | Edge headless, verified NVIDIA RTX 4060 Laptop WebGPU/ANGLE D3D11, 1280x900/device scale 1. Both seven-state protocols pass with no page/asset failures. Neutral, controls, anatomy, reset and independently settled scene regions match exactly. |
| Ten retained browser regressions | Compression barrier, build contract, mechanics, controls, slider range, device-loss recovery, WebGL recovery, forehead detail, CPU/GPU parity and diffuse transport pass. |
| GPU/CPU parity | Maximum dynamic position error 3.137e-9 m; quasistatic error 1.003e-7 m. |
| Device-loss recovery | Maximum preserved checkpoint error 0 m; the recovered CPU worker accepts new controls. |
| Diffuse transport | Original/staged metrics match: 273,096 lit pixels, 24,910 changed pixels; background, iris and restoration errors are zero. |
| Publisher reproducibility | Editable publication repeatedly produces `cff25f32c7645d113a4d3ff0`. Using archived selected source in a disposable tree recomputes exact `4bc8b53a8f15b8b62cf6a86e`. No candidate was promoted. |

These checks complement inspected actual scene captures. Their count does not establish anatomical, visual or performance acceptance. Raw host-specific reports and browser recordings are kept privately and are not part of this repository. Run the retained tests to produce new local reports.

## Unresolved hardware timing gap

**Performance acceptance is incomplete.** In approximately eight-second dynamic samples after static convergence, the consolidated stage measured 30.44 actual scene/solver updates per second versus 40.51 for the original, about 24.9% lower. An earlier pair measured solver rates 33.65 versus 44.98/s. The order was original-then-stage initially and stage-then-original in the final pair; these are short observations, not a controlled repeated acceptance benchmark.

The final paired settings matched: 1280x900/device scale 1, Edge headless, NVIDIA/lovelace WebGPU, Interactive quality, fat 25%, softness 25%, gravity on, three-quarter view, default skin diffusion and dynamic time step 1/60 s. Both began from the same uncapped 13-step settlement: J=0.62837857, residual 0.00040598 N, zero error across 188 sampled surface points, matching settled scene pixels and the same completed 2,145-node forehead solve. No video encoding or provenance badge was active. The shared external observer counted actual skin-scene renders separately from RAF callbacks.

Final GPU medians were close (3.630 ms stage; 3.539 ms original), as were solver wall medians (11.6 vs 11.1 ms). The stage recorded nine wall samples above 100 ms, totaling 3.950 s; the original recorded six, totaling 2.660 s. Several 400-470 ms wall samples contained only 2-5 ms of GPU-timestamp work. The wall timer includes submission and awaited GPU readback, so these timestamps do not cover the entire wait. Extra long wall stalls materially reduce throughput, but their initiating cause is not demonstrated.

Recorded resource-path multisets match. Six representative asset responses have matching content types, lengths, modification times and server/cache headers. Both route-enabled contexts disable HTTP cache. Origins/ports differ, and V8/shader/driver caches, CPU load, GC, thermals/power clocks and OS/readback scheduling were not profiled. These remain possible confounders, not demonstrated explanations. A later controlled matched-order/origin/cache test with traces is needed. Application math was not changed to seek a pass.

Both runs lagged wall time: the final stage advanced 4.083 s of simulation during an 8.049 s observed window; the original advanced 5.4 s during 7.997 s. Neither establishes realtime simulation.

## Retained limits and corrected harness failures

Software CPU solves took roughly 7-15 seconds per viewer solve. Static controls/reset reached the unchanged 45-second cap before convergence. An initial recorded harness timed out waiting for four updates; the harness subsequently checked actual applied parameters and completed state transitions. An external evidence wrapper initially misplaced diffusion JSON needed by its Python companion; after correcting only the wrapper, original and stage diffusion checks passed. No application threshold or solver math was changed for these corrections.

Full geometry regeneration was not performed. Head proportions, eyelids, brows and skin shading remain prototypes. Moving contact/sliding fascia, self-contact, active muscles and jaw coupling are unfinished; forehead detail is one-way and light transport approximate. Positive cell volume and passing numerical tests do not establish photorealistic, anatomical or clinical fidelity.

Run `python manage.py verify --suite offline`. Start a local server before `python manage.py verify --suite browser --port 8841`. Serialize browser/GPU tests with other GPU applications.
