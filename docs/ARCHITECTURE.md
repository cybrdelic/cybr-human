# Architecture and runtime contract

## Entry and identity

`neutral-tissue.html` provides the controls and local Three.js import map. `src/boot.js` reads `output/runtime-current.json`, checks optional `expectedBuild`, then dynamically imports the selected immutable viewer. All runtime workers, modules, WGSL and bound model assets resolve beneath that build. `manifest.json` records the publisher's source identities; runtime JSON URLs are rewritten after hashing. `tests/source-graph.py` accounts for that distinction, checks literal module/worker closure and asset-level hashes, and compares the preserved entry/runtime against the original copied bytes.

Selected build: `4bc8b53a8f15b8b62cf6a86e`. A page keeps the build it booted. A reload resolves the pointer again. Model loading verifies binary hashes and bound GLB identity before installing a solver or deformation surface.

## Geometry and numerical state

The code-authored head uses millimetres in its construction atlas frame, then exports glTF metres. `tools/anatomical_surface.py`, `head_features.py` and `ear_surface.py` define the chart, apertures, eyes, hair and ears. Only the licensed internal atlas geometry and manifest feed anatomy containment checks; exterior scans, photographs and imported ocular meshes are absent from the active generator.

`build_neutral_head.py` exports head v5, internal anatomy, cached authored arrays and registration. `build_neutral_tissue.py` constructs tissue v6 through `build_volume.py`; `build_neutral_tissue_assets.py` adds boundary masks, local tangent anchors, bending hinges and surface bindings; `build_tissue_transfer.py` provides continuous transfer weights. The included NPZ intermediates support rebinding without importing old atlas rigs or historical output. Geometry regeneration is a separate, expensive operation.

The selected model contains 9,228 nodes and 36,450 tetrahedra. The CPU worker and GPU shader use the same prescribed boundary and volumetric Stable Neo-Hookean equations, bending and numerical compression barrier. The barrier is inactive above J=0.6 and singular at the J=0.2 guard; it is a regularizer rather than a calibrated tissue law.

## Interactive solve and rendering

`viewer.js` orchestrates controls, camera, model installation, solve policy and rendering. It chooses WebGPU when available, otherwise the CPU worker. On-demand changes use quasistatic equilibrium with no simulated time or velocity; **Run simulation** accumulates fixed-time dynamic substeps. Adipose growth changes local volume targets; softness changes stiffness; gravity and distributed point springs supply loads. Reset and release operate on the same model.

Both backends publish packed nodal positions and recovered deformation gradients. `layered_fem_surface.js` transfers them to the dense authored skin. `skin_rest_normals.js`, `authored_skin_material.js` and `skin_diffusion.js` handle normals, procedural material and depth-gated RGB transport. `fine_skin_patch.js` and its plate worker add a one-way forehead patch. `gpu_page_lifecycle.js` suspends or disposes unsafe graphics sessions.

Editable source contains newer FEM surface caching, bounded-CG work, lip shading and interactive static-update scheduling. See REPRODUCIBILITY.md for exact content differences and the selected-source reconstruction check. Those modules remain available for future candidate evaluation. The selected immutable build uses its older surface transfer; no source edit changes it until a separate verified promotion. That deliberate seam protects the actual current behavior.

## Dependencies and verification

The browser runtime depends only on included assets, local Three.js 0.169.0 and browser graphics APIs. The minimal Three.js closure is `three.module.js`, `OrbitControls.js`, `GLTFLoader.js` and `BufferGeometryUtils.js`, with MIT attribution. Npm dependencies are Playwright, meshoptimizer and Prettier; Python pins support construction/topology analysis. Dependency installs and generated test output are excluded from source provenance and should not be shipped as source caches.

Verification has three layers: immutable file/asset graph and offline numerical checks; actual browser loading/interaction and CPU/GPU paths; controlled baseline/staged visual comparison and short recordings. Every result must name its selected build, backend and settings. A capped static solve or low measured update rate remains a reported outcome. Do not infer photorealism, anatomical coupling or clinical validity from a test pass.

## Remaining structural cost

`viewer.js` still carries model orchestration, controls, simulation scheduling, rendering and diagnostics. Extracting those responsibilities while keeping worker/state contracts synchronized is future work. This consolidation removes output contamination and ambiguous project boundaries without changing the solver or inventing new extension layers.
