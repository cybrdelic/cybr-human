# CYBR human tissue platform

The supported application is `neutral-tissue.html`. This source repository preserves the selected immutable runtime **4bc8b53a8f15b8b62cf6a86e**, with its code-authored head v5, tissue v6, licensed internal anatomy and procedural skin material. It is a prototype for interactive volumetric deformation, not an accepted photorealistic or clinically calibrated human model.

## Run the preserved application

Python 3 is enough to serve the included assets. No npm install, external model download or network connection is required to run the selected browser application.

```powershell
git clone https://github.com/cybrdelic/cybr-human.git
cd cybr-human
python manage.py serve --port 8841
```

Open `http://127.0.0.1:8841/neutral-tissue.html`. The index and legacy page names redirect here. Use a browser with WebGL2; WebGPU enables the GPU solver. `?backend=cpu` explicitly uses the worker fallback; rendering still requires WebGL2. Controls include adipose growth, softness, gravity, skin pull/release, reset, rest anatomy, camera views and optional forehead detail. On-demand controls solve static equilibrium. **Run simulation** uses fixed-time dynamic integration; measured speed depends on hardware.

## Install development dependencies

Node 20+ and Python are required for tooling. Package versions are pinned in `package-lock.json` and `requirements.txt`. The browser imports the small pinned Three.js distribution from `vendor/`, independently of npm.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
npm ci
```

Browser regression tests use locally installed Microsoft Edge through Playwright (`channel: msedge`). No existing browser profile or login state is needed. Keep browser/GPU tests serialized with other GPU applications.

## Verify

```powershell
python manage.py verify --suite offline
# Start the staged server first, then use its port:
python manage.py verify --suite browser --port 8841
python manage.py verify --suite all --port 8841
```

The offline suite checks selected runtime closure/hashes, known rest-shape defects, actual GLB topology and the selected CPU worker's static/dynamic/reset/substep behavior. The ten browser regressions check compression derivatives, controls, ranges, build identity, failure recovery, WebGL context recovery, forehead detail, CPU/GPU parity, diffuse transport and mechanics. Results are written to `output/verification/`. They complement visual inspection and do not establish visual acceptance by their count.

Supplemental `tests/bounded-cg.mjs` and `tests/surface-cache.mjs` concern newer editable source and require a separately constructed candidate supporting those features. The selected runtime does not include the newer surface cache. See [architecture](docs/ARCHITECTURE.md), [consolidation decisions](docs/CONSOLIDATION.md), [reproducibility](docs/REPRODUCIBILITY.md) and [asset attribution](ASSET-LICENSES.md) and [measured validation](docs/VALIDATION.md).

## Measured correctness and performance

The selected runtime passed offline verification and all ten retained hardware browser regressions on an NVIDIA RTX 4060 Laptop GPU. Actual original/staged control and settled scene pixels matched. **Performance acceptance remains incomplete:** short matched-setting runs measured about 30-34 scene/solver updates per second in the consolidated stage versus 40-45 in the original, and both simulations lagged wall time. Extra wall-time stalls were observed; their cause is unresolved. These measurements do not establish realtime, photorealistic or anatomical acceptance. See [validation and limits](docs/VALIDATION.md).

## Construction and promotion

The editable `src/` retains newer source work separately from the selected runtime. Rebuilding can therefore produce a different build. These commands modify only this staged tree:

```powershell
# Construct a candidate, leaving runtime-current.json alone:
python tools/build_tissue_runtime.py --no-publish
python manage.py verify --candidate --suite offline
python manage.py verify --candidate --suite browser --port 8841
# Explicit release operation: verifies and promotes its new candidate:
python manage.py build --port 8841
# Also regenerates geometry, volume and surface bindings (expensive):
python manage.py build --geometry --port 8841
```

Do not use `build` merely to run the preserved application. Candidate **ef7aee6716ff02e2cc0021f9** from the originals was not promoted or imported as the active build.

## Licensing

Licenses are scoped by component: the inherited GPL-2.0 code notice, CC BY-SA 4.0 anatomical artifacts and Three.js MIT terms are retained separately. See [licensing](LICENSING.md) and [asset attribution](ASSET-LICENSES.md). A code license does not relicense the third-party assets.

## Included boundary and limitations

`src/` owns the viewer, verified model loading, CPU/GPU FEM, surface transfer and procedural shading. `tools/` owns offline authored geometry and runtime publication. `tests/` contains reusable numerical and browser checks. `assets/anatomy/` contains the minimal licensed atlas geometry/manifest needed by the active construction tools; `anatomy-source/` retains attribution and selection provenance. `output/neutral-head/v5` and `output/neutral-tissue/v6` contain only required editable construction outputs/intermediates; `output/runtime/` contains one immutable selected build.

The public package contains curated project files and required runtime assets. Browser profiles, cookies/auth state, credentials, caches, iteration dumps, old captures and inactive experimental pipelines are excluded. `provenance/staged-source.json` retains path-free source-copy hashes. `provenance/public-source.json` records this public package's file hashes. Host paths and raw QA reports are omitted.

Known limitations remain: approximate anatomical chart containment and fixed tangent contact, inactive muscles and jaw coupling, unfinished moving contact/sliding fascia/self-contact, one-way forehead detail, approximate RGB screen-space transport and unaccepted head/skin appearance. Positive cell volume or numerical verification does not establish anatomical fidelity or realtime performance.
