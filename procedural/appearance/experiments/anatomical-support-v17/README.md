# Opt-in anatomical support v17

**This reproducible structural experiment is not judged an overall improvement over preferred v53.** Its reusable gains are the rounded chin and coherent closed nasal construction. The more open upper lids give a startled expression, the nasal tip/alar shadow reads bulbous and pinched, and the diagonal side-cheek/jaw hollow remains conspicuous. Preferred v53 stays the selected baseline. See `VALIDATION.json` and `validation/VISUAL_JUDGMENT.json`.

The module rebuilds one explicit candidate from the exact original preferred head. It changes no default profile, mechanical rest, material law or existing runtime source. No dense generated head, scan, stock head, real-face texture or rejected-iteration archive is shipped.

## Supported platform and tested environment

The build/render launchers currently support **Linux**. They require `fcntl.flock`, `os.sched_getaffinity`, Linux `/proc` accounting and the `taskset` executable. Native Windows and macOS execution is not supported by these launchers. At least two allowed CPUs are required; CPUs 2 and 3 are preferred, otherwise the first two allowed CPUs are chosen.

The exact replay was tested with Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0 and Blender 4.3.2. Blender must be available as `blender` on `PATH`. These are observed tested versions, not asserted minimums. No external asset, network request, Library access or automatic installation is performed by the module.

Stages release the same shared lock between phases. Each geometry/fit phase has a 90-second bound; each appearance frame has a 360-second bound. The admission floor is 2 GiB available memory and the sampled aggregate RSS stop threshold is 900 MiB. This is user-space supervision, not a kernel hard cap. Use the same lock as any other HUMAN work running on that host.

## Repository location and dependencies

Install this entire directory at `procedural/appearance/experiments/anatomical-support-v17` in `cybrdelic/cybr-human`.

`DEPENDENCIES.json` pins the 26 existing appearance/runtime/control files used by the replay and render to both SHA-256 and Git blob hashes. Those bytes were verified against main commit `9218e8696a471018b5c711bd9f167d4af27fd2cb`; the inherited appearance merge is `31fde2fc2afeb6fd3e1209d7ac873a4dbf3f34f0`. The preferred input is obtained from the existing `procedural/checkpoints/preferred-v53` builder. None of those existing files needs modification.

`CONTROL_CONTRACT.json` records the typed NPZ control/attachment inputs and construction interfaces. `SOURCE_MANIFEST.json` pins this module's own files. The inputs are original procedural controls, fixed attachment indices and labels, not an imported head. Tissue points and normals are reconstructed from the candidate surface.

## Lightweight CI from the repository root

This check uses only the Python standard library. It does not run a model, install packages, invoke Blender, or acquire the heavy-compute lock:

```sh
python3 procedural/appearance/experiments/anatomical-support-v17/tests/test_control_contract.py \
  --appearance-root procedural/appearance
```

It checks Python syntax, source hashes/completeness, exact existing dependency bytes, opt-in/output pins, typed NPZ control headers, construction interfaces and absence of baked head/render assets. Passing it does not repeat or replace the recorded numerical replay, contact checks or visual review.

## Exact numerical replay from the repository root

Use fresh output directories. The existing preferred-head builder rejects an already existing output or companion input directory.

```sh
python3 procedural/checkpoints/preferred-v53/build_preferred.py \
  --output /tmp/human-v17-preferred \
  --lock /tmp/cybr-human-cpu.lock

python3 procedural/appearance/experiments/anatomical-support-v17/build.py \
  --preferred /tmp/human-v17-preferred/project/output/coordinated-orbital-scale-v53/head.npz \
  --appearance-root procedural/appearance \
  --output /tmp/human-v17-geometry \
  --lock /tmp/cybr-human-cpu.lock
```

The second command rebuilds the finite orbital/margin geometry, closed nasal volume and shared anatomical control surface. It verifies the full candidate and semantic hashes, then runs intersection and oriented-topology checks. Final outputs are `/tmp/human-v17-geometry/head.npz`, `anatomical_semantics/authored_lip_lid_semantics.npz` and `REPLAY_RECEIPT.json`.

An already available preferred head can replace the first command only if its SHA-256 is `545b8ea2b292107dc2da9f1e90379989eb620c6ee5c2c50b3f9b5316cc77420e`.

## Exact fit and matched appearance

```sh
python3 procedural/appearance/experiments/anatomical-support-v17/render.py \
  --appearance-root procedural/appearance \
  --geometry-root /tmp/human-v17-geometry \
  --output /tmp/human-v17-appearance \
  --lock /tmp/cybr-human-cpu.lock --view head_front --fit-only

python3 procedural/appearance/experiments/anatomical-support-v17/render.py \
  --appearance-root procedural/appearance \
  --geometry-root /tmp/human-v17-geometry \
  --output /tmp/human-v17-appearance \
  --lock /tmp/cybr-human-cpu.lock --view head_front

python3 procedural/appearance/experiments/anatomical-support-v17/render.py \
  --appearance-root procedural/appearance \
  --geometry-root /tmp/human-v17-geometry \
  --output /tmp/human-v17-appearance \
  --lock /tmp/cybr-human-cpu.lock --view head
```

The common published scene retains the original shaders, cameras, lights, finite eyes and groom recipe. Skin/eye/groom fits are freshly checked against this exact surface. Uniform numerical optical conditioning is 1000, with light and SSS compensation; it does not change the authored physical geometry. No iris painting or shadow bypass is introduced. Actual captured evidence uses 560 px, 256 samples and no denoising.

## Editable construction and limits

`surface/source/mandibular_support.py` contains the curved triangular subdivision cage, independent XYZ anatomical handles, bounded bending fit and shared jaw-support constraints. The build writes the source/target cage for inspection. The adjacent orbital, medial-orbital, oral and midface modules retain the coupled finite-feature construction.

Changes invalidate the pinned output, fit and visual checks and must become a separately validated candidate. The fictional generated guide informed qualitative proportions only; it is neither metric reconstruction evidence nor a texture source.

Passing topology, sampled Jacobians and triangle checks is not a realism certificate or a full global geometric proof. This is not a qualified mechanical binding, expression solve or dynamic contact result. A changed rest surface must be explicitly rebound and validated before mechanical coupling. Preserve the repository's existing licensing and component notices.
