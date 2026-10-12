# Procedural HUMAN appearance

The default geometry profile is the exact preferred-v53 rest head. The renderer uses original procedural tissue, iris and grooming sources. The optional orbital and visible-margin variants are separate visual studies; they do not replace the mechanical rest model.

## Reproduce the preferred input and prepare a render

From the repository root, build the original preferred geometry first:

```sh
python checkpoints/preferred-v53/build_preferred.py --output /absolute/fresh-preferred
python appearance/run.py \
  --geometry-root /absolute/fresh-preferred \
  --output /absolute/fresh-render \
  --lock /absolute/HUMAN.lock \
  --view head_front --prepare-only
```

The preferred builder produces `project/output/coordinated-orbital-scale-v53/head.npz` and the matched `anatomical_semantics/authored_lip_lid_semantics.npz`. Both are hash-checked before scene preparation. `--geometry-root` accepts either that final directory or the build output root. Explicit `--head` and `--semantics` paths are also supported. A fresh output directory is required; source data are never overwritten.

Remove `--prepare-only` to render. Use `--view head` for the fixed three-quarter camera. The default profile is `preferred`. The included skin/groom response is an authored diagnostic appearance, not a biological calibration or automatic selection over historical imagery.

Dependencies: Python3 with NumPy, Blender4.3.2 with its NumPy environment, Linux `taskset`/`flock` support via Python `fcntl`. SciPy is additionally needed to evaluate editable optional section curves. No network download, API token, real texture, scanned head or generated-image texture is used.

## Optical conditioning and unchanged controls

The actual layered-eye test exposed a fixed-distance threshold problem in Cycles4.3.2 MNEE at the small numerical scene scale. `uniform_numerical_scale.py` applies one uniform factor of1000 to all scene dimensions, camera/area-light dimensions, strand lengths and SSS/bump lengths. Area-light power scales by the square of that factor. IOR, color, roughness, view angles, exposure and dimensionless geometry stay fixed. The eyeball is never scaled alone, and no shadow-ray transparency substitute or bright iris paint is used.

The full-frame controls are560×560,256 raw samples, seed47, two threads and no denoising. Actual verified images are in `evidence`. Historical field names containing `mm` are legacy numerical labels; the geometry and tissue dimensions remain uncalibrated authored design units. This is not a validated physiological eye, skin or tear-film model.

## Optional finite geometry

```sh
python appearance/geometry/replay.py \
  --preferred /absolute/preferred/head.npz \
  --variant orbital-v2 \
  --output /absolute/fresh-orbital \
  --lock /absolute/HUMAN.lock
python appearance/run.py --profile orbital-v2 \
  --geometry-root /absolute/fresh-orbital \
  --output /absolute/fresh-orbital-render \
  --lock /absolute/HUMAN.lock --prepare-only
```

`--variant margin-v3` restores orbital-v2 followed by the exact anterior-margin update. Use the matching render profile. These commands reproduce hash-pinned complete meshes and regenerated semantics from compact source data; they do not fetch or duplicate a dense head in this directory.

The editable periodic section construction is in each variant's `authoring/orbital_surface.py`. Evaluate it portably with:

```sh
python appearance/geometry/author_sections.py --variant margin-v3 --output /absolute/fresh-sections
```

That command evaluates local finite surfaces on the included exact host interfaces. Edited sections require fresh global assembly, closure, crossing, optic-clearance and visual checks before becoming a new admitted complete head. It does not silently substitute newly edited local geometry into the exact replay. Original preferred input reconstruction belongs to `checkpoints/preferred-v53`.

The orbital-v2 actual front/three-quarter images show a more connected lid body and less doubled lower bearing. Its lower margin remains faint. Margin-v3 improves the exposed rounded turn in a local clay test; a complete shaded comparison is still pending. Neither is the default mechanical rest. Rejected nasal constructions and other replacement heads are deliberately excluded.

## Resource and scope limits

Each render takes the supplied shared mutex, uses allowed CPUs2,3 when available (otherwise the first two permitted CPUs), requires at least2048MiB available memory, and stops its owned process tree above a sampled900MiB RSS or after360seconds. Geometry replay has a90second bound. The memory guard is sampled user-space supervision, not a kernel cap; it never targets other jobs. Concurrent workload can change duration, so timings are not benchmarks.

This directory provides rest appearance. No integrated muscle-driven pose or reciprocal dense-skin force law is implied. Qualified future sparse displacements require explicit rest/semantic scope, optical clearance and groom binding verification.

`VALIDATION.json` records the fresh relocated preparation checks. `SOURCE_MANIFEST.json` pins published files. Generated reference images are qualitative art direction only and are not included as runtime inputs.
