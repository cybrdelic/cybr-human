# Original layered fibre mapping, v1

This source assigns five independently addressable authored fibre families in a
checked tetrahedral mesh. It does not remesh, modify geometry, run mechanics, or
choose material stiffness. Physical units and physiology remain uncalibrated.

## Input contract

- Final positive tetrahedra: `vertices`, `tetrahedra` in a pinned NPZ.
- Original closed regional volumes plus authored centrelines in a pinned route
  `parts.npz`. Use the same route version that produced the final union.
- Exact prior zygomatic `cheek-remesh-v9/geometry.npz`, pinned separately. Its
  saved ring centres/tangents define that direction field. Its original passive
  collar/adapter stays passive by containment in the original passive tets.
- `ANATOMICAL_SOURCE_CONSTRAINTS.json` is read and hashed into provenance.

The executable requires all three geometry SHA-256 values. An existing output
directory is rejected. Neither input files nor solver sources are changed.

## Mapping convention

For each tetrahedron, four symmetric interior quadrature locations are tested
against all original closed volumes. Three rays must agree; source-boundary or
disagreeing results use double-precision triangle solid-angle winding instead.
Nonintegral winding is explicitly ambiguous. There is no nearest-route fallback.

Junction and original zygomatic passive-adapter membership take precedence. A
sample contained in multiple active routes shares a unit occupancy budget
equally among those routes. This is a declared modelling estimate, not measured
fascicle fraction. Per-cell fractions are equal-weight quadrature averages.
The passive material is intended to be evaluated once by the existing shared
continuum. Fibres are not added in the junction just because a nearby route ends
there. Cells crossing a source interface are identified explicitly.

Any missing or ambiguous sample is unassigned and quarantines its cell. Such a
run still writes diagnostics, but exits with status 2 and marks admission false.
No material or fibre identity is invented to hide a Boolean/remesh mismatch.

Each family gets its own C1 centreline tangent estimate and activation channel:
`zygomatic_reference`, `superficial_oral`, `deep_oral`, `buccinator_deep`, and
`mandibular_depressor`. Tangents are obtained only after membership from a bounded
local closest-curve projection. Family direction vectors remain distinct in
overlaps; they are never averaged across muscles. Centrelines are editable by
providing a separately versioned authored route input, never changing history.

## Outputs and mechanics handoff

- `mapping.npz`: full membership statuses; family fractions and directions;
  passive/unresolved fractions; exact cell lists for interface mixtures,
  overlaps, missing/ambiguous membership and quarantine.
- `families.json`: named activation channels, per-tet `volumeFractions` and
  `fibreDirections`, with activation zero. No stiffness is invented.
- `report.json`: source pins, source closure checks, volume budgets, admission
  status, numerical tolerances and explicit limitations.

These records match the family fields of `InterdigitatedMuscleFEM` after a
future assembly explicitly supplies positive stiffness values. Do not pass
overlapping families to the single-family `VolumetricMuscleFEM` class. Independent
activation uses `setMuscleActivation(value, exactFamilyId)` on that existing
solver. Mapping admission is not solve admission; contact, quality, attachments,
constitutive choices and static validation still need their own gates.

Four-point occupancy sampling is bounded and reviewable, but is not exact
volume clipping. Tiny intersections can be missed. Refinement or explicit
internal interface clipping would be needed for a stronger material-boundary
claim. None of this asserts a complete facial-muscle model.

## Run policy

Coordinate the real-mesh run with the owner. Use the existing shared HUMAN
`cybr-human-cpu.lock`, CPU affinity 2,3, all thread counts 2, a sampled aggregate
RSS stop at 900 MiB, and at least 2 GiB available-memory admission.
Use the project's installed Python with mathutils and SciPy, or its existing
Blender runner. Run `test_mapping.py` under the same guard before real geometry.

Arguments are `--tet`, `--parts`, `--zygomatic`, `--output`, `--constraints`,
`--expect-tet-sha256`, `--expect-parts-sha256`, and
`--expect-zygomatic-sha256`. The optional `--chunk-tets` defaults to 1024 and is
bounded to 4096. Use a fresh output subdirectory such as `checked-mesh-map-v1`.
The default `--mesh-status diagnostic-only` does not imply mesh admission.
`--mesh-status admitted-geometry` additionally requires `--mesh-report` with a
matching geometry pin and explicit reference-geometry admission; its quality
statistics and limitations are retained verbatim in the mapping report.

For an independent output check, `audit_mapping_v1.py <mapping-directory>
<new-audit-report.json>` verifies all input/output pins, fraction reconstruction,
JSON/NPZ correspondence and a deterministic balanced sample of direct
double-precision winding classifications. This is a sampled numerical audit,
not complete containment or mechanics certification.
