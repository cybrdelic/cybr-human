# Experimental continuous tissue self-contact

Opt-in reference geometry and solver-transaction checks. This experiment does
not alter the runtime, selected v53 appearance, physical energy/gradient, PCG,
activation, or line-search/determinant gates. It is a rejection filter, not a
complete contact force law or an accepted loaded equilibrium.

## What is checked

- Binary64 endpoint coordinates are lifted exactly onto an integer lattice.
  Strict exact polynomial/Bernstein witnesses cover each entire closed dyadic
  interval; no temporal samples or floating tolerances authorize acceptance.
- Nonincident pairs use exact separation of their swept convex hulls.
  Incident pairs require a verified moving plane through the declared common
  vertex or edge; only the actual shared global indices grant that feature.
  There is no blanket adjacency exemption or centroid trim.
- Complete swept-AABB self-traversal covers every overlapping unordered face
  pair. Unrelated coincident vertices remain unrelated. Work budgets fail closed.
- A stationary fast path checks the complete exact triangle intersection feature
  only when both full trajectories are constant; it never promotes changing
  trajectories from endpoint coincidence. Coordinate caching preserves proofs.
- The new combined wrapper derives its face list from all tetrahedra and pins
  positions, topology, rest and mass. It combines self-contact with the existing
  exact static-bone/moving-root filter at initial guesses, fixed-target writes
  and every inherited line-search trial.
- A private capability binds the genuine unchanged bone collider; source-root
  capabilities retain their original collider identity. Method accessors,
  substituted implementations, forged summaries, stale topology, and direct
  unadmitted coordinate changes fail closed.
- Exceptional inherited transactions restore the original position/velocity/
  predictor buffer references and bytes, mark the report failed, and require
  explicit readmission. Arbitrary subclass caches are not claimed transactional.

## Evidence

Run with Node 24 from the repository root:

```sh
node experiments/human-continuous-self-contact-v1/tests/test_moving_triangle_pair.mjs
node experiments/human-continuous-self-contact-v1/tests/test_exact_corroboration.mjs
node experiments/human-continuous-self-contact-v1/tests/test_tissue_self_bvh.mjs
node experiments/human-continuous-self-contact-v1/tests/test_indexed_equivalence.mjs
node experiments/human-continuous-self-contact-v1/tests/test_actual_rejected_state.mjs
node experiments/human-continuous-self-contact-v1/tests/test_combined_boundary.mjs
```

Coverage includes 26 pair tests, 13 complete-boundary/coverage tests, 200
exhaustive broad-phase comparisons, 900 deterministic integer trajectories,
1,926 independent exact dyadic sample corroborations of admitted trajectories,
900 byte-identical indexed/uncached results, and 25 combined-authority tests.
Samples corroborate the implementation; they do not replace the algebraic proof.
The compact real wider-16% failure fixture rejects both known bad pairs and
interpolated 8%-to-16% paths; no invalid 16% state is a continuation seed.

The private hash-pinned authored-domain probe is separate from small CI fixtures.
It admitted 33,736 boundary faces, all 367,661 self-pairs and 19,390 bone
candidates, and completed a zero-Newton neutral step with minJ exactly 1,
unchanged coordinates and residual 2.5046135190938e-17 N. Observed total time was
63.81 seconds and sampled peak 667.24 MiB on a shared cloud CPU host; this is
not a general speed or hard-memory-limit guarantee. Large anatomical assets
are not included in this patch.

A straight-line rest-to-stored-first-2% diagnostic was safely rejected at
shared-vertex pair 18591/18593 after 6,015 checked self-pairs. The guard left
positions unchanged. The rejection was unresolved witness search, not a proven
collision. It does not reconstruct the historical Newton trajectory. An
independent exact moving-plane extension is being investigated separately;
this patch retains the frozen reference search and its unchanged defaults.

## Limits

No uniform positive tissue clearance, continuous positive-cell determinant,
solid containment, complete collision response, anatomical calibration, or
loaded equilibrium is established. Conservative false rejection is possible.
Complete closed edge incidence does not by itself prove a face list complete;
the solver wrapper supplies the independent complete tetrahedral extraction.
The primitive modules remain stateless and do not themselves grant solver
admission. Endpoint determinant checks are inherited unchanged.

## Provenance and licensing

Existing imported dependencies are pinned in `DEPENDENCIES.json`. The recovered
original FEM remains byte-identical. Repository adaptation changes import paths
only. Preserve
`procedural/mechanics/regional-reference/workspace/human-mechanics-20261010/reference-source/LICENSING.md`
and its `LICENSES/GPL-2.0.txt` when redistributing corresponding source.
