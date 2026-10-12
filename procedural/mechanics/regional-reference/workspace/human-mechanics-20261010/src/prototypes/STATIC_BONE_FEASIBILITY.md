# Static-bone continuous feasibility: bounded opt-in integration

Source-only implementation, 2026-10-10. No main solver/source changes,
installations, rendering, upload, or full-mesh experiment were performed.

## What this stage does

`static_bone_swept_bvh.mjs` supplies scalable candidate discovery around the
existing original whole-triangle interval certificate.
`guarded_static_bone_fem.mjs` makes it an opt-in admission/rejection condition
around the recovered original FEM step.

This stage deliberately does **not** use `StaticTrianglePairBarrier`, change
the existing finite contact energy, add a force, change its exact gradient,
change its metric, project a pose, or replace PCG. It does **not** turn the
existing finite-sample penalty into a complete contact force model. Its purpose
is to prevent an unsampled surface crossing while testing whether the existing
approximate forces can reach equilibrium without hitting that restriction.
If those forces stall at the guard, the solve remains unconverged.

The parent reports that a subsequently built support-aware bilateral rest
domain has passed independent rest bone/self/globe/containment audits with
conservative bone margin at least 0.4164 authored length. This implementation
has not been run against that domain. Its validation here is entirely tiny,
synthetic fixtures, including one actual original-FEM solve.

## API and input contract

    import { StaticBoneSweptBVH } from './static_bone_swept_bvh.mjs';
    import { withStaticBoneFeasibility } from './guarded_static_bone_fem.mjs';

    const bvh = new StaticBoneSweptBVH({
      bones: [
        { id: 'maxilla', vertices: maxillaWorldVertices, triangles: maxillaFaces },
        { id: 'mandible', vertices: mandibleWorldVertices, triangles: mandibleFaces }
      ],
      leafSize: 8
    });
    const Guarded = withStaticBoneFeasibility(ExistingFEMClass);
    const solver = new Guarded(...unchangedExistingConstructorArguments);
    solver.setStaticBoneGuard(bvh, {
      minimumDistance: selectedSolverUnitClearance,
      maxBoxTests: 2000000,
      maxCandidatePairs: 50000,
      maxTotalIntervals: 100000,
      maxDepth: 24,
      maxIntervalsPerPair: 2048
    });
    const result = solver.step(Infinity, unchangedExistingStepOptions);

`GuardedStaticBoneLayeredFEM` is also exported for the base `LayeredFEM` case.
The factory requires `ExistingFEMClass.prototype.step` to be the exact inherited
recovered original method. Existing subclasses that replace `step` are rejected.
The original GPL-2.0 attribution, license and licensing notes must accompany
any distribution. The original source SHA256 is regression-checked.

- Coordinates may be flat arrays/typed arrays or nested triples. Triangle
  indices may be flat arrays/typed arrays or nested triples. They are zero-based.
- All bone coordinates are immutable copied **world coordinates** in exactly
  the same length unit and coordinate frame as `solver.x`. No transform or unit
  conversion is performed. A bone-pose change requires a new BVH and a separate
  admissible motion treatment; replacing its snapshot does not certify motion.
- The bilateral domain convention is 0.001 solver length per authored length.
  For example, a selected positive clearance of 0.020 authored length would
  be 0.000020 solver length. A permitted negative penetration is not the same
  parameter as positive minimum separation.
- The complete tissue boundary is extracted from **all** `solver.nodes`
  tetrahedra, not an external regional face mask. Duplicate tetrahedra and
  faces incident to more than two tetrahedra are rejected. This is not a full
  manifold, self-intersection or volumetric-containment audit.
- `setStaticBoneGuard` certifies the current boundary before installing the
  guard. Failure throws `BoundaryFeasibilityError` and leaves any prior guard
  unchanged. `clearStaticBoneGuard()` explicitly disables it.

For read-only use without a solver:

    bvh.findCandidates(start, end, tissueFaces, sameOptions)
    bvh.certifyStep(start, end, tissueFaces, sameOptions)

`findCandidates` returns `complete`, `pairs`, `reason` and work counts. Each pair
has `tissueFace`, `boneIndex`, `boneId`, and `boneFace`. An incomplete result must
never be used as a complete collision set. `certifyStep` streams narrow-phase
checks without allocating the Cartesian product; it returns `certified` only
when the entire traversal and every candidate certificate succeeded.

## Conservative broad phase and limits

The static median-split BVH stores exact min/max bounds of its represented
triangle vertices. Each tissue query uses all six start/end triangle vertices,
therefore enclosing every barycentric point along every linear vertex path.
Its min and max bounds are padded by minimumDistance with outward-rounded
subtraction/addition. Strictly disjoint padded boxes may be culled; overlapping
or touching boxes remain candidates. There is no normal-direction, sidedness,
bone-label, region, or currently-active-contact exclusion.

Every tissue face is queried against the static tree. Each bone face belongs
to one leaf, so each input triangle pair appears at most once per query.
Duplicate input bone faces can still cause redundant physical geometry checks;
they are not silently removed. This stage does not assemble primitive forces,
so global VF/EE force-term deduplication remains outside its scope.

The defaults in the API example are independent per-certifyStep-call limits.
Box, candidate, total-interval and per-pair-interval exhaustion all reject the
segment. No clipped/truncated candidate list can be called complete. Counts and
the offending pair/reason are returned. Solver-level total work also depends on
the explicitly supplied original iteration and 18-backtracking budgets; there
is no wall-clock promise or asynchronous cancellation in this helper.

Construction rejects more than 100,000 total bone triangles, 300,000 total bone
vertices, or 16 bones. Query data is limited to 300,000 vertices and 50,000 tissue
faces. Flat input size is checked before a static copy is allocated. These are
deliberate safety caps, not claims of measured performance at those limits.

The narrow-phase assumptions and exact-dyadic proof replay are documented in
`COMPLETE_BOUNDARY_CONTACT.md`. Ordinary certified floating-point arithmetic,
conservative rejection and tiny regression coverage are not a formal end-to-end
proof or a full production CCD benchmark.

## How the original step is guarded

1. Save physical state and construct precisely the initial positions the
   original step would assign: fixed targets take precedence, then initialGuess
   for free nodes. Certify the complete old-to-proposed segment **before**
   calling the inherited step. On failure, positions, velocities, predictors,
   trial coordinates, metric buffers and prior solve result remain untouched.
   The requested pin/target configuration is retained for the caller to revise.
2. During each original trial evaluation, certify from the current accepted
   coordinates to the actual represented trial coordinates before calling the
   inherited energy evaluation. Rejected trials return Infinity without
   touching inherited gradient or metric buffers. Original backtracking handles
   them and never performs its `x.set(trial)` acceptance for such a trial.
3. Snapshot admitted coordinates and check them before and after inherited
   evaluations and at step return. This prevents an unexpected subclass
   evaluator from substituting different coordinates after certification.
   Likewise compare the full stored topology: stale boundary topology cannot
   silently remain active. Direct external writes to x are rejected at the next
   step; use an admitted initialGuess or explicitly reconfigure instead.
4. Preserve the original actual-energy, endpoint minimum-J, residual and
   convergence logic. On an unexpected exception, restore x, velocity, predict
   and prior last. Arbitrary subclass caches/diagnostics are not guaranteed
   transactional and should be recomputed before inspection. An unexpected
   topology-mutating subclass remains invalid until the caller restores it.

The result adds `boundaryFeasibility` and exposes the same report through
`solver.lastBoundaryFeasibility`. It includes initial admission, trial checks,
rejections, certified accepted segments, work totals, last rejection, and:

- `forceModelComplete: false`
- `energyAndGradientUnchanged: true`
- `solidContainmentChecked: false`
- `termination: 'converged'`, `'not-converged'`, or
  `'not-converged-with-guard-rejections'`

The report's `certified:true` means the positions accepted by this guarded call
satisfied the configured static-boundary feasibility tests. It does not mean
force equilibrium, contact-force accuracy, biological validity, self-contact
freedom or rendered-skin clearance. A whole volume contained inside another can
have disjoint boundaries; the exterior initial state must be established
independently. Continuous tetrahedral determinant freedom is also separate.

## Tiny verification

    node --test tests/complete_boundary_contact.test.mjs \
      tests/guarded_static_bone_fem.test.mjs

The integration suite covers swept candidates with disjoint endpoint boxes,
outward positive-clearance padding, two bones and reversed winding, copied
static geometry, all budget failures, exhaustive tiny candidate comparison,
complete boundary extraction, initial intersection rejection, transactional
initialGuess and pin admission, pin precedence, exact no-contact equilibrium
parity, unchanged active-fibre energy/gradient, original source identity,
mutating-evaluator and stale-topology rejection, and rollback.

A deliberately inadequate spring-only force model would translate one tet
through the bone in the unguarded original solve. The guarded original line
search rejects crossing trials, accepts only a feasible partial segment and
returns `converged:false`. This is an expected demonstrated failure to reach
equilibrium, not a solved contact fixture. No broad-domain equilibrium or
performance result is claimed.
