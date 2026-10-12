# Complete-boundary contact: bounded original prototype

Status: source-only design and tiny fixtures, 2026-10-10. No production solver,
full-domain solve, authored anatomy, renderer, package installation, or upload
was changed. This is a proposed next integration, not a claim that the existing
regional or connected facial domain is collision-free.

## Recommendation

Use contact on the actual complete FEM boundary, with both vertex–face
directions and edge–edge constraints. Make continuous feasibility a separate,
mandatory condition on every accepted position change. Keep exact contact
energy gradients in the same equilibrium objective as passive and active
tissue. A finite quadrature penalty, including denser quadrature, cannot replace
that condition.

The bounded helper is `complete_boundary_contact.mjs`; tests are
`../../tests/complete_boundary_contact.test.mjs`.

The parent confirms the next connected rest domain still intersects the static
bone until its carrier fit incorporates actual clearances. It is therefore not
an admissible initial state for this barrier/filter. Initial-intersection
rejection is intentionally retained; this work does not attempt a dynamics run
or silently push the initial mesh out of the bone.

This paragraph records the state during the initial primitive prototype. A
later support-aware rest candidate and a feasibility-only BVH/step integration
are described in `STATIC_BONE_FEASIBILITY.md`; the primitive barrier has not
been integrated into the broad-domain force model.

## Exact API and coordinate contract

- `trianglePairFeatures(tissue, bone)` takes two nested 3-by-3 coordinate arrays
  and returns 15 feature records. This is distance/gradient support, not a
  collision test; all of these distances may be positive at an already
  intersecting pair.
- `certifyTriangleStep(start, end, bone, options)` takes three nested 3-by-3
  arrays. Options are `minimumDistance=0`, `maxDepth=24`, `maxIntervals=2048`,
  `collectProof=false`. The result has `certified`, `reason`, work counts and,
  when requested and successful, a complete interval/axis `proof`.
- `certifyBoundaryStep({startVertices, endVertices, tissueTriangles,
  boneVertices, boneTriangles}, options)` takes nested vertex arrays and
  zero-based triangle-index arrays. Options include the single-pair options
  plus `maxTrianglePairs=4096`. It rejects an excessive Cartesian pair count
  before executing pair checks. It neither discovers a boundary nor builds a
  BVH. This must not be used as an 8k-face production implementation.
- `new StaticTrianglePairBarrier({restTissue, bone, minimumDistance=0,
  activationDistance, stiffness=1, mollifierScale=1e-3})` fixes one static bone
  triangle and the rest edge thresholds. `evaluate(tissue)` returns `valid`,
  `energy`, 3-by-3 `gradient` and `boneGradient`, `metricBlocks`, primitive
  counts and a current-separation certificate. Invalid current geometry yields
  `valid:false` and `energy:Infinity`. Gradients are dE/dx, not forces.

Coordinates are passed through unchanged. There is no unit conversion or
biological-unit assumption. All start/end/rest/bone coordinates, minimum
distance and activation distance must use the SAME length unit. If a caller
uses the existing convention `solverLength = authoredLength * 0.001`, an
authored distance 0.020 converts to 0.000020 solver length. However, a tolerated
negative penetration of 0.020 is not itself a positive minimum-separation
parameter: choose and report the latter independently. The helper requires
strict separation greater than its nonnegative minimum distance. The barrier
stiffness coefficient's dimensions are described below.

### Why this is needed in the recovered source

- `../bone_mesh_contact.mjs` intentionally supplies point-to-closed-bone signed
  distance and a finite penalty. It permits penetration.
- `../embedded_surface_contact_fem.mjs` evaluates those penalties at fixed
  barycentric samples. It correctly disclaims unsampled and continuous contact.
- `../../reference-source/src/layered_fem.js:487–496` writes fixed targets and
  initial guesses before the first evaluation. Those writes bypass ordinary
  line-search acceptance.
- The same original file at approximately lines 510–524 builds a trial, checks
  actual energy and endpoint minimum J, then accepts it. Contact protection
  must cover the whole segment from the current state to this actual represented
  trial. Positive endpoint J is separately not a continuous determinant proof.

## What the helper implements

### Contact energy and exact analytic first derivatives

For a tissue triangle A and a static bone triangle B, it enumerates:

1. A's three vertices against B's face
2. B's three vertices against A's face
3. All nine edge pairs

Closest-point queries include lower-dimensional vertex/edge regions. With
`delta = closestA - closestB`, squared distance `d2`, and closest barycentrics
`wa`, `wb`, the envelope derivatives are `2 wa[i] delta` and
`-2 wb[j] delta`. Differentiating the closest barycentrics is unnecessary for
the squared-distance envelope. As usual, numerical evaluation can be
ill-conditioned and a genuinely nonunique nonsmooth feature has no unique
classical derivative; this is not an interval-certified gradient engine.

For `r = minimumDistance`, absolute activation distance `h > r`,
`s = d2 - r²`, and `H = h² - r²`, the prototype uses
`b(s) = -(s-H)² log(s/H)` for `0 < s < H`, zero for `s >= H`, and rejects
`s <= 0`. Its first and second scalar derivatives are implemented analytically.
The coefficient k has dimensions energy/length⁴ (force/length³), so the
existing nodal penalty coefficient must not simply be copied over.

For edge pairs, let `c = |u × v|²` and a fixed rest threshold
`epsilon = mollifierScale |u_rest|² |v_rest|²`. The multiplier is
`m(c) = (c/epsilon)(2-c/epsilon)` below epsilon and one above it. The gradient
includes BOTH `m b' grad(d2)` and `b grad(m)`. In particular,
`grad_u(c) = 2 v × (u × v)` and `grad_v(c) = 2 (u × v) × u`.
Rest thresholds remain fixed during differentiation. Fixed-bone gradient
entries are returned as reaction diagnostics, without introducing bone DOFs.

Returned positive rank-one metric blocks use `k m b'' grad(d2) grad(d2)^T`.
They omit geometric and mollifier Hessian terms. They are explicitly a PSD
search metric, not the exact Hessian. The total energy gradient includes the
omitted first-derivative terms. The original true-energy descent check remains
necessary.

This pair energy is deliberately not mesh-normalized. Adjacent triangle pairs
repeat global primitives, and lower-dimensional classifications can create
further duplicates. A real mesh assembly must canonicalize global VF and
undirected EE IDs, handle point/edge reductions consistently, and explicitly
choose weighting/stiffness. Blindly summing this pair class across all triangles
is geometrically guarded but creates tessellation-dependent mechanics.

### Whole-triangle continuous segment certificate

`certifyTriangleStep(start, end, bone, options)` checks the real linear
trajectory between supplied, finite binary64 vertex coordinates. The static
bone triangle is constant over the segment. For each time interval `[lo,hi]`:

1. Find candidate separating axes from approximate midpoint geometry. Include
   coordinate axes, face normals, edge cross products, coplanar axes and
   closest-feature differences. These are merely proof candidates.
2. For an arbitrary represented nonzero axis n, enclose the projections of all
   triangle vertices at both time endpoints using outward-rounded arithmetic.
   Linear motion and barycentric convexity mean their extrema bound EVERY point
   of the moving triangle throughout the time interval.
3. Bound the stationary triangle's projections. If the two projection intervals
   have a positive gap g and a downward-rounded `g²` exceeds an upward-rounded
   `r² |n|²`, the entire pair is strictly farther apart than r throughout that
   interval.
4. Otherwise bisect time. If the depth/work budget expires, reject. A false
   return means collision OR insufficient proof, never permission to proceed.

Accepted leaves must cover all of `[0,1]`. Optional `collectProof` returns their
time intervals and axes. The tests replay these using a separately implemented
exact BigInt dyadic arithmetic oracle. No closest-distance approximation,
square root, logarithm, finite temporal samples, or finite surface quadrature
is trusted as a separation certificate.

The proof assumes a conforming IEEE-754 binary64 JavaScript implementation with
gradual underflow and correctly rounded primitive arithmetic. Each +, -, and *
operation is enclosed with the adjacent representable numbers. Nonfinite
computations cannot certify. Very small gaps, very large coordinates, poor axes
and hard motion can cause conservative rejection. There is no promised bound
on false positives or useful time of impact. This original implementation has
tiny regression tests, not an independent formal or production robustness audit.

The optional `certifyBoundaryStep` harness exhaustively checks supplied face
pairs and rejects jobs over its default 4,096-pair budget. It is only a tiny-mesh
integration fixture, not the 37k-tet domain implementation.

## Practical next integration

1. Extract all boundary triangles from the connected tetrahedral volume,
   including material-facing inner surfaces. Check manifold connectivity and
   topology. Build unique boundary vertices/edges and global contact IDs.
2. Establish a feasible rest state before enabling a barrier. Check complete
   triangle separation and solid containment in both directions: boundary
   disjointness alone misses a component entirely inside the other material.
   Existing penetrating regional states cannot be made valid just by turning
   on this barrier. Repair or rebuild rest geometry separately, then recheck.
3. Build a static bone triangle BVH. Use conservatively padded swept tissue
   bounds for each proposed move. Prune only by a proven lower bound. A cached
   list of currently close pairs is insufficient for a large proposed step.
4. Assemble a unique primitive barrier potential at each evaluation, with
   exact first derivatives and a PSD search metric. Keep candidate discovery
   geometrically complete as features activate or deactivate.
5. In a new derived solver, preflight the prospective fixed-target/initial-guess
   state against a saved current state before calling the original `step`.
   Guard each trial before inherited trial evaluation and before acceptance.
   On an unresolved certificate return an invalid trial/Infinity and backtrack;
   on an invalid initial move report that the prescribed move is infeasible.
   Never silently mutate the already accepted positions during a failed check.
6. Keep the existing energy/minimum-J tests. If continuous inversion freedom
   is claimed, add a separate conservative determinant-path guard; this helper
   does not supply one. Test initial guesses, fixed targets, restarts and
   any manual coordinate update, not only Newton corrections.
7. A prescribed moving mandible needs motion-aware contact too. Frozen-pose
   static solves do not certify the transition between poses. Linear endpoint
   interpolation also does not certify a true rigid rotation arc.
8. Apply the same scrutiny to the high-resolution rendered skin. The existing
   affine transfer is a separate surface and does not inherit a FEM boundary
   nonintersection guarantee if it extrapolates outside the guarded boundary.
9. Start integration on a tiny closed tissue/bone fixture. Measure equilibrium
   residual, active force balance, minimum certified clearance, search rejection
   reasons and work counts. Then run a small connected anatomical patch. Only
   after those checks should a separate broad-domain solve be authorized/run.

## Verification

Run from `human-mechanics-20261010`:

    node --test tests/complete_boundary_contact.test.mjs

24 tests pass on Node v24.19.0. They cover both VF directions, a purely EE-close
configuration, known initial intersection despite positive values for all 15
primitive distances, endpoint tunneling, strict offset clearance, coplanar
overlap and grazing, resource-limit rejection, degenerate/nonfinite inputs,
gradient finite differences including the mollifier, bone reaction derivatives,
force/torque balance, rigid objectivity, PSD metric status, 96 known crossing
paths, 96 separated paths, and exact replay of accepted certificates. An accepted
three-leaf trajectory demonstrates subdivision, rather than only one separating
plane for the entire step. These tests do not test a real broad-phase BVH,
mesh-level primitive assembly, full FEM equilibrium or the broader anatomy.

## Primary references and scope of borrowing

- [Li et al., Incremental Potential Contact, 2020](https://ipc-sim.github.io/):
  the variational barrier/contact and feasible optimization architecture.
- [IPC technical supplement](https://cims.nyu.edu/gcl/papers/2020-IPC-A-Technical.pdf):
  barrier smoothness, near-parallel edge handling, duplicate primitive issues,
  conservative line-search feasibility and prescribed-boundary updates.
- [Wang et al., Tight Inclusion, 2021](https://continuous-collision-detection.github.io/tight_inclusion/):
  why floating-point CCD must be conservative; missing collisions and excessive
  false positives both matter. Their method is not implemented here.
- [Official Tight-Inclusion source documentation](https://github.com/Continuous-Collision-Detection/Tight-Inclusion):
  its numerical error filters, minimum separation and termination controls are
  essential parts of its contract, not optional informal epsilons.
- [IPC Toolkit](https://ipctk.xyz/): a production-oriented alternative supplying
  contact/derivative and CCD building blocks without replacing the tissue
  constitutive law or solver. Nothing was installed or imported for this work.

The interval swept-separation helper is our bounded design, not a copied IPC or
Tight-Inclusion algorithm. It trades sophistication/performance for an explicit
accept-only-with-proof contract. Full anatomical or biological validation does
not follow from these numerical references.
