# Experimental qualified inverse-BFGS search

This standalone experiment qualifies inverse-BFGS history around one existing PCG search. It does not integrate into any runtime, change defaults, promote a model, or alter energy, force, material, contact or constraint equations. Its current-metric check is conservative: the metric need not be the exact physical Hessian, and the unchanged actual-energy/contact line search remains authoritative.

## Run the focused tests

From repository root:

```sh
node experiments/human-qualified-search-v3/tests/test_qualified_secant_admission_v3.mjs
```

The test prints its quantitative assertions and pass summary. It uses tiny synthetic systems and performs no nonlinear state solve or large-mesh probe. The three-file patch contains only this README, the source and executable JavaScript test. There are no mesh/state/binary assets or additional generated result dependencies.

A separately preserved large-model opposing-mode witness motivated conditional qualification: its historical correction worsened agreement with the unchanged metric and increased the quadratic model while plain PCG gave a much smaller linear residual. That external witness and its numerical regression are not included or required here. The tests in this patch check admission and control flow, not full-model convergence or physiological validity.

## Required integration contract

Call `installQualifiedSecantSearch(s, options)` explicitly. The solver provides `x`, `g`, `direction`, `diagonal`, `pcg(dt, limit, tolerance)` and `multiply(direction, out, dt)`. A three-dimensional solver with fixed nodes also provides a binary node mask `fixed` and full finite target array `target`.

The required `contextFingerprint` callback must cover mesh/array identity, materials and constitutive implementation, constraints and actual fixed targets, contact law and obstacle geometry, current activation, and any externally modified dynamic metric configuration. The qualifier additionally tracks dt itself; changing dt invalidates history. Dynamic mass/inertia and other mutable metric parameters belong in the caller's fingerprint. The `contactSignature` callback must identify every active free-node contact feature, for example sorted node ID, obstacle triangle and closest-point feature.

The fixed-node mask and solver dimensions are captured at wrapper installation. Install the wrapper only once per solver instance. Changing either requires constructing a fresh solver instance and installing once with fresh compatible history; reinstalling onto an already wrapped solver would stack wrappers. Ordinary fixed-target or physical-context changes still require an updated fingerprint and invalidate stored history. The caller must maintain current positions at fixed targets and fixed gradient entries at zero.

## Checkpoint admission

Only schema `qualified-fem-secant-v3` is accepted. Legacy v1/v2 and untagged checkpoints cannot migrate implicitly. Restoration is atomic: a malformed record clears the entire checkpoint. Each record must contain finite, correctly sized dx/dy vectors; positive curvature above the fresh-pair threshold; and finite positive rho satisfying `abs(rho * dot(dx,dy) - 1) <= 1e-12`. Rho is then recomputed. Nonempty records require paired finite previous positions and gradients plus contact and metric signatures. Missing endpoints cannot suppress contact invalidation.

Restored dx/dy must have zero fixed-DOF entries. Fixed endpoint positions must equal current targets; fixed endpoint gradients must be zero. Current fixed positions/gradients are verified before search. Every selected and fallback direction is explicitly projected to exact positive zero at fixed DOFs, because the downstream trial constructor may apply `x + fraction * direction` to all entries without another target projection.

## Direction qualification and budget

Finite positions/gradient and strictly positive finite diagonals are required. Finite-array and diagonal checks use allocation-free loops. A nonzero gradient receives exactly one underlying PCG call. A zero gradient returns a finite zero direction without PCG. A candidate must have finite original-metric diagnostics, negative slope and negative quadratic model change. A rejected historical candidate clears records and reuses the inner PCG direction only when it meets those same checks. Otherwise a finite scaled-Jacobi descent is required, or search fails closed. Qualification adds no PCG iterations.

## Test coverage

The tests retain useful history on a valid quadratic rather than disabling acceleration globally. They reject arbitrary rho, nonpositive curvature, malformed/missing endpoints and legacy checkpoints; invalidate context/contact/dt changes; exercise exact zero gradient, rejected-candidate inner reuse, unsafe-inner Jacobi fallback and nonfinite failures; and check one-PCG accounting. A positive-curvature, reciprocal-rho-valid malicious fixed secant is rejected, and downstream trials preserve fixed target coordinates exactly at multiple line-search fractions.
