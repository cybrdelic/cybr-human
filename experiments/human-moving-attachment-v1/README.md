# Experimental moving-root attachment certificate

This is an explicit opt-in feasibility filter for a linearly moving tissue triangle against static bone. It preserves the existing stationary attachment policy and does not install a runtime/default, change a constitutive law, add contact forces, or qualify an anatomical model.

The scoped BVH and FEM wrapper copies are needed because the original BVH traversal is private. Existing procedural mechanics modules are imported from `procedural/mechanics/regional-reference/workspace`; their hashes are recorded in `DEPENDENCIES.json`. No original source file is modified by this experiment.

## Run from the repository root

Tested with Node 24.19.0. No package install, external asset, network call, or authored head is needed for these synthetic tests.

```sh
node experiments/human-moving-attachment-v1/tests/test_moving_attachment.mjs
node experiments/human-moving-attachment-v1/tests/test_guarded_regression.mjs
node experiments/human-moving-attachment-v1/tests/test_parameterized_bonds.mjs
```

The suites cover 34 focused attachment checks, 25 inherited guard checks, and the original stationary-policy regression. They include a shared root and edge, safe coplanar outward motion, free-vertex crossing, clear endpoints with an interior-time crossing, an unrelated incident-face collision, subnormal positive gaps, barycentric source roots, malformed inputs, forged/stale authority, altered pins/topology/collider, complete-boundary enumeration, and synthetic inherited energy/gradient/PCG equivalence. Deterministic sample checks corroborate the implementation; they are not the continuous-motion proof.

## Opt-in use

Import `StaticBoneSweptBVH` and `extractCompleteFEMBoundary` from `src/static_bone_swept_bvh.mjs`, `withStaticBoneFeasibility` from `src/guarded_static_bone_fem.mjs`, and `createVerifiedMovingBondedPatch` from `src/moving_bonded_patch.mjs`.

Create a fresh immutable bone BVH and a token using the solver's current positions, complete extracted boundary, fixed mask, targets, explicit source root records, root-cap faces and requested `minimumDistance`. Pass that token as `movingBondedPatch` to `setStaticBoneGuard`. Do not also pass `distributedBondedPatch`. The latter still selects the existing all-fixed stationary attachment policy.

The FEM wrapper requires the exact inherited reference `LayeredFEM.step`. It independently extracts the complete tetrahedral boundary before admission. A standalone collider caller is responsible for supplying the complete boundary. No token or public summary can certify a face that was omitted by the caller.

Any change to supports, fixed targets, root declarations, collider, margin or topology requires a fresh admitted token. A search-history wrapper has its own independent context and fresh-solver requirements; this contact module does not weaken them.

## Sufficient exact proof

1. Finite binary64 coordinates are converted exactly to a common rational lattice. A registered barycentric root is the exact normalized weighted point on its immutable source triangle, with its binary64 weights interpreted exactly.
2. A tissue triangle with linear vertex paths lies in the convex hull of its six endpoint vertices. No intermediate-time sampling is needed for that containment.
3. Candidate separating axes are generated lazily, in a deterministic order. Each candidate is checked with exact projections. Axis generation is only a search strategy; an axis is never trusted without its exact witness check.
4. Strict separation of the full endpoint hull from the static bone triangle proves positive separation for the entire path. It does not prove a particular metric clearance.
5. Otherwise, every endpoint must be on the tissue side of valid weak separating planes, and every non-root endpoint must be strictly on that side of at least one. Their sum is rechecked exactly. Any contact point can therefore use only the registered fixed roots.
6. Exact endpoint triangle/triangle feature clipping establishes that the fixed root hull meets this bone triangle only in its registered source-contact feature. Thus a root-root chord cannot cross an unrelated bone triangle and acquire an unregistered exemption. A shared root edge or cap can remain bonded while adjacent free vertices move away.
7. Each candidate bone pair on a moving root-incident face receives this proof anew. No face, region, bone, normal direction or unknown collision is blanket-exempted. Other moving faces retain the inherited full-margin interval certificate. Failure to find a sufficient witness or exhaustion of a traversal budget is rejection.

The convex endpoint hull can be larger than the true swept surface, so this method can conservatively reject a safe path. The lazy implementation only avoids constructing later axes after an earlier witness succeeds; it does not alter candidate order, accepted witnesses, rejection logic or physical laws.

## Authority and numerical conventions

- WeakMap-backed tokens bind immutable bone identity, copied ordered topology and source records, the fixed mask, all admitted fixed coordinates/targets, and requested clearance. Forged/cloned tokens and changed contexts fail closed.
- Literal bone-vertex roots must match their source binary64 coordinates exactly.
- Barycentric roots retain the existing representation allowance of at most `2e-15` solver units, measured by exact rational Euclidean comparison. This is a source representation bound, not a collision tolerance. The proof is about source-parametric roots; rounded solver coordinates are not claimed to lie literally on a bone plane.
- Outward-rounded candidate boxes include that bound, even for a zero requested ordinary margin.
- Individually proved attachment pairs have source-feature contact or exact positive separation. There is no whole-model uniform margin claim. Other moving pairs still use the requested margin.

## What is not certified

This is not a force law, equilibrium test, tissue self-contact certificate, continuous positive-Jacobian proof, solid-containment proof, moving-bone model, friction model, or anatomical calibration. Positive Jacobians and bone clearance alone do not prevent tissue self-intersection. Those requirements must be handled separately by any consuming experiment.

Larger authored-domain neutral admissions and support-sensitivity studies are retained as separate evidence checkpoints. Their states, assets, private workspace paths and solver runners are intentionally absent from this small source/test patch. Passing the tests above does not qualify those studies or promote this experiment into the runtime.

## Licensing

The existing recovered FEM and contact dependencies remain under their repository licensing. Preserve `procedural/mechanics/regional-reference/workspace/human-mechanics-20261010/LICENSING.md` and `reference-source/LICENSES/GPL-2.0.txt` when redistributing the corresponding source. This patch does not replace those notices.
