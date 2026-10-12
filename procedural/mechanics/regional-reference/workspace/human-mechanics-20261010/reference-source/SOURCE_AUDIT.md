# Original mechanical source audit

Verified 2026-10-10 through the GitHub connector. Repository: cybrdelic/cybr-human. Pinned main commit: `c0e6d626dfa48c0b05657bd87b880cb7a6bc92a9` (2026-10-07 merge of PR #3). Every copied upstream file has its Git blob SHA recomputed and matched in `SOURCE_PROVENANCE.json`. Original files remain byte-for-byte unchanged. No atlas, scanned geometry, model binary, or upstream anatomical artifact was downloaded or reused.

## Reusable CPU core

- `src/layered_fem.js`: `LayeredFEM`, `compressionBarrier`, `installTissueBoundary`.
- `src/shell_bending.js`: sole dependency of the core; `hingeState`.
- `src/layered_fem_transfer_math.js`: volume-weighted nodal deformation-gradient recovery.
- `src/layered_fem_surface.js`: reference implementation of dense-surface affine transfer; it needs Three.js, but the mechanical core does not.
- `src/tissue_cpu_backend.js` and `src/tissue_cpu_worker.js`: browser worker wrapper, unnecessary for a bounded Node.js demonstration.

Instantiate `new LayeredFEM(model, arrays)`. Core `model` needs `schema: "cybr-layered-fem-v1"` and `nodes: N`; include `tetrahedra: T` and `surface_nodes: S` for surrounding tools. The core has no requirement that N=3S; that restriction belongs to the original full-head loader/renderer wrapper.

Required flat arrays:

| Field | Storage | Meaning |
|---|---|---|
| rest | Float64, 3N | Rest nodal positions |
| nodes | Uint32, 4T | Four global node IDs per tetrahedron |
| gradients | Float64, 12T | Four row-major 3-vectors, rest shape-function gradients |
| material | Float64, 3T | Rest volume V, shear mu, stable-Neo-Hookean lambda |
| mass | Float64, N | Positive lumped nodal masses |

The constructor recomputes gradient N0 = -(N1+N2+N3) and creates x, velocity, fixed mask, target positions, PCG buffers and precomputed shear blocks. Each shape gradient for local vertex1..3 is a row of the inverse of Dm=[X1-X0, X2-X0, X3-X0]. V=det(Dm)/6 must be positive; swap two tetrahedral node IDs before deriving gradients if necessary. The constructor's length checks do not replace a full mesh validity check.

Optional arrays: hinges[6H]=(n0,n1,n2,n3,theta0,k), faces[3F], layers[T], groups[S], normals[3S], fatWeights[T], fixedNodes[N], contactPlanes[7N]. Original loader uses Uint32 only for nodes/faces/layers/groups and Float64 for other fields. Assets metadata is `{url, sha256, count}` per array. Direct core construction bypasses the asset loader.

API:

- `pin(node,[x,y,z])`, `unpin(node)`; pinned coordinates are imposed when step begins.
- `evaluate(x,dt,blocks=true)` returns `{energy,minJ,maxJ,residual,contactViolation}`; analytic total gradient is left in `solver.g`. With blocks=true it refreshes the PCG metric/preconditioner. Pinned DOFs are zeroed in g.
- `step(dt,{iterations=8,cgIterations=32,cgTolerance=.01,forceTolerance=1e-5,minimumJ=.2,initialGuess=null})`. dt can be Infinity for quasistatics, otherwise 0<dt<=.05. Returns energy, J extrema, free residual, timing, PCG count, line-search rejections/accepted steps and convergence.
- `multiply(v,out,dt)` is the matrix-free SPD quasi-Newton metric; `pcg` searches against it.
- `gravity=[gx,gy,gz]`; `pointSprings=[{node,target,k}]`; `substrateContacts=[{node,point,normal,k,clearance}]`; `solidMeshContacts=[{node,collider}]` where collider.evaluate(point) supplies energy/gradient/normal/violation and collider.stiffness supplies the search block.

Original core source locations: constructor line22, evaluate114, multiply313, PCG362, step464, boundary installer550.

## Exact original constitutive law

Let F=I+sum_a (u_a-u_0) outer grad(N_a), a=1..3; J=det(F), C=cofactor(F). Let q=volumeTargets[t] or1, s=cuberoot(q), h=J-q-mu/(lambda*s), and kB=compressionStiffness[t] (initialized to the original material mu).

E_t = V [ .5 mu (F:F - 3s^2) + .5 lambda h^2 - mu^2/(2 lambda s^2) + kB b(J) ].

b(J)=0 for J>=.6; Infinity for J<=.2; otherwise b(J)=-(J-.6)^2 log((J-.2)/.4).

Writing d=J-.2, z=J-.6 and l=log(d/.4):
b'=-2z l-z^2/d; b''=-2l-4z/d+z^2/d^2.

Exact nodal gradient:
g_a += V [mu F + (lambda h+kB b') C] grad(N_a).

The search metric intentionally omits indefinite second-derivative terms:
H_ab = V mu (gradNa dot gradNb) I + V (lambda+kB b'') h_a outer h_b,
where h_a=C gradNa. The metric is not advertised as the exact Hessian. It is combined with positive inertia, hinge Gauss-Newton blocks and active contact/spring blocks.

At q=1 this is the stable Neo-Hookean rest law. The generator stores lambda=lambda_classic+mu, not just the ordinary first Lame parameter. Baseline authored skin E=120000, nu=.45; adipose E=18000, nu=.48. These are engineering parameters, not fitted biological calibration. If using design-unit geometry, convert to coherent units or explicitly use authored design-unit parameters; a design-unit number is not an anatomical measurement.

Other original energies: inertia sum .5 m/dt^2 |x-predict|^2; gravity -m g dot(x-X); point spring .5 k |x-target|^2; unilateral local support plane .5 k min(gap,0)^2; bending .5 k wrap(theta-theta0)^2. Dynamic velocity damping is 0.98 per1/60 s. Static dt=Infinity removes inertia and sets resulting velocity to zero.

Each step performs at most18 backtracks per nonlinear iteration. It accepts finite true total energy only when every J>minimumJ and Armijo decrease with coefficient1e-4 holds. Pinned boundary inversion throws. The `invertedCells` return field is a 0/1 indicator based on minimum J, not a count of inverted tetrahedra.

## Original tetrahedral construction

`tools/build_volume.py` lines301-313 sorts each top triangle's global IDs (a,b,c) and divides each successive layer prism into `(a,b,c,A)`, `(b,c,A,B)`, `(c,A,B,C)`. Uppercase IDs are the next-depth layer. Sorting gives consistent prism face diagonals. The original has three depth surfaces/two material layers, but the construction and core generalize to more. The original builder repairs its atlas-derived cage; do not run it as a native-geometry generator.

Original positive-cell orientation, V, gradients and mass assembly are lines349-367. The original material/mass values and geometry generation are separate from the reusable FEM equations.

## Boundaries and authored bone support

`installTissueBoundary` uses explicit fixedNodes when present; otherwise it pins the deepest third and a1mm bottom strip. Nonfixed nodes get a one-sided plane at their corresponding deepest-layer point, outward radial normal, k=20000 and tiny clearance. With contactPlanes each record supplies `(point3,normal3,clearance)` instead.

For a native demonstration, build an explicit fixed authored support mask or pin a small authored bone-origin node set. Do not call the default boundary installer on a five-depth-layer mesh without supplying the intended mask and plane mapping: it assumes original three-surface correspondence. A fixed authored origin can also be represented by the new fibre endpoint `{point:[x,y,z]}`. These are prescribed fixed support/attachments, not validated bone collision or reciprocal jaw mechanics. Existing local tangent contact does not establish global inside/outside or moving contact.

## Surface-binding schema and exact rest transfer

Original `surface-embedding.json` schema is `cybr-full-head-surface-embedding-v1` with parts `{index,vertex_count,embedding:{offset,count},active_vertex_count}`. Binary descriptors faces/vertex_faces/barycentric2 have byte_offset, byte_length, length, sha256. faces are Uint32 triangles; vertex_faces is Uint32 per dense vertex, with0xffffffff inactive; barycentric2 is Float32 first two weights, third=max(0,1-w0-w1), normalized. smooth_transfer adds eight node IDs followed by eight Wendland weights in sixteen Float32 values per vertex; runtime uses those to interpolate affine fields. Original support generation rejects geodesic shortcuts across closed lips/eyelids/folded ears.

`recoverSurfaceGradients(x,rest,nodes,gradients,material,S)` averages tetrahedral F with rest-volume weights at node IDs below S and returns Float32[9S], row-major. It works with more than three depth surfaces when the desired outer S nodes are first.

The original shader transfer is exactly:
q_deformed = sum_i w_i [x_i + F_i (q_rest-X_i)].

At rest x=X and F=I, so q_deformed=q_rest. A dense point does not need to lie on its coarse triangle. Its actual rest offset is transported through the mechanical affine field, requiring no sculpt correction. The new CPU helper `../src/affine_surface_transfer.mjs` implements that formula for three barycentric supports or a general support count; it reexports the unchanged original recovery helper. It is one-way skin transfer, not an additional elastic or collision model.

## Explicit active-fibre extension and verified checks

`../src/active_fibre_fem.mjs` subclasses the byte-identical original LayeredFEM. Each fibre uses endpoints a/b (node ID, nonnegative barycentric node weights summing to1, or an explicit fixed point), stiffness, optional restLength and activation in[0,1]. With ell=|p_b-p_a|, Ltarget=(1-a)L0 and e=max(ell-Ltarget,0), E=.5 k e^2. Exact free-DOF gradient is k e beta_n direction, where beta combines b-weights minus a-weights. Search blocks are k q q^T, q_nd=beta_n direction_d. The subclass adds energy and gradient for every line-search trial, updates the diagonal/blocks only when requested, masks fixed DOFs and recomputes the residual. It does not prescribe skin displacement or alter the original volume law.

Test commands:

    node human-mechanics-20261010/tests/active_fibre_fem.test.mjs --write-report
    node human-mechanics-20261010/tests/affine_surface_transfer.test.mjs

Tiny independent six-tetrahedron cube checks passed: total finite-difference gradient max relative error8.68e-9; internal force/torque balance near machine precision; slack branch and zero-activation rest parity; symmetric positive metric; pinned DOF handling; fixed origin gradient. Actual quasistatic activation shortened the embedded10mm segment by0.20945mm, minJ=.99321, residual2.77e-10 and converged. Transfer reproduced rest exactly and an arbitrary affine transform within2.87e-11m. These tests establish implementation behavior, not physiological validation or the native full-domain result. Report: `../results/active-fibre-unit-check.json`.

## License/provenance boundary

Upstream LICENSING.md assigns the inherited GPL version2 notice to original application code/tooling/tests; the complete notice is preserved. Separate atlas-derived data uses CC BY-SA4.0. This recovery copies source only; the existing native domain must remain independent of upstream anatomical assets. Preserve code notices and clearly label changes on redistribution. No assertion of ownership or licensing over independently supplied native geometry is made here.
