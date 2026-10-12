/**
 * Experimental nodal 3x3 block-Jacobi preconditioner for the recovered FEM.
 *
 * Derived PCG flow: cybrdelic/cybr-human@c0e6d626dfa48c0b05657bd87b880cb7a6bc92a9.
 * Preserve reference-source/LICENSES/GPL-2.0.txt and LICENSING.md.
 * The reference source, energy, gradient, multiply(), constraints, line search,
 * determinant guard, and Euclidean residual stopping rule remain unchanged.
 * Only M^-1 r in PCG changes. This is not a different constitutive model.
 *
 * M consists of the exact 3x3 diagonal principal blocks of the current free-DOF
 * search metric. Diagonal entries come from the original assembly; off-diagonal
 * entries are assembled from its rank-one terms. Isotropic inertia/shear/point
 * springs have no within-node off-diagonal contribution. A principal block of
 * an SPD metric is SPD. Normalized Cholesky is used, with an explicit scalar
 * Jacobi fallback if floating-point factorization fails; no metric shift occurs.
 *
 * Scope: unmodified LayeredFEM through VolumetricMuscleFEM multiply methods.
 * An unknown metric extension is rejected rather than silently omitting terms.
 * Call evaluate(x, dt, true) before PCG, exactly as in the original solver.
 */
import { LayeredFEM } from "../reference-source/src/layered_fem.js";
import { ActiveFibreFEM } from "./active_fibre_fem.mjs";
import { EmbeddedSurfaceContactFEM } from "./embedded_surface_contact_fem.mjs";
import { VolumetricMuscleFEM } from "./volumetric_muscle_fem.mjs";

const supportedMultiply = new Set([LayeredFEM.prototype.multiply,
  ActiveFibreFEM.prototype.multiply, EmbeddedSurfaceContactFEM.prototype.multiply,
  VolumetricMuscleFEM.prototype.multiply]);

/** Packed symmetric block order: xx, xy, xz, yy, yz, zz. */
export function buildNodalBlockJacobi(s, workspace = null) {
  if (!supportedMultiply.has(s.multiply))
    throw Error("Nodal block Jacobi requires a supported, unmodified FEM multiply method");
  const n = s.mass.length;
  const w = workspace ?? { blocks: new Float64Array(n * 6),
    factors: new Float64Array(n * 6), scales: new Float64Array(n),
    fallback: new Uint8Array(n) };
  if (w.blocks.length !== n * 6 || w.factors.length !== n * 6 ||
      w.scales.length !== n || w.fallback.length !== n)
    throw Error("Block Jacobi workspace size mismatch");
  const B = w.blocks, L = w.factors;
  B.fill(0); w.fallback.fill(0); w.fallbackNodes = 0;
  for (let node = 0; node < n; node++) {
    const i = node * 6, j = node * 3;
    B[i] = s.diagonal[j]; B[i + 3] = s.diagonal[j + 1]; B[i + 5] = s.diagonal[j + 2];
  }
  const outerOffDiagonal = (node, x, y, z, k) => {
    if (s.fixed[node]) return;
    const i = node * 6;
    B[i + 1] += k * x * y; B[i + 2] += k * x * z; B[i + 4] += k * y * z;
  };
  for (let t = 0; t < s.nodes.length / 4; t++) {
    const k = s.material[t * 3] * s.effectiveBulk[t];
    for (let a = 0; a < 4; a++) {
      const i = t * 12 + a * 3;
      outerOffDiagonal(s.nodes[t * 4 + a], s.hydro[i], s.hydro[i + 1], s.hydro[i + 2], k);
    }
  }
  for (const h of s.hingeBlocks)
    for (let a = 0; a < 4; a++) outerOffDiagonal(h.ids[a], ...h.q[a], h.k);
  for (const b of s.contactBlocks) outerOffDiagonal(b.node, ...b.normal, b.k);
  for (const b of s.fibreBlocks ?? [])
    for (const { node, weight } of b.coefficients)
      outerOffDiagonal(node, ...b.direction, b.k * weight * weight);
  for (const b of s.surfaceContactBlocks ?? [])
    for (let a = 0; a < b.nodes.length; a++)
      outerOffDiagonal(b.nodes[a], ...b.normal, b.k * b.weights[a] ** 2);
  for (const b of s.muscleBlocks ?? [])
    for (let a = 0; a < b.nodes.length; a++)
      outerOffDiagonal(b.nodes[a], ...b.direction, b.k * b.coefficients[a] ** 2);

  for (let node = 0; node < n; node++) {
    const i = node * 6, xx = B[i], yy = B[i + 3], zz = B[i + 5];
    if (!(xx > 0 && yy > 0 && zz > 0 &&
        B.subarray(i, i + 6).every(Number.isFinite)))
      throw Error("Nonpositive or nonfinite FEM nodal block");
    const scale = w.scales[node] = Math.max(xx, yy, zz),
      l00 = Math.sqrt(xx / scale), l10 = B[i + 1] / scale / l00,
      l20 = B[i + 2] / scale / l00, pivot1 = yy / scale - l10 * l10,
      l11 = Math.sqrt(pivot1), l21 = (B[i + 4] / scale - l20 * l10) / l11,
      pivot2 = zz / scale - l20 * l20 - l21 * l21, l22 = Math.sqrt(pivot2);
    if (!(l00 > 0 && pivot1 > 0 && pivot2 > 0 && Number.isFinite(l21))) {
      // Scalar Jacobi remains SPD. This is a preconditioner-only fallback.
      w.fallback[node] = 1; w.fallbackNodes++;
      continue;
    }
    L[i] = l00; L[i + 1] = l10; L[i + 2] = l20;
    L[i + 3] = l11; L[i + 4] = l21; L[i + 5] = l22;
  }
  return w;
}

export function applyNodalBlockJacobi(w, r, out) {
  const L = w.factors, B = w.blocks;
  if (r.length !== w.scales.length * 3 || out.length !== r.length)
    throw Error("Block Jacobi vector size mismatch");
  for (let node = 0; node < w.scales.length; node++) {
    const i = node * 6, j = node * 3;
    if (w.fallback[node]) {
      out[j] = r[j] / B[i]; out[j + 1] = r[j + 1] / B[i + 3]; out[j + 2] = r[j + 2] / B[i + 5];
      continue;
    }
    const y0 = r[j] / L[i], y1 = (r[j + 1] - L[i + 1] * y0) / L[i + 3],
      y2 = (r[j + 2] - L[i + 2] * y0 - L[i + 4] * y1) / L[i + 5],
      z2 = y2 / L[i + 5], z1 = (y1 - L[i + 4] * z2) / L[i + 3],
      z0 = (y0 - L[i + 1] * z1 - L[i + 2] * z2) / L[i], scale = w.scales[node];
    out[j] = z0 / scale; out[j + 1] = z1 / scale; out[j + 2] = z2 / scale;
  }
}

/** May be explicitly assigned to a supported solver INSTANCE for an A/B test.
 * No prototype or legacy source is patched. The return is the CG iteration count,
 * as in the original. Build cost is included in the original step elapsed time. */
export function nodalBlockJacobiPCG(dt, limit, tolerance) {
  const { r, z, p, ap, direction: q, g } = this;
  const w = this.nodalBlockJacobi = buildNodalBlockJacobi(this, this.nodalBlockJacobi);
  q.fill(0);
  // Same Galerkin warm start as the original, with the same acceptance checks.
  if (this.useSearchSubspace && this.searchBasis.length) {
    const B = this.searchBasis, Y = B.map(b => {
      const y = new Float64Array(q.length); this.multiply(b, y, dt); return y;
    }), N = B.length, L = Array.from({ length: N }, () => new Float64Array(N)),
      rhs = new Float64Array(N);
    for (let a = 0; a < N; a++) {
      for (let j = 0; j < g.length; j++) rhs[a] -= B[a][j] * g[j];
      for (let b = 0; b <= a; b++) {
        let value = 0;
        for (let j = 0; j < g.length; j++) value += B[a][j] * Y[b][j];
        for (let k = 0; k < b; k++) value -= L[a][k] * L[b][k];
        if (a === b) {
          if (!(value > 1e-20)) throw Error("Dependent FEM search subspace");
          L[a][b] = Math.sqrt(value);
        } else L[a][b] = value / L[b][b];
      }
    }
    const coeff = new Float64Array(N), y = new Float64Array(N);
    for (let a = 0; a < N; a++) {
      let value = rhs[a];
      for (let b = 0; b < a; b++) value -= L[a][b] * y[b];
      y[a] = value / L[a][a];
    }
    for (let a = N - 1; a >= 0; a--) {
      let value = y[a];
      for (let b = a + 1; b < N; b++) value -= L[b][a] * coeff[b];
      coeff[a] = value / L[a][a];
    }
    for (let a = 0; a < N; a++) for (let j = 0; j < q.length; j++) q[j] += B[a][j] * coeff[a];
  }
  this.multiply(q, ap, dt);
  let initial = 0, fullNorm = 0;
  for (let i = 0; i < r.length; i++) {
    r[i] = -g[i] - ap[i]; initial += r[i] * r[i]; fullNorm += g[i] * g[i];
  }
  if (initial <= fullNorm * tolerance * tolerance || initial < 1e-30) return 0;
  applyNodalBlockJacobi(w, r, z);
  let rz = 0;
  for (let i = 0; i < r.length; i++) { p[i] = z[i]; rz += r[i] * z[i]; }
  for (let it = 0; it < limit; it++) {
    this.multiply(p, ap, dt);
    let den = 0;
    for (let i = 0; i < p.length; i++) den += p[i] * ap[i];
    if (!(den > 0)) throw Error("Non-positive FEM search metric");
    const alpha = rz / den;
    let norm = 0;
    for (let i = 0; i < r.length; i++) {
      q[i] += alpha * p[i]; r[i] -= alpha * ap[i]; norm += r[i] * r[i];
    }
    if (norm <= fullNorm * tolerance * tolerance) {
      this.rememberSearch(q); return it + 1;
    }
    applyNodalBlockJacobi(w, r, z);
    let next = 0;
    for (let i = 0; i < r.length; i++) next += r[i] * z[i];
    const beta = next / rz;
    for (let i = 0; i < p.length; i++) p[i] = z[i] + beta * p[i];
    rz = next;
  }
  this.rememberSearch(q);
  return limit;
}

export class NodalBlockJacobiVolumetricFEM extends VolumetricMuscleFEM {
  pcg(dt, limit, tolerance) { return nodalBlockJacobiPCG.call(this, dt, limit, tolerance); }
}
