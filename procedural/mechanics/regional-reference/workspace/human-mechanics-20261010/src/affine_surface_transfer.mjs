/** CPU form of the original cybr-human layered_fem_surface.js shader formula.
 * Σ_i w_i [x_i + F_i (q-X_i)]: the supplied rest position q is reproduced
 * exactly at rest. The rest residual is transported mechanically, not sculpted.
 * All positions and FEM arrays must use the same coordinate frame and units.
 * Source/terms: reference-source/SOURCE_PROVENANCE.json and LICENSES/GPL-2.0.txt.
 * One-way rendering transfer; no additional contact or thin-feature mechanics.
 */
export { recoverSurfaceGradients } from "../reference-source/src/layered_fem_transfer_math.js";

/** Flat arrays: points[3M], indices[KM], weights[KM], x/rest[3N], F[9S]. */
export function applyAffineTransfer(points, indices, weights, x, rest, F,
  nodesPerVertex = 3, out = new Float64Array(points.length)) {
  if (!Number.isInteger(nodesPerVertex) || nodesPerVertex < 1 || points.length % 3 ||
      indices.length !== points.length / 3 * nodesPerVertex ||
      weights.length !== indices.length || out.length !== points.length)
    throw Error("Invalid affine surface binding layout");
  for (let p = 0; p < points.length / 3; p++) {
    let sum = 0;
    out[p * 3] = out[p * 3 + 1] = out[p * 3 + 2] = 0;
    for (let a = 0; a < nodesPerVertex; a++) {
      const j = p * nodesPerVertex + a, n = indices[j], w = weights[j];
      if (!Number.isInteger(n) || n < 0 || n * 9 + 8 >= F.length ||
          n * 3 + 2 >= x.length || n * 3 + 2 >= rest.length || !Number.isFinite(w))
        throw Error("Invalid affine surface support");
      sum += w;
      for (let d = 0; d < 3; d++) {
        let value = x[n * 3 + d];
        for (let k = 0; k < 3; k++)
          value += F[n * 9 + d * 3 + k] * (points[p * 3 + k] - rest[n * 3 + k]);
        out[p * 3 + d] += w * value;
      }
    }
    if (Math.abs(sum - 1) > 1e-6) throw Error("Affine transfer weights must sum to one");
    // Normalize representation rounding, matching the original barycentric shader.
    for (let d = 0; d < 3; d++) out[p * 3 + d] /= sum;
  }
  return out;
}
