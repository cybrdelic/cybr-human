/**
 * Barycentric boundary-face contact on the original cybr-human CPU FEM.
 *
 * Source solver: cybrdelic/cybr-human@c0e6d626dfa48c0b05657bd87b880cb7a6bc92a9.
 * Preserve reference-source/LICENSES/GPL-2.0.txt and LICENSING.md with this work.
 * Inherits the original passive FEM, embedded active fibres, PCG, determinant
 * guard and actual-energy backtracking without changing any of those methods.
 *
 * A surface-contact record is:
 * { id?: string, nodes: [i,j,k], weights: [a,b,c], collider,
 *   quadratureWeight?: number, referenceArea?: number }
 * The three distinct nodes must be a boundary triangle of the volumetric mesh.
 * The barycentric weights are fixed material coordinates, nonnegative, with
 * partition of unity. Only floating-point-size sum error (<= 1e-12) is allowed.
 * collider uses the BoneMeshCollider evaluate()/stiffness contract.
 *
 * With p = sum_a weights[a] x[nodes[a]], s = quadratureWeight * referenceArea,
 * E = .5 s k min(phi(p) - clearance, 0)^2; g_a = s weights[a] dE_collider/dp.
 * The metric adds s k (weights tensor normal)(weights tensor normal)^T.
 * This is a positive-semidefinite Gauss-Newton search metric, not the exact
 * Hessian at curved closest-point features. The energy gradient is exact away
 * from signed-distance feature transitions and the interior medial axis.
 *
 * quadratureWeight defaults to 1 and is dimensionless. When referenceArea is
 * omitted, collider stiffness k is force/length, as in ordinary nodal contact.
 * When referenceArea is supplied, it is CONSTANT REST area in length^2 and k
 * must be an area-penalty density in force/length^3. Sum quadratureWeight to 1
 * on each triangle if integrating over that triangle's supplied rest area.
 * Do not reuse that area-density collider for unscaled nodal contact. Current
 * deformed area is deliberately not used: its derivative would add terms.
 *
 * This is finite-penalty, finite-sample, one-way rigid-bone contact. It can
 * catch interior-face penetrations missed by nodes, but does NOT certify
 * nonintersection between samples, continuous collision, or tissue self-contact.
 * Reported fixed-bone reactions are diagnostic loads, not bone dynamics.
 */
import { ActiveFibreFEM } from "./active_fibre_fem.mjs";

const faceKey = nodes => nodes.slice().sort((a, b) => a - b).join(":");
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];

function boundaryFaces(cells) {
  const counts = new Map();
  for (let t = 0; t < cells.length; t += 4) {
    const tet = Array.from(cells.subarray ? cells.subarray(t, t + 4) : cells.slice(t, t + 4));
    for (let omit = 0; omit < 4; omit++) {
      const key = faceKey(tet.filter((_, i) => i !== omit));
      counts.set(key, (counts.get(key) || 0) + 1);
    }
  }
  return new Set(Array.from(counts).filter(([, count]) => count === 1).map(([key]) => key));
}

function contactPosition(contact, x) {
  const p = [0, 0, 0];
  for (let a = 0; a < 3; a++)
    for (let d = 0; d < 3; d++) p[d] += contact.weights[a] * x[contact.nodes[a] * 3 + d];
  return p;
}

export class EmbeddedSurfaceContactFEM extends ActiveFibreFEM {
  constructor(model, arrays, fibres = [], surfaceContacts = []) {
    super(model, arrays, fibres);
    this.surfaceContactBlocks = [];
    this.surfaceContactDiagnostics = [];
    this.setSurfaceContacts(surfaceContacts);
  }

  setSurfaceContacts(records) {
    if (!Array.isArray(records)) throw Error("Surface contacts must be an array");
    if (records.length && !this.surfaceBoundaryFaces)
      this.surfaceBoundaryFaces = boundaryFaces(this.nodes);
    const seen = new Set();
    const contacts = records.map((record, index) => {
      const nodes = Array.from(record.nodes || []), weights = Array.from(record.weights || []),
        id = record.id ?? String(index), collider = record.collider,
        quadratureWeight = record.quadratureWeight ?? 1, referenceArea = record.referenceArea ?? 1;
      if (nodes.length !== 3 || new Set(nodes).size !== 3 ||
          !nodes.every(n => Number.isInteger(n) && n >= 0 && n < this.mass.length) ||
          !this.surfaceBoundaryFaces.has(faceKey(nodes)))
        throw Error("Surface contact nodes must identify one volumetric boundary triangle");
      const sum = weights.reduce((a, b) => a + b, 0);
      if (weights.length !== 3 || !weights.every(w => Number.isFinite(w) && w >= 0) ||
          Math.abs(sum - 1) > 1e-12)
        throw Error("Surface contact weights require a nonnegative partition of unity");
      // Remove only input roundoff. Set the largest weight from the other two,
      // so a near-zero weight cannot become negative through normalization.
      for (let a = 0; a < 3; a++) weights[a] /= sum;
      const largest = weights.indexOf(Math.max(...weights));
      weights[largest] = 1 - weights.filter((_, a) => a !== largest).reduce((a, b) => a + b, 0);
      if (!(Number.isFinite(quadratureWeight) && quadratureWeight > 0) ||
          !(Number.isFinite(referenceArea) && referenceArea > 0) ||
          !collider || typeof collider.evaluate !== "function" ||
          !(Number.isFinite(collider.stiffness) && collider.stiffness > 0) ||
          !Number.isFinite(quadratureWeight * referenceArea * collider.stiffness))
        throw Error("Surface contact requires positive finite weighting and a penalty collider");
      if (seen.has(id)) throw Error("Duplicate surface contact ID");
      seen.add(id);
      return { id, nodes, weights, collider, quadratureWeight,
        referenceArea: record.referenceArea, scale: quadratureWeight * referenceArea };
    });
    this.surfaceContacts = contacts;
    this.surfaceContactBlocks.length = 0;
    this.surfaceContactDiagnostics.length = 0;
    this.searchBasis.length = 0;
    return this;
  }

  evaluate(x, dt, blocks = true) {
    const state = super.evaluate(x, dt, blocks);
    if (blocks) this.surfaceContactBlocks.length = 0;
    let surfaceContactEnergy = 0, surfaceContactViolation = 0, activeSurfaceContacts = 0;
    const surfaceBoneReaction = [0, 0, 0], surfaceBoneReactionTorque = [0, 0, 0],
      surfaceTissueContactForce = [0, 0, 0], surfaceTissueContactTorque = [0, 0, 0], diagnostics = [];
    for (const contact of this.surfaceContacts) {
      const { id, nodes, weights, collider, scale } = contact, position = contactPosition(contact, x),
        query = collider.evaluate(position), energy = scale * query.energy,
        gradient = query.gradient.map(v => scale * v), k = scale * collider.stiffness;
      surfaceContactEnergy += energy;
      surfaceContactViolation = Math.max(surfaceContactViolation, query.violation);
      const boneTorque = cross(position, gradient);
      for (let d = 0; d < 3; d++) {
        surfaceBoneReaction[d] += gradient[d];
        surfaceBoneReactionTorque[d] += boneTorque[d];
      }
      diagnostics.push({ id, position, energy, violation: query.violation,
        signedDistance: query.signedDistance, normal: query.normal.slice(),
        closestPoint: query.closestPoint?.slice(), triangle: query.triangle,
        effectiveStiffness: k, boneReaction: gradient.slice() });
      if (query.violation <= 0) continue;
      activeSurfaceContacts++;
      for (let a = 0; a < 3; a++) {
        const node = nodes[a], weight = weights[a], j = node * 3,
          force = gradient.map(v => -weight * v),
          torque = cross([x[j], x[j + 1], x[j + 2]], force);
        for (let d = 0; d < 3; d++) {
          // Include pinned-node contact loads in diagnostic force/torque sums.
          // They are excluded only from the free-DOF gradient and search solve.
          surfaceTissueContactForce[d] += force[d];
          surfaceTissueContactTorque[d] += torque[d];
          if (this.fixed[node]) continue;
          this.g[j + d] += weight * gradient[d];
          if (blocks) this.diagonal[j + d] += k * (weight * query.normal[d]) ** 2;
        }
      }
      if (blocks) this.surfaceContactBlocks.push({ nodes, weights, normal: query.normal, k });
    }
    let residual2 = 0;
    for (let node = 0; node < this.mass.length; node++)
      for (let d = 0; d < 3; d++) {
        const j = node * 3 + d;
        if (this.fixed[node]) {
          this.g[j] = 0;
          if (blocks) this.diagonal[j] = 1;
        } else residual2 += this.g[j] ** 2;
      }
    this.surfaceContactDiagnostics = diagnostics;
    return { ...state, energy: state.energy + surfaceContactEnergy,
      passiveAndExternalEnergy: state.passiveAndExternalEnergy + surfaceContactEnergy,
      energyWithoutSurfaceContacts: state.energy, surfaceContactEnergy, surfaceContactViolation,
      contactViolation: Math.max(state.contactViolation, surfaceContactViolation),
      surfaceContactCount: this.surfaceContacts.length, activeSurfaceContacts,
      surfaceBoneReaction, surfaceBoneReactionTorque, surfaceTissueContactForce,
      surfaceTissueContactTorque, residual: Math.sqrt(residual2) };
  }

  /** Inherited metric plus positive-semidefinite boundary-contact blocks. */
  multiply(v, out, dt) {
    super.multiply(v, out, dt);
    for (const { nodes, weights, normal, k } of this.surfaceContactBlocks) {
      let projection = 0;
      for (let a = 0; a < 3; a++) {
        if (this.fixed[nodes[a]]) continue;
        for (let d = 0; d < 3; d++) projection += weights[a] * normal[d] * v[nodes[a] * 3 + d];
      }
      for (let a = 0; a < 3; a++) {
        if (this.fixed[nodes[a]]) continue;
        for (let d = 0; d < 3; d++) out[nodes[a] * 3 + d] += k * projection * weights[a] * normal[d];
      }
    }
  }
}

export default EmbeddedSurfaceContactFEM;
