/**
 * Authored volumetric active-fibre energy on the original cybr-human CPU FEM.
 *
 * Source solver: cybrdelic/cybr-human@c0e6d626dfa48c0b05657bd87b880cb7a6bc92a9.
 * Preserve reference-source/LICENSES/GPL-2.0.txt and LICENSING.md with this work.
 * This is an uncalibrated active-fibre phenomenology, NOT a Hill muscle model.
 * It has no measured force-length/velocity response, tendon law or activation
 * dynamics. The inherited passive solid supplies the transverse/bulk response.
 * No displacement target, new volume constraint or external force is imposed.
 *
 * Region schema:
 * { id?: string, tetrahedra: number[], stiffness: number, activation?: number,
 *   fibreDirection: [x,y,z] OR fibreDirections: [[x,y,z], ...] }
 * Directions are REST/material coordinates, normalized on input; the optional
 * per-tet list follows tetrahedra order. Each tetrahedron belongs to at most one
 * region. stiffness is energy / rest volume (stress), not cable stiffness.
 * Regions are configured through setMuscleRegions(), not by mutating records.
 * Origin attachments are ordinary inherited pins; a soft insertion uses shared
 * volumetric nodes with the surrounding tissue, distributing the internal load.
 *
 * For each tagged tetrahedron, f0 is its unit rest fibre and V its rest volume:
 * s = ||F f0||, e = max(s - (1-a), 0), E = .5 V k e^2.
 * With b_i = grad(N_i) dot f0 and n = (F f0)/s, g_i = V k e b_i n.
 * The search metric adds the PSD Gauss-Newton block V k (b tensor n)^2.
 * This omits the geometric Hessian term; actual-energy line search remains
 * authoritative. Activation zero is energy/force-free at rest and in fibre
 * compression, but EXTENDED inactive fibres still add anisotropic stiffness.
 *
 * Gradient storage follows the recovered source exactly:
 * gradients[t*12 + localNode*3 + materialAxis]. F is row-major spatial/material.
 * Like the source, F = I + sum_{i=1..3} (u_i-u_0) tensor grad(N_i).
 * Using displacement differences preserves exact identity at stored rest and
 * avoids translation cancellation; finite-precision rest gradients bound the
 * numerical accuracy of rigid objectivity, as in the inherited passive model.
 */
import { EmbeddedSurfaceContactFEM } from "./embedded_surface_contact_fem.mjs";

function activationValue(value) {
  if (!Number.isFinite(value) || value < 0 || value > 1)
    throw Error("Muscle activation must lie in [0,1]");
  return value;
}

function unitDirection(value) {
  if (!value || value.length !== 3 || !Array.from(value).every(Number.isFinite))
    throw Error("Muscle fibre direction must be a finite nonzero 3-vector");
  const length = Math.hypot(...value);
  if (!(Number.isFinite(length) && length > 1e-12))
    throw Error("Muscle fibre direction must be a finite nonzero 3-vector");
  return Array.from(value, v => v / length);
}

export class VolumetricMuscleFEM extends EmbeddedSurfaceContactFEM {
  constructor(model, arrays, fibres = [], surfaceContacts = [], muscleRegions = []) {
    super(model, arrays, fibres, surfaceContacts);
    this.muscleBlocks = [];
    this.muscleDiagnostics = [];
    this.muscleTetDiagnostics = [];
    // Internal forces before the Dirichlet projection, INCLUDING pinned nodes.
    // These are muscle-only loads, not the total bone/support reaction.
    this.muscleForces = new Float64Array(this.x.length);
    this.setMuscleRegions(muscleRegions);
  }

  setMuscleRegions(records) {
    if (!Array.isArray(records)) throw Error("Muscle regions must be an array");
    const seen = new Set(), assigned = new Set(), cellCount = this.nodes.length / 4;
    const regions = records.map((record, index) => {
      if (!record || typeof record !== "object") throw Error("Invalid muscle region");
      const id = record.id ?? String(index), tetrahedra = Array.from(record.tetrahedra || []),
        stiffness = record.stiffness, activation = activationValue(record.activation ?? 0);
      if (typeof id !== "string" || !id.length) throw Error("Muscle region ID must be a nonempty string");
      if (seen.has(id)) throw Error("Duplicate muscle region ID");
      seen.add(id);
      if (!tetrahedra.length || !tetrahedra.every(t => Number.isInteger(t) && t >= 0 && t < cellCount))
        throw Error("Muscle region requires valid tetrahedron IDs");
      if (!(Number.isFinite(stiffness) && stiffness > 0))
        throw Error("Muscle stiffness must be positive and finite");
      const perTet = record.fibreDirections !== undefined;
      if (perTet === (record.fibreDirection !== undefined) ||
          (perTet && (!Array.isArray(record.fibreDirections) || record.fibreDirections.length !== tetrahedra.length)))
        throw Error("Supply either one muscle fibreDirection or one fibreDirections entry per tetrahedron");
      const common = perTet ? null : unitDirection(record.fibreDirection);
      const elements = tetrahedra.map((tetrahedron, localIndex) => {
        if (assigned.has(tetrahedron)) throw Error("Muscle tetrahedron cannot be assigned more than once");
        assigned.add(tetrahedron);
        const fibreDirection = perTet ? unitDirection(record.fibreDirections[localIndex]) : common.slice(),
          nodes = Array.from(this.nodes.slice(tetrahedron * 4, tetrahedron * 4 + 4)),
          restVolume = this.material[tetrahedron * 3], coefficients = [0, 0, 0, 0],
          k = restVolume * stiffness, go = tetrahedron * 12;
        if (!(Number.isFinite(restVolume) && restVolume > 0 && Number.isFinite(k) && k > 0))
          throw Error("Muscle tetrahedron requires positive finite rest volume and weighted stiffness");
        for (let a = 1; a < 4; a++) {
          for (let d = 0; d < 3; d++) coefficients[a] += this.gradients[go + a * 3 + d] * fibreDirection[d];
          coefficients[0] -= coefficients[a];
        }
        if (!coefficients.every(Number.isFinite)) throw Error("Invalid muscle tetrahedron gradients");
        return { tetrahedron, nodes, fibreDirection, restVolume, coefficients, k };
      });
      return { id, stiffness, activation, tetrahedra, elements,
        restVolume: elements.reduce((sum, element) => sum + element.restVolume, 0) };
    });
    this.muscleRegions = regions;
    this.muscleBlocks.length = 0;
    this.muscleDiagnostics.length = 0;
    this.muscleTetDiagnostics.length = 0;
    this.muscleForces.fill(0);
    this.searchBasis.length = 0;
    return this;
  }

  /** Omit selector to change all regions; otherwise use an index or exact ID.
   * This does NOT change the inherited cable-fibre activations. */
  setMuscleActivation(value, selector = null) {
    value = activationValue(value);
    if (selector === null) {
      for (const region of this.muscleRegions) region.activation = value;
    } else {
      const region = Number.isInteger(selector) ? this.muscleRegions[selector]
        : this.muscleRegions.find(r => r.id === selector);
      if (!region) throw Error("Unknown muscle region selector");
      region.activation = value;
    }
    return this;
  }

  evaluate(x, dt, blocks = true) {
    const state = super.evaluate(x, dt, blocks);
    if (blocks) this.muscleBlocks.length = 0;
    this.muscleForces.fill(0);
    let muscleEnergy = 0, muscleRestVolume = 0, muscleVolume = 0, activeMuscleTetrahedra = 0;
    const diagnostics = [], tetDiagnostics = [], F = new Float64Array(9);
    for (const region of this.muscleRegions) {
      const activation = activationValue(region.activation), targetStretch = 1 - activation,
        record = { id: region.id, activation, targetStretch, tetrahedronCount: region.elements.length,
          activeTetrahedra: 0, restVolume: region.restVolume, currentVolume: 0, energy: 0,
          meanStretch: 0, minStretch: Infinity, maxStretch: -Infinity, minJ: Infinity, maxJ: -Infinity };
      for (const element of region.elements) {
        const { tetrahedron, nodes, fibreDirection, restVolume, coefficients, k } = element,
          go = tetrahedron * 12, anchor = nodes[0] * 3;
        F.fill(0);
        F[0] = F[4] = F[8] = 1;
        for (let a = 1; a < 4; a++) {
          const node = nodes[a] * 3;
          for (let d = 0; d < 3; d++) {
            const u = x[node + d] - this.rest[node + d] - (x[anchor + d] - this.rest[anchor + d]);
            for (let c = 0; c < 3; c++) F[d * 3 + c] += u * this.gradients[go + a * 3 + c];
          }
        }
        const fibre = [0, 1, 2].map(d => F[d * 3] * fibreDirection[0] + F[d * 3 + 1] * fibreDirection[1] + F[d * 3 + 2] * fibreDirection[2]),
          stretch = Math.hypot(...fibre), extension = Math.max(stretch - targetStretch, 0),
          energy = .5 * k * extension * extension,
          J = F[0] * (F[4] * F[8] - F[5] * F[7]) + F[1] * (F[5] * F[6] - F[3] * F[8]) + F[2] * (F[3] * F[7] - F[4] * F[6]);
        record.currentVolume += restVolume * J;
        record.energy += energy;
        record.meanStretch += restVolume * stretch;
        record.minStretch = Math.min(record.minStretch, stretch);
        record.maxStretch = Math.max(record.maxStretch, stretch);
        record.minJ = Math.min(record.minJ, J);
        record.maxJ = Math.max(record.maxJ, J);
        tetDiagnostics.push({ region: region.id, tetrahedron, activation, targetStretch,
          stretch, extension, J, restVolume, currentVolume: restVolume * J, energy,
          fibreStress: region.stiffness * extension });
        if (extension === 0) continue;
        record.activeTetrahedra++;
        const direction = fibre.map(v => v / stretch), magnitude = k * extension;
        for (let a = 0; a < 4; a++)
          for (let d = 0; d < 3; d++) {
            const j = nodes[a] * 3 + d, q = coefficients[a] * direction[d], gradient = magnitude * q;
            this.muscleForces[j] -= gradient;
            if (this.fixed[nodes[a]]) continue;
            this.g[j] += gradient;
            if (blocks) this.diagonal[j] += k * q * q;
          }
        if (blocks) this.muscleBlocks.push({ nodes, coefficients, direction, k });
      }
      record.meanStretch /= record.restVolume;
      record.meanJ = record.currentVolume / record.restVolume;
      diagnostics.push(record);
      muscleEnergy += record.energy;
      muscleRestVolume += record.restVolume;
      muscleVolume += record.currentVolume;
      activeMuscleTetrahedra += record.activeTetrahedra;
    }
    let residual2 = 0;
    for (let n = 0; n < this.mass.length; n++)
      for (let d = 0; d < 3; d++) {
        const j = n * 3 + d;
        if (this.fixed[n]) {
          this.g[j] = 0;
          if (blocks) this.diagonal[j] = 1;
        } else residual2 += this.g[j] ** 2;
      }
    this.muscleDiagnostics = diagnostics;
    this.muscleTetDiagnostics = tetDiagnostics;
    return { ...state, energy: state.energy + muscleEnergy, energyWithoutMuscles: state.energy,
      muscleEnergy, muscleRestVolume, muscleVolume, muscleMeanJ: muscleRestVolume ? muscleVolume / muscleRestVolume : 1,
      muscleRegionCount: this.muscleRegions.length, muscleTetrahedronCount: tetDiagnostics.length,
      activeMuscleTetrahedra, residual: Math.sqrt(residual2) };
  }

  /** Inherited metric plus rest-volume-weighted PSD rank-one fibre blocks. */
  multiply(v, out, dt) {
    super.multiply(v, out, dt);
    for (const { nodes, coefficients, direction, k } of this.muscleBlocks) {
      let projection = 0;
      for (let a = 0; a < 4; a++) {
        if (this.fixed[nodes[a]]) continue;
        for (let d = 0; d < 3; d++) projection += coefficients[a] * direction[d] * v[nodes[a] * 3 + d];
      }
      for (let a = 0; a < 4; a++) {
        if (this.fixed[nodes[a]]) continue;
        for (let d = 0; d < 3; d++) out[nodes[a] * 3 + d] += k * projection * coefficients[a] * direction[d];
      }
    }
  }
}

export default VolumetricMuscleFEM;
