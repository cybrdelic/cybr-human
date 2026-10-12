/**
 * Tension-only embedded active fibres on the original cybr-human CPU FEM.
 *
 * Source solver: cybrdelic/cybr-human@c0e6d626dfa48c0b05657bd87b880cb7a6bc92a9.
 * Preserve reference-source/LICENSES/GPL-2.0.txt and LICENSING.md with this work.
 * This is an authored mechanical actuator, not a calibrated muscle model.
 * The inherited volumetric energy, determinant guard, PCG and line search are
 * unchanged. No position target or displacement is imposed by a fibre.
 *
 * Fibre schema:
 * { id?: string, a: Endpoint, b: Endpoint, stiffness: number,
 *   restLength?: number, activation?: number }
 * Endpoint = nodeIndex | { nodes: number[], weights: number[] }
 *                     | { point: [x,y,z] } (fixed external origin).
 * Stiffness is force / length in the same consistent units as the FEM arrays.
 * Missing restLength is measured between endpoints in the FEM rest state.
 * E_f = .5 k max(length - (1-activation) restLength, 0)^2.
 */
import { LayeredFEM } from "../reference-source/src/layered_fem.js";

function endpoint(record, nodeCount) {
  if (Number.isInteger(record)) record = { nodes: [record], weights: [1] };
  if (!record || typeof record !== "object") throw Error("Invalid fibre endpoint");
  if (record.point !== undefined) {
    if (record.nodes !== undefined || record.weights !== undefined ||
        record.point.length !== 3 || !Array.from(record.point).every(Number.isFinite))
      throw Error("Invalid fixed fibre endpoint");
    return { point: Array.from(record.point), nodes: [], weights: [] };
  }
  const nodes = Array.from(record.nodes || []), weights = Array.from(record.weights || []);
  if (!nodes.length || nodes.length !== weights.length ||
      !nodes.every(n => Number.isInteger(n) && n >= 0 && n < nodeCount) ||
      !weights.every(w => Number.isFinite(w) && w >= 0) ||
      Math.abs(weights.reduce((s, w) => s + w, 0) - 1) > 1e-10)
    throw Error("Fibre endpoint requires valid node IDs and nonnegative partition-of-unity weights");
  return { nodes, weights };
}

function position(endpoint, x) {
  if (endpoint.point) return endpoint.point;
  const p = [0, 0, 0];
  for (let a = 0; a < endpoint.nodes.length; a++)
    for (let d = 0; d < 3; d++) p[d] += endpoint.weights[a] * x[endpoint.nodes[a] * 3 + d];
  return p;
}

function activationValue(value) {
  if (!Number.isFinite(value) || value < 0 || value > 1)
    throw Error("Fibre activation must lie in [0,1]");
  return value;
}

export class ActiveFibreFEM extends LayeredFEM {
  constructor(model, arrays, fibres = []) {
    super(model, arrays);
    this.fibreBlocks = [];
    this.fibreDiagnostics = [];
    this.setFibres(fibres);
  }

  setFibres(records) {
    const seen = new Set();
    this.fibres = records.map((record, index) => {
      const a = endpoint(record.a, this.mass.length), b = endpoint(record.b, this.mass.length),
        pa = position(a, this.rest), pb = position(b, this.rest),
        restLength = record.restLength ?? Math.hypot(...pa.map((v, d) => pb[d] - v)),
        stiffness = record.stiffness, id = record.id ?? String(index),
        activation = activationValue(record.activation ?? 0);
      if (!(Number.isFinite(restLength) && restLength > 1e-12) ||
          !(Number.isFinite(stiffness) && stiffness > 0))
        throw Error("Fibre requires positive finite restLength and stiffness");
      if (seen.has(id)) throw Error("Duplicate fibre ID");
      seen.add(id);
      const coefficients = new Map();
      for (const [end, sign] of [[a, -1], [b, 1]])
        for (let j = 0; j < end.nodes.length; j++) {
          const n = end.nodes[j];
          coefficients.set(n, (coefficients.get(n) || 0) + sign * end.weights[j]);
        }
      return { id, a, b, stiffness, restLength, activation,
        coefficients: Array.from(coefficients, ([node, weight]) => ({ node, weight }))
          .filter(({ weight }) => Math.abs(weight) > 1e-15) };
    });
    this.fibreBlocks = [];
    this.searchBasis.length = 0;
    return this;
  }

  /** Omit selector to change all fibres; selector is an index or an exact ID. */
  setActivation(value, selector = null) {
    value = activationValue(value);
    if (selector === null) {
      for (const fibre of this.fibres) fibre.activation = value;
    } else {
      const fibre = Number.isInteger(selector) ? this.fibres[selector]
        : this.fibres.find(f => f.id === selector);
      if (!fibre) throw Error("Unknown fibre selector");
      fibre.activation = value;
    }
    return this;
  }

  evaluate(x, dt, blocks = true) {
    const state = super.evaluate(x, dt, blocks);
    if (blocks) this.fibreBlocks.length = 0;
    let fibreEnergy = 0, activeTensionFibres = 0, maximumFibreTension = 0;
    const diagnostics = [];
    for (const fibre of this.fibres) {
      const { a, b, restLength, stiffness: k, coefficients } = fibre,
        activation = activationValue(fibre.activation),
        pa = position(a, x), pb = position(b, x),
        delta = pa.map((v, d) => pb[d] - v), length = Math.hypot(...delta),
        targetLength = (1 - activation) * restLength,
        extension = Math.max(length - targetLength, 0), tension = k * extension,
        energy = 0.5 * k * extension * extension;
      fibreEnergy += energy;
      maximumFibreTension = Math.max(maximumFibreTension, tension);
      diagnostics.push({ id: fibre.id, activation, length, restLength, targetLength,
        extension, tension, energy });
      if (extension === 0) continue;
      activeTensionFibres++;
      const direction = delta.map(v => v / length);
      for (const { node, weight } of coefficients) {
        if (this.fixed[node]) continue;
        for (let d = 0; d < 3; d++) {
          const q = weight * direction[d], j = node * 3 + d;
          this.g[j] += tension * q;
          if (blocks) this.diagonal[j] += k * q * q;
        }
      }
      if (blocks) this.fibreBlocks.push({ coefficients, direction, k });
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
    this.fibreDiagnostics = diagnostics;
    return { ...state, energy: state.energy + fibreEnergy, fibreEnergy,
      passiveAndExternalEnergy: state.energy, activeTensionFibres,
      maximumFibreTension, residual: Math.sqrt(residual2) };
  }

  /** Inherited metric plus PSD rank-one fibre Gauss-Newton blocks. */
  multiply(v, out, dt) {
    super.multiply(v, out, dt);
    for (const { coefficients, direction, k } of this.fibreBlocks) {
      let projection = 0;
      for (const { node, weight } of coefficients) {
        if (this.fixed[node]) continue;
        for (let d = 0; d < 3; d++) projection += weight * direction[d] * v[node * 3 + d];
      }
      for (const { node, weight } of coefficients) {
        if (this.fixed[node]) continue;
        for (let d = 0; d < 3; d++) out[node * 3 + d] += k * projection * weight * direction[d];
      }
    }
  }
}
