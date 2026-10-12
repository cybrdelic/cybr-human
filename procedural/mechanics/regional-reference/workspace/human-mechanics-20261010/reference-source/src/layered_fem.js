import { hingeState } from "./shell_bending.js";
/** Numerical orientation barrier, inactive above J=.6 and singular at the
 * existing J=.2 guard. This is not a calibrated tissue constitutive term. */
export function compressionBarrier(J) {
  if (J >= 0.6) return [0, 0, 0];
  if (J <= 0.2) return [Infinity, 0, 0];
  const d = J - 0.2,
    q = J - 0.6,
    l = Math.log(d / 0.4);
  return [
    -q * q * l,
    -2 * q * l - (q * q) / d,
    -2 * l - (4 * q) / d + (q * q) / (d * d),
  ];
}
/** Unreduced implicit volumetric FEM. SI units, glTF coordinates.
 * Stable Neo-Hookean energy; exact gradient and an SPD quasi-Newton search
 * metric (shear block + current determinant outer product). Backtracking
 * checks actual energy and every cell determinant. No XPBD pose projection.
 */
export class LayeredFEM {
  constructor(model, arrays) {
    if (model.schema !== "cybr-layered-fem-v1")
      throw Error("Unsupported FEM model");
    Object.assign(this, arrays);
    this.model = model;
    this.gradients = this.gradients.slice();
    for (let t = 0; t < this.nodes.length / 4; t++)
      for (let d = 0; d < 3; d++)
        this.gradients[t * 12 + d] = -(
          this.gradients[t * 12 + 3 + d] +
          this.gradients[t * 12 + 6 + d] +
          this.gradients[t * 12 + 9 + d]
        );
    this.x = this.rest.slice();
    this.velocity = new Float64Array(this.x.length);
    const n = this.x.length,
      t = this.nodes.length / 4;
    if (
      n !== model.nodes * 3 ||
      this.gradients.length !== t * 12 ||
      this.material.length !== t * 3 ||
      this.mass.length !== n / 3 ||
      this.mass.some((m) => !(m > 0))
    )
      throw Error("Invalid FEM arrays");
    this.g = new Float64Array(n);
    this.diagonal = new Float64Array(n);
    this.hydro = new Float64Array(t * 12);
    this.compressionStiffness = Float64Array.from(
      { length: t },
      (_, i) => this.material[i * 3 + 1],
    );
    this.effectiveBulk = new Float64Array(t);
    this.shear = new Float64Array(t * 16);
    this.fixed = new Uint8Array(n / 3);
    this.target = this.rest.slice();
    this.predict = this.x.slice();
    this.trial = this.x.slice();
    this.direction = new Float64Array(n);
    this.r = new Float64Array(n);
    this.z = new Float64Array(n);
    this.p = new Float64Array(n);
    this.ap = new Float64Array(n);
    this.f = new Float64Array(9);
    this.c = new Float64Array(9);
    for (let i = 0; i < t; i++) {
      const w = this.material[i * 3] * this.material[i * 3 + 1];
      for (let a = 0; a < 4; a++)
        for (let b = 0; b < 4; b++) {
          let dot = 0;
          for (let d = 0; d < 3; d++)
            dot +=
              this.gradients[i * 12 + a * 3 + d] *
              this.gradients[i * 12 + b * 3 + d];
          this.shear[i * 16 + a * 4 + b] = w * dot;
        }
    }
    if (this.hinges) {
      this.hinges = this.hinges.slice();
      for (let i = 0; i < this.hinges.length; i += 6) {
        const points = Array.from(this.hinges.subarray(i, i + 4), (n) =>
          Array.from(this.rest.subarray(n * 3, n * 3 + 3)),
        );
        this.hinges[i + 4] = hingeState(points).theta;
      }
    }
    this.hingeBlocks = [];
    this.gravity = [0, 0, 0];
    this.substrateContacts = [];
    this.pointSprings = [];
    this.contacts = [];
    this.solidMeshContacts = [];
    this.contactBlocks = [];
    this.last = null;
    this.useSearchSubspace = false;
    this.searchBasis = [];
  }
  pin(node, point) {
    this.fixed[node] = 1;
    this.target.set(point, node * 3);
  }
  unpin(node) {
    this.fixed[node] = 0;
  }
  /** Analytic spheres for verified benchmark contact. Anatomical mesh contact
   * must be supplied separately; these are not labelled skull constraints. */
  setSphereContacts(records) {
    for (const r of records)
      if (!(r.radius > 0) || !(r.stiffness > 0) || r.center.length !== 3)
        throw Error("Invalid contact");
    this.contacts = records;
  }
  evaluate(x, dt, blocks = true) {
    let energy = 0,
      minJ = Infinity,
      maxJ = -Infinity;
    this.g.fill(0);
    if (blocks) {
      this.diagonal.fill(0);
      this.contactBlocks.length = 0;
      this.hingeBlocks.length = 0;
    }
    const F = this.f,
      C = this.c;
    for (let i = 0; i < this.nodes.length / 4; i++) {
      F.fill(0);
      F[0] = F[4] = F[8] = 1;
      const go = i * 12,
        ni = i * 4,
        vo = i * 3,
        V = this.material[vo],
        mu = this.material[vo + 1],
        lam = this.material[vo + 2];
      const anchor = this.nodes[ni] * 3;
      for (let a = 1; a < 4; a++) {
        const node = this.nodes[ni + a] * 3;
        for (let d = 0; d < 3; d++) {
          const u =
            x[node + d] -
            this.rest[node + d] -
            (x[anchor + d] - this.rest[anchor + d]);
          for (let k = 0; k < 3; k++)
            F[d * 3 + k] += u * this.gradients[go + a * 3 + k];
        }
      }
      C[0] = F[4] * F[8] - F[5] * F[7];
      C[1] = F[5] * F[6] - F[3] * F[8];
      C[2] = F[3] * F[7] - F[4] * F[6];
      C[3] = F[7] * F[2] - F[8] * F[1];
      C[4] = F[8] * F[0] - F[6] * F[2];
      C[5] = F[6] * F[1] - F[7] * F[0];
      C[6] = F[1] * F[5] - F[2] * F[4];
      C[7] = F[2] * F[3] - F[0] * F[5];
      C[8] = F[0] * F[4] - F[1] * F[3];
      const J = F[0] * C[0] + F[1] * C[1] + F[2] * C[2],
        targetJ = this.volumeTargets?.[i] || 1,
        growth = Math.cbrt(targetJ),
        h = J - targetJ - mu / (lam * growth);
      minJ = Math.min(minJ, J);
      maxJ = Math.max(maxJ, J);
      let square = 0;
      for (let k = 0; k < 9; k++) square += F[k] * F[k];
      const barrier = compressionBarrier(J),
        kBarrier = this.compressionStiffness[i];
      if (blocks) this.effectiveBulk[i] = lam + kBarrier * barrier[2];
      energy +=
        V *
        (0.5 * mu * (square - 3 * growth * growth) +
          0.5 * lam * h * h -
          (mu * mu) / (2 * lam * growth * growth) +
          kBarrier * barrier[0]);
      for (let a = 0; a < 4; a++) {
        const node = this.nodes[ni + a] * 3;
        for (let d = 0; d < 3; d++) {
          let force = 0,
            hydro = 0;
          for (let k = 0; k < 3; k++) {
            const grad = this.gradients[go + a * 3 + k];
            force +=
              (mu * F[d * 3 + k] +
                (lam * h + kBarrier * barrier[1]) * C[d * 3 + k]) *
              grad;
            hydro += C[d * 3 + k] * grad;
          }
          this.g[node + d] += V * force;
          if (blocks) {
            this.hydro[go + a * 3 + d] = hydro;
            this.diagonal[node + d] +=
              this.shear[i * 16 + a * 5] +
              V * this.effectiveBulk[i] * hydro * hydro;
          }
        }
      }
    }
    for (let i = 0; i < (this.hinges?.length || 0); i += 6) {
      const ids = Array.from(this.hinges.subarray(i, i + 4)),
        points = ids.map((n) => Array.from(x.subarray(n * 3, n * 3 + 3))),
        k = this.hinges[i + 5],
        state = hingeState(points, this.hinges[i + 4], k);
      energy += state.energy;
      for (let a = 0; a < 4; a++)
        for (let d = 0; d < 3; d++) {
          const j = ids[a] * 3 + d;
          this.g[j] += state.gradient[a][d];
          if (blocks) this.diagonal[j] += k * state.derivative[a][d] ** 2;
        }
      if (blocks) this.hingeBlocks.push({ ids, q: state.derivative, k });
    }
    for (let n = 0; n < this.mass.length; n++)
      for (let d = 0; d < 3; d++) {
        const j = n * 3 + d,
          w = this.mass[n] / (dt * dt),
          v = x[j] - this.predict[j];
        energy += 0.5 * w * v * v;
        this.g[j] += w * v;
        if (blocks) this.diagonal[j] += w;
      }
    // Gravity is a body force, not a prescribed surface displacement.
    for (let n = 0; n < this.mass.length; n++)
      for (let d = 0; d < 3; d++) {
        const j = n * 3 + d,
          force = this.mass[n] * this.gravity[d];
        energy -= force * (x[j] - this.rest[j]);
        this.g[j] -= force;
      }
    let contactViolation = 0;
    for (const record of this.solidMeshContacts) {
      const { node, collider } = record,
        j = node * 3,
        state = collider.evaluate(Array.from(x.subarray(j, j + 3)));
      energy += state.energy;
      contactViolation = Math.max(contactViolation, state.violation);
      for (let d = 0; d < 3; d++) {
        this.g[j + d] += state.gradient[d];
        if (blocks && state.violation > 0)
          this.diagonal[j + d] += collider.stiffness * state.normal[d] ** 2;
      }
      if (blocks && state.violation > 0)
        this.contactBlocks.push({
          node,
          normal: state.normal,
          k: collider.stiffness,
        });
    }
    for (const contact of this.substrateContacts) {
      const { node, point, normal, k, clearance } = contact,
        j = node * 3;
      let gap = -clearance;
      for (let d = 0; d < 3; d++) gap += (x[j + d] - point[d]) * normal[d];
      if (gap >= 0) continue;
      contactViolation = Math.max(contactViolation, -gap);
      energy += 0.5 * k * gap * gap;
      for (let d = 0; d < 3; d++) {
        this.g[j + d] += k * gap * normal[d];
        if (blocks) this.diagonal[j + d] += k * normal[d] ** 2;
      }
      if (blocks) this.contactBlocks.push({ node, normal, k });
    }
    for (const spring of this.pointSprings) {
      const { node, target, k } = spring,
        j = node * 3;
      for (let d = 0; d < 3; d++) {
        const v = x[j + d] - target[d];
        energy += 0.5 * k * v * v;
        this.g[j + d] += k * v;
        if (blocks) {
          this.diagonal[j + d] += k;
          const normal = [0, 0, 0];
          normal[d] = 1;
          this.contactBlocks.push({ node, normal, k });
        }
      }
    }

    for (const sphere of this.contacts)
      for (let node = 0; node < this.mass.length; node++) {
        const j = node * 3,
          dx = x[j] - sphere.center[0],
          dy = x[j + 1] - sphere.center[1],
          dz = x[j + 2] - sphere.center[2],
          r = Math.hypot(dx, dy, dz),
          gap = r - sphere.radius;
        if (gap >= 0) continue;
        if (r < 1e-12) throw Error("Undefined sphere contact normal");
        contactViolation = Math.max(contactViolation, -gap);
        const normal = [dx / r, dy / r, dz / r],
          k = sphere.stiffness;
        energy += 0.5 * k * gap * gap;
        for (let d = 0; d < 3; d++) {
          this.g[j + d] += k * gap * normal[d];
          if (blocks) this.diagonal[j + d] += k * normal[d] ** 2;
        }
        if (blocks) this.contactBlocks.push({ node, normal, k });
      }
    let residual = 0;
    for (let n = 0; n < this.mass.length; n++)
      for (let d = 0; d < 3; d++) {
        const j = n * 3 + d;
        if (this.fixed[n]) {
          this.g[j] = 0;
          if (blocks) this.diagonal[j] = 1;
        } else residual += this.g[j] ** 2;
      }
    return {
      energy,
      minJ,
      maxJ,
      residual: Math.sqrt(residual),
      contactViolation,
    };
  }
  multiply(v, out, dt) {
    for (let n = 0; n < this.mass.length; n++)
      for (let d = 0; d < 3; d++)
        out[n * 3 + d] = this.fixed[n]
          ? 0
          : (this.mass[n] / (dt * dt)) * v[n * 3 + d];
    for (let i = 0; i < this.nodes.length / 4; i++) {
      const ni = i * 4,
        go = i * 12;
      let hv = 0;
      for (let a = 0; a < 4; a++) {
        const n = this.nodes[ni + a];
        if (this.fixed[n]) continue;
        for (let d = 0; d < 3; d++)
          hv += this.hydro[go + a * 3 + d] * v[n * 3 + d];
      }
      hv *= this.material[i * 3] * this.effectiveBulk[i];
      for (let a = 0; a < 4; a++) {
        const n = this.nodes[ni + a];
        if (this.fixed[n]) continue;
        for (let d = 0; d < 3; d++) {
          let value = hv * this.hydro[go + a * 3 + d];
          for (let b = 0; b < 4; b++) {
            const nb = this.nodes[ni + b];
            if (!this.fixed[nb])
              value += this.shear[i * 16 + a * 4 + b] * v[nb * 3 + d];
          }
          out[n * 3 + d] += value;
        }
      }
    }
    for (const h of this.hingeBlocks) {
      let dot = 0;
      for (let a = 0; a < 4; a++)
        if (!this.fixed[h.ids[a]])
          for (let d = 0; d < 3; d++) dot += h.q[a][d] * v[h.ids[a] * 3 + d];
      for (let a = 0; a < 4; a++)
        if (!this.fixed[h.ids[a]])
          for (let d = 0; d < 3; d++)
            out[h.ids[a] * 3 + d] += h.k * dot * h.q[a][d];
    }
    for (const b of this.contactBlocks) {
      if (this.fixed[b.node]) continue;
      let dot = 0;
      for (let d = 0; d < 3; d++) dot += b.normal[d] * v[b.node * 3 + d];
      for (let d = 0; d < 3; d++)
        out[b.node * 3 + d] += b.k * dot * b.normal[d];
    }
  }
  pcg(dt, limit, tolerance) {
    const { r, z, p, ap, direction: q, g, diagonal } = this;
    q.fill(0);
    let rz = 0,
      initial = 0;
    // Galerkin warm start in prior, numerically independent search directions.
    // This reduces search work, never the physical state or energy inventory.
    if (this.useSearchSubspace && this.searchBasis.length) {
      const B = this.searchBasis,
        Y = B.map((b) => {
          const y = new Float64Array(q.length);
          this.multiply(b, y, dt);
          return y;
        }),
        N = B.length,
        L = Array.from({ length: N }, () => new Float64Array(N)),
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
      const coeff = new Float64Array(N),
        y = new Float64Array(N);
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
      for (let a = 0; a < N; a++)
        for (let j = 0; j < q.length; j++) q[j] += B[a][j] * coeff[a];
    }
    this.multiply(q, ap, dt);
    let fullNorm = 0;
    for (let i = 0; i < r.length; i++) {
      r[i] = -g[i] - ap[i];
      z[i] = r[i] / diagonal[i];
      p[i] = z[i];
      rz += r[i] * z[i];
      initial += r[i] * r[i];
      fullNorm += g[i] * g[i];
    }
    if (initial <= fullNorm * tolerance * tolerance) return 0;
    if (initial < 1e-30) return 0;
    for (let it = 0; it < limit; it++) {
      this.multiply(p, ap, dt);
      let den = 0;
      for (let i = 0; i < p.length; i++) den += p[i] * ap[i];
      if (!(den > 0)) throw Error("Non-positive FEM search metric");
      const alpha = rz / den;
      let norm = 0,
        next = 0;
      for (let i = 0; i < r.length; i++) {
        q[i] += alpha * p[i];
        r[i] -= alpha * ap[i];
        norm += r[i] * r[i];
        z[i] = r[i] / diagonal[i];
        next += r[i] * z[i];
      }
      if (norm <= fullNorm * tolerance * tolerance) {
        this.rememberSearch(q);
        return it + 1;
      }
      const beta = next / rz;
      for (let i = 0; i < p.length; i++) p[i] = z[i] + beta * p[i];
      rz = next;
    }
    this.rememberSearch(q);
    return limit;
  }
  rememberSearch(q) {
    if (!this.useSearchSubspace) return;
    const b = q.slice();
    let original = 0;
    for (const v of b) original += v * v;
    if (original < 1e-30) return;
    for (let pass = 0; pass < 2; pass++)
      for (const a of this.searchBasis) {
        let dot = 0;
        for (let j = 0; j < b.length; j++) dot += b[j] * a[j];
        for (let j = 0; j < b.length; j++) b[j] -= dot * a[j];
      }
    let norm = 0;
    for (const v of b) norm += v * v;
    if (norm < original * 1e-8) return;
    norm = Math.sqrt(norm);
    for (let j = 0; j < b.length; j++) b[j] /= norm;
    if (this.searchBasis.length === 8) this.searchBasis.shift();
    this.searchBasis.push(b);
  }
  step(
    dt,
    {
      iterations = 8,
      cgIterations = 32,
      cgTolerance = 0.01,
      forceTolerance = 1e-5,
      minimumJ = 0.2,
      initialGuess = null,
    } = {},
  ) {
    if (!(dt === Infinity || (dt > 0 && dt <= 0.05)))
      throw Error("Invalid FEM time step");
    const begin = performance.now(),
      old = this.x.slice();
    let totalCG = 0,
      rejections = 0,
      accepted = 0;
    if (
      initialGuess &&
      (initialGuess.length !== this.x.length ||
        !initialGuess.every(Number.isFinite))
    )
      throw Error("Invalid FEM initial guess");
    for (let n = 0; n < this.mass.length; n++)
      for (let d = 0; d < 3; d++) {
        const j = n * 3 + d;
        this.predict[j] =
          dt === Infinity ? this.x[j] : this.x[j] + dt * this.velocity[j];
        if (this.fixed[n]) this.x[j] = this.target[j];
        else if (initialGuess) this.x[j] = initialGuess[j];
      }
    let state = this.evaluate(this.x, dt, true);
    if (state.minJ <= minimumJ) {
      this.x.set(old);
      throw Error(
        "Prescribed boundary would invert or excessively compress a FEM cell",
      );
    }
    for (let it = 0; it < iterations && state.residual > forceTolerance; it++) {
      totalCG += this.pcg(dt, cgIterations, cgTolerance);
      let slope = 0;
      for (let j = 0; j < this.g.length; j++)
        slope += this.g[j] * this.direction[j];
      if (!(slope < 0)) throw Error("FEM search is not a descent direction");
      let fraction = 1,
        found = false;
      for (let k = 0; k < 18; k++) {
        for (let j = 0; j < this.x.length; j++)
          this.trial[j] = this.x[j] + fraction * this.direction[j];
        const trial = this.evaluate(this.trial, dt, false);
        if (
          Number.isFinite(trial.energy) &&
          trial.minJ > minimumJ &&
          trial.energy <= state.energy + 1e-4 * fraction * slope
        ) {
          this.x.set(this.trial);
          found = true;
          accepted++;
          break;
        }
        fraction *= 0.5;
        rejections++;
      }
      if (!found) break;
      state = this.evaluate(this.x, dt, true);
    }
    for (let j = 0; j < this.x.length; j++)
      this.velocity[j] =
        dt === Infinity
          ? 0
          : ((this.x[j] - old[j]) / dt) * Math.exp(Math.log(0.98) * 60 * dt);
    this.last = {
      ...state,
      elapsedMs: performance.now() - begin,
      totalCG,
      rejections,
      accepted,
      converged: state.residual <= forceTolerance,
      invertedCells: state.minJ <= 0 ? 1 : 0,
    };
    return this.last;
  }
}

/** Shared prescribed tissue boundary for the worker and reference checks. */
export function installTissueBoundary(solver, model, arrays) {
  const S = model.surface_nodes;
  let bottom = Infinity;
  for (let n = 0; n < S; n++) bottom = Math.min(bottom, arrays.rest[n * 3 + 1]);
  solver.substrateContacts = [];
  for (let n = 0; n < model.nodes; n++) {
    if (
      arrays.fixedNodes
        ? arrays.fixedNodes[n] > 0.5
        : n >= 2 * S || arrays.rest[n * 3 + 1] < bottom + 0.001
    ) {
      solver.pin(n, Array.from(arrays.rest.subarray(n * 3, n * 3 + 3)));
      continue;
    }
    const i = n % S,
      j = (i + 2 * S) * 3,
      point = Array.from(arrays.rest.subarray(j, j + 3));
    const delta = Array.from(arrays.rest.subarray(i * 3, i * 3 + 3)).map(
        (v, d) => v - point[d],
      ),
      length = Math.hypot(...delta);
    const plane = arrays.contactPlanes?.subarray(n * 7, n * 7 + 7);
    solver.substrateContacts.push({
      node: n,
      point: plane ? Array.from(plane.subarray(0, 3)) : point,
      normal: plane
        ? Array.from(plane.subarray(3, 6))
        : delta.map((v) => v / length),
      k: 20000,
      clearance: plane ? plane[6] : Math.min(0.00001, 0.05 * length),
    });
  }
}
