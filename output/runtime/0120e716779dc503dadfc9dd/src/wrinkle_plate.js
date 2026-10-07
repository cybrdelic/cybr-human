/** Fine normal-displacement membrane FEM on a compliant foundation.
 * Linear triangular membrane basis, small-slope plate bending, nonlinear
 * Green strain. This local model is not the full 3D coupled shell solver.
 */
export class WrinklePlate {
  constructor({
    nx = 65,
    ny = 33,
    pitch = 0.00075,
    E = 600000,
    thickness = 0.0002,
    nu = 0.45,
    foundation = 18000 / 0.0035,
  } = {}) {
    Object.assign(this, { nx, ny, pitch, E, thickness, nu, foundation });
    this.h = new Float64Array(nx * ny);
    this.g = this.h.slice();
    this.lap = this.h.slice();
    this.area = pitch * pitch;
    this.mu = (E * thickness) / (2 * (1 + nu));
    this.lambda = (E * thickness * nu) / (1 - nu * nu);
    this.D = (E * thickness ** 3) / (12 * (1 - nu * nu));
    this.triangles = [];
    for (let y = 0; y < ny - 1; y++)
      for (let x = 0; x < nx - 1; x++) {
        const a = y * nx + x;
        this.triangles.push(
          [
            a,
            a + 1,
            a + nx,
            [-1 / pitch, 1 / pitch, 0],
            [-1 / pitch, 0, 1 / pitch],
          ],
          [
            a + 1,
            a + nx + 1,
            a + nx,
            [0, 1 / pitch, -1 / pitch],
            [-1 / pitch, 1 / pitch, 0],
          ],
        );
      }
    this.fixed = Uint8Array.from({ length: this.h.length }, (_, i) =>
      i % nx < 2 ||
      i % nx >= nx - 2 ||
      Math.floor(i / nx) < 2 ||
      Math.floor(i / nx) >= ny - 2
        ? 1
        : 0,
    );
  }
  evaluate(h = this.h, F = [1, 0, 0, 1]) {
    const g = this.g;
    g.fill(0);
    this.lap.fill(0);
    let energy = 0;
    let triangle = 0;
    for (const [a, b, c, dx, dy] of this.triangles) {
      const ids = [a, b, c];
      let hx = 0,
        hy = 0;
      for (let k = 0; k < 3; k++) {
        hx += h[ids[k]] * dx[k];
        hy += h[ids[k]] * dy[k];
      }
      const j = F.length === 4 ? 0 : 4 * triangle++,
        ex = 0.5 * (F[j] ** 2 + F[j + 2] ** 2 + hx * hx - 1),
        ey = 0.5 * (F[j + 1] ** 2 + F[j + 3] ** 2 + hy * hy - 1),
        xy = 0.5 * (F[j] * F[j + 1] + F[j + 2] * F[j + 3] + hx * hy),
        tr = ex + ey,
        A = this.area / 2;
      energy +=
        A *
        (this.mu * (ex * ex + ey * ey + 2 * xy * xy) +
          0.5 * this.lambda * tr * tr);
      const gx =
          (2 * this.mu * ex + this.lambda * tr) * hx + 2 * this.mu * xy * hy,
        gy = (2 * this.mu * ey + this.lambda * tr) * hy + 2 * this.mu * xy * hx;
      for (let k = 0; k < 3; k++) g[ids[k]] += A * (gx * dx[k] + gy * dy[k]);
    }
    const n = this.nx,
      scale = 1 / (this.pitch * this.pitch),
      D = this.D * this.area;
    for (let y = 1; y < this.ny - 1; y++)
      for (let x = 1; x < n - 1; x++) {
        const i = y * n + x,
          L = (h[i - 1] + h[i + 1] + h[i - n] + h[i + n] - 4 * h[i]) * scale;
        this.lap[i] = L;
        energy += 0.5 * D * L * L;
        const v = D * L * scale;
        g[i] -= 4 * v;
        g[i - 1] += v;
        g[i + 1] += v;
        g[i - n] += v;
        g[i + n] += v;
      }
    for (let i = 0; i < h.length; i++) {
      energy += 0.5 * this.foundation * this.area * h[i] * h[i];
      g[i] += this.foundation * this.area * h[i];
      if (this.fixed[i]) g[i] = 0;
    }
    return energy;
  }
  solve(F, { iterations = 600 } = {}) {
    // Deterministic micron-scale imperfection selects an instability direction;
    // it does not prescribe the equilibrium wavelength or crease locations.
    if (Math.max(...this.h.map(Math.abs)) < 1e-7)
      for (let i = 0; i < this.h.length; i++)
        if (!this.fixed[i])
          this.h[i] = 1e-6 * Math.sin(i * 12.9898) * Math.cos(i * 7.233);
    let energy = this.evaluate(this.h, F);
    const initial = energy,
      history = [],
      trial = this.h.slice();
    let completed = 0;
    const dot = (a, b) => {
      let v = 0;
      for (let i = 0; i < a.length; i++) v += a[i] * b[i];
      return v;
    };
    for (let k = 0; k < iterations; k++) {
      const oldG = this.g.slice(),
        oldH = this.h.slice(),
        q = oldG.slice(),
        alphas = [];
      for (let j = history.length - 1; j >= 0; j--) {
        const rec = history[j],
          alpha = dot(rec.s, q) * rec.rho;
        alphas[j] = alpha;
        for (let i = 0; i < q.length; i++) q[i] -= alpha * rec.y[i];
      }
      const last = history.at(-1),
        gamma = last
          ? dot(last.s, last.y) / dot(last.y, last.y)
          : 1 /
            (this.foundation * this.area +
              (20 * this.D) / this.area +
              2 * (this.mu + this.lambda));
      for (let i = 0; i < q.length; i++) q[i] *= gamma;
      for (let j = 0; j < history.length; j++) {
        const rec = history[j],
          beta = dot(rec.y, q) * rec.rho;
        for (let i = 0; i < q.length; i++)
          q[i] += rec.s[i] * (alphas[j] - beta);
      }
      let slope = -dot(q, oldG);
      if (!(slope < 0)) break;
      let step = 1,
        accepted = false;
      for (let line = 0; line < 24; line++) {
        for (let i = 0; i < q.length; i++)
          trial[i] = this.fixed[i] ? 0 : oldH[i] - step * q[i];
        const next = this.evaluate(trial, F);
        if (Number.isFinite(next) && next <= energy + 1e-4 * step * slope) {
          this.h.set(trial);
          energy = next;
          accepted = true;
          break;
        }
        step *= 0.5;
      }
      if (!accepted) {
        this.evaluate(this.h, F);
        break;
      }
      const s = this.h.map((v, i) => v - oldH[i]),
        y = this.g.map((v, i) => v - oldG[i]),
        sy = dot(s, y);
      if (sy > 1e-18) {
        history.push({ s, y, rho: 1 / sy });
        if (history.length > 7) history.shift();
      }
      completed++;
      if (Math.sqrt(dot(this.g, this.g)) < 1e-8) break;
    }
    return {
      initialEnergyJ: initial,
      energyJ: energy,
      iterations: completed,
      maxHeightM: Math.max(...this.h.map(Math.abs)),
      residualN: Math.hypot(...this.g),
      predictedWavelengthM: 2 * Math.PI * (this.D / this.foundation) ** 0.25,
    };
  }
}
