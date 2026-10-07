import { LayeredFEM, installTissueBoundary } from "./layered_fem.js";
import { recoverSurfaceGradients } from "./layered_fem_transfer_math.js";
let solver,
  model,
  base,
  material,
  mass,
  areas,
  frames = 0;
function init(m, a) {
  model = m;
  base = a;
  material = a.material.slice();
  mass = a.mass.slice();
  solver = new LayeredFEM(m, a);
  const S = m.surface_nodes;
  areas = new Float64Array(S);
  for (let f = 0; f < a.faces.length; f += 3) {
    const ids = Array.from(a.faces.subarray(f, f + 3)),
      p = ids.map((n) => Array.from(a.rest.subarray(n * 3, n * 3 + 3))),
      e = p[1].map((v, d) => v - p[0][d]),
      q = p[2].map((v, d) => v - p[0][d]),
      area =
        Math.hypot(
          e[1] * q[2] - e[2] * q[1],
          e[2] * q[0] - e[0] * q[2],
          e[0] * q[1] - e[1] * q[0],
        ) / 6;
    for (const n of ids) areas[n] += area;
  }
  installTissueBoundary(solver, m, a);
}
function step(parameters, options) {
  const started = performance.now(),
    S = model.surface_nodes,
    N = model.nodes,
    { fatPercent = 0, sagPercent = 0, gravity = 1, grab = null } = parameters;
  solver.volumeTargets = new Float64Array(model.tetrahedra);
  solver.mass.set(mass);
  const soft = 10 ** (-1.7 * sagPercent * 0.01);
  for (let t = 0; t < model.tetrahedra; t++) {
    const ratio = 1 + fatPercent * 0.01 * (base.fatWeights?.[t] || 0),
      grow = Math.cbrt(ratio),
      j = t * 3,
      mu = material[j + 1] * soft;
    solver.volumeTargets[t] = ratio;
    solver.material[j + 1] = mu * grow;
    solver.material[j + 2] = (material[j + 2] - material[j + 1] + mu) / ratio;
    for (let a = 0; a < 4; a++) {
      const n = base.nodes[t * 4 + a];
      solver.mass[n] += (material[j] * 950 * (ratio - 1)) / 4;
      for (let b = 0; b < 4; b++) {
        let dot = 0;
        for (let d = 0; d < 3; d++)
          dot +=
            solver.gradients[t * 12 + a * 3 + d] *
            solver.gradients[t * 12 + b * 3 + d];
        solver.shear[t * 16 + a * 4 + b] =
          material[j] * solver.material[j + 1] * dot;
      }
    }
  }
  solver.gravity = [0, -9.81 * gravity, 0];
  solver.pointSprings = [];
  let supports = 0;
  if (grab) {
    const weights = [];
    let sum = 0;
    for (let n = 0; n < S; n++) {
      if (solver.fixed[n] || base.groups[n] !== base.groups[grab.node])
        continue;
      let d2 = 0;
      for (let d = 0; d < 3; d++)
        d2 += (base.rest[n * 3 + d] - base.rest[grab.node * 3 + d]) ** 2;
      if (d2 > 9 * 0.006 ** 2) continue;
      const w = areas[n] * Math.exp(-d2 / (2 * 0.006 ** 2));
      weights.push([n, w]);
      sum += w;
    }
    for (const [n, w] of weights)
      solver.pointSprings.push({
        node: n,
        target: Array.from(base.rest.subarray(n * 3, n * 3 + 3)).map(
          (v, d) => v + grab.target[d] - base.rest[grab.node * 3 + d],
        ),
        k: ((grab.stiffness ?? 100) * w) / sum,
      });
    supports = weights.length;
  }
  const dt = options.dt ?? 1 / 120,
    substeps = options.substeps ?? 2;
  if (!(Number.isInteger(substeps) && substeps >= 1 && substeps <= 16))
    throw Error("Invalid CPU substep count");
  let state,
    totalCG = 0;
  for (let step = 0; step < substeps; step++) {
    state = solver.step(options.quasiStatic ? Infinity : dt, {
      iterations: options.newton ?? 2,
      cgIterations: options.cg ?? 24,
      cgTolerance: 0.01,
      forceTolerance: 1e-4,
    });
    totalCG += state.totalCG;
  }
  if (!Number.isFinite(state.residual) || state.minJ <= 0.2)
    throw Error("CPU tissue solve became invalid");
  const staticState = solver.evaluate(solver.x, Infinity, false);
  let maxNodeSpeedMps = 0,
    kineticEnergyJ = 0;
  for (let n = 0; n < N; n++) {
    if (solver.fixed[n]) continue;
    const speed2 =
      solver.velocity[n * 3] ** 2 +
      solver.velocity[n * 3 + 1] ** 2 +
      solver.velocity[n * 3 + 2] ** 2;
    maxNodeSpeedMps = Math.max(maxNodeSpeedMps, Math.sqrt(speed2));
    kineticEnergyJ += 0.5 * solver.mass[n] * speed2;
  }
  const F = recoverSurfaceGradients(
      solver.x,
      base.rest,
      base.nodes,
      base.gradients,
      base.material,
      S,
    ),
    packed = new Float32Array(S * 24);
  for (let n = 0; n < N; n++)
    for (let d = 0; d < 3; d++) packed[n * 4 + d] = solver.x[n * 3 + d];
  for (let n = 0; n < S; n++)
    for (let c = 0; c < 3; c++)
      for (let d = 0; d < 3; d++)
        packed[N * 4 + c * S * 4 + n * 4 + d] = F[n * 9 + d * 3 + c];
  return {
    packed,
    ...(options.checkpointState ? { velocities: solver.velocity.slice() } : {}),
    stats: {
      elapsedMs: performance.now() - started,
      gpuMs: null,
      minJ: state.minJ,
      residualN: state.residual,
      staticResidualN: staticState.residual,
      maxNodeSpeedMps,
      kineticEnergyJ,
      lineAccepted: state.accepted > 0 || state.converged,
      cgIterationsExecuted: totalCG,
      substepsExecuted: substeps,
      integration: options.quasiStatic ? "quasistatic" : "dynamic",
      simulatedSeconds: options.quasiStatic ? 0 : dt * substeps,
      grabSupportNodes: supports,
      frame: ++frames,
    },
  };
}
self.onmessage = ({ data }) => {
  try {
    if (data.type === "init") {
      init(data.model, data.arrays);
      self.postMessage({ id: data.id, ready: true });
    } else if (data.type === "reset") {
      init(model, { ...base, material: material.slice(), mass: mass.slice() });
      self.postMessage({ id: data.id, ready: true });
    } else if (data.type === "restore") {
      const { positions, velocity } = data;
      if (
        positions.length !== solver.x.length ||
        velocity.length !== solver.velocity.length ||
        !positions.every(Number.isFinite) ||
        !velocity.every(Number.isFinite)
      )
        throw Error("Invalid tissue checkpoint");
      solver.x.set(positions);
      solver.velocity.set(velocity);
      for (let n = 0; n < model.nodes; n++)
        if (solver.fixed[n])
          for (let d = 0; d < 3; d++) {
            const j = n * 3 + d;
            solver.x[j] = solver.target[j];
            solver.velocity[j] = 0;
          }
      if (data.verifyCheckpoint) {
        const state = solver.evaluate(solver.x, Infinity, false);
        if (!Number.isFinite(state.energy) || !Number.isFinite(state.residual) || !Number.isFinite(state.minJ) || state.minJ <= 0.2)
          throw Error("Restored tissue checkpoint failed numerical validation");
        const positions = solver.x.slice(), velocities = solver.velocity.slice();
        self.postMessage({ id: data.id, ready: true, positions, velocities,
          minJ: state.minJ, energyJ: state.energy, residualN: state.residual }, [positions.buffer, velocities.buffer]);
      } else self.postMessage({ id: data.id, ready: true });
    } else {
      const result = step(data.parameters, data.options);
      self.postMessage({ id: data.id, ...result }, [result.packed.buffer, ...(result.velocities ? [result.velocities.buffer] : [])]);
    }
  } catch (e) {
    self.postMessage({ id: data.id, error: e.message });
  }
};
