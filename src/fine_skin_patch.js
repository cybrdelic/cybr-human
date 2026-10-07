import * as THREE from "three";
/** Experimental one-way, local mechanical detail; not a full-face shell. */
export function createFineSkinPatch(
  model,
  arrays,
  onChange,
  onError = () => {},
) {
  const width = 65,
    height = 33,
    values = new Float32Array(width * height),
    texture = new THREE.DataTexture(
      values,
      width,
      height,
      THREE.RedFormat,
      THREE.FloatType,
    );
  texture.minFilter = texture.magFilter = THREE.LinearFilter;
  texture.needsUpdate = true;
  const domain = new THREE.Vector4(-0.024, 0.064, 0.048, 0.024),
    worker = new Worker(new URL("./wrinkle_worker.js", import.meta.url), {
      type: "module",
    });
  const candidates = [];
  for (let n = 0; n < model.surface_nodes; n++) {
    const j = n * 3;
    if (
      Math.abs(arrays.rest[j]) < 0.045 &&
      arrays.rest[j + 1] > 0.045 &&
      arrays.rest[j + 1] < 0.11 &&
      arrays.rest[j + 2] > 0.02
    )
      candidates.push(n);
  }
  const normals = new Float64Array(model.surface_nodes * 3);
  for (let j = 0; j < arrays.faces.length; j += 3) {
    const ids = Array.from(arrays.faces.subarray(j, j + 3)),
      p = ids.map((n) => Array.from(arrays.rest.subarray(n * 3, n * 3 + 3))),
      a = p[1].map((v, d) => v - p[0][d]),
      b = p[2].map((v, d) => v - p[0][d]),
      n = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
      ];
    for (const id of ids)
      for (let d = 0; d < 3; d++) normals[id * 3 + d] += n[d];
  }
  const supports = [];
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const px = domain.x + x * 0.00075,
        py = domain.y + y * 0.00075,
        near = candidates
          .map((n) => ({
            n,
            d: Math.hypot(arrays.rest[n * 3] - px, arrays.rest[n * 3 + 1] - py),
          }))
          .sort((a, b) => a.d - b.d)
          .slice(0, 9),
        radius = Math.max(near.at(-1).d, 1e-9),
        weights = near.slice(0, 8).map((p) => {
          const r = Math.min(1, p.d / radius);
          return (1 - r) ** 4 * (4 * r + 1);
        }),
        sum = weights.reduce((s, w) => s + w, 0),
        normal = [0, 0, 0];
      weights.forEach((w, i) => {
        for (let d = 0; d < 3; d++)
          normal[d] += (w * normals[near[i].n * 3 + d]) / sum;
      });
      const l = Math.hypot(...normal);
      for (let d = 0; d < 3; d++) normal[d] /= l;
      const t1 = [1, 0, -normal[0] / normal[2]],
        a = Math.hypot(...t1);
      for (let d = 0; d < 3; d++) t1[d] /= a;
      const t2 = [
        normal[1] * t1[2],
        normal[2] * t1[0] - normal[0] * t1[2],
        -normal[1] * t1[0],
      ];
      supports.push({
        ids: near.slice(0, 8).map((p) => p.n),
        weights: weights.map((w) => w / sum),
        t1,
        t2,
      });
    }
  const triangles = [];
  for (let y = 0; y < height - 1; y++)
    for (let x = 0; x < width - 1; x++) {
      const a = y * width + x;
      triangles.push([a, a + 1, a + width], [a + 1, a + width + 1, a + width]);
    }
  let busy = false,
    last = null,
    pending = null,
    disposed = false;
  const state = {
    scope:
      "One-way forehead membrane FEM / compliant foundation; no volume feedback",
    ready: false,
    report: null,
    errors: [],
    nodes: width * height,
  };
  function request(F) {
    if (disposed || state.failed) return;
    if (busy) {
      pending = F;
      return;
    }
    if (last && Math.max(...F.map((v, i) => Math.abs(v - last[i]))) < 0.001)
      return;
    last = F;
    busy = true;
    state.busy = true;
    worker.postMessage({ F });
  }
  function fail(reason) {
    if (disposed) return;
    state.errors.push(reason);
    state.failed = true;
    state.ready = false;
    busy = false;
    state.busy = false;
    pending = null;
    values.fill(0);
    texture.needsUpdate = true;
    worker.terminate();
    onChange();
    onError(reason);
  }
  worker.onerror = (e) => fail(e.message || "Fine skin worker failed");
  worker.onmessage = ({ data }) => {
    busy = false;
    state.busy = false;
    if (data.error) {
      fail(data.error);
      return;
    } else {
      values.set(data.height);
      texture.needsUpdate = true;
      state.ready = true;
      state.report = data.report;
      onChange();
    }
    if (pending) {
      const F = pending;
      pending = null;
      request(F);
    }
  };
  return {
    texture,
    domain,
    state,
    heatmap: { value: 0 },
    update(packed) {
      if (disposed || state.failed || !candidates.length) return;
      const metrics = new Float64Array(supports.length * 4),
        N = model.nodes,
        S = model.surface_nodes;
      for (let i = 0; i < supports.length; i++) {
        const s = supports[i],
          F = new Float64Array(9);
        s.ids.forEach((n, j) => {
          for (let c = 0; c < 3; c++)
            for (let d = 0; d < 3; d++)
              F[d * 3 + c] +=
                s.weights[j] * packed[N * 4 + c * S * 4 + n * 4 + d];
        });
        // Match the full tangent metric, including shear, rather than dropping the
        // out-of-plane entries of the volume deformation gradient.
        const apply = (v) =>
            [0, 1, 2].map(
              (d) =>
                F[d * 3] * v[0] + F[d * 3 + 1] * v[1] + F[d * 3 + 2] * v[2],
            ),
          a = apply(s.t1),
          b = apply(s.t2),
          l = Math.hypot(...a),
          dot = a.reduce((s, v, i) => s + v * b[i], 0) / l,
          t = Math.sqrt(
            Math.max(1e-8, b.reduce((s, v) => s + v * v, 0) - dot * dot),
          );
        metrics.set([l, dot, 0, t], i * 4);
      }
      const fields = new Float64Array(triangles.length * 4);
      triangles.forEach((ids, i) => {
        for (const n of ids)
          for (let d = 0; d < 4; d++)
            fields[i * 4 + d] += metrics[n * 4 + d] / 3;
      });
      request(fields);
    },
    dispose() {
      disposed = true;
      worker.terminate();
      texture.dispose();
    },
  };
}
