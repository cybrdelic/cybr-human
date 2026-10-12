export function recoverSurfaceGradients(
  x,
  rest,
  nodes,
  gradients,
  material,
  count,
) {
  const result = new Float32Array(count * 9),
    weight = new Float64Array(count),
    F = new Float64Array(9);
  for (let t = 0; t < nodes.length / 4; t++) {
    F.fill(0);
    F[0] = F[4] = F[8] = 1;
    const anchor = nodes[t * 4] * 3;
    for (let a = 1; a < 4; a++) {
      const j = nodes[t * 4 + a] * 3;
      for (let d = 0; d < 3; d++)
        for (let k = 0; k < 3; k++)
          F[d * 3 + k] +=
            (x[j + d] - rest[j + d] - (x[anchor + d] - rest[anchor + d])) *
            gradients[t * 12 + a * 3 + k];
    }
    const V = material[t * 3];
    for (let a = 0; a < 4; a++) {
      const n = nodes[t * 4 + a];
      if (n >= count) continue;
      weight[n] += V;
      for (let k = 0; k < 9; k++) result[n * 9 + k] += V * F[k];
    }
  }
  for (let n = 0; n < count; n++) {
    if (!(weight[n] > 0)) throw Error("Unconnected transfer node");
    for (let k = 0; k < 9; k++) result[n * 9 + k] /= weight[n];
  }
  return result;
}
