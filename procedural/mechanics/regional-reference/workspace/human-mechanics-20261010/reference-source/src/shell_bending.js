// Rotationally invariant discrete skin bending at a shared triangle edge.
// ids: edge endpoints then the two opposite vertices, consistently oriented.
const sub = (a, b) => a.map((v, i) => v - b[i]),
  dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0),
  cross = (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ],
  scale = (a, k) => a.map((v) => v * k);
export function hingeState(p, theta0 = 0, k = 1) {
  const e = sub(p[1], p[0]),
    e2 = dot(e, e),
    length = Math.sqrt(e2),
    n1 = cross(e, sub(p[2], p[0])),
    n2 = cross(sub(p[3], p[0]), e),
    s1 = dot(n1, n1),
    s2 = dot(n2, n2);
  if (length < 1e-12 || Math.min(s1, s2) < 1e-24)
    throw Error("Collapsed shell hinge");
  const theta = Math.atan2(dot(cross(n1, n2), e) / length, dot(n1, n2)),
    delta = Math.atan2(Math.sin(theta - theta0), Math.cos(theta - theta0));
  const q2 = scale(n1, -length / s1),
    q3 = scale(n2, -length / s2),
    q0 = q2.map(
      (v, i) =>
        (dot(sub(p[2], p[1]), e) / e2) * v +
        (dot(sub(p[3], p[1]), e) / e2) * q3[i],
    ),
    q1 = q0.map((v, i) => -v - q2[i] - q3[i]);
  const derivative = [q0, q1, q2, q3];
  return {
    theta,
    energy: 0.5 * k * delta * delta,
    derivative,
    gradient: derivative.map((q) => scale(q, k * delta)),
  };
}
