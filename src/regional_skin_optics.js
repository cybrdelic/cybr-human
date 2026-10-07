// Authored optical fields in rest-space millimetres, independent of tissue mechanics.
const clamp = (x) => Math.max(0, Math.min(1, x));
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };
export function regionalSkinOptics(x, height, depth) {
  if (![x, height, depth].every(Number.isFinite)) throw new TypeError('Finite rest coordinates required');
  const front = smooth(20, 35, depth);
  const q = clamp(Math.abs(x) / 25.5), span = Math.max(0, 1 - q * q);
  const line = -35.5 + 0.5 * q * q;
  const upper = (4 + 1.3 * Math.exp(0 -((Math.abs(x) - 7) / 4) ** 2)) * span ** 0.7;
  const lower = 5.5 * span ** 0.7;
  const distance = Math.abs(height - line) / Math.max(0.05, height >= line ? upper : lower);
  const lip = front * (1 - smooth(0.78, 1.12, distance)) * smooth(0, 0.08, span);
  const nose = front * Math.exp(0 -(x / 16) ** 2 - ((height + 9) / 14) ** 2);
  const cheek = front * Math.exp(0 -((Math.abs(x) - 40) / 20) ** 2 - ((height - 1) / 22) ** 2);
  const forehead = front * Math.exp(0 -(x / 24) ** 2 - ((height - 48) / 25) ** 2);
  const eye = front * Math.exp(0 -((Math.abs(x) - 31) / 17) ** 2 - ((height - 30) / 9) ** 2);
  const roughness = 0.6 - 0.15 * nose - 0.065 * forehead - 0.1 * eye;
  return { lip, nose, cheek, roughness: roughness * (1 - lip) + 0.3 * lip };
}
