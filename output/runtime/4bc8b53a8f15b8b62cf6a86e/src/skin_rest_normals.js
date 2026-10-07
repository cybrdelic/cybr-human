import * as THREE from "three";
/** Weld UV aliases for normal accumulation while preserving authored creases.
 * Render indices and UV coordinates are not welded or changed. */
export function prepareSkinNormals(mesh) {
  const g = mesh.geometry,
    p = g.attributes.position;
  if (!g.index) return;
  if (!g.attributes.normal) {
    g.computeVertexNormals();
    for (const m of Array.isArray(mesh.material)
      ? mesh.material
      : [mesh.material]) {
      m.flatShading = false;
      m.needsUpdate = true;
    }
  }
  const n = g.attributes.normal;
  const representative = new Uint32Array(p.count),
    first = new Map();
  for (let i = 0; i < p.count; i++) {
    const key = `${Math.round(p.getX(i) * 1e7)},${Math.round(p.getY(i) * 1e7)},${Math.round(p.getZ(i) * 1e7)}:${Math.round(n.getX(i) * 1e4)},${Math.round(n.getY(i) * 1e4)},${Math.round(n.getZ(i) * 1e4)}`;
    let j = first.get(key);
    if (j === undefined) {
      j = i;
      first.set(key, j);
    }
    representative[i] = j;
  }
  mesh.userData.restNormalRepresentatives = representative;
}
export function rebuildSkinNormals(mesh) {
  const g = mesh.geometry,
    r = mesh.userData.restNormalRepresentatives;
  if (!r) {
    g.computeVertexNormals();
    return;
  }
  const index = new Uint32Array(g.index.count);
  for (let i = 0; i < index.length; i++) index[i] = r[g.index.getX(i)];
  const scratch = new THREE.BufferGeometry();
  scratch.setAttribute("position", g.attributes.position);
  scratch.setIndex(new THREE.BufferAttribute(index, 1));
  scratch.computeVertexNormals();
  const n = scratch.attributes.normal;
  for (let i = 0; i < n.count; i++)
    if (r[i] !== i) n.setXYZ(i, n.getX(r[i]), n.getY(r[i]), n.getZ(r[i]));
  g.setAttribute("normal", n);
  n.needsUpdate = true;
}
