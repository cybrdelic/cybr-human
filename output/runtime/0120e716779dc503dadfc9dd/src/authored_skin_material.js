import * as THREE from "three";
import { regionalSkinOptics } from "./regional_skin_optics.js";
/** Code-authored pigmentation and micron-scale relief. No skin image inputs.
 * Diffuse radiance is separated for screen-space transport; fine relief is not a mechanical wrinkle model.
 */
export function improveAuthoredMaterials(mesh, options = {}) {
  const regional = options.regional === true;
  const g = mesh.geometry,
    p = g.attributes.position;
  if (/Continuous_(neutral|anatomical)_skin/.test(mesh.name)) {
    const colors = new Float32Array(p.count * 3),
      roughnessField = regional ? new Float32Array(p.count) : null,
      color = new THREE.Color();
    for (let i = 0; i < p.count; i++) {
      const x = p.getX(i) * 1000,
        z = p.getY(i) * 1000,
        front = p.getZ(i) > 0.03 ? 1 : 0,
        q = Math.min(1, Math.abs(x) / 25.5),
        span = Math.max(0, 1 - q * q),
        line = /anatomical/.test(mesh.name)
          ? -35.5 + 0.5 * q * q
          : -42 - 0.65 * Math.exp(0 - (x / 5) ** 2) + 0.65 * q * q;
      const optics = regional ? regionalSkinOptics(x, z, p.getZ(i) * 1000) : null;
      const lip = optics ? optics.lip : front * Math.exp(0 - ((z - line) / 4.3) ** 4) * span ** 0.45;
      if (roughnessField) roughnessField[i] = optics.roughness;
      const cheek =
          front *
          Math.exp(0 - ((Math.abs(x) - 40) / 20) ** 2 - ((z - 1) / 22) ** 2),
        nose = front * Math.exp(0 - (x / 16) ** 2 - ((z + 9) / 14) ** 2);
      const variation =
        0.015 * Math.sin(x * 0.13 + z * 0.07) * Math.sin(z * 0.17 - x * 0.09);
      color.setRGB(
        0.66 + variation + 0.02 * cheek - 0.1 * lip,
        0.43 + variation - 0.025 * cheek - 0.02 * nose - 0.16 * lip,
        0.34 + variation - 0.02 * cheek - 0.01 * nose - 0.11 * lip,
        THREE.SRGBColorSpace,
      );
      color.toArray(colors, i * 3);
    }
    g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    if (roughnessField) g.setAttribute("authoredRoughness", new THREE.BufferAttribute(roughnessField, 1));
    const noise = new Uint8Array(256 * 256);
    let seed = 271828;
    for (let i = 0; i < noise.length; i++) {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      noise[i] = seed >>> 24;
    }
    const micro = new THREE.DataTexture(
      noise,
      256,
      256,
      THREE.RedFormat,
      THREE.UnsignedByteType,
    );
    micro.wrapS = micro.wrapT = THREE.RepeatWrapping;
    micro.minFilter = micro.magFilter = THREE.LinearFilter;
    micro.needsUpdate = true;
    const Material = regional ? THREE.MeshPhysicalMaterial : THREE.MeshStandardMaterial;
    const m = new Material({
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.55,
      metalness: 0,
    });
    if (regional) { m.ior = 1.4; m.specularIntensity = 1; }
    mesh.userData.authoredMicroTexture = micro;
    const diffusePass = (mesh.userData.skinDiffusePass = { value: 0 });
    m.onBeforeCompile = (shader) => {
      shader.uniforms.authoredMicroTexture = { value: micro };
      shader.uniforms.skinDiffusePass = diffusePass;
      if (regional) {
        shader.vertexShader = shader.vertexShader.replace("#include <common>", "#include <common>\nattribute float authoredRoughness;\nvarying float vAuthoredRoughness;").replace("#include <begin_vertex>", "#include <begin_vertex>\nvAuthoredRoughness=authoredRoughness;");
        shader.fragmentShader = shader.fragmentShader.replace("#include <common>", "#include <common>\nvarying float vAuthoredRoughness;").replace("#include <roughnessmap_fragment>", "#include <roughnessmap_fragment>\nroughnessFactor=clamp(vAuthoredRoughness,.2,.8);");
      }
      shader.vertexShader = shader.vertexShader
        .replace(
          "#include <common>",
          "#include <common>\nvarying vec3 vAuthoredSkinPosition;",
        )
        .replace(
          "#include <begin_vertex>",
          "#include <begin_vertex>\nvAuthoredSkinPosition=position;",
        );
      shader.fragmentShader = shader.fragmentShader.replace(
        "#include <common>",
        `#include <common>
varying vec3 vAuthoredSkinPosition;
uniform sampler2D authoredMicroTexture;
uniform float skinDiffusePass;
`,
      );
      shader.fragmentShader = shader.fragmentShader.replace(
        "#include <normal_fragment_begin>",
        `#include <normal_fragment_begin>
float skinRelief=.000003*(texture2D(authoredMicroTexture,vAuthoredSkinPosition.xy*(6500./256.)).r+.35*texture2D(authoredMicroTexture,vAuthoredSkinPosition.xy*(13000./256.)).r);
vec3 skinDx=dFdx(-vViewPosition),skinDy=dFdy(-vViewPosition);
float skinScaleX=max(length(skinDx),1e-7),skinScaleY=max(length(skinDy),1e-7);
skinDx/=skinScaleX;skinDy/=skinScaleY;
vec3 skinR1=cross(skinDy,normal),skinR2=cross(normal,skinDx);float skinDet=dot(skinDx,skinR1);
vec3 skinPerturbed=abs(skinDet)*normal-sign(skinDet)*(dFdx(skinRelief)/skinScaleX*skinR1+dFdy(skinRelief)/skinScaleY*skinR2);
if(abs(skinDet)>1e-5 && dot(skinPerturbed,skinPerturbed)>1e-8)normal=normalize(skinPerturbed);
`,
      );
      shader.fragmentShader = shader.fragmentShader.replace(
        "#include <opaque_fragment>",
        "if(skinDiffusePass>.5)outgoingLight=reflectedLight.directDiffuse+reflectedLight.indirectDiffuse;\n#include <opaque_fragment>",
      );
    };
    m.customProgramCacheKey = () => regional ? "authored-regional-optics-v1" : "authored-diffuse-transport-v3";
    mesh.material = m;
  } else if (/nasal_vestibule_rim/.test(mesh.name)) {
    mesh.material = new THREE.MeshPhysicalMaterial({
      color: 0xa9755e,
      roughness: 0.48,
      specularIntensity: 0.35,
    });
  } else if (/_iris$/.test(mesh.name)) {
    g.computeBoundingBox();
    const c = g.boundingBox.getCenter(new THREE.Vector3()),
      colors = new Float32Array(p.count * 3),
      color = new THREE.Color();
    for (let i = 0; i < p.count; i++) {
      const x = (p.getX(i) - c.x) * 1000,
        y = (p.getY(i) - c.y) * 1000,
        r = Math.hypot(x, y) / 5.72,
        a = Math.atan2(y, x),
        fiber = 0.5 + 0.5 * Math.sin(a * 143 + Math.sin(a * 31) * 2 + r * 11),
        limbus = 1 - 0.5 * Math.exp(0 - ((r - 0.98) / 0.055) ** 2);
      color.setRGB(
        (0.28 + 0.12 * fiber) * limbus,
        (0.14 + 0.09 * fiber) * limbus,
        (0.065 + 0.045 * fiber) * limbus,
        THREE.SRGBColorSpace,
      );
      color.toArray(colors, i * 3);
    }
    g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    mesh.material = new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      vertexColors: true,
      // The pigmented iris lies behind the cornea; keep the wet specular
      // response on the authored corneal shell rather than doubling it.
      roughness: 0.8,
      specularIntensity: 0.05,
    });
    const centre = c.clone();
    centre.z = g.boundingBox.max.z - 0.0112;
    const vertices = [],
      indices = [],
      rings = 24,
      segments = 96;
    for (let j = 0; j <= rings; j++)
      for (let k = 0; k <= segments; k++) {
        const r = (0.0059 * j) / rings,
          a = (2 * Math.PI * k) / segments;
        vertices.push(
          centre.x + r * Math.cos(a),
          centre.y + r * Math.sin(a),
          centre.z + 0.0052 + Math.sqrt(0.0078 ** 2 - r * r),
        );
      }
    for (let j = 0; j < rings; j++)
      for (let k = 0; k < segments; k++) {
        const a = j * (segments + 1) + k;
        indices.push(
          a,
          a + segments + 1,
          a + 1,
          a + 1,
          a + segments + 1,
          a + segments + 2,
        );
      }
    const cg = new THREE.BufferGeometry();
    cg.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
    cg.setIndex(indices);
    cg.computeVertexNormals();
    const cornea = new THREE.Mesh(
      cg,
      new THREE.MeshPhysicalMaterial({
        color: 0xffffff,
        transparent: options.refractiveEyes !== true,
        opacity: options.refractiveEyes === true ? 1 : 0.025,
        depthWrite: options.refractiveEyes === true,
        transmission: options.refractiveEyes === true ? 1 : 0,
        thickness: options.refractiveEyes === true ? 0.00055 : 0,
        roughness: 0.035,
        clearcoat: options.refractiveEyes === true ? 0 : 1,
        clearcoatRoughness: 0.02,
        ior: 1.376,
        specularIntensity: 1,
      }),
    );
    cornea.name = mesh.name + "_authored_corneal_specular_shell";
    mesh.parent.add(cornea);
  }
}
