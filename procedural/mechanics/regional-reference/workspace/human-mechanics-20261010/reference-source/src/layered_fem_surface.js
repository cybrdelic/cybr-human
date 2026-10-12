import * as THREE from "three";
/** Continuous nodal deformation-gradient transfer. The FEM worker solves the
 * volume; CPU gradient recovery is linear in its cells; dense skin stays GPU. */
export function installFEMSurface(
  mesh,
  bundle,
  bytes,
  rest,
  partIndex = 0,
  pool = null,
) {
  const read = (rec, Type) => new Type(bytes, rec.byte_offset, rec.length),
    faces = read(bundle.faces, Uint32Array),
    support = read(bundle.vertex_faces, Uint32Array),
    bary = read(bundle.barycentric2, Float32Array),
    part = bundle.parts.find((p) => p.index === partIndex),
    count = mesh.geometry.attributes.position.count,
    N = rest.length / 9;
  if (!part || part.vertex_count !== count)
    throw Error(
      `FEM surface binding differs from source skin: ${mesh.name}, part ${partIndex}, expected ${part?.vertex_count}, loaded ${count}`,
    );
  const indices = new Float32Array(count * 3),
    weights = new Float32Array(count * 2),
    active = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const j = part.embedding.offset + i,
      f = support[j];
    if (f === 0xffffffff) continue;
    indices.set(faces.subarray(f * 3, f * 3 + 3), i * 3);
    weights.set(bary.subarray(j * 2, j * 2 + 2), i * 2);
    active[i] = 1;
  }
  for (const [name, array, size] of [
    ["femNodes", indices, 3],
    ["femWeights", weights, 2],
    ["femActive", active, 1],
  ])
    mesh.geometry.setAttribute(name, new THREE.BufferAttribute(array, size));
  if (!bundle.smoothData) throw Error("Smooth surface transfer is required");
  const fine = partIndex === 0 ? pool?.fine : null;
  {
    const attrs = Array.from({ length: 4 }, () => new Float32Array(count * 4));
    for (let i = 0; i < count; i++) {
      const j = (part.embedding.offset + i) * 16;
      for (let k = 0; k < 4; k++)
        attrs[k].set(
          bundle.smoothData.subarray(j + k * 4, j + k * 4 + 4),
          i * 4,
        );
    }
    for (let k = 0; k < 4; k++)
      mesh.geometry.setAttribute(
        ["femSupport0", "femSupport1", "femBlend0", "femBlend1"][k],
        new THREE.BufferAttribute(attrs[k], 4),
      );
  }
  let resources = pool?.resources;
  if (!resources) {
    const width = 256,
      height = Math.ceil(rest.length / 3 / width),
      current = new Float32Array(width * height * 4),
      reference = current.slice(),
      frameHeight = Math.ceil(N / width),
      columns = Array.from(
        { length: 3 },
        () => new Float32Array(width * frameHeight * 4),
      );
    const copy = (xyz, dst) => {
      for (let i = 0; i < xyz.length / 3; i++)
        for (let d = 0; d < 3; d++) dst[i * 4 + d] = xyz[i * 3 + d];
    };
    copy(rest, current);
    copy(rest, reference);
    const texture = (a, h) => {
        const t = new THREE.DataTexture(
          a,
          width,
          h,
          THREE.RGBAFormat,
          THREE.FloatType,
        );
        t.needsUpdate = true;
        return t;
      },
      posed = texture(current, height),
      initial = texture(reference, height),
      frames = columns.map((a) => texture(a, frameHeight));
    // The deformation gradient of the undeformed reference state is identity.
    for (let n = 0; n < N; n++)
      for (let c = 0; c < 3; c++) columns[c][n * 4 + c] = 1;
    resources = {
      width,
      height,
      current,
      reference,
      frameHeight,
      columns,
      posed,
      initial,
      frames,
    };
    if (pool) pool.resources = resources;
  }
  const {
    width,
    height,
    current,
    reference,
    frameHeight,
    columns,
    posed,
    initial,
    frames,
  } = resources;
  mesh.updateWorldMatrix(true, false);
  const world = mesh.matrixWorld.clone(),
    inverse = world.clone().invert(),
    nw = new THREE.Matrix3().getNormalMatrix(world),
    nl = nw.clone().invert();
  const heatmap = pool?.heatmap || { value: 0 };
  if (pool) pool.heatmap = heatmap;
  const common = `
 varying float vFEMJ;varying float vFEMActive;vec3 femTransferredNormal;
 attribute vec3 femNodes;attribute vec2 femWeights;attribute float femActive;
 uniform sampler2D femCurrent;uniform sampler2D femRest;uniform sampler2D femF0;uniform sampler2D femF1;uniform sampler2D femF2;uniform vec2 femSize;uniform vec2 femFrameSize;
 uniform mat4 femWorld;uniform mat4 femInverse;uniform mat3 femNormalWorld;uniform mat3 femNormalLocal;
 vec3 femRead(sampler2D tex,float id,vec2 size){return texture2D(tex,(vec2(mod(id,size.x),floor(id/size.x))+.5)/size).xyz;}
 vec3 femW(){vec3 w=vec3(femWeights,max(0.,1.-femWeights.x-femWeights.y));return w/(w.x+w.y+w.z);}
 vec3 femBlend(sampler2D tex,vec3 w,vec2 size){return w.x*femRead(tex,femNodes.x,size)+w.y*femRead(tex,femNodes.y,size)+w.z*femRead(tex,femNodes.z,size);}
 mat3 femF(vec3 w){return mat3(femBlend(femF0,w,femFrameSize),femBlend(femF1,w,femFrameSize),femBlend(femF2,w,femFrameSize));}
 vec3 femNormal(mat3 f,vec3 n){vec3 a=cross(f[1],f[2]),b=cross(f[2],f[0]),c=cross(f[0],f[1]);float det=dot(f[0],a);return mat3(a,b,c)*n/max(det,.00001);}
 attribute vec4 femSupport0;attribute vec4 femSupport1;attribute vec4 femBlend0;attribute vec4 femBlend1;
 mat3 femNodeF(float n){return mat3(femRead(femF0,n,femFrameSize),femRead(femF1,n,femFrameSize),femRead(femF2,n,femFrameSize));}
 mat3 femSmoothF(){mat3 f=mat3(0.);for(int i=0;i<8;i++){float n=i<4?femSupport0[i]:femSupport1[i-4];float w=i<4?femBlend0[i]:femBlend1[i-4];f+=w*femNodeF(n);}return f;}
 vec3 femSmoothPosition(vec3 q){vec3 p=vec3(0.);for(int i=0;i<8;i++){float n=i<4?femSupport0[i]:femSupport1[i-4];float w=i<4?femBlend0[i]:femBlend1[i-4];p+=w*(femRead(femCurrent,n,femSize)+femNodeF(n)*(q-femRead(femRest,n,femSize)));}return p;}
 ${
   fine
     ? `
 uniform sampler2D femFineHeight;uniform vec4 femFineDomain;varying float vFineHeight;
 float femFine(vec3 q){vec2 uv=(q.xy-femFineDomain.xy)/femFineDomain.zw;if(q.z<.02||any(lessThan(uv,vec2(0.)))||any(greaterThan(uv,vec2(1.))))return 0.;return texture2D(femFineHeight,uv).r;}
 vec3 femFineNormal(vec3 q,vec3 n){float dx=femFineDomain.z/64.,dy=femFineDomain.w/32.;float hx=(femFine(q+vec3(dx,0.,0.))-femFine(q-vec3(dx,0.,0.)))/(2.*dx);float hy=(femFine(q+vec3(0.,dy,0.))-femFine(q-vec3(0.,dy,0.)))/(2.*dy);return normalize(vec3(n.x-hx*n.z,n.y-hy*n.z,n.z));}
 `
     : ""
 }
 `;
  const hook = (shader) => {
    const hasVertexNormal = shader.vertexShader.includes(
      "#include <beginnormal_vertex>",
    );
    Object.assign(shader.uniforms, {
      femHeatmap: heatmap,
      femCurrent: { value: posed },
      femRest: { value: initial },
      femF0: { value: frames[0] },
      femF1: { value: frames[1] },
      femF2: { value: frames[2] },
      femWorld: { value: world },
      femInverse: { value: inverse },
      femNormalWorld: { value: nw },
      femNormalLocal: { value: nl },
      femSize: { value: new THREE.Vector2(width, height) },
      femFrameSize: { value: new THREE.Vector2(width, frameHeight) },
    });
    if (fine)
      Object.assign(shader.uniforms, {
        femFineHeight: { value: fine.texture },
        femFineDomain: { value: fine.domain },
        femFineHeatmap: fine.heatmap,
      });
    shader.vertexShader = shader.vertexShader.replace(
      "#include <common>",
      "#include <common>\n" + common,
    );
    shader.vertexShader = shader.vertexShader.replace(
      "#include <beginnormal_vertex>",
      `#include <beginnormal_vertex>\nif(femActive>.5){vec3 n=femNormal(femSmoothF(),femNormalWorld*objectNormal);femTransferredNormal=n;${fine ? "n=femFineNormal((femWorld*vec4(position,1.)).xyz,n);" : ""}objectNormal=femNormalLocal*n;}`,
    );
    shader.vertexShader = shader.vertexShader.replace(
      "#include <begin_vertex>",
      `#include <begin_vertex>
 vFEMJ=1.;vFEMActive=femActive;${fine ? "vFineHeight=femFine((femWorld*vec4(position,1.)).xyz);" : ""}if(femActive>.5){mat3 f=femF(femW());vFEMJ=dot(f[0],cross(f[1],f[2]));}
 if(femActive>.5){
   vec3 p=femSmoothPosition((femWorld*vec4(position,1.)).xyz);
   ${fine ? `vec3 n=normalize(${hasVertexNormal ? "femTransferredNormal" : "femNormal(femSmoothF(),femNormalWorld*normal)"});p+=n*femFine((femWorld*vec4(position,1.)).xyz);` : ""}
   transformed=(femInverse*vec4(p,1.)).xyz;
 }
`,
    );
  };
  const shadingHook = (shader) => {
    hook(shader);
    // Preserve the inverse-transpose transported authored smooth normal and
    // fine-patch normal. Keep the geometric override for paired visual QA.
    if (pool?.geometricNormals) shader.fragmentShader = shader.fragmentShader.replace(
      "#include <normal_fragment_begin>",
      `#include <normal_fragment_begin>
if(vFEMActive>.5){vec3 geometricNormal=cross(dFdx(-vViewPosition),dFdy(-vViewPosition));float lengthSquared=dot(geometricNormal,geometricNormal);if(lengthSquared>1e-24)normal=(gl_FrontFacing?1.:-1.)*geometricNormal*inversesqrt(lengthSquared);}
`,
    );
    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <common>",
      "#include <common>\nuniform float femHeatmap;varying float vFEMJ;varying float vFEMActive;" +
        (fine ? "uniform float femFineHeatmap;varying float vFineHeight;" : ""),
    );
    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <opaque_fragment>",
      `vec3 heat=vFEMJ<1.?mix(vec3(.68,.75,.74),vec3(.10,.4,.9),clamp((1.-vFEMJ)/.05,0.,1.)):mix(vec3(.68,.75,.74),vec3(.95,.22,.08),clamp((vFEMJ-1.)/.05,0.,1.));outgoingLight=mix(outgoingLight,heat,femHeatmap*vFEMActive);${fine ? "float amplitude=clamp(abs(vFineHeight)/.0001,0.,1.);vec3 detail=mix(vec3(.4,.4,.4),vFineHeight<0.?vec3(.08,.28,.95):vec3(.95,.18,.05),amplitude);outgoingLight=mix(outgoingLight,detail,femFineHeatmap);" : ""}\n#include <opaque_fragment>`,
    );
  };
  const authoredHook = mesh.material.onBeforeCompile,
    authoredKey = mesh.material.customProgramCacheKey(),
    material = mesh.material.clone();
  material.onBeforeCompile = (shader, renderer) => {
    authoredHook(shader, renderer);
    shadingHook(shader);
  };
  material.customProgramCacheKey = () =>
    "layered-fem-affine-surface" + !!fine + !!pool?.geometricNormals + authoredKey;
  mesh.material = material;
  mesh.frustumCulled = false;
  mesh.customDepthMaterial = new THREE.MeshDepthMaterial({
    depthPacking: THREE.RGBADepthPacking,
  });
  mesh.customDepthMaterial.onBeforeCompile = hook;
  mesh.customDepthMaterial.customProgramCacheKey = () =>
    "layered-fem-affine-depth" + !!fine;
  return {
    setHeatmap: (value) => {
      heatmap.value = value ? 1 : 0;
    },
    activeVertices: active.reduce((a, b) => a + b, 0),
    nodeTextureBytes:
      current.byteLength + columns.reduce((s, a) => s + a.byteLength, 0),
    updatePacked(packed) {
      const nodeFloats = (rest.length / 3) * 4;
      current.set(packed.subarray(0, nodeFloats));
      posed.needsUpdate = true;
      for (let c = 0; c < 3; c++) {
        columns[c].set(
          packed.subarray(nodeFloats + c * N * 4, nodeFloats + (c + 1) * N * 4),
        );
        frames[c].needsUpdate = true;
      }
    },
  };
}
