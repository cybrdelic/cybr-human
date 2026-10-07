import * as THREE from "three";

/** Evaluate the same surface transfer once per changed state, then reuse its
 * full-precision result in beauty, diffuse and shadow passes. No mesh reduction. */
export function createFEMSurfaceCache(renderer, mesh, common, uniforms, fine) {
  const count = mesh.geometry.attributes.position.count;
  const width = Math.min(1024, renderer.capabilities.maxTextureSize);
  const height = Math.ceil(count / width);
  const target = new THREE.WebGLRenderTarget(width, height, {
    count: 2,
    type: THREE.FloatType,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    depthBuffer: false,
  });
  const ids = Float32Array.from({ length: count }, (_, i) => i);
  mesh.geometry.setAttribute("femVertexId", new THREE.BufferAttribute(ids, 1));
  const geometry = new THREE.BufferGeometry();
  for (const [name, attribute] of Object.entries(mesh.geometry.attributes))
    geometry.setAttribute(name, attribute);
  const material = new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3,
    depthTest: false,
    depthWrite: false,
    blending: THREE.NoBlending,
    uniforms: { ...uniforms, cacheSize: { value: new THREE.Vector2(width, height) } },
    vertexShader: `precision highp float;
    precision highp int;
    in vec3 position;in vec3 normal;in float femVertexId;
    uniform vec2 cacheSize;
    ${common.replaceAll("varying ", "out ").replaceAll("attribute ", "in ").replaceAll("texture2D", "texture")}
    out vec4 cachePosition;out vec4 cacheNormal;
    void main(){
      vec3 q=(femWorld*vec4(position,1.)).xyz;
      vec3 p=position,n=normal;float j=1.,h=0.;
      if(femActive>.5){
        mat3 f=mat3(0.);vec3 posed=vec3(0.);
        for(int i=0;i<8;i++){
          float id=i<4?femSupport0[i]:femSupport1[i-4];
          float w=i<4?femBlend0[i]:femBlend1[i-4];
          mat3 nf=femNodeF(id);f+=w*nf;
          posed+=w*(femRead(femCurrent,id,femSize)+nf*(q-femRead(femRest,id,femSize)));
        }
        vec3 worldNormal=femNormal(f,femNormalWorld*normal);
        ${fine ? "h=femFine(q);posed+=normalize(worldNormal)*h;worldNormal=femFineNormal(q,worldNormal);" : ""}
        p=(femInverse*vec4(posed,1.)).xyz;n=femNormalLocal*worldNormal;
        mat3 heat=femF(femW());j=dot(heat[0],cross(heat[1],heat[2]));
      }
      cachePosition=vec4(p,j);cacheNormal=vec4(n,h);
      vec2 texel=vec2(mod(femVertexId,cacheSize.x),floor(femVertexId/cacheSize.x));
      gl_Position=vec4((texel+.5)/cacheSize*2.-1.,0.,1.);gl_PointSize=1.;
    }`,
    fragmentShader: `precision highp float;
    in vec4 cachePosition;in vec4 cacheNormal;
    layout(location=0) out vec4 positionOut;
    layout(location=1) out vec4 normalOut;
    void main(){positionOut=cachePosition;normalOut=cacheNormal;}`,
  });
  const points = new THREE.Points(geometry, material);
  points.frustumCulled = false;
  const scene = new THREE.Scene();
  scene.add(points);
  const camera = new THREE.Camera();
  let signature = "";
  const state = { evaluations: 0, vertices: count, cpuSubmitMs: 0 };
  return {
    state,
    readPositions() {
      const values = new Float32Array(width * height * 4);
      renderer.readRenderTargetPixels(target, 0, 0, width, height, values, 0, 0);
      return values.subarray(0, count * 4);
    },
    uniforms: {
      femCachedPosition: { value: target.textures[0] },
      femCachedNormal: { value: target.textures[1] },
      femCacheSize: { value: new THREE.Vector2(width, height) },
    },
    update() {
      const next = uniforms.femCurrent.value.version + ":" + (fine?.texture.version ?? 0);
      if (signature === next) return;
      const start = performance.now();
      const previous = renderer.getRenderTarget();
      const shadow = renderer.shadowMap.autoUpdate;
      try {
        renderer.shadowMap.autoUpdate = false;
        renderer.setRenderTarget(target);
        renderer.render(scene, camera);
        signature = next;
        state.evaluations++;
        state.cpuSubmitMs = performance.now() - start;
      } finally {
        renderer.shadowMap.autoUpdate = shadow;
        renderer.setRenderTarget(previous);
      }
    },
    invalidate() { signature = ""; },
    dispose() { target.dispose();geometry.dispose();material.dispose(); },
  };
}
