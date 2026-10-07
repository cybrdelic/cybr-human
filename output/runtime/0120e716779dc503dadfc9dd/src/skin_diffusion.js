import * as THREE from "three";

/** Two-pass, depth-gated RGB diffuse transport, in linear radiance.
 * Inspired by Jimenez et al., Separable Subsurface Scattering (2015).
 * This is a screen-space approximation, not a layered spectral BSSRDF.
 * Specular radiance is excluded. Radius is measured in metres, not pixels.
 */
export function createSkinDiffusion(renderer, scene, camera, options = {}) {
  // Canvas MSAA does not apply to the offscreen beauty/diffuse targets. Query
  // samples supported by BOTH actual attachment formats, not just MAX_SAMPLES.
  const gl = renderer.getContext();
  const requestedSamples = options.coverageSamples === 4 ? 4 : 0;
  const supportedSamples = requestedSamples
    ? [gl.RGBA16F, gl.DEPTH_COMPONENT24].map(format =>
      Array.from(gl.getInternalformatParameter(gl.RENDERBUFFER, format, gl.SAMPLES) ?? []))
    : [];
  const coverageSamples = requestedSamples
    ? [4, 2].find(n => supportedSamples.every(counts => counts.includes(n))) ?? 0
    : 0;
  const state = {
    enabled: true,
    supported: renderer.extensions.has("EXT_color_buffer_float"),
    radiusMm: 1.2,
    renders: 0,
    cpuSubmitMs: 0,
    requestedCoverageSamples: requestedSamples,
    coverageSamples,
  };
  const target = (depth = false) => {
    const rt = new THREE.WebGLRenderTarget(1, 1, {
      type: THREE.HalfFloatType,
      depthBuffer: depth,
      samples: depth ? coverageSamples : 0,
      resolveDepthBuffer: true,
    });
    if (depth)
      rt.depthTexture = new THREE.DepthTexture(1, 1, THREE.UnsignedIntType);
    return rt;
  };
  const beauty = target(true),
    diffuse = target(true),
    horizontal = target(),
    vertical = target();
  const quadScene = new THREE.Scene(),
    quadCamera = new THREE.Camera();
  const geometry = new THREE.PlaneGeometry(2, 2);
  const vertexShader =
    "varying vec2 vUv;void main(){vUv=uv;gl_Position=vec4(position.xy,0.,1.);}";
  const blur = new THREE.ShaderMaterial({
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
    uniforms: {
      image: { value: null },
      coverage: { value: diffuse.texture },
      depth: { value: diffuse.depthTexture },
      sceneDepth: { value: beauty.depthTexture },
      direction: { value: new THREE.Vector2() },
      projection: { value: new THREE.Vector2() },
      nearFar: { value: new THREE.Vector2() },
      radius: { value: 0.0012 },
    },
    vertexShader,
    fragmentShader: `varying vec2 vUv;uniform sampler2D image,coverage,depth,sceneDepth;uniform vec2 direction,projection,nearFar;uniform float radius;
float eyeDepth(vec2 uv){float d=texture2D(depth,uv).r;return nearFar.x*nearFar.y/(nearFar.y-d*(nearFar.y-nearFar.x));}
float visibleSkin(vec2 uv){return step(.5,texture2D(coverage,uv).a)*step(texture2D(depth,uv).r,texture2D(sceneDepth,uv).r+1e-7);}
void main(){vec4 center=texture2D(image,vUv);float mask=visibleSkin(vUv);if(mask<.5){gl_FragColor=center;return;}
float z=eyeDepth(vUv);vec2 stepUv=direction*projection*radius/max(z,nearFar.x);
vec3 sum=vec3(0.),normalization=vec3(0.);
for(int i=-8;i<=8;i++){float t=float(i)/4.;vec2 uv=vUv+stepUv*t;
vec3 w=exp(-.5*vec3(t*t,t*t/.25,t*t/.0625));
float allowed=visibleSkin(uv)*exp(-pow(abs(eyeDepth(uv)-z)/max(radius*.75,.0001),2.));
allowed*=float(all(greaterThanEqual(uv,vec2(0.)))&&all(lessThanEqual(uv,vec2(1.))));
sum+=w*allowed*texture2D(image,uv).rgb;normalization+=w*allowed;}
gl_FragColor=vec4(sum/max(normalization,vec3(1e-6)),mask);}`,
  });
  const composite = new THREE.ShaderMaterial({
    depthTest: false,
    depthWrite: false,
    uniforms: {
      beauty: { value: beauty.texture },
      beautyDepth: { value: beauty.depthTexture },
      background: { value: new THREE.Color() },
      original: { value: diffuse.texture },
      skinDepth: { value: diffuse.depthTexture },
      scattered: { value: vertical.texture },
    },
    vertexShader,
    fragmentShader: `varying vec2 vUv;uniform sampler2D beauty,original,scattered,beautyDepth,skinDepth;uniform vec3 background;
void main(){vec4 b=texture2D(beauty,vUv),d=texture2D(original,vUv),s=texture2D(scattered,vUv);
float mask=d.a*step(texture2D(skinDepth,vUv).r,texture2D(beautyDepth,vUv).r+1e-7);
gl_FragColor=vec4(max(b.rgb+(s.rgb-d.rgb)*mask,vec3(0.)),1.);
#include <tonemapping_fragment>
if(texture2D(beautyDepth,vUv).r>=1.)gl_FragColor.rgb=background;
#include <colorspace_fragment>
}`,
  });
  const quad = new THREE.Mesh(geometry, blur);
  quad.frustumCulled = false;
  quadScene.add(quad);
  let width = 0,
    height = 0;
  const size = new THREE.Vector2(),
    clear = new THREE.Color();
  function render(bypass = false) {
    if (!state.enabled || !state.supported || bypass) {
      renderer.render(scene, camera);
      return;
    }
    const start = performance.now();
    renderer.getDrawingBufferSize(size);
    if (width !== size.x || height !== size.y) {
      width = size.x;
      height = size.y;
      for (const rt of [beauty, diffuse, horizontal, vertical])
        rt.setSize(width, height);
    }
    const previousTarget = renderer.getRenderTarget(),
      tone = renderer.toneMapping,
      alpha = renderer.getClearAlpha(),
      shadow = renderer.shadowMap.autoUpdate;
    renderer.getClearColor(clear);
    composite.uniforms.background.value.copy(clear);
    const hidden = [],
      passes = [];
    try {
      // Offscreen targets must remain linear until the final composite.
      renderer.toneMapping = THREE.NoToneMapping;
      renderer.setRenderTarget(beauty);
      renderer.render(scene, camera);
      scene.traverse((object) => {
        if (!object.isMesh) return;
        const pass = object.userData.skinDiffusePass;
        if (pass) {
          passes.push(pass);
          pass.value = 1;
        } else if (object.visible) {
          hidden.push(object);
          object.visible = false;
        }
      });
      renderer.shadowMap.autoUpdate = false;
      renderer.setClearColor(0x000000, 0);
      renderer.setRenderTarget(diffuse);
      renderer.render(scene, camera);
      for (const pass of passes) pass.value = 0;
      for (const object of hidden) object.visible = true;
      blur.uniforms.projection.value.set(
        camera.projectionMatrix.elements[0] * 0.5,
        camera.projectionMatrix.elements[5] * 0.5,
      );
      blur.uniforms.nearFar.value.set(camera.near, camera.far);
      blur.uniforms.radius.value = state.radiusMm * 0.001;
      quad.material = blur;
      blur.uniforms.image.value = diffuse.texture;
      blur.uniforms.direction.value.set(1, 0);
      renderer.setRenderTarget(horizontal);
      renderer.render(quadScene, quadCamera);
      blur.uniforms.image.value = horizontal.texture;
      blur.uniforms.direction.value.set(0, 1);
      renderer.setRenderTarget(vertical);
      renderer.render(quadScene, quadCamera);
      renderer.toneMapping = tone;
      renderer.setClearColor(clear, alpha);
      renderer.setRenderTarget(previousTarget);
      quad.material = composite;
      renderer.render(quadScene, quadCamera);
      state.renders++;
      state.cpuSubmitMs = performance.now() - start;
    } finally {
      for (const pass of passes) pass.value = 0;
      for (const object of hidden) object.visible = true;
      renderer.toneMapping = tone;
      renderer.shadowMap.autoUpdate = shadow;
      renderer.setClearColor(clear, alpha);
      renderer.setRenderTarget(previousTarget);
    }
  }
  return {
    state,
    render,
    dispose() {
      for (const rt of [beauty, diffuse, horizontal, vertical]) rt.dispose();
      blur.dispose();
      composite.dispose();
      geometry.dispose();
    },
  };
}
