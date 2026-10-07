import * as THREE from "three";

import {
  loadModel,
  loadSkin,
  loadInternalAnatomy,
  readJSON,
  verifiedBytes,
} from "./model.js";

import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import { prepareSkinNormals, rebuildSkinNormals } from "./skin_rest_normals.js";

import { installFEMSurface } from "./layered_fem_surface.js";

import { GPUHeadFEM } from "./full_head_fem_gpu.js";
import { CPUHeadFEM } from "./tissue_cpu_backend.js";
import { createFineSkinPatch } from "./fine_skin_patch.js";
import { createSkinDiffusion } from "./skin_diffusion.js";
import { improveAuthoredMaterials } from "./authored_skin_material.js";

import { guardGPUPage } from "./gpu_page_lifecycle.js";
import { createLazyAnatomy } from "./lazy_anatomy.js";
import { createFrameMetrics } from "./frame_metrics.js";

const stage = document.querySelector("#stage"),
  message = document.querySelector("#message"),
  status = document.querySelector("#tissueStatus");

const debug = {
  ready: false,
  errors: [],
  samples: [],
  coverage: null,
  engine: "initializing",
};
window.__fullHeadFEM = debug;
const startupBegin = performance.now();
debug.startup = { startedAtMs: startupBegin };
const frameMetrics = createFrameMetrics();
debug.measurement = () => frameMetrics.snapshot(debug.clock);

let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true });
} catch (e) {
  message.textContent = e.message;
  throw e;
}

renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
renderer.setSize(stage.clientWidth, stage.clientHeight);
renderer.setClearColor(0x111d21);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
stage.append(renderer.domElement);
const gl = renderer.getContext();
status.dataset.graphics = JSON.stringify({
  renderer: gl.getParameter(gl.RENDERER),
  floatLinear: !!gl.getExtension("OES_texture_float_linear"),
  precision: gl.getShaderPrecisionFormat(gl.FRAGMENT_SHADER, gl.HIGH_FLOAT)
    .precision,
});

const scene = new THREE.Scene(),
  camera = new THREE.PerspectiveCamera(
    35,
    stage.clientWidth / stage.clientHeight,
    0.001,
    10,
  ),
  controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, -0.025);
camera.position.set(0, 0.025, 0.555);
controls.update();

scene.add(new THREE.HemisphereLight(0xe8eeff, 0x4b5054, 0.8));
for (const [x, y, z, power] of [
  [-0.3, 0.4, 0.5, 3],
  [0.3, 0.1, 0.2, 0.7],
  [0, -0.12, 0.42, 0.8],
]) {
  const light = new THREE.DirectionalLight(0xffffff, power);
  light.position.set(x, y, z);
  light.target.position.set(0, 0, 0.06);
  scene.add(light, light.target);
  if (x < 0) {
    light.castShadow = true;
    light.shadow.mapSize.set(2048, 2048);
    Object.assign(light.shadow.camera, {
      left: -0.25,
      right: 0.25,
      top: 0.25,
      bottom: -0.25,
      near: 0.2,
      far: 1,
    });
    light.shadow.bias = -0.0001;
    light.shadow.normalBias = 0.0005;
  }
}

// Repeatable inspection lights; normal application lighting remains the default.
const inspectionLights = scene.children.filter(item => item.isDirectionalLight);
debug.setInspectionLighting = (preset = "studio") => {
  const setups = {
    studio: [[-0.3, 0.4, 0.5, 3], [0.3, 0.1, 0.2, .7], [0, -.12, .42, .8]],
    side: [[-.45, .08, .2, 3], [.3, .1, .2, .2], [0, -.12, .42, .3]],
    overhead: [[-.1, .5, .15, 3], [.3, .1, .2, .4], [0, -.12, .42, .2]],
  };
  if (!setups[preset]) throw Error("Unknown inspection lighting: " + preset);
  inspectionLights.forEach((light, index) => {
    const [x, y, z, intensity] = setups[preset][index];
    light.position.set(x, y, z); light.intensity = intensity;
  });
  debug.inspectionLighting = preset;
  requestRender();
};

const skinDiffusion = createSkinDiffusion(renderer, scene, camera);
debug.skinDiffusion = skinDiffusion.state;
const scatteringControl = document.querySelector("#scattering");
scatteringControl.checked = skinDiffusion.state.supported;
scatteringControl.disabled = !skinDiffusion.state.supported;
scatteringControl.onchange = () => {
  skinDiffusion.state.enabled = scatteringControl.checked;
  requestRender();
};

let activeSolver = null,
  finePatch = null,
  surfacePool = null;
function disposeSceneResources() {
  const textures = new Set(),
    materials = new Set(),
    geometries = new Set();
  scene.traverse((item) => {
    if (!item.isMesh) return;
    geometries.add(item.geometry);
    if (item.userData.authoredMicroTexture)
      textures.add(item.userData.authoredMicroTexture);
    for (const material of [item.material, item.customDepthMaterial].flat()) {
      if (!material) continue;
      materials.add(material);
      for (const value of Object.values(material))
        if (value?.isTexture) textures.add(value);
    }
  });
  if (surfacePool?.resources) {
    const r = surfacePool.resources;
    for (const t of [r.posed, r.initial, ...r.frames]) textures.add(t);
  }
  for (const t of textures) t.dispose();
  for (const m of materials) m.dispose();
  for (const g of geometries) g.dispose();
}

const lifecycle = guardGPUPage({
  document,
  window,
  canvas: renderer.domElement,
  onContextLost: () => {
    debug.graphicsSuspended = true;
    status.textContent =
      "Graphics context lost. Tissue state is preserved while the browser restores rendering.";
    message.hidden = false;
    message.textContent = status.textContent;
  },
  onContextRestored: () => {
    debug.graphicsSuspended = false;
    debug.contextRestorations = (debug.contextRestorations || 0) + 1;
    message.hidden = true;
    requestRender();
  },
  onStop: (reason) => {
    debug.errors.push(reason);
    debug.graphicsStopped = true;
    status.textContent = reason;
    document.querySelector("#running").checked = false;
    for (const input of document.querySelectorAll(
      "aside input,aside select,aside button",
    ))
      input.disabled = true;
    message.hidden = false;
    message.textContent = "The tissue solver stopped. " + reason;
    message.style.background = "#111d21e8";
  },
  onDispose: () => {
    skinDiffusion.dispose();
    activeSolver?.dispose();
    finePatch?.dispose();
    disposeSceneResources();
    controls.dispose();
    renderer.dispose();
  },
});

let pending = false;
function render() {
  if (lifecycle.canSubmit) {
    const begin = performance.now();
    skinDiffusion.render(
      document.querySelector("#heatmap").checked ||
        document.querySelector("#fineHeatmap").checked ||
        document.querySelector("#cage").checked,
    );
    frameMetrics.submitted(begin);
    if (debug.startup.firstFaceSubmittedMs === undefined && debug.facePrepared)
      debug.startup.firstFaceSubmittedMs = performance.now() - startupBegin;
  }
}
function requestRender() {
  if (pending || lifecycle.disposed) return;
  pending = true;
  requestAnimationFrame(() => {
    pending = false;
    render();
  });
}
controls.addEventListener("change", requestRender);

document.addEventListener("visibilitychange", () => {
  frameMetrics.suspend();
  if (!document.hidden) requestRender();
});

debug.focusForehead = () => {
  controls.target.set(0, 0.078, 0.055);
  // Keep the enlarged authored forehead in front of the camera and include
  // its silhouette; the previous close-up filled the entire viewport.
  camera.position.set(0, 0.078, 0.34);
  controls.update();
  requestRender();
};
if (document.querySelector("#foreheadDetail"))
  document.querySelector("#foreheadDetail").onclick = debug.focusForehead;
debug.setView = (view) => {
  controls.target.set(0, 0, -0.025);
  const r = Math.max(0.64, 0.35 / camera.aspect);
  camera.position
    .copy(controls.target)
    .add(
      new THREE.Vector3(
        ...(view === "profile"
          ? [r, 0, 0]
          : view === "back"
            ? [0, 0, -r]
            : view === "three-quarter"
              ? [r * 0.65, 0, r * 0.76]
              : [0, 0, r]),
      ),
    );
  controls.update();
  requestRender();
};
for (const button of document.querySelectorAll("[data-view]"))
  button.onclick = () => debug.setView(button.dataset.view);

new ResizeObserver(() => {
  camera.aspect = stage.clientWidth / stage.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(stage.clientWidth, stage.clientHeight);
  if (debug.ready) debug.setView("front");
  requestRender();
}).observe(stage);

async function start() {
  const query = new URLSearchParams(location.search);
  const { model, arrays, bundle, bytes } = await loadModel(
    window.__tissueRuntime?.tissueBase || "output/neutral-tissue/v1/",
  );
  message.textContent = "Loading the authored full head…";
  const skin = await loadSkin(model);
  scene.add(skin.scene);
  skin.scene.updateMatrixWorld(true);
  const anatomyControl = document.querySelector("#anatomy");
  anatomyControl.disabled = !model.internal_anatomy_url;
  const anatomy = createLazyAnatomy({
    load: () => loadInternalAnatomy(model),
    attach: internal => scene.add(internal.scene),
    show: (internal, visible) => {
      if (internal) internal.scene.visible = visible;
      skin.scene.visible = !visible;
      requestRender();
    },
    discard: internal => internal.scene.traverse(item => {
      item.geometry?.dispose();
      for (const material of [item.material].flat()) material?.dispose();
    }),
    isDisposed: () => lifecycle.disposed,
    onState: state => {
      debug.anatomyLoading = state;
      if (state.phase === "failed") {
        anatomyControl.checked = false;
        anatomyControl.title = "Anatomy could not load. Select again to retry: " + state.error;
        status.textContent = "Anatomy could not load. Select anatomy again to retry.";
      } else anatomyControl.title = state.phase === "loading" ? "Loading registered anatomy" : "Show registered anatomy";
    },
  });
  debug.anatomyLoading = { ...anatomy.state };
  anatomyControl.onchange = () => anatomy.setVisible(anatomyControl.checked);

  const authoredMeshes = [];
  skin.scene.traverse((item) => {
    if (item.isMesh) authoredMeshes.push(item);
  });
  debug.eyeSamplePixels = () =>
    authoredMeshes
      .filter((m) => /_iris$/.test(m.name))
      .flatMap((mesh) => {
        const box = new THREE.Box3().setFromObject(mesh),
          centre = box.getCenter(new THREE.Vector3()),
          radius = (box.max.x - box.min.x) * 0.3;
        return [-1, 1].map((side) => {
          const p = centre.clone();
          p.x += side * radius;
          p.project(camera);
          return [
            Math.round((p.x * 0.5 + 0.5) * renderer.domElement.clientWidth),
            Math.round((0.5 - p.y * 0.5) * renderer.domElement.clientHeight),
          ];
        });
      });
  for (const item of authoredMeshes) {
    improveAuthoredMaterials(item);
    item.castShadow = !/corneal|tear|iris|pupil/i.test(item.name);
    item.receiveShadow = true;
    prepareSkinNormals(item);
    for (const m of Array.isArray(item.material)
      ? item.material
      : [item.material])
      if (m.normalMap && !item.geometry.attributes.tangent)
        m.normalScale.y = Math.abs(m.normalScale.y);
  }

  message.textContent = "Preparing skin and full-head tissue…";
  const shape = await readJSON(model.skin_shape_url);

  for (const part of shape.parts) {
    const item = skin.scene.getObjectByName(part.node),
      b = await verifiedBytes(part.url, part.sha256, "authored head shape");
    const delta = new Float32Array(b),
      p = item.geometry.attributes.position,
      linear = new THREE.Matrix3().setFromMatrix4(
        item.matrixWorld.clone().invert(),
      ),
      point = new THREE.Vector3();
    for (let j = 0; j < delta.length; j += 3) {
      point.fromArray(delta, j).applyMatrix3(linear);
      p.array[j] += point.x;
      p.array[j + 1] += point.y;
      p.array[j + 2] += point.z;
    }
    p.needsUpdate = true;
    if (delta.some((v) => v !== 0)) {
      rebuildSkinNormals(item);
      item.geometry.computeBoundingSphere();
    }
  }

  if (bundle.smoothData) {
    finePatch = createFineSkinPatch(model, arrays, requestRender, (reason) => {
      debug.fineSkinFailure = reason;
      const control = document.querySelector("#fineHeatmap");
      if (control) control.disabled = true;
      status.textContent = "Fine skin detail stopped: " + reason;
    });
    debug.fineSkinPatch = finePatch.state;
    if (document.querySelector("#fineHeatmap"))
      document.querySelector("#fineHeatmap").onchange = (e) => {
        finePatch.heatmap.value = e.target.checked ? 1 : 0;
        requestRender();
      };
  }
  const pool = (surfacePool = { fine: finePatch, geometricNormals: query.get("normals") === "geometric" }),
    transfers = [];
  for (const part of bundle.parts)
    if (part.active_vertex_count) {
      const item = skin.scene.getObjectByName(shape.parts[part.index].node);
      transfers.push(
        installFEMSurface(item, bundle, bytes, arrays.rest, part.index, pool),
      );
    }

  debug.coverage = {
    nodes: model.nodes,
    cells: model.tetrahedra,
    mainSkinVertices: bundle.parts[0].active_vertex_count,
    softVertices: bundle.parts.reduce((s, p) => s + p.active_vertex_count, 0),
    bounds: model.construction.bounds_m,
    regions: [
      "scalp",
      "forehead",
      "eyelid surrounds",
      "nose",
      "cheeks",
      "lips",
      "jaw",
      "both pinnae",
      "neck",
      "shoulders",
    ],
    thinFeatureCellsHomogenized: true,
  };
  debug.sampleSurface = (ids) => {
    const mesh = skin.scene.getObjectByName(model.skin_main_node),
      p = mesh.geometry.attributes.position,
      packed = engine.latestPacked,
      N = model.nodes,
      S = model.surface_nodes;
    return ids.map((id) => {
      if (!Number.isInteger(id) || id < 0 || id >= p.count)
        throw Error(
          `Skin sample ${id} is outside the current ${p.count}-vertex mesh`,
        );
      const q = new THREE.Vector3()
          .fromBufferAttribute(p, id)
          .applyMatrix4(mesh.matrixWorld)
          .toArray(),
        position = [0, 0, 0],
        j = id * 16;
      if (!packed) return { id, rest: q, position: q.slice() };
      for (let a = 0; a < 8; a++) {
        const n = bundle.smoothData[j + a],
          w = bundle.smoothData[j + 8 + a];
        for (let d = 0; d < 3; d++) {
          let v = packed[n * 4 + d];
          for (let c = 0; c < 3; c++)
            v +=
              packed[N * 4 + c * S * 4 + n * 4 + d] *
              (q[c] - arrays.rest[n * 3 + c]);
          position[d] += w * v;
        }
      }
      return { id, rest: q, position };
    });
  };
  const layers = [];
  for (let l = 0; l < 3; l++) {
    const g = new THREE.BufferGeometry(),
      p = new Float32Array(model.surface_nodes * 3);
    p.set(arrays.rest.subarray(l * p.length, (l + 1) * p.length));
    g.setAttribute("position", new THREE.BufferAttribute(p, 3));
    g.setIndex(new THREE.BufferAttribute(arrays.faces, 1));
    const mesh = new THREE.Mesh(
      g,
      new THREE.MeshBasicMaterial({
        color: [0x77ccbb, 0xe5b274, 0xcc7766][l],
        wireframe: true,
        transparent: true,
        opacity: 0.45,
      }),
    );
    mesh.visible = false;
    layers.push(mesh);
    scene.add(mesh);
  }

  layers[0].geometry.computeBoundingSphere();
  layers[0].geometry.boundingSphere.radius += 0.03;
  const main = skin.scene.getObjectByName(model.skin_main_node);
  document.querySelector("#cage").onchange = (e) => {
    for (const layer of layers) layer.visible = e.target.checked;
    main.material.transparent = e.target.checked;
    main.material.opacity = e.target.checked ? 0.28 : 1;
    main.material.depthWrite = !e.target.checked;
    requestRender();
  };
  document.querySelector("#heatmap").onchange = (e) => {
    transfers[0].setHeatmap(e.target.checked);
    requestRender();
  };

  debug.capabilities = model.capabilities || null;
  status.dataset.capabilities = JSON.stringify(debug.capabilities);
  status.dataset.build = window.__tissueRuntime?.build || "legacy";
  message.hidden = true;
  debug.facePrepared = true;
  debug.setView("front");
  requestRender();
  status.textContent = "Starting the tissue solver…";
  if (lifecycle.disposed || debug.graphicsStopped) return;
  let engine;
  try {
    if (query.get("backend") === "cpu")
      throw Error("CPU backend explicitly selected");
    engine = await GPUHeadFEM.create(model, arrays);
  } catch (error) {
    debug.gpuBootFailure = error.message;
    engine = await CPUHeadFEM.create(model, arrays, error.message);
  }
  if (engine.boundedSubmissions && ["8", "16"].includes(query.get("gpuBatch")))
    engine.cgBatchSize = Number(query.get("gpuBatch"));
  activeSolver = engine;
  if (lifecycle.disposed) {
    engine.dispose();
    return;
  }
  engine.device?.lost.then((info) => {
    if (!lifecycle.disposed && activeSolver === engine) {
      debug.gpuDeviceLoss = info.message || info.reason;
      requestControlUpdate();
    }
  });
  debug.engine = engine.device
    ? "WebGPU implicit FEM"
    : "CPU worker implicit FEM";
  debug.adapter = engine.adapterInfo;
  status.dataset.compute = JSON.stringify(engine.adapterInfo);
  debug.timestampQueries = engine.timestamp && !engine.boundedSubmissions;
  debug.ready = true;
  debug.startup.solverReadyMs = performance.now() - startupBegin;
  debug.solver = engine;

  for (const name of [
    "fat",
    "sag",
    "gravity",
    "grab",
    "release",
    "resetShape",
    "running",
    "quality",
  ])
    document.querySelector("#" + name).disabled = false;

  let controlUpdates = 0,
    settling = false,
    settleSteps = 0,
    quietSteps = 0,
    settleStart = 0,
    timeAccumulator = 0,
    resetPending = false;
  const fixedDt = 1 / 60;
  debug.clock = {
    simulationSeconds: 0,
    activeWallSeconds: 0,
    discardedSeconds: 0,
    backlogSeconds: 0,
  };
  const requestControlUpdate = () => {
    debug.settlement = null;
    quietSteps = 0;
    controlUpdates = Math.max(controlUpdates, 12);
    if (!settling) {
      settleSteps = 0;
      quietSteps = 0;
      settleStart = performance.now();
    }
    settling = true;
  };
  let requested = { fatPercent: 0, sagPercent: 0, gravity: 1 },
    applied = { ...requested },
    grab = null,
    currentGrab = null,
    running = false,
    lastTime = performance.now(),
    lastStatus = 0;
  const frameSamples = [],
    positions = new Float32Array(model.nodes * 3),
    checkpointVelocity = new Float64Array(model.nodes * 3);
  positions.set(arrays.rest);
  document.querySelector("#running").checked = running;
  if (!running)
    status.textContent =
      "Tissue ready · " +
      debug.engine +
      ". Change fat/softness or drag skin to solve.\n" +
      model.tetrahedra.toLocaleString() +
      " volume cells · " +
      (model.shell_hinges || 0).toLocaleString() +
      " bending hinges.\nOn-demand static FEM equilibrium; Run simulation uses fixed-time dynamics. Rest anatomy shares the frame; full anatomical coupling remains pending.";

  const update = () => {
    requested = {
      fatPercent: Number(document.querySelector("#fat").value),
      sagPercent: Number(document.querySelector("#sag").value),
      gravity: document.querySelector("#gravity").checked ? 1 : 0,
    };
    document.querySelector("#fatValue").textContent =
      "+" + requested.fatPercent + "%";
    document.querySelector("#sagValue").textContent =
      requested.sagPercent + "%";
    requestControlUpdate();
  };
  for (const name of ["fat", "sag"])
    document.querySelector("#" + name).oninput = update;
  document.querySelector("#gravity").onchange = update;

  if (document.querySelector("#grabStiffness"))
    document.querySelector("#grabStiffness").oninput = (e) => {
      document.querySelector("#grabStiffnessValue").textContent =
        e.target.value + " N/m";
      requestControlUpdate();
    };

  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(0.0018, 12, 8),
    new THREE.MeshBasicMaterial({
      color: 0xf1c578,
      wireframe: true,
      depthTest: false,
    }),
  );
  marker.visible = false;
  marker.renderOrder = 20;
  scene.add(marker);

  document.querySelector("#release").onclick = () => {
    grab = null;
    currentGrab = null;
    marker.visible = false;
    requestControlUpdate();
    requestRender();
  };
  document.querySelector("#resetShape").onclick = () => {
    resetPending = true;
    requestControlUpdate();
  };

  document.addEventListener("visibilitychange", () => {
    lastTime = performance.now();
    timeAccumulator = 0;
  });
  document.querySelector("#running").onchange = (e) => {
    running = e.target.checked;
    timeAccumulator = 0;
    lastTime = performance.now();
  };
  debug.pause = () => {
    running = false;
    document.querySelector("#running").checked = false;
  };
  debug.resume = () => {
    timeAccumulator = 0;
    running = true;
    document.querySelector("#running").checked = true;
    lastTime = performance.now();
  };
  debug.setParameters = (p) => {
    document.querySelector("#fat").value = p.fatPercent;
    document.querySelector("#sag").value = p.sagPercent;
    document.querySelector("#gravity").checked = !!p.gravity;
    update();
  };

  const ray = new THREE.Raycaster(),
    mouse = new THREE.Vector2(),
    plane = new THREE.Plane(),
    point = new THREE.Vector3();
  let dragNode = null;
  const pointerRay = (e) => {
    const r = renderer.domElement.getBoundingClientRect();
    mouse.set(
      ((e.clientX - r.left) / r.width) * 2 - 1,
      1 - ((e.clientY - r.top) / r.height) * 2,
    );
    ray.setFromCamera(mouse, camera);
  };
  document.querySelector("#grab").onchange = (e) => {
    controls.enabled = !e.target.checked;
    renderer.domElement.style.cursor = e.target.checked ? "grab" : "";
  };

  const closest = (world) => {
    const p = layers[0].geometry.attributes.position;
    let best = -1,
      distance = Infinity;
    for (let n = 0; n < p.count; n++) {
      const d =
        (p.array[n * 3] - world.x) ** 2 +
        (p.array[n * 3 + 1] - world.y) ** 2 +
        (p.array[n * 3 + 2] - world.z) ** 2;
      if (d < distance) {
        best = n;
        distance = d;
      }
    }
    return best;
  };

  renderer.domElement.addEventListener("pointerdown", (e) => {
    if (!document.querySelector("#grab").checked) return;
    pointerRay(e);
    const hit = ray.intersectObject(layers[0])[0];
    if (!hit) return;
    dragNode = closest(hit.point);
    const p = layers[0].geometry.attributes.position;
    grab = {
      node: dragNode,
      target: Array.from(p.array.subarray(dragNode * 3, dragNode * 3 + 3)),
    };
    currentGrab = { node: dragNode, target: [...grab.target] };
    marker.position.copy(hit.point);
    marker.visible = true;
    plane.setFromNormalAndCoplanarPoint(
      camera.getWorldDirection(new THREE.Vector3()),
      hit.point,
    );
    renderer.domElement.setPointerCapture(e.pointerId);
    renderer.domElement.style.cursor = "grabbing";
  });

  renderer.domElement.addEventListener("pointermove", (e) => {
    if (dragNode === null) return;
    pointerRay(e);
    if (ray.ray.intersectPlane(plane, point)) {
      const origin = new THREE.Vector3().fromArray(arrays.rest, dragNode * 3),
        delta = point.clone().sub(origin);
      if (delta.length() > 0.02) delta.setLength(0.02);
      grab.target = origin.add(delta).toArray();
      marker.position.fromArray(grab.target);
      if (!running) {
        requestControlUpdate();
        requestRender();
      }
    }
  });
  renderer.domElement.addEventListener("pointerup", () => {
    dragNode = null;
    renderer.domElement.style.cursor = "grab";
  });
  renderer.domElement.addEventListener("pointercancel", () => {
    dragNode = null;
  });

  debug.grabPoint = (position, offset) => {
    const node = closest(new THREE.Vector3(...position));
    grab = {
      node,
      target: Array.from(
        layers[0].geometry.attributes.position.array.subarray(
          node * 3,
          node * 3 + 3,
        ),
      ).map((v, k) => v + offset[k]),
    };
    currentGrab = {
      node,
      target: Array.from(
        layers[0].geometry.attributes.position.array.subarray(
          node * 3,
          node * 3 + 3,
        ),
      ),
    };
    marker.position.fromArray(grab.target);
    marker.visible = true;
    requestControlUpdate();
    return node;
  };
  debug.release = () => {
    grab = null;
    currentGrab = null;
    marker.visible = false;
    requestControlUpdate();
  };

  debug.projectPoint = (position) => {
    const p = new THREE.Vector3(...position).project(camera),
      r = renderer.domElement.getBoundingClientRect();
    return {
      x: r.left + ((p.x + 1) * r.width) / 2,
      y: r.top + ((1 - p.y) * r.height) / 2,
    };
  };

  async function tick() {
    if (lifecycle.disposed || debug.graphicsStopped) return;

    if (resetPending) {
      resetPending = false;
      await engine.reset();
      positions.set(arrays.rest);
      checkpointVelocity.fill(0);
      timeAccumulator = 0;
      for (const name of ["fat", "sag"])
        document.querySelector("#" + name).value = 0;
      document.querySelector("#gravity").checked = true;
      update();
      applied = { ...requested };
      grab = null;
      currentGrab = null;
      marker.visible = false;
      lastTime = performance.now();
      debug.resetCount = (debug.resetCount || 0) + 1;
    }
    const now = performance.now();
    if ((!running && controlUpdates === 0) || !lifecycle.canSubmit) {
      lastTime = now;
      requestAnimationFrame(tick);
      return;
    }

    const elapsed = Math.max(0, (now - lastTime) / 1000);
    lastTime = now;
    const controlDt = Math.min(0.1, elapsed);
    if (running) {
      debug.clock.activeWallSeconds += elapsed;
      timeAccumulator += elapsed;
      if (timeAccumulator > 0.25) {
        debug.clock.discardedSeconds += timeAccumulator - 0.25;
        timeAccumulator = 0.25;
      }
      if (timeAccumulator < fixedDt) {
        requestAnimationFrame(tick);
        return;
      }
      timeAccumulator -= fixedDt;
    } else timeAccumulator = 0;
    const frameDt = fixedDt;
    debug.clock.backlogSeconds = timeAccumulator;
    const stepBegin = performance.now();

    for (const name of ["fatPercent", "sagPercent"])
      applied[name] += Math.max(
        -100 * controlDt,
        Math.min(100 * controlDt, requested[name] - applied[name]),
      );
    applied.gravity = requested.gravity;
    if (
      ["fatPercent", "sagPercent"].some(
        (name) => Math.abs(requested[name] - applied[name]) > 0.001,
      )
    )
      requestControlUpdate();

    if (grab) {
      if (!currentGrab || currentGrab.node !== grab.node)
        currentGrab = { node: grab.node, target: [...grab.target] };
      currentGrab.stiffness = Number(
        document.querySelector("#grabStiffness")?.value || 100,
      );
      for (let d = 0; d < 3; d++)
        currentGrab.target[d] += Math.max(
          -0.08 * frameDt,
          Math.min(0.08 * frameDt, grab.target[d] - currentGrab.target[d]),
        );
    }

    const unsettled =
      debug.samples.at(-1)?.staticResidualN > 0.001 ||
      debug.samples.at(-1)?.residualN > 0.01;
    const high =
      unsettled || document.querySelector("#quality").value === "reference";
    let result;

    const solveParameters = {
      ...applied,
      grab: currentGrab
        ? { ...currentGrab, target: [...currentGrab.target] }
        : null,
    };
    try {
      result = await engine.step(solveParameters, {
        dt: frameDt,
        quasiStatic: !running,
        substeps: 1,
        newton: high ? 4 : 2,
        cg: !running ? (high ? 256 : 64) : high ? 64 : 24,
        block: true,
      });
    } catch (e) {
      if (engine.device && model.skin_url && !lifecycle.disposed) {
        const failed = engine;
        status.textContent =
          "Restoring the last solved tissue state on the CPU…";
        try {
          const replacement = await CPUHeadFEM.create(model, arrays, e.message);
          await replacement.restore(positions, checkpointVelocity);
          engine = replacement;
          activeSolver = replacement;
          debug.solver = replacement;
          debug.engine = "CPU worker implicit FEM";
          debug.adapter = replacement.adapterInfo;
          status.dataset.compute = JSON.stringify(replacement.adapterInfo);
          debug.recovery = {
            reason: e.message,
            positionsPreserved: true,
            frame: debug.samples.length,
          };
          failed.dispose();
          lastTime = performance.now();
          requestControlUpdate();
          requestAnimationFrame(tick);
          return;
        } catch (recoveryError) {
          running = false;
          lifecycle.fail("Tissue recovery failed: " + recoveryError.message);
          return;
        }
      }
      running = false;
      lifecycle.fail("Tissue FEM stopped: " + e.message);
      return;
    }

    if (lifecycle.disposed || debug.graphicsStopped) return;

    if (
      !Number.isFinite(result.stats.minJ) ||
      result.stats.minJ <= 0 ||
      !Number.isFinite(result.stats.residualN)
    ) {
      running = false;
      lifecycle.fail(
        "The tissue solve became invalid. Reload this page before continuing.",
      );
      return;
    }

    debug.clock.simulationSeconds += result.stats.simulatedSeconds ?? frameDt;
    const uploadBegin = performance.now();
    transfers[0].updatePacked(result.packed);
    finePatch?.update(result.packed);

    let maxSurfaceStepM = 0;
    for (let n = 0; n < model.surface_nodes; n++)
      maxSurfaceStepM = Math.max(
        maxSurfaceStepM,
        Math.hypot(
          result.packed[n * 4] - positions[n * 3],
          result.packed[n * 4 + 1] - positions[n * 3 + 1],
          result.packed[n * 4 + 2] - positions[n * 3 + 2],
        ),
      );
    for (let n = 0; n < model.nodes; n++) {
      for (let d = 0; d < 3; d++)
        checkpointVelocity[n * 3 + d] =
          ((result.packed[n * 4 + d] - positions[n * 3 + d]) / frameDt) *
          Math.exp(Math.log(0.98) * 60 * frameDt);
      positions[n * 3] = result.packed[n * 4];
      positions[n * 3 + 1] = result.packed[n * 4 + 1];
      positions[n * 3 + 2] = result.packed[n * 4 + 2];
    }
    debug.positions = positions;

    // The coarse cage is also used for picking; the dense skin never comes back to CPU.

    for (let l = 0; l < 3; l++) {
      const p = layers[l].geometry.attributes.position;
      p.array.set(
        positions.subarray(l * p.array.length, (l + 1) * p.array.length),
      );
      p.needsUpdate = true;
    }
    document.querySelector("#fatValue").textContent =
      "+" +
      applied.fatPercent.toFixed(0) +
      "%" +
      (Math.abs(requested.fatPercent - applied.fatPercent) > 0.5
        ? " / target " + requested.fatPercent + "%"
        : "");
    document.querySelector("#sagValue").textContent =
      applied.sagPercent.toFixed(0) +
      "%" +
      (Math.abs(requested.sagPercent - applied.sagPercent) > 0.5
        ? " / target " + requested.sagPercent + "%"
        : "");
    render();
    controlUpdates = Math.max(0, controlUpdates - 1);
    const sample = {
      ...result.stats,
      maxSurfaceStepM,
      frameWorkMs: performance.now() - stepBegin,
      uploadAndSubmitMs: performance.now() - uploadBegin,
      frameIntervalMs: now - debug.lastFrameTime || null,
      applied: {
        fatPercent: solveParameters.fatPercent,
        sagPercent: solveParameters.sagPercent,
        gravity: solveParameters.gravity,
      },
      grabbed: !!solveParameters.grab,
    };
    const atTarget = ["fatPercent", "sagPercent"].every(
      (k) => Math.abs(requested[k] - applied[k]) < 0.001,
    );
    quietSteps =
      atTarget &&
      sample.residualN < 0.001 &&
      sample.staticResidualN < 0.001 &&
      sample.maxNodeSpeedMps < 0.00012 &&
      maxSurfaceStepM < 0.000002
        ? quietSteps + 1
        : 0;
    if (!settling && quietSteps >= 3) {
      debug.settlement = {
        converged: true,
        capped: false,
        steps: settleSteps,
        maxSurfaceStepM,
        residualN: sample.residualN,
        staticResidualN: sample.staticResidualN,
        maxNodeSpeedMps: sample.maxNodeSpeedMps,
        kineticEnergyJ: sample.kineticEnergyJ,
      };
      if (!running) controlUpdates = 0;
    }
    if (settling) {
      settleSteps++;
      const capped =
        settleSteps >= 120 || performance.now() - settleStart > 45000;
      if (quietSteps >= 3 || capped) {
        settling = false;
        if (quietSteps >= 3) controlUpdates = 0;
        debug.settlement = {
          converged: quietSteps >= 3,
          capped,
          steps: settleSteps,
          maxSurfaceStepM,
          residualN: sample.residualN,
          staticResidualN: sample.staticResidualN,
          maxNodeSpeedMps: sample.maxNodeSpeedMps,
          kineticEnergyJ: sample.kineticEnergyJ,
        };
      } else if (!running) controlUpdates = Math.max(controlUpdates, 1);
    }
    status.dataset.lastSolve = JSON.stringify(sample);
    status.dataset.settlement = JSON.stringify(debug.settlement);
    debug.lastFrameTime = now;
    debug.samples.push(sample);
    if (debug.samples.length > 600) debug.samples.shift();
    frameSamples.push(sample);
    if (frameSamples.length > 90) frameSamples.shift();

    if (now - lastStatus > 400 || (!running && controlUpdates === 0)) {
      const good = frameSamples.slice(5),
        intervals = good
          .map((s) => s.frameIntervalMs)
          .filter(Number.isFinite)
          .sort((a, b) => a - b),
        median = intervals[Math.floor(intervals.length / 2)],
        gpu = good
          .map((s) => s.gpuMs)
          .filter(Number.isFinite)
          .sort((a, b) => a - b),
        gpuMedian = gpu[Math.floor(gpu.length / 2)];
      status.textContent = `${median ? (1000 / median).toFixed(1000 / median < 10 ? 1 : 0) + " measured updates/s" : "Measuring…"} · ${engine.adapterInfo.vendor}${engine.device ? " GPU" : ""}\n${model.nodes.toLocaleString()} nodes · ${model.tetrahedra.toLocaleString()} cells\n${bundle.parts[0].active_vertex_count.toLocaleString()} main skin vertices · full head\n${engine.device && Number.isFinite(gpuMedian) ? "GPU timestamp " + gpuMedian.toFixed(1) : "Solver elapsed " + sample.elapsedMs.toFixed(1)} ms\nMinimum J ${sample.minJ.toFixed(3)} · residual ${sample.residualN.toExponential(1)} N\nStatic force ${sample.staticResidualN.toExponential(1)} N · speed ${(sample.maxNodeSpeedMps * 1000).toFixed(3)} mm/s\nApplied fat +${applied.fatPercent.toFixed(0)}% · softness ${applied.sagPercent.toFixed(0)}%\n${settling ? "Settling tissue…" : debug.settlement?.capped ? "Solve budget reached; tissue is not settled" : "Last solve complete"} · surface step ${(maxSurfaceStepM * 1000).toFixed(3)} mm`;
      if (debug.fineSkinFailure)
        status.textContent +=
          "\nFine skin detail unavailable: " + debug.fineSkinFailure;
      lastStatus = now;
      debug.performance = {
        medianUpdateMs: median,
        medianGpuMs: gpuMedian,
        samples: good.length,
      };
      if (running)
        status.textContent +=
          "\nSimulated " +
          debug.clock.simulationSeconds.toFixed(2) +
          " s · behind by " +
          (debug.clock.backlogSeconds + debug.clock.discardedSeconds).toFixed(
            2,
          ) +
          " s";
      if (debug.recovery)
        status.textContent +=
          "\nRecovered on CPU; last tissue state preserved.";
    }

    requestAnimationFrame(tick);
  }

  if (model.skin_url) requestControlUpdate();
  requestAnimationFrame(tick);
}

start().catch((e) => {
  activeSolver?.dispose();
  debug.errors.push(e.message);
  status.textContent = "Controls unavailable: " + e.message + ".";
  message.hidden = true;
  requestRender();
});
requestRender();
