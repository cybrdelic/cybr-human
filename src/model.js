import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

async function response(url) {
  const result = await fetch(url);
  if (!result.ok) throw Error(`Missing ${url}`);
  return result;
}

export async function readJSON(url) {
  return (await response(url)).json();
}

async function digest(bytes) {
  const hash = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(hash), (value) =>
    value.toString(16).padStart(2, "0"),
  ).join("");
}

export async function verifiedBytes(url, expected, label) {
  const bytes = await (await response(url)).arrayBuffer();
  if (!expected || (await digest(bytes)) !== expected)
    throw Error(`Stale ${label}`);
  return bytes;
}

/** Load one complete tissue model. Alternate experiments do not enter this path. */
export async function loadModel(base) {
  const model = await readJSON(base + "model.json");
  if (
    model.schema !== "cybr-layered-fem-v1" ||
    model.nodes !== model.surface_nodes * 3
  ) {
    throw Error("Unsupported tissue model layout");
  }
  const integers = new Set(["nodes", "faces", "layers", "groups"]);
  const arrays = {};
  await Promise.all(
    Object.entries(model.assets).map(async ([name, asset]) => {
      const bytes = await verifiedBytes(asset.url, asset.sha256, name);
      const values = new (integers.has(name) ? Uint32Array : Float64Array)(
        bytes,
      );
      if (values.length !== asset.count) throw Error(`Wrong ${name} length`);
      arrays[name] = values;
    }),
  );
  if (!arrays.fatWeights || arrays.fatWeights.length !== model.tetrahedra) {
    throw Error("Tissue model requires one adipose growth weight per cell");
  }
  if (
    arrays.contactPlanes &&
    (arrays.contactPlanes.length !== model.nodes * 7 ||
      !arrays.contactPlanes.every(Number.isFinite))
  ) {
    throw Error("Invalid anatomical contact planes");
  }
  if (
    arrays.fixedNodes &&
    (arrays.fixedNodes.length !== model.nodes ||
      !arrays.fixedNodes.every((v) => v === 0 || v === 1))
  )
    throw Error("Invalid tissue boundary mask");
  const bundle = await readJSON(base + "surface-embedding.json");
  const bytes = await (
    await response(base + "surface-embedding.bin")
  ).arrayBuffer();
  for (const name of ["faces", "vertex_faces", "barycentric2"]) {
    const record = bundle[name];
    if (
      (await digest(
        bytes.slice(
          record.byte_offset,
          record.byte_offset + record.byte_length,
        ),
      )) !== record.sha256
    ) {
      throw Error("Stale full-head embedding");
    }
  }
  if (!bundle.smooth_transfer)
    throw Error("Tissue model requires smooth surface transfer");
  const transfer = bundle.smooth_transfer;
  bundle.smoothData = new Float32Array(
    await verifiedBytes(
      transfer.url,
      transfer.sha256,
      "smooth tissue transfer",
    ),
  );
  return { model, arrays, bundle, bytes };
}

export async function loadSkin(model) {
  const loader = new GLTFLoader();
  const bytes = await (await response(model.skin_url)).arrayBuffer();
  if (
    !model.skin_glb_sha256 ||
    (await digest(bytes)) !== model.skin_glb_sha256
  ) {
    throw Error("Rendered skin differs from the bound solver build");
  }
  return loader.parseAsync(
    bytes,
    new URL(".", new URL(model.skin_url, location.href)).href,
  );
}

export async function loadInternalAnatomy(model) {
  if (!model.internal_anatomy_url) return null;
  const bytes = await verifiedBytes(
    model.internal_anatomy_url,
    model.internal_anatomy_sha256,
    "registered internal anatomy",
  );
  return new GLTFLoader().parseAsync(
    bytes,
    new URL(".", new URL(model.internal_anatomy_url, location.href)).href,
  );
}
