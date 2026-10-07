export class GPUHeadFEM {
  static async create(model, arrays, adapter = null) {
    if (!navigator.gpu) throw Error("WebGPU is unavailable");
    adapter ??= await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter)
      throw Error(
        "No WebGPU adapter; real-time FEM requires a working GPU browser session",
      );
    const intel = /intel/i.test(
        adapter.info.vendor + " " + adapter.info.architecture,
      ),
      timestamp = !intel && adapter.features.has("timestamp-query"),
      device = await adapter.requestDevice({
        requiredFeatures: timestamp ? ["timestamp-query"] : [],
      }),
      engine = new GPUHeadFEM(device, model, arrays, timestamp);
    engine.adapterInfo = {
      vendor: adapter.info.vendor,
      architecture: adapter.info.architecture,
      isFallbackAdapter: adapter.info.isFallbackAdapter,
    };
    engine.boundedSubmissions = intel;
    engine.cgBatchSize = engine.boundedSubmissions ? 16 : 8;
    engine.boundedEarlyExit = true;
    try {
      await engine.initialize();
      return engine;
    } catch (error) {
      engine.dispose();
      throw error;
    }
  }
  constructor(device, model, arrays, timestamp) {
    this.device = device;
    this.model = model;
    this.arrays = arrays;
    this.timestamp = timestamp;
    this.errors = [];
    this.frames = 0;
    this.samples = [];
    device.addEventListener("uncapturederror", (e) =>
      this.errors.push(e.error.message),
    );
    device.lost.then((info) => {
      this.lost = info;
    });
  }
  async initialize() {
    const { device: d, model: m, arrays: a } = this,
      N = m.nodes,
      volumeT = m.tetrahedra,
      T = volumeT + (a.hinges?.length / 6 || 0),
      S = m.surface_nodes;
    const nodeIds = new Uint32Array(T * 4);
    nodeIds.set(a.nodes);
    for (let h = 0; h < T - volumeT; h++)
      for (let k = 0; k < 4; k++)
        nodeIds[(volumeT + h) * 4 + k] = a.hinges[h * 6 + k];
    this.N = N;
    this.T = T;
    this.S = S;
    this.tetStride = 304;
    this.tetMetricOffset = 68;
    this.groups = Math.ceil(Math.max(N, T) / 128);
    const nodeData = new ArrayBuffer(N * 256),
      nf = new Float32Array(nodeData),
      nu = new Uint32Array(nodeData),
      tetData = new ArrayBuffer(T * 304),
      tf = new Float32Array(tetData),
      tu = new Uint32Array(tetData),
      counts = new Uint32Array(N),
      fatMass = new Float32Array(N);
    for (let t = 0; t < T; t++) {
      const j = t * 76;
      let x = 0,
        y = 0,
        z = 0;
      for (let n = 0; n < 4; n++) {
        const id = nodeIds[t * 4 + n];
        tu[j + n] = id;
        counts[id]++;
        x += a.rest[id * 3] / 4;
        y += a.rest[id * 3 + 1] / 4;
        z += a.rest[id * 3 + 2] / 4;
        for (let k = 0; k < 3; k++)
          tf[j + 4 + n * 4 + k] =
            t < volumeT ? a.gradients[t * 12 + n * 3 + k] : 0;
      }
      if (t >= volumeT) {
        const h = (t - volumeT) * 6;
        tf.set([1, 0, a.hinges[h + 5], -1], j + 20);
        tf[j + 72] = a.hinges[h + 4];
        continue;
      }
      const weight = a.fatWeights[t];
      tf.set(
        [
          a.material[t * 3],
          a.material[t * 3 + 1],
          a.material[t * 3 + 2],
          weight,
        ],
        j + 20,
      );
      for (let n = 0; n < 4; n++)
        fatMass[nodeIds[t * 4 + n]] += (a.material[t * 3] * 950 * weight) / 4;
    }
    let bottom = Infinity;
    for (let n = 0; n < S; n++) bottom = Math.min(bottom, a.rest[n * 3 + 1]);
    const pinned = (n) =>
      a.fixedNodes
        ? a.fixedNodes[n] > 0.5
        : n >= 2 * S || (m.full_head && a.rest[n * 3 + 1] < bottom + 0.001);
    const edgeMaps = Array.from({ length: N }, () => new Map());
    for (let t = 0; t < T; t++)
      for (let i = 0; i < 4; i++) {
        const n = nodeIds[t * 4 + i];
        if (pinned(n)) continue;
        for (let k = 0; k < 4; k++) {
          const neighbor = nodeIds[t * 4 + k];
          if (pinned(neighbor)) continue;
          let codes = edgeMaps[n].get(neighbor);
          if (!codes) {
            codes = [];
            edgeMaps[n].set(neighbor, codes);
          }
          codes.push(t * 16 + i * 4 + k);
        }
      }
    const edgeStarts = new Uint32Array(N + 1);
    for (let n = 0; n < N; n++)
      edgeStarts[n + 1] = edgeStarts[n] + edgeMaps[n].size;
    this.edges = edgeStarts[N];
    const edgeData = new ArrayBuffer(this.edges * 64),
      edgeU = new Uint32Array(edgeData),
      edgeCodes = new Uint32Array(T * 16);
    let edge = 0,
      codeAt = 0;
    for (let n = 0; n < N; n++)
      for (const [neighbor, codes] of edgeMaps[n]) {
        edgeU.set([neighbor, codeAt, codeAt + codes.length, 0], edge * 16);
        edgeCodes.set(codes, codeAt);
        codeAt += codes.length;
        edge++;
      }
    const starts = new Uint32Array(N + 1);
    for (let n = 0; n < N; n++) starts[n + 1] = starts[n] + counts[n];
    const cursor = starts.slice(),
      adjacency = new Uint32Array(T * 4);
    for (let t = 0; t < T; t++)
      for (let k = 0; k < 4; k++) {
        const n = nodeIds[t * 4 + k];
        adjacency[cursor[n]++] = t * 4 + k;
      }
    const areas = new Float32Array(S);
    for (let f = 0; f < a.faces.length; f += 3) {
      const ids = [a.faces[f], a.faces[f + 1], a.faces[f + 2]],
        q = ids.map((i) => [
          a.rest[i * 3],
          a.rest[i * 3 + 1],
          a.rest[i * 3 + 2],
        ]),
        ab = q[1].map((v, k) => v - q[0][k]),
        ac = q[2].map((v, k) => v - q[0][k]);
      const share =
        Math.hypot(
          ab[1] * ac[2] - ab[2] * ac[1],
          ab[2] * ac[0] - ab[0] * ac[2],
          ab[0] * ac[1] - ab[1] * ac[0],
        ) / 6;
      for (const n of ids) areas[n] += share;
    }
    this.surfaceAreas = areas;
    this.fixed = Uint8Array.from({ length: N }, (_, n) => (pinned(n) ? 1 : 0));
    this.patchNode = null;
    for (let n = 0; n < N; n++) {
      const j = n * 64;
      for (let k = 0; k < 3; k++) {
        nf[j + k] = a.rest[n * 3 + k];
        nf[j + 4 + k] = 0;
        nf[j + 8 + k] = 0;
        nf[j + 56 + k] = a.rest[n * 3 + k];
      }
      nf[j + 3] = a.mass[n];
      nf[j + 59] = pinned(n) ? 1 : 0;
      const i = n % S,
        inner = (i + 2 * S) * 3,
        v = [
          a.rest[i * 3] - a.rest[inner],
          a.rest[i * 3 + 1] - a.rest[inner + 1],
          a.rest[i * 3 + 2] - a.rest[inner + 2],
        ],
        length = Math.hypot(...v);
      nf.set(
        v.map((x) => x / length),
        j + 52,
      );
      nf[j + 55] = fatMass[n];
      nf[j + 62] = v.reduce(
        (sum, value, k) =>
          sum + ((a.rest[n * 3 + k] - a.rest[inner + k]) * value) / length,
        0,
      );
      if (a.contactPlanes) {
        const plane = a.contactPlanes.subarray(n * 7, n * 7 + 7);
        nf.set(plane.subarray(3, 6), j + 52);
        nf[j + 62] =
          plane[3] * (a.rest[n * 3] - plane[0]) +
          plane[4] * (a.rest[n * 3 + 1] - plane[1]) +
          plane[5] * (a.rest[n * 3 + 2] - plane[2]);
      }
      nu[j + 60] = starts[n];
      nu[j + 61] = starts[n + 1];
      nu[j + 63] = edgeStarts[n];
      if (!pinned(n)) {
        nf[j + 56] = n < S ? areas[n] : 0;
        nf[j + 57] = a.groups ? a.groups[n % S] : 0;
        nf[j + 58] = 0;
      }
    }
    const edgeStorage = new Uint32Array(codeAt);
    edgeStorage.set(edgeCodes.subarray(0, codeAt));
    this.initialNodeData = nodeData.slice(0);
    const buffer = (
      name,
      data,
      usage = GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    ) => {
      const b = d.createBuffer({
        label: name,
        size: Math.max(16, data.byteLength),
        usage,
      });
      d.queue.writeBuffer(b, 0, data);
      return b;
    };
    this.buffers = [
      buffer("FEM nodes", new Uint8Array(nodeData)),
      buffer("FEM tetrahedra", new Uint8Array(tetData)),
      buffer("Node-to-cell incidence", adjacency),
      buffer("Parallel reductions", new Float32Array(this.groups * 4)),
      buffer("Solver scalars", new Float32Array(16)),
      buffer("GPU position/gradient transfer", new Float32Array(S * 24)),
      buffer("Sparse stiffness blocks", new Uint8Array(edgeData)),
      buffer("Block cell incidence", edgeStorage),
    ];
    this.params = d.createBuffer({
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.paramBytes = new ArrayBuffer(80);
    this.paramF = new Float32Array(this.paramBytes);
    this.paramU = new Uint32Array(this.paramBytes);
    this.paramU.set([N, T, S, this.groups]);
    this.paramU[12] = 0xffffffff;
    const layout = d.createBindGroupLayout({
      entries: [
        ...this.buffers.map((_, binding) => ({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type:
              binding === 2 || binding === 7 ? "read-only-storage" : "storage",
          },
        })),
        {
          binding: 8,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
      ],
    });
    this.bindGroup = d.createBindGroup({
      layout,
      entries: [
        ...this.buffers.map((buffer, binding) => ({
          binding,
          resource: { buffer },
        })),
        { binding: 8, resource: { buffer: this.params } },
      ],
    });
    this.indirectArgs = d.createBuffer({
      label: "GPU convergence dispatch arguments",
      size: 192,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.INDIRECT,
    });
    const controlLayout = d.createBindGroupLayout({
      entries: [0, 3, 4]
        .map((binding) => ({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        }))
        .concat([
          {
            binding: 8,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "uniform" },
          },
          {
            binding: 9,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage" },
          },
        ]),
    });
    this.controlBindGroup = d.createBindGroup({
      layout: controlLayout,
      entries: [0, 3, 4]
        .map((binding) => ({
          binding,
          resource: { buffer: this.buffers[binding] },
        }))
        .concat([
          { binding: 8, resource: { buffer: this.params } },
          { binding: 9, resource: { buffer: this.indirectArgs } },
        ]),
    });
    this.controlNames = new Set([
      "predict",
      "cgStart",
      "cgBeta",
      "currentEnergy",
      "acceptTrial",
    ]);
    const controlPipelineLayout = d.createPipelineLayout({
      bindGroupLayouts: [controlLayout],
    });
    const code = await (
        await fetch(new URL("./full_head_fem.wgsl", import.meta.url))
      ).text(),
      shader = d.createShaderModule({ code, label: "Implicit full-head FEM" }),
      info = await shader.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === "error");
    if (errors.length)
      throw Error(errors.map((m) => `${m.lineNum}: ${m.message}`).join("\n"));
    const pipelineLayout = d.createPipelineLayout({
      bindGroupLayouts: [layout],
    });
    this.pipelines = {};
    const names = [
      "predict",
      "evaluateTets",
      "gatherGradient",
      "cgInit",
      "rzPartial",
      "cgStart",
      "cgAlpha",
      "cgBeta",
      "cgDirection",
      "currentPartial",
      "currentEnergy",
      "trialNodes",
      "trialTets",
      "trialPartial",
      "acceptTrial",
      "commit",
      "finish",
      "pack",
      "assembleBlocks",
      "multiplyReduce",
      "updateReduce",
      "updateMaterials",
      "validationPartial",
      "validationFinish",
    ];
    for (const name of names)
      this.pipelines[name] = await d.createComputePipelineAsync({
        layout: this.controlNames.has(name)
          ? controlPipelineLayout
          : pipelineLayout,
        compute: { module: shader, entryPoint: name },
        label: name,
      });
    this.outputBytes = S * 24 * 4;
    this.readback = d.createBuffer({
      size: this.outputBytes + 80,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.cgControlReadback = d.createBuffer({
      size: 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    if (this.timestamp) {
      this.queries = d.createQuerySet({ type: "timestamp", count: 2 });
      this.queryBuffer = d.createBuffer({
        size: 16,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
    }
  }
  reset() {
    this.device.queue.writeBuffer(this.buffers[0], 0, this.initialNodeData);
    this.device.queue.writeBuffer(this.buffers[4], 0, new Float32Array(16));
  }
  async step(
    { fatPercent = 0, sagPercent = 0, gravity = 1, grab = null } = {},
    {
      dt = 1 / 120,
      substeps = 2,
      newton = 2,
      cg = 24,
      block = true,
      indirect = true,
      quasiStatic = false,
    } = {},
  ) {
    if (this.lost) throw Error("GPU device lost: " + this.lost.message);
    const begin = performance.now(),
      d = this.device,
      bounded = this.boundedSubmissions;
    let submissionChunks = 1, cgIterationsScheduled = 0, cgIterationsSkipped = 0;
    this.paramF.set(
      [dt, gravity, fatPercent * 0.01, Math.pow(10, -1.7 * sagPercent * 0.01)],
      4,
    );
    this.paramU[16] = quasiStatic ? 1 : 0;
    this.paramU[12] = grab?.node ?? 0xffffffff;
    this.paramU[13] = block ? 1 : 0;
    this.paramU[14] = this.edges;
    this.paramF[15] = grab ? this.grabNormalization(grab.node) : 0;
    this.paramF.set(
      grab ? [...grab.target, grab.stiffness ?? 100] : [0, 0, 0, 0],
      8,
    );
    d.queue.writeBuffer(this.params, 0, this.paramBytes);
    let encoder = d.createCommandEncoder(),
      pass;
    const beginPass = (first = false, last = false) => {
      const timestampWrites =
        this.timestamp && !bounded && (first || last)
          ? {
              querySet: this.queries,
              ...(first ? { beginningOfPassWriteIndex: 0 } : {}),
              ...(last ? { endOfPassWriteIndex: 1 } : {}),
            }
          : undefined;
      pass = encoder.beginComputePass(
        timestampWrites ? { timestampWrites } : {},
      );
      pass.setBindGroup(0, this.bindGroup);
    };
    beginPass(true);
    const flush = async () => {
      pass.end();
      if (this.boundedEarlyExit)
        encoder.copyBufferToBuffer(this.buffers[4], 0, this.cgControlReadback, 0, 16);
      d.queue.submit([encoder.finish()]);
      let active = true;
      if (this.boundedEarlyExit) {
        await this.cgControlReadback.mapAsync(GPUMapMode.READ);
        active = new Float32Array(this.cgControlReadback.getMappedRange())[3] > .5;
        this.cgControlReadback.unmap();
      } else await d.queue.onSubmittedWorkDone();
      if (this.lost) throw Error("GPU device lost: " + this.lost.message);
      encoder = d.createCommandEncoder();
      beginPass();
      submissionChunks++;
      return active;
    };
    const nodeGroups = Math.ceil(this.N / 128),
      tetGroups = Math.ceil(this.T / 128),
      edgeGroups = Math.ceil(this.edges / 128);
    const run = (name, count = nodeGroups) => {
      pass.setPipeline(this.pipelines[name]);
      pass.setBindGroup(
        0,
        this.controlNames.has(name) ? this.controlBindGroup : this.bindGroup,
      );
      pass.dispatchWorkgroups(count);
    };
    const work = (name, slot, count = nodeGroups) => {
      if (!indirect) {
        run(name, count);
        return;
      }
      pass.setPipeline(this.pipelines[name]);
      pass.setBindGroup(0, this.bindGroup);
      pass.dispatchWorkgroupsIndirect(this.indirectArgs, slot * 16);
    };
    run("updateMaterials", tetGroups);
    for (let step = 0; step < substeps; step++) {
      run("predict");
      for (let it = 0; it < newton; it++) {
        work("evaluateTets", 9, tetGroups);
        work("gatherGradient", 8);
        work("cgInit", 8);
        work("rzPartial", 8);
        run("cgStart", 1);
        work("assembleBlocks", 3, edgeGroups);
        for (let k = 0; k < cg; k++) {
          cgIterationsScheduled++;
          work("multiplyReduce", 0);
          work("cgAlpha", 2, 1);
          work("updateReduce", 0);
          run("cgBeta", 1);
          work("cgDirection", 0);
          if (bounded && (k + 1) % (this.cgBatchSize || 8) === 0) {
            if (!(await flush())) {
              cgIterationsSkipped += cg - k - 1;
              break;
            }
          }
        }
        work("currentPartial", 10, this.groups);
        run("currentEnergy", 1);
        for (let line = 0; line < 18; line++) {
          work("trialNodes", 4);
          work("trialTets", 5, tetGroups);
          work("trialPartial", 6, this.groups);
          run("acceptTrial", 1);
        }
        work("commit", 8);
      }
      run("finish");
    }
    pass.end();
    beginPass(false, true);
    run("evaluateTets", tetGroups);
    run("gatherGradient");
    run("cgInit");
    run("rzPartial");
    run("cgStart", 1);
    run("validationPartial");
    run("validationFinish", 1);
    run("pack");
    pass.end();
    encoder.copyBufferToBuffer(
      this.buffers[5],
      0,
      this.readback,
      0,
      this.outputBytes,
    );
    encoder.copyBufferToBuffer(
      this.buffers[4],
      0,
      this.readback,
      this.outputBytes,
      64,
    );
    if (this.timestamp && !bounded) {
      encoder.resolveQuerySet(this.queries, 0, 2, this.queryBuffer, 0);
      encoder.copyBufferToBuffer(
        this.queryBuffer,
        0,
        this.readback,
        this.outputBytes + 64,
        16,
      );
    }
    d.queue.submit([encoder.finish()]);
    await this.readback.mapAsync(GPUMapMode.READ);
    const bytes = this.readback.getMappedRange(),
      packed = new Float32Array(bytes, 0, this.outputBytes / 4).slice(),
      stats = new Float32Array(bytes, this.outputBytes, 16).slice();
    let gpuMs = null;
    if (this.timestamp && !bounded) {
      const stamps = new BigUint64Array(bytes, this.outputBytes + 64, 2);
      gpuMs = Number(stamps[1] - stamps[0]) / 1e6;
    }
    this.readback.unmap();
    const elapsedMs = performance.now() - begin;
    this.frames++;
    const sample = {
      elapsedMs,
      gpuMs,
      minJ: stats[9],
      residualN: stats[10],
      staticResidualN: stats[13],
      maxNodeSpeedMps: stats[14],
      kineticEnergyJ: stats[15],
      lineAccepted: stats[8] > 0.5,
      directionalDerivativeJ: stats[6],
      lastStepFraction: stats[7],
      cgIterationsExecuted: stats[11],
      grabSupportNodes: grab ? this.grabSupportNodes : 0,
      indirect,
      submissionChunks,
      cgIterationsScheduled,
      cgIterationsSkipped,
      integration: quasiStatic ? "quasistatic" : "dynamic",
      simulatedSeconds: quasiStatic ? 0 : dt * substeps,
      frame: this.frames,
    };
    this.samples.push(sample);
    if (this.samples.length > 240) this.samples.shift();
    if (this.errors.length) throw Error(this.errors.at(-1));
    this.latestPacked = packed;
    return { packed, stats: sample };
  }
  grabNormalization(node) {
    if (this.patchNode === node) return this.patchScale;
    let sum = 0,
      count = 0;
    const a = this.arrays,
      S = this.S,
      sigma = 0.006;
    for (let n = 0; n < S; n++) {
      if (this.fixed[n] || (a.groups && a.groups[n] !== a.groups[node]))
        continue;
      const d2 =
        (a.rest[n * 3] - a.rest[node * 3]) ** 2 +
        (a.rest[n * 3 + 1] - a.rest[node * 3 + 1]) ** 2 +
        (a.rest[n * 3 + 2] - a.rest[node * 3 + 2]) ** 2;
      if (d2 > 9 * sigma * sigma) continue;
      sum += this.surfaceAreas[n] * Math.exp(-d2 / (2 * sigma * sigma));
      count++;
    }
    this.patchNode = node;
    this.patchScale = sum > 0 ? 1 / sum : 0;
    this.grabSupportNodes = count;
    return this.patchScale;
  }
  async readState() {
    const d = this.device,
      b = d.createBuffer({
        size: this.N * 256,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
      e = d.createCommandEncoder();
    e.copyBufferToBuffer(this.buffers[0], 0, b, 0, b.size);
    d.queue.submit([e.finish()]);
    await b.mapAsync(GPUMapMode.READ);
    const f = new Float32Array(b.getMappedRange()).slice();
    b.unmap();
    b.destroy();
    return f;
  }
  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    for (const b of this.buffers ?? []) b.destroy();
    this.indirectArgs?.destroy();
    this.params?.destroy();
    this.readback?.destroy();
    this.cgControlReadback?.destroy();
    this.queryBuffer?.destroy();
    this.queries?.destroy();
    this.device.destroy();
  }
}
