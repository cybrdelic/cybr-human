export class GPUHeadFEM {
  static async create(model, arrays, adapter = null, { checkpointState = false } = {}) {
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
    engine.checkpointState = checkpointState;
    engine.adapterInfo = {
      vendor: adapter.info.vendor,
      architecture: adapter.info.architecture,
      isFallbackAdapter: adapter.info.isFallbackAdapter,
    };
    engine.boundedSubmissions = intel;
    engine.cgBatchSize = engine.boundedSubmissions ? 16 : 8;
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
    this.submissionEpoch = 0;
    this.submissionSequence = 0;
    this.acceptedSubmissionSequence = 0;
    this.submissionSlots = new Set();
    this.submissionBanks = [];
    this.peakSubmissionSlotBytes = 0;
    device.addEventListener("uncapturederror", (e) =>
      this.errors.push(e.error.message),
    );
    device.lost.then((info) => {
      this.lost = info;
      this.cancelSubmittedSteps();
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
      buffer("GPU position/gradient transfer", new Float32Array(S * 24 + (this.checkpointState ? N * 4 : 0))),
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
    this.submissionLayout = layout;
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
    this.submissionControlLayout = controlLayout;
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
    if (this.checkpointState) names.push("packCheckpointVelocity");
    for (const name of names)
      this.pipelines[name] = await d.createComputePipelineAsync({
        layout: this.controlNames.has(name)
          ? controlPipelineLayout
          : pipelineLayout,
        compute: { module: shader, entryPoint: name },
        label: name,
      });
    this.outputBytes = S * 24 * 4;
    this.velocityBytes = this.checkpointState ? N * 16 : 0;
    this.statsOffset = this.outputBytes + this.velocityBytes;
    this.readback = d.createBuffer({
      size: this.statsOffset + 80,
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
    this.cancelSubmittedSteps();
    this.submissionNeedsRecovery = false;
    this.device.queue.writeBuffer(this.buffers[0], 0, this.initialNodeData);
    this.device.queue.writeBuffer(this.buffers[4], 0, new Float32Array(16));
  }
  async step(parameters = {}, options = {}) {
    if (this.submissionNeedsRecovery) throw Error("GPU submission requires recovery");
    if (this.submissionSlots.size) throw Error("Pending speculative GPU steps");
    return this._runStep(parameters, options);
  }
  async _runStep(
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
    slot = null,
  ) {
    if (this.disposed) throw Error("GPU solver disposed");
    if (this.lost) throw Error("GPU device lost: " + this.lost.message);
    const begin = performance.now(),
      d = this.device,
      bounded = this.boundedSubmissions;
    let submissionChunks = 1;
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
    d.queue.writeBuffer(slot?.params ?? this.params, 0, this.paramBytes);
    const readback = slot?.readback ?? this.readback,
      queries = slot?.queries ?? this.queries,
      queryBuffer = slot?.queryBuffer ?? this.queryBuffer,
      bindGroup = slot?.bindGroup ?? this.bindGroup,
      controlBindGroup = slot?.controlBindGroup ?? this.controlBindGroup,
      grabSupportNodes = grab ? this.grabSupportNodes : 0;
    let encoder = d.createCommandEncoder(),
      pass;
    const beginPass = (first = false, last = false) => {
      const timestampWrites =
        this.timestamp && !bounded && (first || last)
          ? {
              querySet: queries,
              ...(first ? { beginningOfPassWriteIndex: 0 } : {}),
              ...(last ? { endOfPassWriteIndex: 1 } : {}),
            }
          : undefined;
      pass = encoder.beginComputePass(
        timestampWrites ? { timestampWrites } : {},
      );
      pass.setBindGroup(0, bindGroup);
    };
    beginPass(true);
    const flush = async () => {
      pass.end();
      d.queue.submit([encoder.finish()]);
      await d.queue.onSubmittedWorkDone();
      if (this.lost) throw Error("GPU device lost: " + this.lost.message);
      encoder = d.createCommandEncoder();
      beginPass();
      submissionChunks++;
    };
    const nodeGroups = Math.ceil(this.N / 128),
      tetGroups = Math.ceil(this.T / 128),
      edgeGroups = Math.ceil(this.edges / 128);
    const run = (name, count = nodeGroups) => {
      pass.setPipeline(this.pipelines[name]);
      pass.setBindGroup(
        0,
        this.controlNames.has(name) ? controlBindGroup : bindGroup,
      );
      pass.dispatchWorkgroups(count);
    };
    const work = (name, slot, count = nodeGroups) => {
      if (!indirect) {
        run(name, count);
        return;
      }
      pass.setPipeline(this.pipelines[name]);
      pass.setBindGroup(0, bindGroup);
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
          work("multiplyReduce", 0);
          work("cgAlpha", 2, 1);
          work("updateReduce", 0);
          run("cgBeta", 1);
          work("cgDirection", 0);
          if (bounded && (k + 1) % (this.cgBatchSize || 8) === 0) await flush();
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
    if (this.checkpointState) run("packCheckpointVelocity");
    pass.end();
    encoder.copyBufferToBuffer(
      this.buffers[5],
      0,
      readback,
      0,
      this.statsOffset,
    );
    encoder.copyBufferToBuffer(
      this.buffers[4],
      0,
      readback,
      this.statsOffset,
      64,
    );
    if (this.timestamp && !bounded) {
      encoder.resolveQuerySet(queries, 0, 2, queryBuffer, 0);
      encoder.copyBufferToBuffer(
        queryBuffer,
        0,
        readback,
        this.statsOffset + 64,
        16,
      );
    }
    const finalSubmitBegin = performance.now();
    d.queue.submit([encoder.finish()]);
    const mapBegin = performance.now();
    const complete = async () => {
      await readback.mapAsync(GPUMapMode.READ);
      if (slot) slot.mapped = true;
      if (slot && (slot.cancelled || slot.epoch !== this.submissionEpoch)) throw Error("Stale GPU submission");
      const mappedAt = performance.now();
      const bytes = readback.getMappedRange(),
        packed = new Float32Array(bytes, 0, this.outputBytes / 4).slice(),
        stats = new Float32Array(bytes, this.statsOffset, 16).slice();
      let velocities;
      if (this.checkpointState) {
        const packedVelocity = new Float32Array(bytes, this.outputBytes, this.N * 4);
        velocities = new Float32Array(this.N * 3);
        for (let n = 0; n < this.N; n++) for (let d = 0; d < 3; d++) velocities[n * 3 + d] = packedVelocity[n * 4 + d];
      }
      let gpuMs = null;
      if (this.timestamp && !bounded) {
        const stamps = new BigUint64Array(bytes, this.statsOffset + 64, 2);
        gpuMs = Number(stamps[1] - stamps[0]) / 1e6;
      }
      readback.unmap();
      if (slot) slot.mapped = false;
      const elapsedMs = performance.now() - begin;
      if (!slot) this.frames++;
      const sample = {
        elapsedMs,
        gpuMs,
        hostBeforeFinalSubmitMs: finalSubmitBegin - begin,
        finalSubmitCpuMs: mapBegin - finalSubmitBegin,
        readbackWaitMs: mappedAt - mapBegin,
        mappedCopyCpuMs: performance.now() - mappedAt,
        readbackBytes: this.statsOffset + 64 + (this.timestamp && !bounded ? 16 : 0),
        minJ: stats[9],
        residualN: stats[10],
        staticResidualN: stats[13],
        maxNodeSpeedMps: stats[14],
        kineticEnergyJ: stats[15],
        lineAccepted: stats[8] > 0.5,
        directionalDerivativeJ: stats[6],
        lastStepFraction: stats[7],
        cgIterationsExecuted: stats[11],
        grabSupportNodes,
        indirect,
        submissionChunks,
        integration: quasiStatic ? "quasistatic" : "dynamic",
        simulatedSeconds: quasiStatic ? 0 : dt * substeps,
        frame: slot?.sequence ?? this.frames,
      };
      if (!slot) {
        this.samples.push(sample);
        if (this.samples.length > 240) this.samples.shift();
      }
      if (this.errors.length) throw Error(this.errors.at(-1));
      if (!slot) this.latestPacked = packed;
      if (slot) {
        sample.inputVersion = slot.inputVersion;
        sample.inputAgeAtSubmitMs = slot.inputAgeAtSubmitMs;
        sample.inputAgeAtCompletionMs = performance.now() - slot.inputAt;
        sample.submissionSlotBytes = slot.bytes;
        sample.peakSubmissionSlotBytes = this.peakSubmissionSlotBytes;
        sample.allocatedReusableBankBytes = this.allocatedReusableBankBytes;
        sample.solverOwnedBufferBytes = this.bufferMemory.ownedBufferBytes;
        sample.peakSolverOwnedBufferBytes = this.bufferMemory.peakOwnedBufferBytes;
      }
      const result = { packed, stats: sample, ...(velocities ? { velocities } : {}) };
      if (slot) slot.result = result;
      return result;
    };
    if (!slot) return complete();
    slot.completion = complete().catch(error => {
      slot.failed = true;
      if (slot.epoch === this.submissionEpoch && !slot.cancelled) this.cancelSubmittedSteps();
      else slot.cancel();
      throw error;
    }).finally(() => slot.release());
    // Cancel/reset can reject before the ordered consumer reaches this ticket.
    slot.completion.catch(() => {});
    return slot;
  }
  /** Serial queue solves with private per-step parameters/readback/timestamps.
   * Does not accept/present a result: the ordered host coordinator owns that gate.
   * Bounded Intel submissions retain their original awaited reference path.
   */
  async submitStep(parameters = {}, options = {}, metadata = {}) {
    if (this.boundedSubmissions) throw Error("Speculation unavailable for bounded submissions");
    if (!this.checkpointState) throw Error("Speculation requires actual checkpoint velocities");
    if (this.disposed || this.lost || this.submissionNeedsRecovery) throw Error("GPU submission requires recovery");
    if (this.submissionSlots.size >= 2) throw Error("GPU submission backpressure");
    const d = this.device, inputAt = metadata.inputAt ?? performance.now();
    if (!Number.isFinite(inputAt) || !Number.isSafeInteger(metadata.inputVersion ?? 0))
      throw Error("Invalid submission input metadata");
    const slot = { epoch: this.submissionEpoch, sequence: ++this.submissionSequence,
      inputVersion: metadata.inputVersion ?? 0, inputAt,
      inputAgeAtSubmitMs: performance.now() - inputAt, cancelled: false,
      bytes: 80 + this.statsOffset + 80 + (this.timestamp ? 16 : 0) };
    // Banks persist across accepted frames. A copied result still reserves its
    // bank until ordered acceptance; no frame allocation churn in steady state.
    slot.release = () => {
      if (slot.mapped) {
        try { slot.readback.unmap(); } finally { slot.mapped = false; }
      }
      slot.released = true;
    };
    slot.cancel = () => {
      slot.cancelled = true;
      try { slot.release(); } finally {
        if (slot.bank?.owner === slot) slot.bank.destroy();
      }
    };
    this.submissionSlots.add(slot);
    try {
      let bank = this.submissionBanks.find(b => !b.destroyed && !b.inUse);
      if (!bank) {
        if (this.submissionBanks.filter(b => !b.destroyed).length >= 2)
          throw Error("GPU staging bank backpressure");
        bank = { bytes: slot.bytes, inUse: false, destroyed: false };
        bank.destroy = () => {
          if (bank.destroyed) return;
          bank.destroyed = true;
          bank.params?.destroy(); bank.readback?.destroy();
          bank.queryBuffer?.destroy(); bank.queries?.destroy();
        };
        // Associate before allocation, so partial allocation failure cleans up.
        slot.bank = bank;
        bank.owner = slot;
        this.submissionBanks = this.submissionBanks.filter(b => !b.destroyed);
        this.submissionBanks.push(bank);
        bank.params = d.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        bank.readback = d.createBuffer({ size: this.statsOffset + 80,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        bank.bindGroup = d.createBindGroup({ layout: this.submissionLayout,
          entries: [...this.buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
            { binding: 8, resource: { buffer: bank.params } }] });
        bank.controlBindGroup = d.createBindGroup({ layout: this.submissionControlLayout,
          entries: [0, 3, 4].map(binding => ({ binding, resource: { buffer: this.buffers[binding] } }))
            .concat([{ binding: 8, resource: { buffer: bank.params } },
              { binding: 9, resource: { buffer: this.indirectArgs } }]) });
        if (this.timestamp) {
          bank.queries = d.createQuerySet({ type: "timestamp", count: 2 });
          bank.queryBuffer = d.createBuffer({ size: 16,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
        }
      }
      bank.inUse = true;
      bank.owner = slot;
      slot.bank = bank;
      for (const key of ['params', 'readback', 'bindGroup', 'controlBindGroup', 'queries', 'queryBuffer'])
        slot[key] = bank[key];
      // Count allocated GPU buffer bytes, including free reusable banks. Excludes
      // querySet driver memory and immutable copied CPU result arrays.
      this.peakSubmissionSlotBytes = Math.max(this.peakSubmissionSlotBytes,
        this.submissionBanks.filter(b => !b.destroyed).reduce((n, b) => n + b.bytes, 0));
      return await this._runStep(parameters, options, slot);
    } catch (error) {
      if (slot.epoch === this.submissionEpoch && !slot.cancelled) this.cancelSubmittedSteps();
      else slot.cancel();
      throw error;
    }
  }
  acceptSubmittedStep(slot, result) {
    if (this.disposed || this.lost || this.submissionNeedsRecovery || slot.cancelled ||
        slot.epoch !== this.submissionEpoch || slot.sequence !== this.acceptedSubmissionSequence + 1 ||
        !this.submissionSlots.has(slot) || slot.result !== result ||
        !slot.released || slot.mapped || slot.bank.destroyed)
      throw Error("GPU submission acceptance order rejected");
    this.acceptedSubmissionSequence = slot.sequence;
    this.submissionSlots.delete(slot);
    slot.bank.inUse = false;
    this.frames++;
    result.stats.frame = this.frames;
    this.samples.push(result.stats);
    if (this.samples.length > 240) this.samples.shift();
    this.latestPacked = result.packed;
  }
  /** GPU may already have advanced speculatively. Cancellation does not claim
   * rollback: further submissions require reset or a validated checkpoint restore.
   */
  cancelSubmittedSteps() {
    this.submissionNeedsRecovery = true;
    for (const slot of this.submissionSlots) slot.cancel();
    this.submissionSlots.clear();
    for (const bank of this.submissionBanks) bank.destroy();
    this.submissionBanks = [];
    this.submissionEpoch++;
    this.acceptedSubmissionSequence = this.submissionSequence;
  }
  get reusableBankAllocations() {
    return this.submissionBanks.flatMap((bank, bankIndex) => bank.destroyed ? [] :
      ['params', 'readback', 'queryBuffer'].filter(key => bank[key]).map(key => ({
        label: `submission-bank-${bankIndex}-${key}`, bytes: bank[key].size,
        bankIndex, reserved: bank.inUse,
      })));
  }
  get allocatedReusableBankBytes() {
    return this.reusableBankAllocations.reduce((bytes, allocation) => bytes + allocation.bytes, 0);
  }
  /** Actual owned GPUBuffer sizes, not whole-process/renderer/driver memory.
   * QuerySets and copied CPU arrays have no measured size and are excluded.
   * Temporary readState() diagnostic readbacks are not part of this pool.
   */
  get bufferMemory() {
    const base = [
      ...(this.buffers ?? []).map((buffer, index) => ({ label: `solver-storage-${index}`, bytes: buffer.size })),
      ...['params', 'readback', 'queryBuffer', 'indirectArgs'].filter(key => this[key])
        .map(key => ({ label: `solver-${key}`, bytes: this[key].size })),
    ];
    const historicalBaseBytes = base.reduce((bytes, allocation) => bytes + allocation.bytes, 0);
    const baseBytes = this.disposed ? 0 : historicalBaseBytes;
    const banks = this.reusableBankAllocations;
    return { allocations: [...(this.disposed ? [] : base), ...banks], baseBufferBytes: baseBytes,
      allocatedReusableBankBytes: this.allocatedReusableBankBytes,
      peakSubmissionSlotBytes: this.peakSubmissionSlotBytes,
      ownedBufferBytes: baseBytes + this.allocatedReusableBankBytes,
      peakOwnedBufferBytes: historicalBaseBytes + this.peakSubmissionSlotBytes,
      excludes: ['querySets', 'CPU snapshots', 'renderer', 'driver', 'temporary readState diagnostic buffer'] };
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
    this.cancelSubmittedSteps();
    for (const b of this.buffers ?? []) b.destroy();
    this.indirectArgs?.destroy();
    this.params?.destroy();
    this.readback?.destroy();
    this.queryBuffer?.destroy();
    this.queries?.destroy();
    this.device.destroy();
  }
}
