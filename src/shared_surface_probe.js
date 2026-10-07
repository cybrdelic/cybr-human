/** Diagnostic shared-device dense transfer. Not imported by the public viewer.
 * Input positions/normals are already in world space. Fine wrinkles, skin
 * diffusion, shadows and eye materials are deliberately outside this probe. */
export function packProbeVertices(positions, normals, smooth, surfaceNodes) {
  const count = positions.length / 3;
  if (!Number.isInteger(count) || count <= 0 || !Number.isInteger(surfaceNodes) || surfaceNodes <= 0 || normals.length !== positions.length || smooth.length !== count * 16)
    throw Error("Probe vertex/normal/transfer lengths disagree");
  const bytes = new ArrayBuffer(count * 96), floats = new Float32Array(bytes), ids = new Uint32Array(bytes);
  for (let i = 0; i < count; i++) {
    const offset = i * 24;
    for (let d = 0; d < 3; d++) {
      if (!Number.isFinite(positions[i * 3 + d]) || !Number.isFinite(normals[i * 3 + d])) throw Error("Nonfinite probe geometry");
      floats[offset + d] = positions[i * 3 + d]; floats[offset + 4 + d] = normals[i * 3 + d];
    }
    if (Math.hypot(...normals.subarray(i * 3, i * 3 + 3)) < 1e-12) throw Error("Probe normal must have direction");
    let sum = 0;
    for (let k = 0; k < 8; k++) {
      const node = smooth[i * 16 + k], weight = smooth[i * 16 + 8 + k];
      if (!Number.isInteger(node) || node < 0 || node >= surfaceNodes || !Number.isFinite(weight) || weight < 0)
        throw Error("Invalid probe support/weight");
      ids[offset + 8 + k] = node; floats[offset + 16 + k] = weight; sum += weight;
    }
    if (Math.abs(sum - 1) > 1e-4) throw Error("Probe weights must sum to one");
  }
  return bytes;
}

const transferWGSL = `
struct Vertex { p:vec4f, n:vec4f, ids0:vec4u, ids1:vec4u, w0:vec4f, w1:vec4f }
struct Surface { p:vec4f, n:vec4f }
@group(0) @binding(0) var<storage,read> packed:array<vec4f>;
@group(0) @binding(1) var<storage,read> rest:array<vec4f>;
@group(0) @binding(2) var<storage,read> vertices:array<Vertex>;
@group(0) @binding(3) var<storage,read_write> surface:array<Surface>;
@group(0) @binding(4) var<uniform> counts:vec4u;
@group(0) @binding(5) var<storage,read> scalars:array<f32>;
@compute @workgroup_size(128) fn transfer(@builtin(global_invocation_id) id:vec3u) {
 let i=id.x;if(i>=counts.x){return;}
 let v=vertices[i];var p=vec3f(0.);var f=mat3x3f(vec3f(0.),vec3f(0.),vec3f(0.));
 for(var k=0u;k<8u;k++) {
  var node:u32;var w:f32;
  if(k<4u){node=v.ids0[k];w=v.w0[k];}else{node=v.ids1[k-4u];w=v.w1[k-4u];}
  let nf=mat3x3f(packed[counts.y+node].xyz,packed[counts.y+counts.z+node].xyz,packed[counts.y+2u*counts.z+node].xyz);
  p+=w*(packed[node].xyz+nf*(v.p.xyz-rest[node].xyz));f+=w*nf;
 }
 let a=cross(f[1],f[2]);let b=cross(f[2],f[0]);let c=cross(f[0],f[1]);
 let n=normalize(mat3x3f(a,b,c)*v.n.xyz/max(dot(f[0],a),.00001));
 // Keep diagnostic rendering gated by numerical validity already calculated
 // by the solver. This supplements, never replaces, host/reference checks.
 let valid=scalars[9]>.2 && scalars[9]<1e20 && scalars[10]>=0. && scalars[10]<1e20
  && all(abs(p)<vec3f(1e10)) && all(abs(n)<vec3f(1e10));
 surface[i].p=vec4f(p,select(0.,1.,valid));surface[i].n=vec4f(n,0.);
}`;

const renderWGSL = `
struct Surface {p:vec4f,n:vec4f}
@group(0) @binding(0) var<storage,read> surface:array<Surface>;
@group(0) @binding(1) var<uniform> viewProjection:mat4x4f;
struct Varying {@builtin(position) p:vec4f,@location(0) n:vec3f,@location(1) valid:f32}
@vertex fn vertex(@builtin(vertex_index) id:u32)->Varying {
 var out:Varying;out.p=viewProjection*vec4f(surface[id].p.xyz,1.);out.n=surface[id].n.xyz;out.valid=surface[id].p.w;return out;
}
@fragment fn fragment(in:Varying)->@location(0) vec4f {
 if(in.valid<.5){discard;}let light=max(0.,dot(normalize(in.n),normalize(vec3f(-.3,.4,.5))));
 return vec4f(vec3f(.62,.39,.29)*(.2+.8*light),1.);
}`;

export async function createSharedSurfaceProbe(engine, data, format = "rgba8unorm") {
  const device = engine.device, count = data.positions.length / 3;
  const input = packProbeVertices(data.positions, data.normals, data.smooth, engine.S);
  if (!data.indices.length || data.indices.length % 3 || !data.indices.every(index => Number.isInteger(index) && index >= 0 && index < count)) throw Error("Invalid probe triangles");
  const owned = [], buffer = (label, bytes, usage) => {
    if (bytes.byteLength > device.limits.maxStorageBufferBindingSize && (usage & GPUBufferUsage.STORAGE)) throw Error("Probe exceeds storage binding limit");
    const b = device.createBuffer({label,size:bytes.byteLength,usage:usage|GPUBufferUsage.COPY_DST});
    owned.push(b);device.queue.writeBuffer(b,0,bytes);return b;
  };
  try {
    const rest = new Float32Array(engine.N * 4);
    for(let i=0;i<engine.N;i++) rest.set(engine.arrays.rest.subarray(i*3,i*3+3),i*4);
    const restBuffer=buffer("Probe rest",rest,GPUBufferUsage.STORAGE);
    const inputBuffer=buffer("Probe dense binding",input,GPUBufferUsage.STORAGE);
    const outputBuffer=buffer("Probe dense surface",new Float32Array(count*8),GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
    const counts=buffer("Probe counts",new Uint32Array([count,engine.N,engine.S,0]),GPUBufferUsage.UNIFORM);
    const matrix=buffer("Probe camera",new Float32Array(16),GPUBufferUsage.UNIFORM);
    const indices=buffer("Probe triangles",new Uint32Array(data.indices),GPUBufferUsage.INDEX);
    const compute=await device.createComputePipelineAsync({layout:"auto",compute:{module:device.createShaderModule({code:transferWGSL}),entryPoint:"transfer"}});
    const computeGroup=device.createBindGroup({layout:compute.getBindGroupLayout(0),entries:[engine.buffers[5],restBuffer,inputBuffer,outputBuffer,counts,engine.buffers[4]].map((b,binding)=>({binding,resource:{buffer:b}}))});
    const render=await device.createRenderPipelineAsync({layout:"auto",vertex:{module:device.createShaderModule({code:renderWGSL}),entryPoint:"vertex"},fragment:{module:device.createShaderModule({code:renderWGSL}),entryPoint:"fragment",targets:[{format}]},primitive:{topology:"triangle-list",cullMode:"back"},depthStencil:{format:"depth24plus",depthWriteEnabled:true,depthCompare:"less"}});
    const renderGroup=device.createBindGroup({layout:render.getBindGroupLayout(0),entries:[outputBuffer,matrix].map((b,binding)=>({binding,resource:{buffer:b}}))});
    return {
      outputBuffer, vertices:count, ownedBufferBytes:owned.reduce((s,b)=>s+b.size,0),
      encodeTransfer(encoder){const pass=encoder.beginComputePass();pass.setPipeline(compute);pass.setBindGroup(0,computeGroup);pass.dispatchWorkgroups(Math.ceil(count/128));pass.end();},
      encodeRender(encoder,colorView,depthView,viewProjection){device.queue.writeBuffer(matrix,0,new Float32Array(viewProjection));const pass=encoder.beginRenderPass({colorAttachments:[{view:colorView,clearValue:{r:.067,g:.114,b:.13,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:depthView,depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});pass.setPipeline(render);pass.setBindGroup(0,renderGroup);pass.setIndexBuffer(indices,"uint32");pass.drawIndexed(data.indices.length);pass.end();},
      dispose(){for(const b of owned)b.destroy();},
    };
  } catch(error){for(const b of owned)b.destroy();throw error;}
}
