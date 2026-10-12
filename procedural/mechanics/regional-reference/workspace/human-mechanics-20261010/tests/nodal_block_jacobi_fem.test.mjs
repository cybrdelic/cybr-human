import assert from "node:assert/strict";
import fs from "node:fs";
import crypto from "node:crypto";
import { LayeredFEM } from "../reference-source/src/layered_fem.js";
import { VolumetricMuscleFEM } from "../src/volumetric_muscle_fem.mjs";
import { BoneMeshCollider } from "../src/bone_mesh_contact.mjs";
import { buildNodalBlockJacobi, applyNodalBlockJacobi, nodalBlockJacobiPCG,
  NodalBlockJacobiVolumetricFEM } from "../src/nodal_block_jacobi_fem.mjs";

const hash = path => crypto.createHash("sha256").update(fs.readFileSync(path)).digest("hex");
const referenceURL = new URL("../reference-source/src/layered_fem.js", import.meta.url);
const referenceSHA256 = hash(referenceURL);
assert.equal(referenceSHA256, "8618d5517de196dd3dae905d598a69eaf14555682895c80dcfc994e2501dbc71");
assert.equal(NodalBlockJacobiVolumetricFEM.prototype.step, LayeredFEM.prototype.step);
assert.equal(NodalBlockJacobiVolumetricFEM.prototype.evaluate, VolumetricMuscleFEM.prototype.evaluate);
assert.equal(NodalBlockJacobiVolumetricFEM.prototype.multiply, VolumetricMuscleFEM.prototype.multiply);
const norm = x => Math.sqrt(x.reduce((s, v) => s + v * v, 0));
const dot = (x, y) => x.reduce((s, v, i) => s + v * y[i], 0);
const maxDifference = (x, y) => Math.max(...x.map((v, i) => Math.abs(v - y[i])));
const report = { schema: "experimental-nodal-block-jacobi-check-v1", originalCoreSHA256: referenceSHA256 };

function oneTet() {
  const a = .038, b = .038, c = .032, volume = a * b * c / 6;
  return { model: { schema: "cybr-layered-fem-v1", nodes: 4, surface_nodes: 4, tetrahedra: 1 },
    arrays: { rest: Float64Array.from([-.008, -.008, .008, .030, -.008, .008,
      -.008, .030, .008, -.008, -.008, .040]), nodes: Uint32Array.from([0, 1, 2, 3]),
      gradients: Float64Array.from([-1/a, -1/b, -1/c, 1/a, 0, 0, 0, 1/b, 0, 0, 0, 1/c]),
      material: Float64Array.from([volume, 120, 900]), mass: new Float64Array(4).fill(1000 * volume / 4) } };
}
function cube(stiffness = 240) {
  return new BoneMeshCollider({ stiffness, clearance: .0001,
    vertices: Array.from({ length: 8 }, (_, i) => [i & 1, (i >> 1) & 1, (i >> 2) & 1].map(v => v * .01)),
    triangles: [[0,4,6],[0,6,2],[1,3,7],[1,7,5],[0,1,5],[0,5,4],
      [2,6,7],[2,7,3],[0,2,3],[0,3,1],[4,5,7],[4,7,6]] });
}
function fullMatrix(s, dt) {
  const A = Array.from({ length: s.x.length }, () => new Float64Array(s.x.length));
  for (let column = 0; column < s.x.length; column++) {
    const v = new Float64Array(s.x.length), out = new Float64Array(s.x.length); v[column] = 1;
    s.multiply(v, out, dt);
    for (let row = 0; row < s.x.length; row++) A[row][column] = out[row];
  }
  return A;
}
function directSolve(s, A) {
  const free = Array.from({ length: s.x.length }, (_, i) => i).filter(i => !s.fixed[Math.floor(i / 3)]),
    n = free.length, L = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let v = A[free[i]][free[j]];
    for (let k = 0; k < j; k++) v -= L[i][k] * L[j][k];
    if (i === j) { assert(v > 0); L[i][j] = Math.sqrt(v); }
    else L[i][j] = v / L[j][j];
  }
  const y = new Float64Array(n), q = new Float64Array(n), result = new Float64Array(s.x.length);
  for (let i = 0; i < n; i++) {
    let v = -s.g[free[i]];
    for (let j = 0; j < i; j++) v -= L[i][j] * y[j];
    y[i] = v / L[i][i];
  }
  for (let i = n - 1; i >= 0; i--) {
    let v = y[i];
    for (let j = i + 1; j < n; j++) v -= L[j][i] * q[j];
    q[i] = v / L[i][i]; result[free[i]] = q[i];
  }
  return result;
}
function trueRelativeResidual(s, dt) {
  const aq = new Float64Array(s.x.length); s.multiply(s.direction, aq, dt);
  return norm(aq.map((v, i) => v + s.g[i])) / norm(s.g);
}

// Tiny exact principal-block audit covering every supported metric contribution.
const f = oneTet(); f.arrays.hinges = Float64Array.from([0, 1, 2, 3, 0, .004]);
const normal = [1, 2, 3].map(v => v / Math.sqrt(14));
const plane = { stiffness: 2e4, evaluate(p) {
  const gap = dot(p, normal) - .08, violation = Math.max(-gap, 0);
  return { energy: .5 * this.stiffness * violation ** 2, violation,
    gradient: normal.map(v => -this.stiffness * violation * v), normal: normal.slice(), signedDistance: gap };
} };
function complete(Solver) {
  const s = new Solver(f.model, f.arrays, [{ a: 0, b: 3, activation: .65, stiffness: 500 }],
    [{ nodes: [0, 1, 2], weights: [.27, .36, .37], collider: plane, quadratureWeight: .7 }],
    [{ id: "fixture-muscle", tetrahedra: [0], fibreDirection: [1, 2, 1], stiffness: 1e4, activation: .7 }]);
  s.pin(0, Array.from(s.rest.slice(0, 3)));
  s.substrateContacts = [{ node: 1, point: [0, .1, 0], normal, k: 17, clearance: .001 }];
  s.solidMeshContacts = [{ node: 2, collider: plane }];
  s.pointSprings = [{ node: 3, target: Array.from(s.rest.slice(9, 12)), k: 11 }];
  s.setSphereContacts([{ center: [.05, .05, .05], radius: .2, stiffness: 13 }]);
  s.volumeTargets = Float64Array.from([1.04]); s.gravity = [.1, -.2, .05];
  return s;
}
const base = complete(VolumetricMuscleFEM), block = complete(NodalBlockJacobiVolumetricFEM), dt = .02,
  compressed = Float64Array.from(f.arrays.rest, (v, i) => f.arrays.rest[i % 3] + .7 * (v - f.arrays.rest[i % 3]));
const bs = base.evaluate(compressed, dt, true), ns = block.evaluate(compressed, dt, true);
assert(bs.minJ > .2 && bs.minJ < .6, "Compression-barrier metric must be exercised");
assert.deepEqual(ns, bs); assert.deepEqual(block.g, base.g);
assert(block.surfaceContactBlocks.length && block.fibreBlocks.length && block.muscleBlocks.length && block.hingeBlocks.length);
const A = fullMatrix(base, dt), w = buildNodalBlockJacobi(block);
let maxBlockError = 0, maxSymmetryError = 0;
const packed = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]];
for (let node = 0; node < 4; node++) for (let k = 0; k < 6; k++) {
  const [a, b] = packed[k], expected = block.fixed[node] ? Number(a === b) : A[node * 3 + a][node * 3 + b];
  maxBlockError = Math.max(maxBlockError, Math.abs(w.blocks[node * 6 + k] - expected));
}
for (let i = 0; i < A.length; i++) for (let j = 0; j < A.length; j++)
  maxSymmetryError = Math.max(maxSymmetryError, Math.abs(A[i][j] - A[j][i]));
assert(maxBlockError < 1e-10 && maxSymmetryError < 1e-10); assert.equal(w.fallbackNodes, 0);
const r = Float64Array.from({ length: 12 }, (_, i) => Math.sin(i + .13)), z = new Float64Array(12);
applyNodalBlockJacobi(w, r, z);
let inverseError = 0;
for (let node = 0; node < 4; node++) for (let a = 0; a < 3; a++) {
  let value = 0;
  for (let b = 0; b < 3; b++) value += (block.fixed[node] ? Number(a === b) : A[node * 3 + a][node * 3 + b]) * z[node * 3 + b];
  inverseError = Math.max(inverseError, Math.abs(value - r[node * 3 + a]));
}
assert(inverseError < 1e-12); assert(dot(r, z) > 0);
const exact = directSolve(base, A), linearTolerance = 1e-10;
const diagIterations = base.pcg(dt, 120, linearTolerance), blockIterations = block.pcg(dt, 120, linearTolerance);
const diagTrueResidual = trueRelativeResidual(base, dt), blockTrueResidual = trueRelativeResidual(block, dt);
assert(diagTrueResidual < 1.1 * linearTolerance && blockTrueResidual < 1.1 * linearTolerance);
assert(maxDifference(exact, base.direction) < 1e-9 && maxDifference(exact, block.direction) < 1e-9);
for (let i = 0; i < 3; i++) assert.equal(block.direction[i], 0);
const savedBlocks = JSON.stringify(block.surfaceContactBlocks), savedWorkspace = Array.from(block.nodalBlockJacobi.blocks);
block.evaluate(block.rest, dt, false);
assert.equal(JSON.stringify(block.surfaceContactBlocks), savedBlocks);
assert.deepEqual(Array.from(block.nodalBlockJacobi.blocks), savedWorkspace);
report.exactBlockAudit = { maxBlockError, maxSymmetryError, inverseError, fallbackNodes: w.fallbackNodes,
  diagIterations, blockIterations, diagTrueResidual, blockTrueResidual };

// Existing Galerkin warm start remains compatible and uses the original rule.
block.evaluate(compressed, dt, true); block.useSearchSubspace = true; block.searchBasis.length = 0;
block.pcg(dt, 120, linearTolerance); const beforeWarm = block.direction.slice();
const warmIterations = block.pcg(dt, 120, linearTolerance);
assert.equal(warmIterations, 0); assert(maxDifference(beforeWarm, block.direction) < 1e-12);
const unsupported = complete(VolumetricMuscleFEM); unsupported.multiply = () => {};
assert.throws(() => buildNodalBlockJacobi(unsupported), /supported/);
const badWorkspace = {blocks:new Float64Array(1),factors:new Float64Array(1),scales:new Float64Array(1),fallback:new Uint8Array(1)};
assert.throws(() => buildNodalBlockJacobi(base,badWorkspace), /workspace size/);
assert.throws(() => applyNodalBlockJacobi(w,new Float64Array(1),new Float64Array(1)), /vector size/);
const invalid = complete(VolumetricMuscleFEM); invalid.evaluate(compressed,dt,true); invalid.diagonal[3]=NaN;
assert.throws(() => buildNodalBlockJacobi(invalid), /Nonpositive or nonfinite/);

// Deliberately extreme stiffness loses the small transverse eigenvalue in
// floating point. Report and use scalar Jacobi; never regularize the metric.
const ff = oneTet(), fallbackSolver = new VolumetricMuscleFEM(ff.model,ff.arrays);
fallbackSolver.substrateContacts = [{node:1,point:[2,2,2],normal:[0,Math.cos(.1),Math.sin(.1)],k:1e24,clearance:0}];
fallbackSolver.evaluate(fallbackSolver.x,Infinity,true);
const fallbackDiagonal = fallbackSolver.diagonal.slice(), fw=buildNodalBlockJacobi(fallbackSolver), fz=new Float64Array(12);
assert(fw.fallbackNodes>0,"Extreme ill-conditioning must exercise the disclosed scalar fallback");
applyNodalBlockJacobi(fw,r,fz); assert(fz.every(Number.isFinite)); assert(dot(r,fz)>0);
assert.deepEqual(fallbackSolver.diagonal,fallbackDiagonal);
for(let node=0;node<4;node++)if(fw.fallback[node])for(let d=0;d<3;d++)
  assert.equal(fz[node*3+d],r[node*3+d]/fallbackDiagonal[node*3+d]);
report.numericalFallback={fallbackNodes:fw.fallbackNodes,metricUnchanged:true,preconditionerQuadratic:dot(r,fz)};
const allPinned = complete(NodalBlockJacobiVolumetricFEM);
for (let node = 0; node < 4; node++) allPinned.pin(node, Array.from(allPinned.rest.slice(node * 3, node * 3 + 3)));
allPinned.evaluate(allPinned.rest, Infinity, true); assert.equal(allPinned.pcg(Infinity, 5, .0015), 0);
assert(allPinned.direction.every(v => v === 0));

// A 64-node procedural rotated grid; no regional case is loaded or solved.
function gridFixture(n = 3) {
  const a = .47, b = .39, ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b),
    R = [ca*cb, -sa, ca*sb, sa*cb, ca, sa*sb, -sb, 0, cb], spacing = .008,
    rotate = p => [0,1,2].map(d => R[d*3]*p[0] + R[d*3+1]*p[1] + R[d*3+2]*p[2]),
    normal = [R[2], R[5], R[8]], rest = [], local = [], nodes = [], gradients = [], material = [],
    index = (x,y,z) => x + (n+1) * (y + (n+1)*z),
    permutations = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]];
  for (let z = 0; z <= n; z++) for (let y = 0; y <= n; y++) for (let x = 0; x <= n; x++) {
    local.push([x,y,z]); rest.push(...rotate([x*spacing,y*spacing,z*spacing]));
  }
  for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) for (const permutation of permutations) {
    const p = [x,y,z], ids = [index(...p)];
    for (const axis of permutation) { p[axis]++; ids.push(index(...p)); }
    const D = [0,1,2].map(d => [1,2,3].map(a => rest[ids[a]*3+d] - rest[ids[0]*3+d]));
    const [a,b,c] = D[0], [d,e,f] = D[1], [g,h,i] = D[2],
      det = a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g),
      inv = [[e*i-f*h,c*h-b*i,b*f-c*e],[f*g-d*i,a*i-c*g,c*d-a*f],[d*h-e*g,b*g-a*h,a*e-b*d]].map(row=>row.map(v=>v/det));
    nodes.push(...ids); gradients.push(...[0,1,2].map(d => -inv.reduce((sum,row)=>sum+row[d],0)), ...inv.flat());
    material.push(Math.abs(det)/6, 300, 2400);
  }
  const faces = new Map();
  for (let t = 0; t < nodes.length; t += 4) for (let omit = 0; omit < 4; omit++) {
    const ids = nodes.slice(t,t+4).filter((_,a)=>a!==omit), key = ids.slice().sort((a,b)=>a-b).join(":");
    const record = faces.get(key); if (record) record.count++; else faces.set(key,{ids,count:1});
  }
  const topFaces = Array.from(faces.values()).filter(f=>f.count===1&&f.ids.every(node=>local[node][2]===n)).map(f=>f.ids);
  const stiffness = 1e5, collider = { stiffness, evaluate(p) {
    const gap = dot(p, normal) - n * spacing - .0001, violation = Math.max(-gap, 0);
    return { energy: .5 * stiffness * violation ** 2, violation, signedDistance: gap, normal: normal.slice(),
      gradient: normal.map(v => -stiffness * violation * v) };
  } };
  const contacts = topFaces.flatMap((ids,face)=>Array.from({length:3},(_,a)=>({ id:`${face}:${a}`,nodes:ids,
    weights:Array.from({length:3},(_,b)=>b===a?2/3:1/6),quadratureWeight:1/3,collider })));
  const model = { schema:"cybr-layered-fem-v1",nodes:local.length,surface_nodes:local.length,tetrahedra:nodes.length/4 },
    arrays = {rest:Float64Array.from(rest),nodes:Uint32Array.from(nodes),gradients:Float64Array.from(gradients),
      material:Float64Array.from(material),mass:new Float64Array(local.length).fill(.001)};
  return {model,arrays,contacts,local};
}
function benchmark() {
  const f = gridFixture(), make = Solver => {
    const s = new Solver(f.model,f.arrays,[],f.contacts,[]);
    for (let n = 0; n < f.local.length; n++) if (f.local[n][2] === 0) s.pin(n,Array.from(s.rest.slice(3*n,3*n+3)));
    s.evaluate(s.x,Infinity,true);
    // Deterministic mixed-direction RHS probes a frozen physical metric.
    for (let i = 0; i < s.g.length; i++) s.g[i] = s.fixed[Math.floor(i/3)] ? 0 : Math.sin((i+1)*.731)+.31*Math.cos((i+1)*.217);
    return s;
  };
  const diag = make(VolumetricMuscleFEM), block = make(NodalBlockJacobiVolumetricFEM), tolerance = 1e-8, limit = 1200,
    A = fullMatrix(diag,Infinity), exact = directSolve(diag,A);
  const run = s => {const begin=performance.now(),iterations=s.pcg(Infinity,limit,tolerance), elapsedMs=performance.now()-begin;
    return {iterations,elapsedMs,trueRelativeResidual:trueRelativeResidual(s,Infinity),relativeSolutionError:norm(s.direction.map((v,i)=>v-exact[i]))/norm(exact)};};
  const d = run(diag), b = run(block);
  assert(d.iterations < limit && b.iterations < limit, "Both comparisons must actually reach the same residual criterion");
  assert(d.trueRelativeResidual < tolerance*1.2 && b.trueRelativeResidual < tolerance*1.2);
  assert(d.relativeSolutionError < 1e-7 && b.relativeSolutionError < 1e-7);
  assert(b.iterations < d.iterations, "The rotated stiff-contact fixture should benefit from retaining local vector coupling");
  assert.equal(block.nodalBlockJacobi.fallbackNodes,0);
  // Repeat, including preconditioner rebuild cost, for stable iteration evidence.
  const d2 = run(diag), b2 = run(block);
  assert.equal(d2.iterations,d.iterations); assert.equal(b2.iterations,b.iterations);
  assert.equal(d2.trueRelativeResidual,d.trueRelativeResidual); assert.equal(b2.trueRelativeResidual,b.trueRelativeResidual);
  const productionTolerance=.0015,productionLimit=90,
    productionScalarIterations=diag.pcg(Infinity,productionLimit,productionTolerance),
    productionScalarResidual=trueRelativeResidual(diag,Infinity),
    productionBlockIterations=block.pcg(Infinity,productionLimit,productionTolerance),
    productionBlockResidual=trueRelativeResidual(block,Infinity);
  assert.equal(productionScalarIterations,productionLimit);
  assert(productionScalarResidual>productionTolerance,"Legacy capped solve is not labeled linearly converged");
  assert(productionBlockIterations<productionLimit && productionBlockResidual<productionTolerance);
  return {nodes:f.model.nodes,tetrahedra:f.model.tetrahedra,embeddedContacts:f.contacts.length,tolerance,
    scalar:d,block:b,repeatedScalar:d2,repeatedBlock:b2,iterationReductionFraction:1-b.iterations/d.iterations,
    sameRegionalBudget:{tolerance:productionTolerance,iterationLimit:productionLimit,
      scalar:{iterations:productionScalarIterations,trueRelativeResidual:productionScalarResidual},
      block:{iterations:productionBlockIterations,trueRelativeResidual:productionBlockResidual}},
    timingCaveat:"Tiny frozen-metric diagnostic, including rebuild. Not a regional nonlinear speedup measurement."};
}
report.rotatedEmbeddedContact = benchmark();

// Actual nonlinear equilibrium against the same closed cube and all original gates.
function nonlinear(Solver) {
  const f = oneTet(), s = new Solver(f.model,f.arrays,[],
    [{nodes:[0,1,2],weights:[1/3,1/3,1/3],collider:cube()}],[]);
  for (const node of [1,2,3]) s.pin(node,Array.from(s.rest.slice(node*3,node*3+3)));
  const initial = s.evaluate(s.x,Infinity,true),
    options = {iterations:180,cgIterations:40,cgTolerance:1e-10,forceTolerance:1e-9,minimumJ:.2},
    result = s.step(Infinity,options), final = s.evaluate(s.x,Infinity,true);
  assert(result.converged && final.residual <= options.forceTolerance);
  assert(final.energy < initial.energy && final.minJ > options.minimumJ);
  assert(final.surfaceContactViolation < initial.surfaceContactViolation * .2);
  for (const node of [1,2,3]) for (let d=0;d<3;d++) assert.equal(s.x[node*3+d],s.rest[node*3+d]);
  return {s,initial,result,options};
}
const nd = nonlinear(VolumetricMuscleFEM), nb = nonlinear(NodalBlockJacobiVolumetricFEM);
assert(maxDifference(nd.s.x,nb.s.x) < 1e-10);
report.nonlinearClosedCube = {options:nb.options,scalar:nd.result,block:nb.result,
  maximumPositionDifference:maxDifference(nd.s.x,nb.s.x),energyDifference:Math.abs(nd.result.energy-nb.result.energy)};
// Explicit instance opt-in leaves prototype identities and other instances intact.
const opt = complete(VolumetricMuscleFEM); opt.pcg = nodalBlockJacobiPCG;
opt.evaluate(compressed,dt,true); assert.equal(opt.pcg(dt,120,linearTolerance),blockIterations);
assert.equal(base.pcg,LayeredFEM.prototype.pcg);
assert.equal(hash(referenceURL),referenceSHA256);
report.checks = ["all supported exact nodal principal blocks", "symmetric positive definite dense free metric",
  "Cholesky inverse and positive quadratic form", "same Euclidean residual stopping criterion and direct solution",
  "energy/gradient/multiply/step unchanged", "compression barrier and every existing rank-one family",
  "pins and zero-RHS case", "line-search trial does not overwrite current metric", "Galerkin warm start",
  "unknown metric extensions and invalid inputs rejected", "explicit scalar numerical fallback without metric mutation",
  "repeatable tiny rotated embedded-contact comparison", "unchanged regional CG cap/tolerance fixture",
  "same nonlinear cube equilibrium at unchanged tolerances", "legacy core SHA256 unchanged"];
report.peakRSSKiB = process.resourceUsage().maxRSS;
assert(report.peakRSSKiB < 300*1024,"Tiny-fixture RSS bound exceeded");
report.passed = true;
console.log(JSON.stringify(report,null,2));
