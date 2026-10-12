import assert from "node:assert/strict";
import fs from "node:fs";
import crypto from "node:crypto";
import { VolumetricMuscleFEM } from "../src/volumetric_muscle_fem.mjs";
import { EmbeddedSurfaceContactFEM } from "../src/embedded_surface_contact_fem.mjs";
import { LayeredFEM } from "../reference-source/src/layered_fem.js";

const close = (a, b, tolerance = 1e-10) => assert(Math.abs(a-b) <= tolerance, `${a} != ${b}`);
const dot = (a, b) => a.reduce((sum, value, i) => sum + value*b[i], 0);
const norm = a => Math.hypot(...a);
const sub = (a, b) => Array.from(a, (v, i) => v-b[i]);
function inverse3(m) {
  const [a,b,c,d,e,f,g,h,i] = m, det = a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);
  return { det, inverse: [e*i-f*h,c*h-b*i,b*f-c*e,f*g-d*i,a*i-c*g,c*d-a*f,d*h-e*g,b*g-a*h,a*e-b*d].map(v => v/det) };
}
function transform(x, A, translation = [0,0,0]) {
  return Float64Array.from(x, (_, i) => {
    const n = i-i%3, d = i%3;
    return translation[d] + A[d*3]*x[n] + A[d*3+1]*x[n+1] + A[d*3+2]*x[n+2];
  });
}

/** Independently authored rectangular solid, six tets per length segment. */
function fixture({segments=1, length=.01, skew=false, mu=1500, lambda=30000} = {}) {
  let rest = Float64Array.from(Array.from({length:(segments+1)*4}, (_, n) =>
    [Math.floor(n/4)*length/segments, (n%4&1)*.01, ((n%4>>1)&1)*.01]).flat());
  if (skew) rest = transform(rest, [1.1,.15,.12, .2,.9,-.08, -.1,.17,1.2], [.012,-.006,.004]);
  const cells = [], gradients = [], material = [], mass = new Float64Array(rest.length/3);
  for (let segment=0; segment<segments; segment++) {
    const cube = Array.from({length:8}, (_,i) => (segment+(i&1))*4+(i>>1));
    for (const local of [[0,1,3,7],[0,3,2,7],[0,2,6,7],[0,6,4,7],[0,4,5,7],[0,5,1,7]]) {
      const ids = local.map(i => cube[i]);
      const matrix = () => [0,1,2].flatMap(d => [1,2,3].map(a => rest[ids[a]*3+d]-rest[ids[0]*3+d]));
      if (inverse3(matrix()).det < 0) [ids[1],ids[2]] = [ids[2],ids[1]];
      const {det,inverse} = inverse3(matrix()), V = det/6;
      cells.push(...ids);
      gradients.push(...[0,1,2].map(d => -inverse[d]-inverse[3+d]-inverse[6+d]), ...inverse);
      material.push(V,mu,lambda);
      for (const node of ids) mass[node] += 1000*V/4;
    }
  }
  return {model:{schema:"cybr-layered-fem-v1",nodes:rest.length/3,surface_nodes:rest.length/3,tetrahedra:cells.length/4},
    arrays:{rest,nodes:Uint32Array.from(cells),gradients:Float64Array.from(gradients),material:Float64Array.from(material),mass}};
}
function regions(f, activation=.18, stiffness=8000) {
  return [{id:"muscle",tetrahedra:Array.from({length:f.model.tetrahedra},(_,i)=>i),
    fibreDirection:[1,0,0],stiffness,activation}];
}
const sourceHash = crypto.createHash("sha256").update(fs.readFileSync(new URL("../reference-source/src/layered_fem.js", import.meta.url))).digest("hex");
assert.equal(sourceHash, "8618d5517de196dd3dae905d598a69eaf14555682895c80dcfc994e2501dbc71");
assert.equal(VolumetricMuscleFEM.prototype.step, LayeredFEM.prototype.step);
assert.equal(VolumetricMuscleFEM.prototype.pcg, LayeredFEM.prototype.pcg);

// Arbitrary non-axis-aligned rest geometry and nonuniform fibre directions expose
// transposed-gradient mistakes that an axis-aligned one-tet test can miss.
const f = fixture({skew:true}), region = {...regions(f,.23,7300)[0],fibreDirection:undefined,
  fibreDirections:Array.from({length:6},(_,i)=>[.9,.1+.03*i,.2-.01*i])},
  solver = new VolumetricMuscleFEM(f.model,f.arrays,[],[],[region]),
  base = new EmbeddedSurfaceContactFEM(f.model,f.arrays),
  posed = Float64Array.from(f.arrays.rest,(v,i)=>v+.00012*Math.sin(i*.79));
const state = solver.evaluate(posed,.02,true), gradient = solver.g.slice(), muscleForces = solver.muscleForces.slice();
let gradientMaxAbsolute = 0, gradientMaxRelative = 0, muscleGradientMaxAbsolute = 0;
for (let j=0;j<posed.length;j++) {
  const old = posed[j], h = 1e-8;
  posed[j]=old+h; const plus=solver.evaluate(posed,.02,false);
  posed[j]=old-h; const minus=solver.evaluate(posed,.02,false);
  posed[j]=old;
  const fd=(plus.energy-minus.energy)/(2*h), error=Math.abs(fd-gradient[j]),
    fdMuscle=(plus.muscleEnergy-minus.muscleEnergy)/(2*h);
  gradientMaxAbsolute=Math.max(gradientMaxAbsolute,error);
  gradientMaxRelative=Math.max(gradientMaxRelative,error/Math.max(1e-6,Math.abs(fd),Math.abs(gradient[j])));
  muscleGradientMaxAbsolute=Math.max(muscleGradientMaxAbsolute,Math.abs(fdMuscle+muscleForces[j]));
}
assert(gradientMaxRelative<1e-5); assert(muscleGradientMaxAbsolute<1e-9);
close(state.energy,state.passiveAndExternalEnergy+state.fibreEnergy+state.muscleEnergy,1e-14);

// Active and total material energies are objective; internal nodal forces rotate.
// Inertia is excluded here because its fixed predictor is an external frame.
const a=.48,b=-.37, ca=Math.cos(a),sa=Math.sin(a),cb=Math.cos(b),sb=Math.sin(b),
  R=[ca*cb,-sa,ca*sb, sa*cb,ca,sa*sb, -sb,0,cb], rotated=transform(posed,R,[-.005,.02,-.01]),
  originalState=solver.evaluate(posed,Infinity,true), originalGradient=solver.g.slice(), originalMuscle=solver.muscleForces.slice(),
  rotatedState=solver.evaluate(rotated,Infinity,true), expectedGradient=transform(originalGradient,R), expectedForces=transform(originalMuscle,R);
const objectivityEnergyError=Math.abs(originalState.energy-rotatedState.energy),
  objectivityMuscleEnergyError=Math.abs(originalState.muscleEnergy-rotatedState.muscleEnergy),
  objectivityForceError=Math.max(...sub(solver.g,expectedGradient).map(Math.abs)),
  objectivityMuscleForceError=Math.max(...sub(solver.muscleForces,expectedForces).map(Math.abs));
assert(objectivityEnergyError<1e-14); assert(objectivityMuscleEnergyError<1e-14);
assert(objectivityForceError<1e-10); assert(objectivityMuscleForceError<1e-11);
const forceSum=[0,0,0], torqueSum=[0,0,0];
for(let node=0;node<f.model.nodes;node++) {
  const p=Array.from(posed.subarray(node*3,node*3+3)), force=Array.from(originalMuscle.subarray(node*3,node*3+3));
  for(let d=0;d<3;d++) forceSum[d]+=force[d];
  torqueSum[0]+=p[1]*force[2]-p[2]*force[1];
  torqueSum[1]+=p[2]*force[0]-p[0]*force[2];
  torqueSum[2]+=p[0]*force[1]-p[1]*force[0];
}
assert(norm(forceSum)<1e-12); assert(norm(torqueSum)<1e-12);

// Explicit dense assembly on the tiny fixture checks every GN metric entry,
// its diagonal preconditioner, and Dirichlet row AND column exclusion.
solver.pin(0,Array.from(solver.rest.subarray(0,3))); base.pin(0,Array.from(base.rest.subarray(0,3)));
solver.evaluate(posed,.02,true); base.evaluate(posed,.02,true);
const size=posed.length, matrix=[], delta=[], expected=Array.from({length:size},()=>new Float64Array(size));
for(const block of solver.muscleBlocks) for(let a=0;a<4;a++) for(let b=0;b<4;b++) {
  if(solver.fixed[block.nodes[a]]||solver.fixed[block.nodes[b]]) continue;
  for(let d=0;d<3;d++) for(let e=0;e<3;e++) expected[block.nodes[a]*3+d][block.nodes[b]*3+e]
    +=block.k*block.coefficients[a]*block.direction[d]*block.coefficients[b]*block.direction[e];
}
let metricSymmetryError=0, metricAssemblyError=0, metricDiagonalError=0;
for(let column=0;column<size;column++) {
  const v=new Float64Array(size), out=new Float64Array(size), without=new Float64Array(size); v[column]=1;
  solver.multiply(v,out,.02); base.multiply(v,without,.02);
  matrix.push(out); delta.push(out.map((v,row)=>v-without[row]));
}
for(let i=0;i<size;i++) for(let j=0;j<size;j++) {
  metricSymmetryError=Math.max(metricSymmetryError,Math.abs(matrix[i][j]-matrix[j][i]));
  metricAssemblyError=Math.max(metricAssemblyError,Math.abs(delta[j][i]-expected[i][j]));
}
for(let i=0;i<size;i++) if(!solver.fixed[Math.floor(i/3)]) metricDiagonalError=Math.max(metricDiagonalError,Math.abs(matrix[i][i]-solver.diagonal[i]));
assert(metricSymmetryError<1e-10); assert(metricAssemblyError<1e-10); assert(metricDiagonalError<1e-10);
let minimumMuscleQuadratic=Infinity, minimumFullQuadratic=Infinity;
for(let sample=0;sample<32;sample++) {
  const v=Float64Array.from({length:size},(_,i)=>Math.sin((i+1)*(sample+1)*.317)),out=new Float64Array(size),without=new Float64Array(size);
  solver.multiply(v,out,.02); base.multiply(v,without,.02);
  minimumFullQuadratic=Math.min(minimumFullQuadratic,dot(v,out));
  minimumMuscleQuadratic=Math.min(minimumMuscleQuadratic,dot(v,out.map((value,i)=>value-without[i])));
}
assert(minimumFullQuadratic>0); assert(minimumMuscleQuadratic>=-1e-10);
for(let d=0;d<3;d++) {
  assert.equal(solver.g[d],0); assert.equal(solver.diagonal[d],1);
  for(let i=0;i<size;i++) {assert.equal(matrix[d][i],0);assert.equal(matrix[i][d],0);}
}
const savedBlocks=JSON.stringify(solver.muscleBlocks);
solver.evaluate(transform(posed,[.65,0,0,0,1,0,0,0,1]),Infinity,false);
assert.equal(JSON.stringify(solver.muscleBlocks),savedBlocks,"Line-search trials must preserve current metric blocks");

// Composition exercises all three authored energies in the same solve without
// coupling their activation controls. A planar penalty is the exact limiting
// contact feature of a locally flat bone face and keeps finite differences away
// from nearest-triangle feature switches (mesh contact has its separate tests).
const plane={stiffness:5,evaluate(p) {
  const signedDistance=p[2]-.012, violation=Math.max(-signedDistance,0),normal=[0,0,1];
  return {energy:.5*this.stiffness*violation**2,gradient:[0,0,-this.stiffness*violation],
    violation,signedDistance,normal,closestPoint:[p[0],p[1],.012]};
}}, cables=[{id:"cable",a:0,b:7,stiffness:2,activation:.1}],
  combined=new VolumetricMuscleFEM(f.model,f.arrays,cables,[{nodes:[0,4,5],weights:[.2,.3,.5],collider:plane}],[region]),
  combinedState=combined.evaluate(posed,.02,true),combinedGradient=combined.g.slice();
assert(combinedState.fibreEnergy>0&&combinedState.surfaceContactEnergy>0&&combinedState.muscleEnergy>0);
let combinedGradientMaxAbsolute=0;
for(let j=0;j<posed.length;j++) {
  const old=posed[j],h=1e-8;
  posed[j]=old+h;const plus=combined.evaluate(posed,.02,false).energy;
  posed[j]=old-h;const minus=combined.evaluate(posed,.02,false).energy;
  posed[j]=old;
  combinedGradientMaxAbsolute=Math.max(combinedGradientMaxAbsolute,Math.abs((plus-minus)/(2*h)-combinedGradient[j]));
}
assert(combinedGradientMaxAbsolute<1e-9);
combined.setMuscleActivation(.3);assert.equal(combined.fibres[0].activation,.1);
combined.setActivation(.2);assert.equal(combined.muscleRegions[0].activation,.3);
for(let n=0;n<f.model.nodes;n++) combined.pin(n,Array.from(combined.rest.subarray(n*3,n*3+3)));
const allPinned=combined.evaluate(posed,.02,true),zero=new Float64Array(size);
combined.multiply(Float64Array.from({length:size},(_,i)=>Math.sin(i)),zero,.02);
assert.equal(allPinned.residual,0);assert(zero.every(v=>v===0));assert(combined.diagonal.every(v=>v===1));
assert(combined.muscleForces.some(v=>v!==0),"Pinned diagnostic muscle loads must survive free-DOF projection");

// Zero-activation rest/slack branches, empty-region parity, and region controls.
const regular=fixture(), inactive=new VolumetricMuscleFEM(regular.model,regular.arrays,[],[],regions(regular,0)),
  inherited=new EmbeddedSurfaceContactFEM(regular.model,regular.arrays),
  inactiveState=inactive.evaluate(inactive.rest,Infinity,true), inheritedState=inherited.evaluate(inherited.rest,Infinity,true);
assert.equal(inactiveState.muscleEnergy,0); assert.equal(inactiveState.energy,inheritedState.energy);
assert(inactive.muscleForces.every(v=>v===0)); assert.equal(inactive.muscleBlocks.length,0);
assert.equal(inactive.evaluate(transform(inactive.rest,[.7,0,0,0,1,0,0,0,1]),Infinity,true).muscleEnergy,0);
assert(inactive.evaluate(transform(inactive.rest,[1.1,0,0,0,1,0,0,0,1]),Infinity,true).muscleEnergy>0,
  "Inactive extended fibres still contribute the documented anisotropic term");
const empty=new VolumetricMuscleFEM(f.model,f.arrays);
close(empty.evaluate(posed,.02,true).energy,new EmbeddedSurfaceContactFEM(f.model,f.arrays).evaluate(posed,.02,true).energy,0);
const splitRegions=[{id:"first",tetrahedra:[0,1,2],fibreDirection:[1,0,0],stiffness:5000,activation:0},
  {id:"second",tetrahedra:[3,4,5],fibreDirection:[1,0,0],stiffness:5000,activation:0}],
  controlled=new VolumetricMuscleFEM(regular.model,regular.arrays,[],[],splitRegions);
controlled.setMuscleActivation(.2,"second"); assert.deepEqual(controlled.muscleRegions.map(r=>r.activation),[0,.2]);
controlled.setMuscleActivation(.1,0); assert.deepEqual(controlled.muscleRegions.map(r=>r.activation),[.1,.2]);
controlled.setMuscleActivation(0); assert.deepEqual(controlled.muscleRegions.map(r=>r.activation),[0,0]);

// Two different tet tessellations of the same rest volume integrate the same
// homogeneous deformation energy. No current-volume weighting is substituted.
const one=fixture({segments:1,length:.02}),two=fixture({segments:2,length:.02}),
  oneSolver=new VolumetricMuscleFEM(one.model,one.arrays,[],[],regions(one,.2)),
  twoSolver=new VolumetricMuscleFEM(two.model,two.arrays,[],[],regions(two,.2)),
  stretchMatrix=[.91,.04,0,.02,1.04,0,0,0,1.03],
  oneState=oneSolver.evaluate(transform(one.arrays.rest,stretchMatrix),Infinity,true),
  twoState=twoSolver.evaluate(transform(two.arrays.rest,stretchMatrix),Infinity,true),
  expectedEnergy=.5*(.02*.01*.01)*8000*(Math.hypot(.91,.02)-.8)**2;
close(oneState.muscleEnergy,expectedEnergy,1e-15); close(twoState.muscleEnergy,expectedEnergy,1e-15);

// Real finite muscle: first two segments contract, third is passive insertion
// tissue. Four shared insertion nodes carry load into that passive material.
// The origin plane is fixed, and four weak end springs represent a surrounding
// tissue load. No fibre/cable, skin position target, or contraction displacement.
function contractionFixture(activation=.25) {
  const f=fixture({segments:3,length:.03}), r={...regions(f,activation)[0],tetrahedra:Array.from({length:12},(_,i)=>i)},
    fem=new VolumetricMuscleFEM(f.model,f.arrays,[],[],[r]);
  for(let n=0;n<4;n++) fem.pin(n,Array.from(fem.rest.subarray(n*3,n*3+3)));
  for(let n=12;n<16;n++) fem.pointSprings.push({node:n,target:Array.from(fem.rest.subarray(n*3,n*3+3)),k:2});
  return fem;
}
const contraction=contractionFixture(),before=contraction.evaluate(contraction.x,Infinity,true),
  insertionNodes=[8,9,10,11], endNodes=[12,13,14,15], initialInsertionLoads=insertionNodes.map(n=>norm(contraction.muscleForces.subarray(n*3,n*3+3))),
  result=contraction.step(Infinity,{iterations:800,cgIterations:100,cgTolerance:1e-9,forceTolerance:1e-9});
const final=contraction.evaluate(contraction.x,Infinity,true), finalRegion={...contraction.muscleDiagnostics[0]},
  insertionDisplacements=insertionNodes.map(n=>contraction.x[n*3]-contraction.rest[n*3]),
  passiveEndDisplacements=endNodes.map(n=>contraction.x[n*3]-contraction.rest[n*3]),
  shortening=-insertionDisplacements.reduce((a,b)=>a+b,0)/4,
  maxDisplacement=Math.max(...sub(contraction.x,contraction.rest).map(Math.abs));
assert(result.converged,"Small finite muscle should reach equilibrium");
assert(result.energy<before.energy); assert(result.minJ>.2);
assert(shortening>1e-4); assert(finalRegion.meanStretch<.98);
assert(initialInsertionLoads.every(v=>v>1e-5),"Every shared insertion node should carry active load");
assert(insertionDisplacements.every(v=>v<0)); assert(passiveEndDisplacements.every(v=>v<0),"Passive tissue must be pulled by shared insertion");
assert(finalRegion.meanJ>.8&&finalRegion.meanJ<1.1,"Inherited bulk elasticity should limit finite muscle volume change");
for(let n=0;n<4;n++) for(let d=0;d<3;d++) assert.equal(contraction.x[n*3+d],contraction.rest[n*3+d]);
contraction.setMuscleActivation(0);
const unloading=contraction.step(Infinity,{iterations:600,cgIterations:100,cgTolerance:1e-9,forceTolerance:1e-9}),
  unloadError=Math.max(...sub(contraction.x,contraction.rest).map(Math.abs));
assert(unloading.converged); assert(unloadError<1e-8);

// Bounded finite-time activation/unloading cycle, with the original implicit
// inertia, damping, PCG and determinant-safe actual-energy backtracking.
const dynamic=contractionFixture(0),dynamicStates=[];
for(let frame=0;frame<40;frame++) {
  dynamic.setMuscleActivation(frame===39?0:.3*Math.sin(Math.PI*frame/39));
  const r=dynamic.step(.01,{iterations:160,cgIterations:70,cgTolerance:1e-8,forceTolerance:1e-8});
  assert(r.converged, JSON.stringify({frame,...r})); assert(Number.isFinite(r.energy)); assert(r.minJ>.2);
  assert(dynamic.x.every(Number.isFinite));
  dynamicStates.push({minJ:r.minJ,maxJ:r.maxJ,residual:r.residual,maxDisplacement:Math.max(...sub(dynamic.x,dynamic.rest).map(Math.abs)),rejections:r.rejections});
}
const dynamicMaximumDisplacement=Math.max(...dynamicStates.map(s=>s.maxDisplacement));
assert(dynamicMaximumDisplacement<.015,"Bounded activation should not cause runaway motion");

// Invalid tags and weights fail before replacing a working muscle configuration.
assert.throws(()=>new VolumetricMuscleFEM(f.model,f.arrays,[],[],[{...region,tetrahedra:[6]}]),/tetrahedron/);
assert.throws(()=>new VolumetricMuscleFEM(f.model,f.arrays,[],[],[{...region,tetrahedra:[0,0]}]),/fibreDirections|assigned/);
assert.throws(()=>new VolumetricMuscleFEM(f.model,f.arrays,[],[],[region,region]),/Duplicate/);
assert.throws(()=>new VolumetricMuscleFEM(regular.model,regular.arrays,[],[],[{...regions(regular)[0],fibreDirection:[0,0,0]}]),/nonzero/);
assert.throws(()=>inactive.setMuscleActivation(1.1),/activation/); assert.throws(()=>inactive.setMuscleActivation(.1,"missing"),/selector/);
const beforeInvalid=inactive.muscleRegions;
assert.throws(()=>inactive.setMuscleRegions([{...regions(regular)[0],stiffness:0}]),/stiffness/);
assert.equal(inactive.muscleRegions,beforeInvalid);

const report={schema:"volumetric-muscle-unit-check-v1",passed:true,originalSolverSHA256:sourceHash,
  fixture:"Authored six-tet skew cube and 18-tet muscle/passive insertion bar; no anatomical assets",
  gradientMaxAbsolute,gradientMaxRelative,muscleGradientMaxAbsolute,combinedGradientMaxAbsolute,
  objectivityEnergyError,objectivityMuscleEnergyError,objectivityForceError,objectivityMuscleForceError,
  muscleForceSum:forceSum,muscleTorqueSum:torqueSum,metricSymmetryError,metricAssemblyError,metricDiagonalError,
  minimumMuscleQuadratic,minimumFullQuadratic,homogeneousTessellationEnergyError:Math.abs(oneState.muscleEnergy-twoState.muscleEnergy),
  contraction:{...result,initialEnergy:before.energy,shortening,relativeShortening:shortening/.02,maxDisplacement,
    initialInsertionLoads,insertionDisplacements,passiveEndDisplacements,muscle:finalRegion},
  unloading:{converged:unloading.converged,residual:unloading.residual,maximumRestError:unloadError},
  dynamic:{frames:dynamicStates.length,dt:.01,minimumJ:Math.min(...dynamicStates.map(s=>s.minJ)),
    maximumJ:Math.max(...dynamicStates.map(s=>s.maxJ)),maximumResidual:Math.max(...dynamicStates.map(s=>s.residual)),dynamicMaximumDisplacement,
    allConverged:true,totalLineSearchRejections:dynamicStates.reduce((sum,s)=>sum+s.rejections,0)},
  peakRSSKiB:process.resourceUsage().maxRSS,
  checks:["original source hash and inherited step/PCG", "total and muscle-only energy finite differences",
    "active and passive energy/force rigid objectivity", "internal force and torque balance", "exact rank-one GN assembly and diagonal",
    "metric symmetry and sampled PSD", "pins excluded from rows and columns", "trial evaluations preserve metric blocks",
    "combined passive/inertial/cable/contact/muscle gradient and independent activation", "all-pinned muscle diagnostics",
    "inactive/rest/slack and empty parity", "independent region activation", "rest-volume tessellation invariance",
    "finite muscle contraction with pinned origin and four-node passive insertion", "volume J response and original determinant guard",
    "deactivation returns to rest", "40-step implicit activation stability", "invalid inputs fail atomically"],
  limitations:["Authored target-shortening energy, not a calibrated Hill-type muscle", "Near-incompressibility comes from inherited passive material and is finite",
    "Inactive extension retains fibre stiffness", "Clamped origin and shared-node insertion omit tendon/sliding mechanics", "Tiny-fixture numerical verification is not anatomical validation"]};
if(process.argv.includes("--write-report")) fs.writeFileSync(new URL("../results/volumetric-muscle-unit-check.json",import.meta.url),JSON.stringify(report,null,2)+"\n");
console.log(JSON.stringify(report,null,2));
