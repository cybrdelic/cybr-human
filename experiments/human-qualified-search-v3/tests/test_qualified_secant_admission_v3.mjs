import assert from 'node:assert/strict';
import {installQualifiedSecantSearch} from '../src/qualified_secant_search_v3.mjs';
const make=()=>({x:Float64Array.from([0,0]),g:Float64Array.from([1,1]),direction:new Float64Array(2),diagonal:Float64Array.from([1,1]),calls:0,multiply(v,o){o.set(v);},pcg(){this.calls++;this.direction.set(this.g.map(v=>-v));return 1;}});
const options={contextFingerprint:()=>'mesh-material-constraints-activation-static',contactSignature:()=>'free-contact-face-1'};
const cp=()=>({schema:'qualified-fem-secant-v3',contextFingerprint:options.contextFingerprint(),previousContactSignature:options.contactSignature(),previousMetricSignature:'static:Infinity',records:[{dx:[3,0],dy:[1,0],rho:1/3}],previousX:[0,0],previousG:[1,1]});
const install=(s=make(),extra={})=>installQualifiedSecantSearch(s,{...options,...extra});
const run=s=>s.pcg(Infinity,1024,.005),checks=[];
const test=(name,f)=>{f();checks.push(name);};
test('valid history is retained, reciprocal rho is recomputed, and one PCG is used',()=>{
 const s=install();run(s);s.x.set([-.1,-.2]);s.g.set([.8,.4]);run(s);assert.equal(s.qualifiedSecantReport.qualifiedCandidates,1);assert.equal(s.calls,2);
 const saved=s.exportQualifiedSecantCheckpoint(),r=install(make(),{restore:saved});assert.equal(r.qualifiedSecantReport.restoredPairs,1);assert.equal(r.qualifiedSecantReport.restoreRejected,null);
});
test('malformed rho, curvature, vectors and oversized histories are rejected atomically',()=>{
 for(const edit of [p=>p.records[0].rho=1,p=>p.records[0].rho=NaN,p=>p.records[0].rho=-1,p=>p.records[0].dy=[-1,0],p=>p.records[0].dy=[0,1],p=>p.records[0].dx=[Infinity,0],p=>p.records[0].dx=[3],p=>p.records=Array(13).fill(p.records[0]),p=>p.records=null]){const p=cp();edit(p);const s=install(make(),{restore:p});assert.ok(s.qualifiedSecantReport.restoreRejected);assert.equal(s.exportQualifiedSecantCheckpoint().records.length,0);}
});
test('nonempty history cannot lack paired finite endpoints or metric/contact signatures',()=>{
 for(const edit of [p=>{p.previousX=null;p.previousG=null;},p=>p.previousX=null,p=>p.previousG=[NaN,0],p=>p.previousX=[0],p=>delete p.previousG,p=>p.previousMetricSignature=null,p=>delete p.previousMetricSignature,p=>p.previousContactSignature=null]){const p=cp();edit(p);const s=install(make(),{restore:p});assert.ok(s.qualifiedSecantReport.restoreRejected);assert.equal(s.exportQualifiedSecantCheckpoint().records.length,0);}
});
test('legacy v1 checkpoints cannot migrate implicitly',()=>{const p=cp();p.schema='qualified-fem-secant-v1';assert.match(install(make(),{restore:p}).qualifiedSecantReport.restoreRejected,/legacy/);assert.throws(()=>install(make(),{restoreLegacy:true}),/implicitly/);});
test('restored contact, context and dt changes invalidate all history before use',()=>{
 for(const [kind,extra,dt]of [['contact',{contactSignature:()=>'different-face'},Infinity],['context',{contextFingerprint:()=>'other-law'},Infinity],['metric',{},.1]]){const s=install(make(),{restore:cp(),...extra});s.pcg(dt,1024,.005);assert.equal(s.qualifiedSecantReport.last.usedPairs,0);if(kind==='context')assert.match(s.qualifiedSecantReport.restoreRejected,/context/);else assert.equal(s.qualifiedSecantReport[kind+'Invalidations'],1);}
 let context='a';const s=install(make(),{contextFingerprint:()=>context});run(s);context='b';run(s);assert.equal(s.qualifiedSecantReport.contextInvalidations,1);
});
test('zero gradient returns exact finite zero without PCG',()=>{const s=make();s.g.fill(0);install(s);assert.equal(run(s),0);assert.equal(s.calls,0);assert.deepEqual(Array.from(s.direction),[0,0]);assert.equal(s.qualifiedSecantReport.stationaryReturns,1);});
test('rejected candidate reuses safe inner direction without another PCG',()=>{
 const s=install(make(),{restore:cp()});run(s);const r=s.qualifiedSecantReport;assert.equal(r.rejectedCandidates,1);assert.equal(r.innerDirectionFallbacks,1);assert.equal(r.last.candidate.quadraticModelChange,1);assert.equal(r.last.selectedMetrics.quadraticModelChange,-.5);assert.deepEqual(Array.from(s.direction),[-0,-1]);assert.equal(s.calls,1);
});
test('unsafe inner direction falls back to finite scaled Jacobi with and without history',()=>{
 for(const restore of [null,cp()]){const raw=make();raw.pcg=function(){this.calls++;this.direction.set([1,1]);return 1;};const s=install(raw,{restore});run(s);assert.equal(s.qualifiedSecantReport.jacobiFallbacks,1);assert.deepEqual(Array.from(s.direction),[-1,-1]);assert.equal(s.calls,1);assert.ok(s.qualifiedSecantReport.last.selectedMetrics.quadraticModelChange<0);}
});
test('nonfinite PCG output is rejected and finite Jacobi remains available',()=>{for(const bad of [NaN,Infinity]){const raw=make();raw.pcg=function(){this.calls++;this.direction.set([bad,0]);return 1;};const s=install(raw,{restore:cp()});run(s);assert.equal(s.qualifiedSecantReport.jacobiFallbacks,1);assert.ok(Array.from(s.direction).every(Number.isFinite));assert.equal(s.calls,1);}});
test('nonfinite positions/gradient and nonpositive/nonfinite diagonals fail closed',()=>{
 for(const [key,value]of [['g',NaN],['g',Infinity],['x',NaN],['diagonal',0],['diagonal',-1],['diagonal',NaN],['diagonal',Infinity]]){const raw=make();raw[key][0]=value;const s=install(raw);assert.throws(()=>run(s),/finite/);assert.equal(s.calls,0);}
});
test('nonfinite metric, underflow/overflow fallback and invalid dt fail closed',()=>{
 for(const kind of ['metric','underflow','overflow']){const raw=make();raw.pcg=function(){this.calls++;this.direction.fill(NaN);return 1;};if(kind==='metric')raw.multiply=(_,o)=>o.fill(NaN);else if(kind==='underflow')raw.diagonal.fill(Number.MIN_VALUE);else raw.g.fill(1e308);const s=install(raw);assert.throws(()=>run(s),/finite|fallback/);assert.ok(s.calls<=1);}
 for(const dt of [NaN,0,-1,-Infinity])assert.throws(()=>install().pcg(dt,1024,.005),/time step/);
});

test('fixed-DOF secants and endpoints are rejected even with valid rho',()=>{
 const raw=()=>({x:Float64Array.from([2,3,4,0,0,0]),target:Float64Array.from([2,3,4,0,0,0]),fixed:Uint8Array.from([1,0]),g:Float64Array.from([0,0,0,1,1,1]),direction:new Float64Array(6),diagonal:new Float64Array(6).fill(1),calls:0,multiply(v,o){o.set(v);},pcg(){this.calls++;this.direction.set(this.g.map(v=>-v));return 1;}});
 const base=()=>({...cp(),previousX:[2,3,4,0,0,0],previousG:[0,0,0,1,1,1],records:[{dx:[1,0,0,1,0,0],dy:[1,0,0,1,0,0],rho:.5}]});
 for(const edit of [()=>{},p=>{p.records=[{dx:[0,0,0,1,0,0],dy:[0,0,0,1,0,0],rho:1}];p.previousX[0]=2.1;},p=>{p.records=[];p.previousG[0]=1;}]){const p=base();edit(p);const s=install(raw(),{restore:p});assert.match(s.qualifiedSecantReport.restoreRejected,/fixed/);assert.equal(s.exportQualifiedSecantCheckpoint().records.length,0);}
 const s=raw();s.pcg=function(){this.calls++;this.direction.set([8,-9,10,-1,-1,-1]);return 1;};install(s);run(s);
 for(let i=0;i<3;i++)assert.ok(Object.is(s.direction[i],0));
 for(const fraction of [1,.5,.001]){const trial=Float64Array.from(s.x,(x,i)=>x+fraction*s.direction[i]);for(let i=0;i<3;i++)assert.ok(Object.is(trial[i],s.target[i]));}
 assert.equal(s.calls,1);
 const bad=raw();bad.x[0]+=.1;assert.throws(()=>run(install(bad)),/fixed/);
 const badGradient=raw();badGradient.g[0]=1;assert.throws(()=>run(install(badGradient)),/fixed/);
});
console.log(JSON.stringify({passed:true,testCount:checks.length,checks,nonlinearStateSolvePerformed:false,largeMetricProbePerformed:false},null,2));
