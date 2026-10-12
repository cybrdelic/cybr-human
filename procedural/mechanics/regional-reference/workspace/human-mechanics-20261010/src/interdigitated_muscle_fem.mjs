/** Multiple finite fibre families in a shared tissue element.
 * Original authored extension; preserve GPL source provenance from the base.
 * The passive volumetric matrix is evaluated exactly once. Each family's
 * active energy uses V*fraction; total fractions in any tet must not exceed 1.
 * Fractions are an authored fibre-mixture measure, not resolved histology.
 * This does not model fascicle sliding, remodelling, or a calibrated Hill law.
 */
import {VolumetricMuscleFEM} from './volumetric_muscle_fem.mjs';
export class InterdigitatedMuscleFEM extends VolumetricMuscleFEM {
 setMuscleRegions(records){
  if(!Array.isArray(records))throw Error('Muscle families must be an array');
  const occupancy=new Map(),ids=new Set(),fractions=[];
  for(const [i,r] of records.entries()){
   const id=r.id??String(i);if(ids.has(id))throw Error('Duplicate fibre-family ID');ids.add(id);
   const tt=Array.from(r.tetrahedra||[]),w=r.volumeFractions===undefined?tt.map(()=>r.volumeFraction??1):Array.from(r.volumeFractions);
   if(r.volumeFractions!==undefined&&r.volumeFraction!==undefined)throw Error('Use one volumeFraction or per-tet volumeFractions');
   if(w.length!==tt.length||!w.every(v=>Number.isFinite(v)&&v>0&&v<=1))throw Error('Fibre fractions must be positive and <=1');
   for(let j=0;j<tt.length;j++){const total=(occupancy.get(tt[j])||0)+w[j];if(total>1+1e-12)throw Error('Total fibre fraction in a tet exceeds 1');occupancy.set(tt[j],total);}
   fractions.push(w);
  }
  // Let the existing tested configurator validate every family independently.
  // Its single-family element/gradient construction stays unchanged.
  const all=[];
  for(const [i,r] of records.entries()){
   super.setMuscleRegions([{...r,id:r.id??String(i)}]);const region=this.muscleRegions[0];
   region.elements.forEach((e,j)=>{e.physicalRestVolume=e.restVolume;e.volumeFraction=fractions[i][j];e.restVolume*=e.volumeFraction;e.k*=e.volumeFraction;});
   region.restVolume=region.elements.reduce((v,e)=>v+e.restVolume,0);region.volumeFractions=fractions[i].slice();all.push(region);
  }
  if(!records.length)super.setMuscleRegions([]);
  this.muscleRegions=all;this.muscleMixtureOccupancy=occupancy;
  this.muscleFamilyFraction=new Map(all.flatMap(r=>r.elements.map(e=>[`${r.id}:${e.tetrahedron}`,e.volumeFraction])));
  this.muscleBlocks.length=0;this.muscleDiagnostics.length=0;this.muscleTetDiagnostics.length=0;this.searchBasis.length=0;
  return this;
 }
 evaluate(x,dt,blocks=true){
  const s=super.evaluate(x,dt,blocks);const unique=new Map();
  for(const t of this.muscleTetDiagnostics){t.volumeFraction=this.muscleFamilyFraction.get(`${t.region}:${t.tetrahedron}`);unique.set(t.tetrahedron,t.J);}
  let unionRestVolume=0,unionCurrentVolume=0;
  for(const [t,J] of unique){const V=this.material[3*t];unionRestVolume+=V;unionCurrentVolume+=V*J;}
  return {...s,muscleFamilyElementCount:this.muscleTetDiagnostics.length,muscleUniqueTetrahedronCount:unique.size,
   muscleUnionRestVolume:unionRestVolume,muscleUnionCurrentVolume:unionCurrentVolume,
   maximumFibreMixtureFraction:this.muscleMixtureOccupancy.size?Math.max(...this.muscleMixtureOccupancy.values()):0,
   mixtureConvention:'Active energies/rest-volume diagnostics weighted by family fraction; unique geometric union volume reported separately. Passive solid counted once.'};
 }
}
