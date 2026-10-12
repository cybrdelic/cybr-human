/**
 * Original feasibility-only static-bone BVH for linear FEM boundary motion.
 * No force model, energy, quadrature, collision response or rest repair here.
 * Static geometry is privately copied in WORLD coordinates. Rebuild the BVH
 * for any bone change; a moving/rotating bone trajectory is not supported.
 *
 * Swept AABBs contain all six represented endpoint triangle vertices. Padding
 * by the required minimum separation is rounded OUTWARD. Only disjoint padded
 * boxes are culled, without sidedness/normal/region exclusions. Every surviving
 * pair is sent to the separate whole-triangle certificate. All work limits are
 * fail-closed. The broad phase never calls an incomplete candidate set complete.
 * No certification of solid containment, self-contact, or valid tetrahedra.
 */
import {nextDown,nextUp,certifyTriangleStep} from './complete_boundary_contact.mjs';

const MAX_BONE_TRIANGLES=100000,MAX_VERTICES=300000,MAX_TISSUE_TRIANGLES=50000;
const DEFAULT_BUDGETS=Object.freeze({maxBoxTests:2000000,maxCandidatePairs:50000,maxTotalIntervals:100000,maxDepth:24,maxIntervalsPerPair:2048});
const point=(v,i)=>[v[3*i],v[3*i+1],v[3*i+2]];
const emptyBox=()=>[Infinity,Infinity,Infinity,-Infinity,-Infinity,-Infinity];
function include(box,other){for(let d=0;d<3;d++){box[d]=Math.min(box[d],other[d]);box[d+3]=Math.max(box[d+3],other[d+3]);}}
function vertexArray(values,name,copy=false) {
  if(!values||!Number.isInteger(values.length)||values.length<1)throw Error(`${name} requires vertices`);
  let v=values;
  if(typeof values[0]!=='number') {
    if(values.length>MAX_VERTICES)throw Error(`${name} vertex budget exceeded`);
    v=new Float64Array(values.length*3);
    for(let i=0;i<values.length;i++) {
      if(!values[i]||values[i].length!==3)throw Error(`${name} requires triples`);
      for(let d=0;d<3;d++)v[3*i+d]=values[i][d];
    }
  }else {
    // Check before copying so an oversized flat input cannot allocate first.
    if(values.length%3||values.length/3>MAX_VERTICES)throw Error(`${name} vertex budget or layout invalid`);
    if(copy)v=Float64Array.from(values);
  }
  if(v.length%3||v.length/3>MAX_VERTICES)throw Error(`${name} vertex budget or layout invalid`);
  for(const x of v)if(!Number.isFinite(x))throw Error(`${name} must be finite`);
  return v;
}
function faceArray(values,count,name,maxCount) {
  if(!values||!Number.isInteger(values.length)||values.length<1)throw Error(`${name} requires triangles`);
  let f=values;
  if(typeof values[0]==='number') {
    if(values.length%3||values.length/3>maxCount)throw Error(`${name} triangle budget or layout invalid`);
    f=Array.from({length:values.length/3},(_,i)=>[values[3*i],values[3*i+1],values[3*i+2]]);
  }
  if(f.length>maxCount)throw Error(`${name} triangle budget exceeded`);
  for(const a of f)if(!a||a.length!==3||new Set(a).size!==3||!Array.from(a).every(i=>Number.isInteger(i)&&i>=0&&i<count))throw Error(`${name} invalid triangle`);
  return f;
}
function faceBox(vertices,face) {
  const b=emptyBox();for(const i of face)for(let d=0;d<3;d++){b[d]=Math.min(b[d],vertices[3*i+d]);b[d+3]=Math.max(b[d+3],vertices[3*i+d]);}return b;
}
function sweptBox(start,end,face,minimumDistance) {
  const b=faceBox(start,face);include(b,faceBox(end,face));
  for(let d=0;d<3;d++){b[d]=nextDown(b[d]-minimumDistance);b[d+3]=nextUp(b[d+3]+minimumDistance);}return b;
}
const overlap=(a,b)=>[0,1,2].every(d=>a[d]<=b[d+3]&&b[d]<=a[d+3]);
function bvh(records,ids,leafSize) {
  const bounds=emptyBox();for(const i of ids)include(bounds,records[i].bounds);
  if(ids.length<=leafSize)return {bounds,ids};
  let axis=0;for(let d=1;d<3;d++)if(bounds[d+3]-bounds[d]>bounds[axis+3]-bounds[axis])axis=d;
  // Center ordering is heuristic only; its rounding cannot change node bounds.
  const center=i=>records[i].bounds[axis]/2+records[i].bounds[axis+3]/2;
  ids.sort((a,b)=>center(a)-center(b)||a-b);const mid=ids.length>>1;
  return {bounds,left:bvh(records,ids.slice(0,mid),leafSize),right:bvh(records,ids.slice(mid),leafSize)};
}
function options(input={}) {
  const {minimumDistance=0,...rest}=input,out={...DEFAULT_BUDGETS,...rest,minimumDistance};
  if(!(Number.isFinite(minimumDistance)&&minimumDistance>=0))throw Error('Invalid minimumDistance');
  for(const name of ['maxBoxTests','maxCandidatePairs','maxTotalIntervals','maxIntervalsPerPair'])
    if(!Number.isInteger(out[name])||out[name]<1||out[name]>5000000)throw Error(`Invalid ${name}`);
  if(out.maxIntervalsPerPair>1000000)throw Error('Invalid maxIntervalsPerPair');
  if(!Number.isInteger(out.maxDepth)||out.maxDepth<0||out.maxDepth>48)throw Error('Invalid maxDepth');
  return out;
}

export class StaticBoneSweptBVH {
  #bones;#records;#root;
  constructor({bones,leafSize=8}={}) {
    if(!Array.isArray(bones)||!bones.length||bones.length>16)throw Error('Supply 1 to 16 static bone meshes');
    if(!Number.isInteger(leafSize)||leafSize<1||leafSize>32)throw Error('Invalid leafSize');
    const ids=new Set();let totalTriangles=0,totalVertices=0;
    this.#bones=bones.map((b,index)=>{
      const id=b.id??String(index);if(typeof id!=='string'||!id.length||ids.has(id))throw Error('Invalid or duplicate bone ID');ids.add(id);
      const vertices=vertexArray(b.vertices,`bone ${id}`,true),faces=faceArray(b.triangles,vertices.length/3,`bone ${id}`,MAX_BONE_TRIANGLES).map(f=>Array.from(f));
      totalTriangles+=faces.length;totalVertices+=vertices.length/3;
      if(totalTriangles>MAX_BONE_TRIANGLES||totalVertices>MAX_VERTICES)throw Error('Combined static geometry budget exceeded');
      return {id,vertices,faces};
    });
    this.#records=[];
    for(let bone=0;bone<this.#bones.length;bone++) {
      const b=this.#bones[bone];for(let face=0;face<b.faces.length;face++)this.#records.push({bone,face,bounds:faceBox(b.vertices,b.faces[face])});
    }
    this.#root=bvh(this.#records,Array.from({length:totalTriangles},(_,i)=>i),leafSize);
    this.summary=Object.freeze({boneCount:this.#bones.length,boneIds:Object.freeze(this.#bones.map(b=>b.id)),triangleCount:totalTriangles,vertexCount:totalVertices,leafSize,staticWorldCoordinates:true});
  }
  #traverse(start,end,faces,input,visit) {
    const o=options(input),a=vertexArray(start,'start'),z=vertexArray(end,'end');
    if(a.length!==z.length)throw Error('Start/end size mismatch');
    const fs=faceArray(faces,a.length/3,'tissue boundary',MAX_TISSUE_TRIANGLES),stats={tissueTriangles:fs.length,boneTriangles:this.#records.length,boxTests:0,candidatePairs:0,intervals:0};
    const stop=(reason,extra={})=>({complete:false,reason,...stats,...extra});
    for(let tissueFace=0;tissueFace<fs.length;tissueFace++) {
      const f=fs[tissueFace],bounds=sweptBox(a,z,f,o.minimumDistance),stack=[this.#root];
      while(stack.length) {
        const node=stack.pop();if(stats.boxTests>=o.maxBoxTests)return stop('box-test-budget');stats.boxTests++;
        if(!overlap(bounds,node.bounds))continue;
        if(!node.ids){stack.push(node.right,node.left);continue;}
        for(const index of node.ids) {
          if(stats.boxTests>=o.maxBoxTests)return stop('box-test-budget');stats.boxTests++;
          const record=this.#records[index];if(!overlap(bounds,record.bounds))continue;
          if(stats.candidatePairs>=o.maxCandidatePairs)return stop('candidate-pair-budget');stats.candidatePairs++;
          const bone=this.#bones[record.bone],pair={tissueFace,boneIndex:record.bone,boneId:bone.id,boneFace:record.face},
            result=visit({pair,startTriangle:f.map(i=>point(a,i)),endTriangle:f.map(i=>point(z,i)),boneTriangle:bone.faces[record.face].map(i=>point(bone.vertices,i)),options:o,stats});
          if(result)return stop(result.reason,{pair,...result});
        }
      }
    }
    return {complete:true,reason:'complete-swept-bvh-traversal',...stats};
  }
  /** complete:false means the list is unusable as a complete candidate set. */
  findCandidates(start,end,tissueTriangles,input={}) {
    const pairs=[],r=this.#traverse(start,end,tissueTriangles,input,({pair})=>{pairs.push(pair);return null;});
    return {...r,pairs};
  }
  /** Streaming certification: no O(tissueFaces * boneFaces) pair allocation. */
  certifyStep(start,end,tissueTriangles,input={}) {
    const r=this.#traverse(start,end,tissueTriangles,input,({startTriangle,endTriangle,boneTriangle,options:o,stats})=>{
      const remaining=o.maxTotalIntervals-stats.intervals;
      if(remaining<1)return {reason:'total-interval-budget'};
      const q=certifyTriangleStep(startTriangle,endTriangle,boneTriangle,{minimumDistance:o.minimumDistance,maxDepth:o.maxDepth,maxIntervals:Math.min(o.maxIntervalsPerPair,remaining)});
      stats.intervals+=q.intervals;
      return q.certified?null:{reason:'uncertified-triangle-segment',narrowPhase:q};
    });
    return {...r,certified:r.complete,forceModelComplete:false,solidContainmentChecked:false};
  }
}

/** Extract every face with exactly one incident tetrahedron; no region mask. */
export function extractCompleteFEMBoundary(nodes,vertexCount) {
  if(!nodes||nodes.length%4||!Number.isInteger(vertexCount)||vertexCount<4)throw Error('Invalid tetrahedral topology');
  const faces=new Map(),tets=new Set();
  for(let i=0;i<nodes.length;i+=4) {
    const ids=Array.from(nodes.slice(i,i+4));
    if(new Set(ids).size!==4||!ids.every(n=>Number.isInteger(n)&&n>=0&&n<vertexCount))throw Error('Invalid tetrahedron');
    const key=ids.toSorted((a,b)=>a-b).join(':');if(tets.has(key))throw Error('Duplicate tetrahedron');tets.add(key);
    for(let omit=0;omit<4;omit++) {
      const face=ids.filter((_,j)=>j!==omit),k=face.toSorted((a,b)=>a-b).join(':'),record=faces.get(k);
      if(record){if(++record.count>2)throw Error('Nonmanifold tetrahedral face');}else faces.set(k,{count:1,face});
    }
  }
  const boundary=Array.from(faces.values()).filter(f=>f.count===1).map(f=>f.face);
  if(!boundary.length||boundary.length>MAX_TISSUE_TRIANGLES)throw Error('Empty or oversized tetrahedral boundary');
  return boundary;
}
