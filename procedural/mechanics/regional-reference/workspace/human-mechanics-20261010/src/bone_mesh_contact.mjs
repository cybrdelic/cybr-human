/**
 * Original finite-penalty node-to-authored-bone contact for LayeredFEM.
 * No geometry or collision library is imported. The original solver is untouched.
 *
 * Local mesh: closed, outward-wound, non-self-intersecting triangle surface.
 * Multiple components must bound disjoint solids; nested shells/cavities are not
 * supported. Topology, winding, vertex fans and signed volumes are checked here;
 * geometric self-intersection or overlap is NOT certified by those checks.
 *
 * E = .5 * stiffness * min(signedDistance - clearance, 0)^2.
 * Outside distance is positive. The returned gradient is dE/d(world point),
 * not force. Signed-distance derivatives are exact away from surface feature
 * transitions and the interior medial axis. At a nondifferentiable tie a
 * deterministic incident branch is returned, not a claimed unique derivative.
 *
 * A median-split AABB BVH accelerates exact closest-triangle queries. Containment
 * uses outward face / edge-sum / angle-weighted vertex pseudonormals, with a
 * bounded BVH ray-parity fallback for numerically ambiguous signs. Prescribed
 * rigid transforms change queries only; the local BVH never needs refitting.
 * Finite stiffness permits penetration. This is one-way nodal contact, not
 * triangle-triangle CCD, tissue self-contact, or a reciprocal bone dynamics solve.
 *
 * Usage:
 *   const collider = new BoneMeshCollider({vertices, triangles, stiffness: 800});
 *   solver.solidMeshContacts.push({node, collider});
 *   collider.setTransform({rotation: rowMajor3x3, translation: [x,y,z]});
 */

const dot = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const sub = (a,b) => [a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const length = a => Math.hypot(a[0],a[1],a[2]);
const scale = (a,s) => [a[0]*s,a[1]*s,a[2]*s];
const unit = a => { const n=length(a); return n>0 ? scale(a,1/n) : [0,0,0]; };
const edgeKey = (a,b) => a<b ? `${a}:${b}` : `${b}:${a}`;
const IDENTITY = [1,0,0,0,1,0,0,0,1];
const rayDirections = [[1,.3713906763541037,.6947465906068658],
  [.193421768445,.932954203383,1],[.754877666247,1,.569840290998],
  [1,-.618033988750,.414213562373],[-.317837245196,.735758882343,1]].map(unit);

function triples(values,name) {
  if (!values || typeof values.length !== 'number') throw Error(`${name} must contain triples`);
  const flat = Array.isArray(values[0]) || ArrayBuffer.isView(values[0])
    ? Array.from(values).flatMap(v => {
      if (v.length!==3) throw Error(`${name} must contain triples`);
      return Array.from(v);
    }) : Array.from(values);
  if (!flat.length || flat.length%3) throw Error(`${name} must contain triples`);
  return flat;
}
function point3(p,name='point') {
  if (!p || p.length!==3 || !Array.from(p).every(Number.isFinite)) throw Error(`Invalid ${name}`);
  return Array.from(p);
}
function boundsDistanceSquared(p,b) {
  let d2=0;
  for(let d=0;d<3;d++) {const v=p[d]<b[d]?b[d]-p[d]:p[d]>b[d+3]?p[d]-b[d+3]:0;d2+=v*v;}
  return d2;
}
function makeBounds() { return [Infinity,Infinity,Infinity,-Infinity,-Infinity,-Infinity]; }
function includeBounds(bounds,other) {
  for(let d=0;d<3;d++){bounds[d]=Math.min(bounds[d],other[d]);bounds[d+3]=Math.max(bounds[d+3],other[d+3]);}
}

// Closest point by the seven vertex/edge/face Voronoi regions. Barycentrics are
// retained for reproducible inspection and for selecting the matching normal.
function nearestTriangle(p,t) {
  const ap=sub(p,t.a),d1=dot(t.ab,ap),d2=dot(t.ac,ap);
  let w,feature;
  if(d1<=0&&d2<=0){w=[1,0,0];feature=0;}
  else {
    const bp=sub(p,t.b),d3=dot(t.ab,bp),d4=dot(t.ac,bp);
    if(d3>=0&&d4<=d3){w=[0,1,0];feature=1;}
    else {
      const vc=d1*d4-d3*d2;
      if(vc<=0&&d1>=0&&d3<=0){const v=d1/(d1-d3);w=[1-v,v,0];feature=3;}
      else {
        const cp=sub(p,t.c),d5=dot(t.ab,cp),d6=dot(t.ac,cp);
        if(d6>=0&&d5<=d6){w=[0,0,1];feature=2;}
        else {
          const vb=d5*d2-d1*d6;
          if(vb<=0&&d2>=0&&d6<=0){const v=d2/(d2-d6);w=[1-v,0,v];feature=5;}
          else {
            const va=d3*d6-d5*d4;
            if(va<=0&&(d4-d3)>=0&&(d5-d6)>=0){const v=(d4-d3)/((d4-d3)+(d5-d6));w=[0,1-v,v];feature=4;}
            else {const inv=1/(va+vb+vc),v=vb*inv,u=vc*inv;w=[1-v-u,v,u];feature=6;}
          }
        }
      }
    }
  }
  const q=[0,1,2].map(d=>t.a[d]+w[1]*t.ab[d]+w[2]*t.ac[d]);
  const delta=sub(p,q);
  return {point:q,barycentric:w,feature,distanceSquared:dot(delta,delta)};
}

function buildBVH(triangles,ids,leafSize,depth=0) {
  const bounds=makeBounds(),centroidBounds=makeBounds();
  for(const i of ids) {
    const t=triangles[i];includeBounds(bounds,t.bounds);
    includeBounds(centroidBounds,[...t.centroid,...t.centroid]);
  }
  if(ids.length<=leafSize) return {bounds,ids,depth};
  let axis=0;
  for(let d=1;d<3;d++)if(centroidBounds[d+3]-centroidBounds[d]>centroidBounds[axis+3]-centroidBounds[axis])axis=d;
  ids.sort((a,b)=>triangles[a].centroid[axis]-triangles[b].centroid[axis]||a-b);
  const mid=ids.length>>1;
  return {bounds,left:buildBVH(triangles,ids.slice(0,mid),leafSize,depth+1),
    right:buildBVH(triangles,ids.slice(mid),leafSize,depth+1),depth};
}

function rayBox(p,dir,b) {
  let near=0,far=Infinity;
  for(let d=0;d<3;d++) {
    let a=(b[d]-p[d])/dir[d],c=(b[d+3]-p[d])/dir[d];
    if(a>c)[a,c]=[c,a];near=Math.max(near,a);far=Math.min(far,c);
    if(far<near)return false;
  }
  return far>=0;
}

export class BoneMeshCollider {
  #triangles; #vertices; #root; #rotation; #translation; #surfaceTolerance;
  constructor({vertices,triangles,stiffness,clearance=0,rotation=IDENTITY,
    translation=[0,0,0],leafSize=8}={}) {
    if(!(Number.isFinite(stiffness)&&stiffness>0))throw Error('stiffness must be finite and positive');
    if(!(Number.isFinite(clearance)&&clearance>=0))throw Error('clearance must be finite and nonnegative');
    if(!Number.isInteger(leafSize)||leafSize<1||leafSize>64)throw Error('leafSize must be an integer in [1,64]');
    const v=triples(vertices,'vertices'),f=triples(triangles,'triangles');
    if(!v.every(Number.isFinite))throw Error('vertices must be finite');
    const vertexCount=v.length/3,triangleCount=f.length/3;
    if(vertexCount<4||triangleCount<4)throw Error('A closed mesh requires at least four vertices and triangles');
    if(!f.every(i=>Number.isInteger(i)&&i>=0&&i<vertexCount))throw Error('Invalid triangle index');
    this.#vertices=Array.from({length:vertexCount},(_,i)=>v.slice(i*3,i*3+3));
    const bounds=makeBounds();for(const p of this.#vertices)includeBounds(bounds,[...p,...p]);
    const extent=length(sub(bounds.slice(3),bounds.slice(0,3)));
    if(!(Number.isFinite(extent)&&extent>0))throw Error('Invalid mesh extent');
    this.#surfaceTolerance=extent*2e-13;
    const incident=Array.from({length:vertexCount},()=>[]),edges=new Map(),duplicates=new Set();
    const vn=Array.from({length:vertexCount},()=>[0,0,0]);
    this.#triangles=Array.from({length:triangleCount},(_,i)=>{
      const ids=f.slice(i*3,i*3+3),key=ids.slice().sort((a,b)=>a-b).join(':');
      if(new Set(ids).size!==3||duplicates.has(key))throw Error(`Degenerate or duplicate triangle ${i}`);
      duplicates.add(key);
      const [a,b,c]=ids.map(j=>this.#vertices[j]),ab=sub(b,a),ac=sub(c,a),n=cross(ab,ac),area2=length(n);
      if(!(area2>1e-14*extent*extent))throw Error(`Degenerate or numerically ill-conditioned triangle ${i}`);
      const normal=scale(n,1/area2),tb=makeBounds();for(const p of [a,b,c])includeBounds(tb,[...p,...p]);
      for(let j=0;j<3;j++){
        const ia=ids[j],ib=ids[(j+1)%3],k=edgeKey(ia,ib),record=edges.get(k)||{ids:[Math.min(ia,ib),Math.max(ia,ib)],faces:[],orientation:0};
        record.faces.push(i);record.orientation+=ia<ib?1:-1;edges.set(k,record);incident[ia].push(i);
        const pa=this.#vertices[ia],u=sub(this.#vertices[ids[(j+1)%3]],pa),w=sub(this.#vertices[ids[(j+2)%3]],pa);
        const angle=Math.atan2(length(cross(u,w)),dot(u,w));
        for(let d=0;d<3;d++)vn[ia][d]+=angle*normal[d];
      }
      return {id:i,ids,a,b,c,ab,ac,normal,bounds:tb,centroid:[0,1,2].map(d=>(a[d]+b[d]+c[d])/3),neighbors:[]};
    });
    for(const e of edges.values()){
      if(e.faces.length!==2)throw Error(`Mesh is not closed edge-manifold at edge ${e.ids.join(':')}: ${e.faces.length} faces`);
      if(e.orientation!==0)throw Error(`Inconsistent winding at edge ${e.ids.join(':')}`);
      const [a,b]=e.faces;this.#triangles[a].neighbors.push(b);this.#triangles[b].neighbors.push(a);
      e.normal=unit(this.#triangles[a].normal.map((v,d)=>v+this.#triangles[b].normal[d]));
    }
    // Edge-manifold alone misses two otherwise-closed fans sharing one vertex.
    for(let i=0;i<vertexCount;i++){
      if(!incident[i].length)throw Error(`Unused vertex ${i}`);
      const remaining=new Set(incident[i]),stack=[incident[i][0]];remaining.delete(stack[0]);
      while(stack.length){const t=this.#triangles[stack.pop()];for(const j of t.neighbors){
        if(remaining.has(j)&&this.#triangles[j].ids.includes(i)){remaining.delete(j);stack.push(j);}
      }}
      if(remaining.size)throw Error(`Non-manifold vertex fan at vertex ${i}`);
    }
    const seen=new Uint8Array(triangleCount),componentVolumes=[];
    for(let seed=0;seed<triangleCount;seed++)if(!seen[seed]){
      const anchor=this.#triangles[seed].a,stack=[seed],cb=makeBounds();let volume=0;seen[seed]=1;
      while(stack.length){const t=this.#triangles[stack.pop()];includeBounds(cb,t.bounds);
        volume+=dot(sub(t.a,anchor),cross(sub(t.b,anchor),sub(t.c,anchor)))/6;
        for(const i of t.neighbors)if(!seen[i]){seen[i]=1;stack.push(i);}
      }
      const size=length(sub(cb.slice(3),cb.slice(0,3)));
      if(!(volume>1e-14*size**3))throw Error('Mesh component is inward-wound or has zero/ill-conditioned signed volume');
      componentVolumes.push(volume);
    }
    for(const t of this.#triangles){
      t.featureNormals=[...t.ids.map(i=>unit(vn[i])),
        ...[0,1,2].map(j=>edges.get(edgeKey(t.ids[j],t.ids[(j+1)%3])).normal),t.normal];
      delete t.neighbors;
    }
    this.#root=buildBVH(this.#triangles,Array.from({length:triangleCount},(_,i)=>i),leafSize);
    let nodes=0,maxDepth=0;const todo=[this.#root];
    while(todo.length){const n=todo.pop();nodes++;maxDepth=Math.max(maxDepth,n.depth);if(n.left)todo.push(n.left,n.right);}
    Object.defineProperties(this,{stiffness:{value:stiffness,enumerable:true},clearance:{value:clearance,enumerable:true}});
    this.validation=Object.freeze({vertexCount,triangleCount,edgeCount:edges.size,components:componentVolumes.length,
      componentSignedVolumes:Object.freeze(componentVolumes),closedEdgeManifold:true,consistentWinding:true,
      vertexManifold:true,outwardSignedVolume:true,selfIntersectionsChecked:false,nestedComponentsSupported:false,
      bvhNodes:nodes,bvhDepth:maxDepth,leafSize});
    this.setTransform({rotation,translation});
  }

  /** R is row-major local-to-world, orthonormal and orientation-preserving. */
  setTransform({rotation=IDENTITY,translation=[0,0,0]}={}) {
    const r=Array.from(rotation||[]),t=point3(translation,'translation');
    if(r.length!==9||!r.every(Number.isFinite))throw Error('Invalid rotation');
    for(let a=0;a<3;a++)for(let b=0;b<3;b++){
      let v=0;for(let d=0;d<3;d++)v+=r[d*3+a]*r[d*3+b];
      if(Math.abs(v-(a===b?1:0))>1e-10)throw Error('Transform must be rigid: rotation must be orthonormal');
    }
    const det=r[0]*(r[4]*r[8]-r[5]*r[7])-r[1]*(r[3]*r[8]-r[5]*r[6])+r[2]*(r[3]*r[7]-r[4]*r[6]);
    if(Math.abs(det-1)>1e-10)throw Error('Rotation must have determinant +1');
    this.#rotation=r;this.#translation=t;return this;
  }
  get transform(){return {rotation:this.#rotation.slice(),translation:this.#translation.slice()};}
  #toLocal(p){const q=sub(point3(p),this.#translation),r=this.#rotation;return [0,1,2].map(d=>r[d]*q[0]+r[3+d]*q[1]+r[6+d]*q[2]);}
  #toWorldVector(p){const r=this.#rotation;return [0,1,2].map(d=>r[3*d]*p[0]+r[3*d+1]*p[1]+r[3*d+2]*p[2]);}
  #toWorldPoint(p){return this.#toWorldVector(p).map((v,d)=>v+this.#translation[d]);}

  #nearest(p) {
    let best=null,bestD2=Infinity,triangleTests=0,nodeVisits=0;
    const stack=[this.#root];
    while(stack.length){
      const node=stack.pop();nodeVisits++;
      if(boundsDistanceSquared(p,node.bounds)>bestD2)continue;
      if(node.ids){for(const id of node.ids){triangleTests++;const q=nearestTriangle(p,this.#triangles[id]);
        if(q.distanceSquared<bestD2||(q.distanceSquared===bestD2&&(!best||id<best.triangle))){bestD2=q.distanceSquared;best={...q,triangle:id};}
      }}else{
        const a=boundsDistanceSquared(p,node.left.bounds),b=boundsDistanceSquared(p,node.right.bounds);
        if(a<=b){if(b<=bestD2)stack.push(node.right);if(a<=bestD2)stack.push(node.left);}
        else{if(a<=bestD2)stack.push(node.left);if(b<=bestD2)stack.push(node.right);}
      }
    }
    return {...best,queryStats:{triangleTests,nodeVisits,parityRays:0}};
  }

  // Only used when the nearest-feature pseudonormal cannot determine a sign.
  // A ray touching an edge, vertex, or near-parallel triangle is discarded;
  // at most five deterministic non-axis directions are attempted.
  #parityInside(p,stats) {
    for(const dir of rayDirections){
      stats.parityRays++;let hits=0,ambiguous=false;const stack=[this.#root];
      while(stack.length&&!ambiguous){const node=stack.pop();if(!rayBox(p,dir,node.bounds))continue;
        if(!node.ids){stack.push(node.left,node.right);continue;}
        for(const id of node.ids){
          const t=this.#triangles[id],h=cross(dir,t.ac),det=dot(t.ab,h),denom=length(t.ab)*length(t.ac);
          if(Math.abs(det)<denom*1e-13){
            if(Math.abs(dot(sub(p,t.a),t.normal))<this.#surfaceTolerance){ambiguous=true;break;}
            continue;
          }
          const inv=1/det,s=sub(p,t.a),u=dot(s,h)*inv,q=cross(s,t.ab),v=dot(dir,q)*inv,travel=dot(t.ac,q)*inv;
          if(travel<=this.#surfaceTolerance||u< -1e-12||v< -1e-12||u+v>1+1e-12)continue;
          if(Math.min(u,v,1-u-v)<1e-12){ambiguous=true;break;}
          hits++;
        }
      }
      if(!ambiguous)return Boolean(hits%2);
    }
    throw Error('Numerically ambiguous mesh containment; inspect mesh quality or query location');
  }

  /** Exact nearest surface sample, in world coordinates; statistics are per query. */
  closestPoint(point) {
    const p=this.#toLocal(point),q=this.#nearest(p),distance=Math.sqrt(q.distanceSquared);
    const t=this.#triangles[q.triangle],pseudo=t.featureNormals[q.feature],delta=sub(p,q.point),score=dot(delta,pseudo);
    let inside=false,localNormal,signedDistance;
    if(distance===0){signedDistance=0;localNormal=length(pseudo)>0?pseudo:t.normal;}
    else{
      inside=Math.abs(score)>1e-12*distance
        ? score<0 : this.#parityInside(p,q.queryStats);
      signedDistance=inside?-distance:distance;
      localNormal=scale(delta,(inside?-1:1)/distance);
    }
    return {point:this.#toWorldPoint(q.point),distance,signedDistance,inside,
      normal:this.#toWorldVector(localNormal),triangle:q.triangle,barycentric:q.barycentric,
      feature:['vertex-a','vertex-b','vertex-c','edge-ab','edge-bc','edge-ca','face'][q.feature],queryStats:q.queryStats};
  }
  signedDistance(point){return this.closestPoint(point).signedDistance;}
  evaluate(point) {
    const q=this.closestPoint(point),gap=q.signedDistance-this.clearance,violation=Math.max(-gap,0);
    const gradient=violation>0?scale(q.normal,-this.stiffness*violation):[0,0,0];
    return {energy:.5*this.stiffness*violation*violation,gradient,normal:q.normal,violation,
      signedDistance:q.signedDistance,closestPoint:q.point,triangle:q.triangle,inside:q.inside,queryStats:q.queryStats};
  }
}

export default BoneMeshCollider;
