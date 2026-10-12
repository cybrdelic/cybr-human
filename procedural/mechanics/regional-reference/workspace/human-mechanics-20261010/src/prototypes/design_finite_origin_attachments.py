#!/usr/bin/env python3
"""Original tiny rest-only bonded-origin feasibility prototype.
Never edits the source mesh, bone, main domain or solver. No solve/render.
Run under cybr-heavy.lock with timeout 120s, OPENBLAS_NUM_THREADS=1.
Writes only additive nodes/tets and explicit face-contact declarations.
The numerical audit is Float64, NOT interval-certified motion admission.
"""
from pathlib import Path
import hashlib,itertools,json,resource,sys,time,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'src/prototypes'))
from mandibular_collision_audit import intersect_sat
from audit_connected_midface_boundary import shared_intersection,clip_coplanar
TOL=1e-9
START=time.monotonic()
def guard():
 if time.monotonic()-START>105:raise RuntimeError('105-second work budget')
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>300*1024:raise RuntimeError('300 MiB RSS limit')
 if shutil.disk_usage(ROOT).free<120*1024**2:raise RuntimeError('120 MiB disk reserve')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def boundary(t):
 f=t[:,np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])].reshape(-1,3)
 _,ix,c=np.unique(np.sort(f,axis=1),axis=0,return_index=True,return_counts=True)
 assert c.max()<=2
 return f[ix[c==1]],np.repeat(np.arange(len(t)),4)[ix[c==1]]
def determinants(x,t):
 dm=x[t[:,1:]]-x[t[:,0,None]]
 return np.einsum('ij,ij->i',dm[:,0],np.cross(dm[:,1],dm[:,2]))
def prism(root,tip):
 # Three conforming tetrahedra per slab, with the same side diagonals.
 frac=np.array([0.,.2,.5,1.]); x=np.array([(1-f)*root+f*tip for f in frac]).reshape(-1,3)
 tet=[]
 for k in range(3):
  a,b,c=3*k+np.arange(3);A,B,C=3*(k+1)+np.arange(3)
  tet.extend([[a,b,c,A],[b,c,A,B],[c,B,C,A]])
 tet=np.array(tet);dt=determinants(x,tet)
 if np.all(dt<0):tet[:,[1,2]]=tet[:,[2,1]];dt=-dt
 return x,tet,dt
EDGES=np.array(list(itertools.combinations(range(4),2)))
FACES=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
def strict_tet_overlap(a,bb):
 # Complete convex tetrahedral SAT; face axes plus all 36 edge crosses.
 if not len(bb):return np.zeros(0,bool)
 aa=a[FACES];bfaces=bb[:,FACES];ae=a[EDGES[:,1]]-a[EDGES[:,0]];be=bb[:,EDGES[:,1]]-bb[:,EDGES[:,0]]
 an=np.cross(aa[:,1]-aa[:,0],aa[:,2]-aa[:,0]);bn=np.cross(bfaces[:,:,1]-bfaces[:,:,0],bfaces[:,:,2]-bfaces[:,:,0])
 axes=[np.broadcast_to(z,(len(bb),3)) for z in an]+[bn[:,j] for j in range(4)]
 axes.extend(np.cross(e,be[:,j]) for e in ae for j in range(6))
 hit=np.ones(len(bb),bool)
 for axis in axes:
  norm=np.linalg.norm(axis,axis=1);valid=norm>1e-13;axis=np.divide(axis,norm[:,None],out=np.zeros_like(axis),where=valid[:,None])
  pa=axis@a.T;pb=np.einsum('ni,nji->nj',axis,bb)
  overlap=np.minimum(pa.max(1),pb.max(1))-np.maximum(pa.min(1),pb.min(1))
  hit&=~(valid&(overlap<=TOL))
 return hit
RAYS=np.array([[1,.3713906763541037,.6947465906068658],[.193421768445,.932954203383,1],[.754877666247,1,.569840290998]])
def ray_parities(p,tri):
 e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0];out=[]
 for dr in RAYS:
  h=np.cross(dr,e2);det=np.einsum('ij,ij->i',e1,h);ok=abs(det)>1e-13
  inv=np.divide(1.,det,out=np.zeros_like(det),where=ok);s=p-tri[:,0];u=inv*np.einsum('ij,ij->i',s,h);q=np.cross(s,e1);vv=inv*(q@dr);tt=inv*np.einsum('ij,ij->i',e2,q)
  yes=ok&(u>=0)&(vv>=0)&(u+vv<=1)&(tt>TOL)
  hits=np.sort(tt[yes]);count=int(len(hits) and 1+np.sum(np.diff(hits)>1e-8))
  out.append(count%2)
 return out

def evaluate(x,t,tipids,boneid,btri,basev,baset,basef):
 guard();f,_=boundary(t);surf=x[f];bmin=btri.min(1);bmax=btri.max(1);allowed=[];unexpected=[];pairs=0
 for fi,q in enumerate(surf):
  ids=np.flatnonzero(np.all(bmax>=q.min(0)-TOL,1)&np.all(bmin<=q.max(0)+TOL,1));pairs+=len(ids)
  for bi in ids[intersect_sat(q,btri[ids])]:
   roots=f[fi][f[fi]<3]
   if int(bi)!=boneid or not len(roots):unexpected.append([int(fi),int(bi),'undeclared']);continue
   if len(roots)==3:
    # Every root vertex has fixed positive barycentrics in this actual bone face.
    status='bonded-face'
   else:
    status,_=shared_intersection(q,btri[bi],x[roots])
   if status in ('bonded-face','expected_shared_feature'):allowed.append([int(fi),int(bi),status])
   else:unexpected.append([int(fi),int(bi),status])
 if unexpected:return None
 # Complete convex-volume pair check against existing tissue, including neighbors.
 original=basev[baset];mn=original.min(1);mx=original.max(1);tetpairs=0
 for k,q in enumerate(x[t]):
  ids=np.flatnonzero(np.all(mx>=q.min(0)-TOL,1)&np.all(mn<=q.max(0)+TOL,1));tetpairs+=len(ids)
  bad=ids[strict_tet_overlap(q,original[ids])]
  if len(bad):return None
 # Explicit boundary intersections may only equal the existing shared tip feature.
 bsurf=basev[basef];mn=bsurf.min(1);mx=bsurf.max(1);basepairs=0;expected=0
 for k,q in enumerate(surf):
  ids=np.flatnonzero(np.all(mx>=q.min(0)-TOL,1)&np.all(mn<=q.max(0)+TOL,1));basepairs+=len(ids)
  for j in ids[intersect_sat(q,bsurf[ids])]:
   shared=np.intersect1d(tipids[np.maximum(f[k]-9,0)][f[k]>=9],basef[j]) if np.any(f[k]>=9) else np.array([],int)
   if len(shared)==3 and set(f[k])=={9,10,11}:expected+=1;continue
   if not len(shared):return None
   status,_=shared_intersection(q,bsurf[j],basev[shared])
   if status!='expected_shared_feature':return None
   expected+=1
 # New finite volume must not self overlap.
 for k,q in enumerate(x[t]):
  if strict_tet_overlap(q,x[t[k+1:]]).any():return None
 parities=[ray_parities(p,btri) for p in x[3:]]
 if np.any(parities):return None
 bn=np.cross(btri[boneid,1]-btri[boneid,0],btri[boneid,2]-btri[boneid,0]);bn/=np.linalg.norm(bn)
 if np.min((x[3:]-x[0])@bn)<=TOL:return None
 return dict(bone_candidate_pairs=pairs,declared_bone_contacts=allowed,unexpected_bone_contacts=[],base_tet_candidate_pairs=tetpairs,base_tet_positive_volume_overlaps=0,base_boundary_candidate_pairs=basepairs,expected_tip_shared_feature_contacts=expected,unexpected_base_boundary_contacts=0,new_tet_positive_volume_self_overlaps=0,nonroot_bone_ray_parities=parities,minimum_nonroot_root_plane_height=float(np.min((x[3:]-x[0])@bn)))

def main():
 # Bounded numerical fixtures: a translated straight prism is positive,
 # touching tets have no positive-volume overlap, and a duplicate does.
 seed=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
 fx,ft,fd=prism(seed,seed+[0,0,1])
 assert np.all(fd>0)
 assert not strict_tet_overlap(fx[ft[0]],fx[ft[1:]]).any()
 assert strict_tet_overlap(fx[ft[0]],fx[ft[:1]])[0]
 guard();out=ROOT/'output/finite-origin-attachment-feasibility-v1';out.mkdir(exist_ok=True)
 mesh=ROOT/'output/bilateral-midface-bone-aware-v2/rest_geometry.npz';bone=ROOT/'output/authored-bone-support-v1/bone_support.npz';domain=ROOT/'output/bilateral-mechanics-v1/domain.json'
 a=np.load(mesh);v=a['vertices'];t=a['tetrahedra'];f,owner=boundary(t);b=np.load(bone);bt=b['vertices'][b['triangles']];d=json.loads(domain.read_text());land=json.loads((bone.parent/'landmarks.json').read_text())['zygomaticus_major_origin_patch_center'];patch=np.array(land['attachment_patch_triangles']);inner=a['midface_column_nodes'][:,-1];deep=np.isin(f,inner).all(1);reports=[];arrays={};trials=0
 for si,side in enumerate([1,-1]):
  s=np.array([side,1,1]);bv=bt*s
  if side<0:bv=bv[:,[0,2,1]]
  muscle=d['muscles'][si];candidate=np.flatnonzero(deep & np.isin(owner,muscle['tetrahedra']));target=np.array(land['point'])*s;candidate=candidate[np.argsort(np.linalg.norm(v[f[candidate]].mean(1)-target,axis=1))[:8]];found=None
  for fi in candidate:
   tipids=f[fi];tip=v[tipids];rank=np.argsort(np.linalg.norm(bv[patch].mean(1)-tip.mean(0),axis=1));roots=patch[rank[:40]]
   # Prefer finite-area faces; skinny marching-tet slivers are deliberately rejected.
   ar=np.linalg.norm(np.cross(bv[roots,1]-bv[roots,0],bv[roots,2]-bv[roots,0]),axis=1)/2
   roots=roots[ar>=.15]
   for bi in roots:
    bc=np.eye(3)*.8+np.ones((3,3))*(.2/3);raw=bc@bv[bi]
    for perm in itertools.permutations(range(3)):
     trials+=1;root=raw[list(perm)];x,tt,det=prism(root,tip)
     if det.min()<=1e-5:continue
     audit=evaluate(x,tt,tipids,int(bi),bv,v,t,f)
     if audit is None:continue
     rootbary=bc[list(perm)]
     if side<0:rootbary=rootbary[:,[0,2,1]]
     q=dict(side=side,tissue_boundary_face_index=int(fi),tissue_tip_nodes=tipids.tolist(),tissue_owner_tet=int(owner[fi]),muscle_id=muscle['id'],bone_triangle=int(bi),bilateral_bone_triangle=int(bi)+(len(bt) if side<0 else 0),bone_source_vertex_ids=b['triangles'][bi].tolist(),bone_region_id=int(b['region_ids'][bi]),root_barycentric_weights=rootbary.tolist(),root_points_design_units=root.tolist(),tip_points_design_units=tip.tolist(),root_face_area=float(np.linalg.norm(np.cross(root[1]-root[0],root[2]-root[0]))/2),tip_face_area=float(np.linalg.norm(np.cross(tip[1]-tip[0],tip[2]-tip[0]))/2),root_to_tip_lengths=np.linalg.norm(tip-root,axis=1).tolist(),minimum_tet_volume=float(det.min()/6),total_volume=float(det.sum()/6),new_nodes=9,new_tets=len(tt),layer_fractions=[0,.2,.5,1],fixed_local_nodes=[0,1,2,3,4,5],root_local_face=[0,1,2],tip_local_face=[9,10,11],audit=audit)
     q['bilateral_triangle_barycentric_weights']=(rootbary if side>0 else rootbary[:,[0,2,1]]).tolist()
     q['minimum_tet_mean_ratio']=float(np.min(12*(det/2)**(2/3)/np.sum((x[tt[:,EDGES[:,0]]]-x[tt[:,EDGES[:,1]]])**2,axis=(1,2))))
     q['tip_to_root_area_ratio']=q['tip_face_area']/q['root_face_area']
     found=(x,tt,q);break
    if found:break
   if found:break
  if not found:raise RuntimeError('No feasible bounded candidate for side '+str(side))
  x,tt,q=found;prefix='right' if side>0 else 'left';arrays[prefix+'_vertices']=x;arrays[prefix+'_tetrahedra']=tt;arrays[prefix+'_tip_source_node_ids']=np.array(q['tissue_tip_nodes']);arrays[prefix+'_root_barycentric_weights']=np.array(q['root_barycentric_weights']);reports.append(q);print(json.dumps(q),flush=True)
 np.savez_compressed(out/'additive_attachments.npz',**arrays)
 from build_bilateral_mechanics_case import surface_audit
 av=v.copy();at=t.copy()
 for prefix in ['right','left']:
  ids=np.r_[np.arange(len(av),len(av)+9),arrays[prefix+'_tip_source_node_ids']]
  av=np.r_[av,arrays[prefix+'_vertices'][:9]];at=np.r_[at,ids[arrays[prefix+'_tetrahedra']]]
 af,_=boundary(at);full=at[:,FACES].reshape(-1,3)
 _,inv,c=np.unique(np.sort(full,axis=1),axis=0,return_inverse=True,return_counts=True)
 parity=np.prod(np.sign(full[:,[1,2,2]]-full[:,[0,0,1]]),axis=1);orientation=np.bincount(inv,weights=parity)
 topo=dict(nodes_after_hypothetical_append=len(av),tets_after_hypothetical_append=len(at),maximum_face_incidence=int(c.max()),inconsistent_internal_face_orientations=int(np.sum((c==2)&(orientation!=0))),boundary=surface_audit(af),unchanged_existing_vertex_coordinates=bool(np.array_equal(av[:len(v)],v)),unchanged_existing_tetrahedra=bool(np.array_equal(at[:len(t)],t)),new_nodes=len(av)-len(v),new_tets=len(at)-len(t),opposite_bone_sides_separated_by_x=bool(arrays['right_vertices'][:,0].min()>0>b['vertices'][:,0].max()*-1 and arrays['left_vertices'][:,0].max()<0<b['vertices'][:,0].min()))
 assert topo['maximum_face_incidence']==2 and topo['inconsistent_internal_face_orientations']==0
 assert not any(topo['boundary'][k] for k in ['open_or_nonmanifold_edges','inconsistent_winding_edges','nonmanifold_vertex_count'])
 assert topo['opposite_bone_sides_separated_by_x']
 (out/'append_topology_audit.json').write_text(json.dumps(topo,indent=2)+'\n')
 declarations=[]
 for i,(prefix,q) in enumerate(zip(['right','left'],reports)):
  lf,_=boundary(arrays[prefix+'_tetrahedra']);mapping=np.r_[np.arange(len(v)+9*i,len(v)+9*(i+1)),arrays[prefix+'_tip_source_node_ids']]
  pairs=[]
  for fi,bi,_ in q['audit']['declared_bone_contacts']:
   local=lf[fi];feature=local[local<3]
   pairs.append(dict(local_boundary_face_index=fi,local_boundary_face_nodes=local.tolist(),hypothetical_appended_boundary_face_nodes=mapping[local].tolist(),combined_bone_face_id=q['bilateral_bone_triangle'],allowed_contact_dimension=len(feature)-1,allowed_root_local_node_ids=feature.tolist(),allowed_root_appended_node_ids=mapping[feature].tolist(),all_face_nodes_fixed=bool(np.all(local<6))))
  declarations.append(dict(side=q['side'],bone_source_npz_sha256=sha(bone),combined_bone_binary_sha256=sha(domain.parent/'bones.bin'),source_bone_triangle=q['bone_triangle'],source_bone_triangle_vertex_ids=q['bone_source_vertex_ids'],combined_bone_triangle=q['bilateral_bone_triangle'],root_local_nodes=[0,1,2],root_appended_nodes=mapping[:3].tolist(),root_barycentric_in_original_triangle_order=q['root_barycentric_weights'],root_barycentric_in_combined_bilateral_triangle_order=q['bilateral_triangle_barycentric_weights'],fixed_collar_local_nodes=list(range(6)),fixed_collar_appended_nodes=mapping[:6].tolist(),fixed_collar_world_points_design_units=arrays[prefix+'_vertices'][:6].tolist(),shared_tissue_tip_nodes=q['tissue_tip_nodes'],declared_stationary_contacts=pairs,requirement='Validate exact identity and stationary geometry; authorize only listed face/edge/vertex strata; continue complete nonpenetration coverage of every other pair.'))
 (out/'bonded_contact_declarations.json').write_text(json.dumps(dict(schema='proposed-static-bonded-strata-v1',status='Explicit future admission contract only; existing guard unchanged',append_mapping_order='right then left, after all 8405 original nodes',coordinate_scale_solver_per_design_unit=.001,declarations=declarations),indent=2)+'\n')
 report=dict(schema='finite-conforming-origin-feasibility-v1',status='rest-only numerical source feasibility; not admitted for mechanics',inputs={str(p.relative_to(ROOT)):sha(p) for p in [mesh,bone,domain]},attachments=reports,candidate_permutations_tested=trials,elapsed_seconds=time.monotonic()-START,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,semantics=dict(bonded_root='Root is a finite subtriangle of one actual source bone triangle; all three root nodes have immutable barycentrics and zero geometric offset.',declared_contact_strata='The root face, its incident side faces at root edges and root vertices touch only the explicitly declared bone face. Never discard entire incident side faces from nonpenetration coverage.',tip='Existing tissue triangle becomes internal after append. Its 3 nodes are reused, not duplicated, and preserve all exterior native coordinates.',fixed_collar='Root and first 20-percent cross-section are a rigid fixed collar. Its side faces still meet the bone at root edges and need stratum-aware contact admission.',mechanical_support_change='On eventual integration, remove all 6 obsolete per-side proxy origin pins unless independently needed by an omitted-boundary support. Fix the root/collar only; keeping old pins would bypass the new finite load path.'),limits=['No main geometry/domain/solver edits, motion, solve, rendering, installation or upload.','Float64 SAT and ray-parity diagnostics; not interval/exact-arithmetic certification.','Source bone is an authored uncalibrated regional support, not measured anatomy.','One small root face per side is a finite geometric seed, not a calibrated enthesis area or material.','Distal zygomatic insertion remains the existing shared perioral body region; this does not complete a genuine modiolus/lip insertion.','Existing strict-positive-separation guard cannot admit these literal bonded contacts. An explicit bonded-stratum guard and tests are prerequisites to any mechanics run.'])
 (out/'feasibility.json').write_text(json.dumps(report,indent=2)+'\n');assert sum(p.stat().st_size for p in out.iterdir())<1024**2;guard();print('Saved '+str(out),flush=True)
if __name__=='__main__':main()
