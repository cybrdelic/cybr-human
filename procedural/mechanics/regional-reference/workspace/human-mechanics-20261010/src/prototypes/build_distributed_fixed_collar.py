#!/usr/bin/env python3
"""Separate source-only exact authored-bone footprint and fixed finite collar.
A distributed soft transition is NOT supplied or claimed accepted.
"""
import sys,json,hashlib,time,itertools,resource
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'src/prototypes'))
from design_distributed_origin_attachment import guard,area,meanratio
from design_finite_origin_attachments import boundary,determinants,strict_tet_overlap,ray_parities
from mandibular_collision_audit import intersect_sat
from audit_connected_midface_boundary import shared_intersection
TOL=1e-9
OUT=ROOT/'output/distributed-origin-attachment-feasibility-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build(side,bv,bf,patch):
 ids=np.unique(bf[patch]);mp={int(g):i for i,g in enumerate(ids)};rf=np.array([[mp[int(g)] for g in q] for q in bf[patch]]);x=bv[ids]*[side,1,1];n=len(x)
 if side<0:rf=rf[:,[0,2,1]]
 x=np.r_[x,x+[.24*side,0,.32]];tt=[]
 for q in rf:
  a,b,c=sorted(q);A,B,C=np.array([a,b,c])+n;z=np.array([[a,b,c,A],[b,c,A,B],[c,B,C,A]])
  # Bone faces point toward the collar. Sorting selects shared diagonals;
  # face permutation parity, never an observed determinant, sets winding.
  odd=sum(int(q[i])>int(q[j]) for i,j in ((0,1),(0,2),(1,2)))%2
  if odd:z=z[:,[0,2,1,3]]
  tt.extend(z)
 tt=np.array(tt)
 assert determinants(x,tt).min()>0, 'Declared collar winding or geometry is invalid'
 return x,tt,rf,ids

def audit(side,x,t,rf,source_ids,bv,bf,oldv,oldt):
 guard();n=len(source_ids);surface,_=boundary(t);bs=bv[bf]*[side,1,1];bnodes=bf if side>0 else bf[:,[0,2,1]]
 if side<0:bs=bs[:,[0,2,1]]
 mn=bs.min(1);mx=bs.max(1);contacts=[];bad=[];candidate_pairs=0
 for fi,q in enumerate(x[surface]):
  ids=np.flatnonzero(np.all(mx>=q.min(0)-TOL,1)&np.all(mn<=q.max(0)+TOL,1));candidate_pairs+=len(ids)
  for bi in ids[intersect_sat(q,bs[ids])]:
   rn=surface[fi][surface[fi]<n];common_source=np.intersect1d(source_ids[rn],bf[bi]);common_local=np.array([int(np.flatnonzero(source_ids==k)[0]) for k in common_source],int)
   if len(common_local)==3:kind='identical-bonded-face' if set(source_ids[surface[fi]])==set(bf[bi]) else 'bad-three-common'
   elif len(common_local):kind,_=shared_intersection(q,bs[bi],x[common_local])
   else:kind='undeclared-contact'
   record=dict(collar_boundary_face=fi,local_nodes=surface[fi].tolist(),bone_triangle=int(bi),shared_source_bone_vertices=common_source.tolist(),dimension=len(common_local)-1,kind=kind)
   (contacts if kind in ('identical-bonded-face','expected_shared_feature') else bad).append(record)
  if fi%32==0:guard()
 # Independent nonroot vertex outside witnesses plus complete boundary contact classification.
 parity=[]
 for i,p in enumerate(x[n:]):
  parity.append(ray_parities(p,bs))
  if i%16==0:guard()
 original=oldv[oldt];mn=original.min(1);mx=original.max(1);oldhits=[];oldpairs=0
 for i,q in enumerate(x[t]):
  ids=np.flatnonzero(np.all(mx>=q.min(0)-TOL,1)&np.all(mn<=q.max(0)+TOL,1));oldpairs+=len(ids);hits=ids[strict_tet_overlap(q,original[ids])]
  if len(hits):oldhits.extend([[i,int(j)] for j in hits])
  if i%32==0:guard()
 own=x[t];mn=own.min(1);mx=own.max(1);selfhits=[];selfpairs=0
 for i,q in enumerate(own):
  ids=np.flatnonzero(np.all(mx>=q.min(0)-TOL,1)&np.all(mn<=q.max(0)+TOL,1)&(np.arange(len(t))>i));selfpairs+=len(ids);hits=ids[strict_tet_overlap(q,own[ids])]
  if len(hits):selfhits.extend([[i,int(j)] for j in hits])
  if i%32==0:guard()
 normal=np.cross(bs[np.array(patch),1]-bs[np.array(patch),0],bs[np.array(patch),2]-bs[np.array(patch),0]);normal/=np.linalg.norm(normal,axis=1)[:,None]
 return dict(side=side,status='pass' if not bad and not np.any(parity) and not oldhits and not selfhits else 'rejected',bone_aabb_candidates=candidate_pairs,declared_feature_contacts=len(contacts),unexpected_contacts=bad,upper_vertices_outside_all_three_rays=bool(not np.any(parity)),minimum_collar_translation_dot_incident_root_normal=float(min(normal@np.array([.24*side,0,.32]))),base_tet_candidates=oldpairs,base_positive_volume_overlaps=oldhits,self_tet_candidates=selfpairs,self_positive_volume_overlaps=selfhits,minimum_tet_volume=float(determinants(x,t).min()/6),minimum_quality=float(meanratio(x,t).min()),quality_1pct=float(np.percentile(meanratio(x,t),1)),contact_records=contacts)
if __name__=='__main__':
 start=time.monotonic();guard();OUT.mkdir(exist_ok=True);bp=ROOT/'output/authored-bone-support-v1/bone_support.npz';b=np.load(bp);land=json.loads((bp.parent/'landmarks.json').read_text())['zygomaticus_major_origin_patch_center'];patch=np.array(land['attachment_patch_triangles']);gp=ROOT/'output/bilateral-midface-bone-aware-v2/rest_geometry.npz';g=np.load(gp);reports=[];contracts=[]
 for side,name in [(1,'right'),(-1,'left')]:
  x,t,rf,ids=build(side,b['vertices'],b['triangles'],patch);f,_=boundary(t);rep=audit(side,x,t,rf,ids,b['vertices'],b['triangles'],g['vertices'],g['tetrahedra']);reports.append(rep);np.savez_compressed(OUT/(name+'_fixed_collar.npz'),vertices=x,tetrahedra=t,root_faces=rf,boundary_faces=f,root_source_bone_vertex_ids=ids,root_source_bone_triangle_ids=patch,fixed_local_nodes=np.arange(len(x)))
  contracts.append(dict(side=side,fixture_file=name+'_fixed_collar.npz',root_nodes=list(range(len(ids))),root_combined_bone_vertex_ids=(ids+(len(b['vertices']) if side<0 else 0)).tolist(),root_source_bone_triangle_ids=patch.tolist(),root_combined_bone_triangle_ids=(patch+(len(b['triangles']) if side<0 else 0)).tolist(),root_face_local_nodes=rf.tolist(),fixed_local_nodes=list(range(len(x))),boundary_local_faces=f.tolist(),vertices_solver_units=(x*.001).tolist(),tetrahedra=t.tolist(),status='Standalone fixed collar only; no accepted soft transition or main integration'))
  print(json.dumps({k:v for k,v in rep.items() if k not in ['contact_records']}),flush=True)
 (OUT/'fixed_collar_audit.json').write_text(json.dumps(dict(schema='distributed-authored-fixed-collar-audit-v1',scope='No accepted soft transition; Float64 static checks only',source_bone_sha256=sha(bp),source_tissue_sha256=sha(gp),coordinate_units='authored design units',root_patch_area=float(area(b['vertices'][b['triangles'][patch]]).sum()),reports=reports,elapsed_seconds=time.monotonic()-start,rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024),separators=(',',':'))+'\n')
 (OUT/'fixed_collar_contract.json').write_text(json.dumps(dict(schema='distributed-original-vertex-fixed-collar-v1',coordinate_scale_solver_per_design_unit=.001,source_bone_sha256=sha(bp),combined_bone_binary_sha256=sha(ROOT/'output/bilateral-mechanics-v1/bones.bin'),combined_bone_vertex_count=2*len(b['vertices']),combined_bone_triangle_count=2*len(b['triangles']),status='Source-only standalone collar, not admitted into a main solver',sides=contracts),separators=(',',':'))+'\n')
 assert sum(p.stat().st_size for p in OUT.iterdir())<3*1024**2
