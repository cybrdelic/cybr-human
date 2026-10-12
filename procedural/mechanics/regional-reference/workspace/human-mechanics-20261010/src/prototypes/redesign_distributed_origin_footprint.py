#!/usr/bin/env python3
"""One bounded original footprint redesign, adding unchanged neighboring faces.
Targets a simple projected disk visible from its original area centroid.
"""
import sys,json,numpy as np,hashlib,time
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/prototypes'));import design_distributed_origin_attachment as d
r=ROOT/'output';out=r/'distributed-origin-attachment-feasibility-v1';b=np.load(r/'authored-bone-support-v1/bone_support.npz');v=b['vertices'];f=b['triangles'];tri=v[f];areas=d.area(tri);centers=tri.mean(1);norm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);norm/=np.linalg.norm(norm,axis=1)[:,None];l=json.load(open(r/'authored-bone-support-v1/landmarks.json'))['zygomaticus_major_origin_patch_center'];base=np.array(l['attachment_patch_triangles']);target=np.average(centers[base],weights=areas[base],axis=0);axis=np.array([1.,0.,1.])/2**.5;allowed=(np.linalg.norm(centers-np.array(l['point']),axis=1)<=6)&(norm@axis>.5)&np.isin(b['region_ids'],[0,4]);edges=defaultdict(list)
for i,q in enumerate(f):
 for a,z in zip(q,np.roll(q,-1)):edges[tuple(sorted([int(a),int(z)]))].append(i)
patch=set(map(int,base));history=[];reason=None
for iteration in range(180):
 d.guard();ids=np.array(sorted(patch));rf=f[ids];loop=d.ring(rf)
 if loop is None:raise RuntimeError('Lost disk topology')
 e=v[np.roll(loop,-1)]-v[loop];no=np.cross(e,axis);no/=np.linalg.norm(no,axis=1)[:,None];slack=np.einsum('ij,ij->i',no,v[loop]-target)
 if min(slack)>.05:reason='strict_projected_visibility';break
 order=np.argsort(slack);added=False
 for k in order:
  if slack[k]>.05:break
  edge=tuple(sorted([int(loop[k]),int(loop[(k+1)%len(loop)])]));opts=[j for j in edges[edge] if j not in patch and allowed[j]]
  for j in opts:
   if areas[ids].sum()+areas[j]>55:continue
   trial=np.array(sorted(patch|{j}))
   if d.ring(f[trial]) is None:continue
   history.append(dict(added_source_triangle=int(j),removed_indented_boundary_edge=list(edge),prior_margin=float(slack[k]),added_area=float(areas[j])));patch.add(j);added=True;break
  if added:break
 if not added:reason='no_allowed_disk_preserving_fill';break
else:reason='bounded_iteration_limit'
ids=np.array(sorted(patch));loop=d.ring(f[ids]);e=v[np.roll(loop,-1)]-v[loop];no=np.cross(e,axis);no/=np.linalg.norm(no,axis=1)[:,None];margin=float(min(np.einsum('ij,ij->i',no,v[loop]-target)));cent=np.average(centers[ids],weights=areas[ids],axis=0);record=dict(status=reason,baseline_triangles=len(base),baseline_area=float(areas[base].sum()),baseline_area_centroid=target.tolist(),revised_triangles=len(ids),revised_vertices=len(np.unique(f[ids])),revised_area=float(areas[ids].sum()),area_change=float(areas[ids].sum()-areas[base].sum()),area_change_percent=float(100*(areas[ids].sum()/areas[base].sum()-1)),revised_area_centroid=cent.tolist(),centroid_change=(cent-target).tolist(),centroid_shift=float(np.linalg.norm(cent-target)),common_projection_axis=axis.tolist(),minimum_projected_kernel_margin=margin,root_boundary_vertices=len(loop),unchanged_source_geometry=True,region_ids=np.unique(b['region_ids'][ids]).tolist(),area_by_region={str(int(k)):float(areas[ids[b['region_ids'][ids]==k]].sum()) for k in np.unique(b['region_ids'][ids])},added_triangles=sorted(patch-set(base)),removed_triangles=[],history=history,source_triangle_ids=ids.tolist())
(out/'revised_footprint.json').write_text(json.dumps(record,indent=2)+'\n');np.savez_compressed(out/'revised_footprint.npz',source_triangle_ids=ids,source_vertex_ids=np.unique(f[ids]),boundary_source_vertex_ids=loop,projection_axis=axis,visibility_target=target);print(json.dumps({k:v for k,v in record.items() if k not in ['history','source_triangle_ids','added_triangles']}),flush=True)
