#!/usr/bin/env python3
"""Complete Float64 static source audit of the revised distributed attachments.
No main writes, solver runs, renderer or activation. Complete arrays stay local.
"""
import sys,json,numpy as np,itertools,hashlib,time,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'src/prototypes'))
import build_distributed_fixed_collar as ca
from build_bilateral_mechanics_case import surface_audit
from design_finite_origin_attachments import boundary,determinants
from design_distributed_origin_attachment import guard,meanratio
R=ROOT/'output';OUT=R/'distributed-origin-attachment-feasibility-v1/revised'
def condition(x,t):
 q=x[t];dm=q[:,1:]-q[:,:1];c=np.linalg.cond(dm);vol=np.linalg.det(dm)/6;mr=meanratio(x,t);best=np.full(len(t),np.inf)
 for base in range(4):best=np.minimum(best,np.linalg.cond(q[:,[k for k in range(4) if k!=base]]-q[:,base,None]))
 bad=np.argsort(mr)[:5];return dict(reference_Dm_condition=dict(max=float(c.max()),p99=float(np.percentile(c,99)),p999=float(np.percentile(c,99.9)),median=float(np.median(c))),best_vertex_base_Dm_condition=dict(max=float(best.max()),p99=float(np.percentile(best,99))),minimum_positive_volume_design_units_cubed=float(vol.min()),minimum_mean_ratio=float(mr.min()),worst_condition_tet=int(np.argmax(c)),worst_condition_location=q[np.argmax(c)].mean(0).tolist(),counts_over_accepted_tissue_worst_501_39=int(np.sum(c>501.39)),lowest_quality_tets=[dict(tet=int(i),local_nodes=t[i].tolist(),centroid_design_units=q[i].mean(0).tolist(),vertices_design_units=q[i].tolist(),mean_ratio=float(mr[i]),reference_Dm_condition=float(c[i]),best_vertex_base_Dm_condition=float(best[i]),volume_design_units_cubed=float(vol[i])) for i in bad])
if __name__=='__main__':
 start=time.monotonic();g=np.load(R/'bilateral-midface-bone-aware-v2/rest_geometry.npz');oldv=g['vertices'];oldt=g['tetrahedra'];b=np.load(R/'authored-bone-support-v1/bone_support.npz');reports=[];conds=[];av=oldv.copy();at=oldt.copy()
 for side,name in [(1,'right'),(-1,'left')]:
  a=np.load(OUT/(name+'_attachment_candidate.npz'));x=a['vertices'];t=a['tetrahedra'];ca.patch=a['root_source_bone_triangle_ids'];report=ca.audit(side,x,t,a['root_faces'],a['root_source_bone_vertex_ids'],b['vertices'],b['triangles'],oldv,oldt);records=report.pop('contact_records');ct=np.full((len(records),6),-1,np.int32)
  for i,q in enumerate(records):ct[i,:3]=[q['collar_boundary_face'],q['bone_triangle'],q['dimension']];ct[i,3:3+len(q['shared_source_bone_vertices'])]=q['shared_source_bone_vertices']
  np.savez_compressed(OUT/(name+'_declared_contact_records.npz'),records=ct);report['contact_record_schema']=['local_boundary_face','source_bone_triangle','shared_feature_dimension','shared_source_vertex_0','shared_source_vertex_1','shared_source_vertex_2'];reports.append(report);co=condition(x,t);co['side']=side;conds.append(co)
  mapping=np.full(len(x),-1,np.int64);shared=a['shared_tissue_local_node_ids'];mapping[shared]=a['shared_tissue_source_node_ids'];new=np.flatnonzero(mapping<0);mapping[new]=len(av)+np.arange(len(new));av=np.r_[av,x[new]];at=np.r_[at,mapping[t]]
  print(json.dumps({k:v for k,v in report.items() if k!='contact_record_schema'}),flush=True);print('CONDITION '+json.dumps({k:v for k,v in co.items() if k!='lowest_quality_tets'}),flush=True)
 f,_=boundary(at);topo=surface_audit(f);full=at[:,np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])].reshape(-1,3);_,inv,count=np.unique(np.sort(full,axis=1),axis=0,return_inverse=True,return_counts=True);parity=np.prod(np.sign(full[:,[1,2,2]]-full[:,[0,0,1]]),axis=1);orient=np.bincount(inv,weights=parity);topo.update(hypothetical_base_nodes=len(oldv),hypothetical_appended_nodes=len(av),hypothetical_appended_tets=len(at),old_coordinates_unchanged=bool(np.array_equal(av[:len(oldv)],oldv)),old_tets_unchanged=bool(np.array_equal(at[:len(oldt)],oldt)),maximum_face_incidence=int(count.max()),inconsistent_shared_face_orientation=int(np.sum((count==2)&(orient!=0))),scope='In-memory append to unchanged old base only. New global IDs are not an integration contract; add after any verified base node count. Combined overlap with new cavity geometry is not tested here.')
 (OUT/'static_geometry_audit.json').write_text(json.dumps(dict(status='static source geometry diagnostics only; conditioning must be reviewed separately',reports=reports,topology=topo,elapsed_seconds=time.monotonic()-start,rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024),indent=2)+'\n');(OUT/'reference_conditioning_audit.json').write_text(json.dumps(conds,indent=2)+'\n');guard();assert sum(p.stat().st_size for p in OUT.parent.rglob('*') if p.is_file())<3*1024**2
