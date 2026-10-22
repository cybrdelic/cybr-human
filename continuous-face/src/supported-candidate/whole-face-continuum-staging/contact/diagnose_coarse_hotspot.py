import sys,json,time,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;sys.path[:0]=[str(P),str(P.parent/'mechanics')]
from source_transfer import load_geometry_with_source_fibres
from coarse_physical_site_contact import occupied_site_map
from coarse_rigid_query import NativeRigidQuery
s=load_geometry_with_source_fibres(P.parent/'geometry/whole-face-continuum-final.npz',P.parent/'geometry/whole-face-muscle-phases-final.npz')
with np.load(P.parent/'geometry/whole-face-integration-cells.npz') as z:pts=z['grid_origin']+(z['cell_ijk']+.5)*float(z['grid_spacing'])
C,_=occupied_site_map(s,pts);column=C[:,5438].tocoo();ids=column.row;w=column.data;points=pts[ids]
r=NativeRigidQuery.from_native();report=json.load(open(P.parent/'mechanics/coarse-physical-site-v1/substep-1ms/report.json'));hist=report['runs']['passive']['reports'][0]['history'];positions=[np.zeros(3)]+[np.array(x['dominant_dynamic_residual']['position_mm'])-s.X[5438] for x in hist if x['dominant_dynamic_residual']['node']==5438]
old=[];new=[]
for d in positions:
 p=points+w[:,None]*d
 old.append(r.query(p));new.append(r.query_band(p))
rows=[]
for i in range(len(points)):
 o=np.array([x['normal'][i] for x in old]);n=np.array([x['normal'][i] for x in new]);g=np.array([x['signed_gap'][i] for x in old]);gg=np.array([x['signed_gap'][i] for x in new]);flip=bool(np.any((o@o.T)<-.5));nflip=bool(np.any((n@n.T)<-.5));delta=float(np.max(np.linalg.norm((.1-g).clip(0)[:,None]*o-(.1-gg).clip(0)[:,None]*n,axis=1))*.01*w[i])
 if (flip or nflip or delta>1e-5) and min(g.min(),gg.min())<.1:
  rows.append(dict(site=int(ids[i]),reference=points[i].tolist(),carrier_weight=float(w[i]),old_gaps=g.tolist(),old_owners=[int(x['owner_id'][i]) for x in old],old_faces=[int(x['face_id'][i]) for x in old],old_normals=o.tolist(),complete_gaps=gg.tolist(),complete_owners=[int(x['owner_id'][i]) for x in new],complete_faces=[int(x['face_id'][i]) for x in new],complete_normals=n.tolist(),maximum_weighted_force_difference_N=delta,old_flip=flip,complete_flip=nflip))
rows.sort(key=lambda x:x['maximum_weighted_force_difference_N'],reverse=True)
out=dict(scope='Bounded node5438 support-site diagnosis: recorded hotspot nodal displacements only; ALL other nodes held at reference. Not reconstructed failed full-state trajectory or a solve.',node=5438,support_sites=len(ids),displacements_mm=[x.tolist() for x in positions],rows=rows,owners=r.report['owners'])
(P/'coarse-hotspot-feature-diagnosis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(sites=len(ids),rows=len(rows),top=rows[:3]),indent=2))
