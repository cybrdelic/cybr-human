"""Reconstruct pinned original v46 directly from delivered v8 and authored controls.

Run after restoring the exact v8 dependency. The original nasal control
network, finite inlet builder, cut contract and lower-head/face controls are
retained separately for editing. This pins one rough rest-shape checkpoint,
not validated anatomy, a complete rig, or the unavailable r20 application.
"""
from pathlib import Path
import hashlib,io,json,os,numpy as np
from regional_cervical_surface_v44 import build
from compose_anatomical_stack import compose
from integrate_nasal_v8_shared import integrate
from oral_mental_continuity_v30 import deform as oral
from midface_spline_envelope_v35 import deform as midface
from upper_lid_rest_pose_v36 import deform as rest
from refresh_anatomical_semantics import refresh
R=Path(__file__).resolve().parents[1];I=R/'output/coordinated-lid-body-v8/head.npz'
O=R/'output/coordinated-nasal-cervical-v46';O.mkdir(exist_ok=True);C=R/'output/coordinated-cervical-v31/controls'
P=R/'output/nasal-fresh-basal-control-net-study/supported-basal-control-net-v4/fitted_nasal_patch.npz'
PIN='dd20de3843df48b43c0ecdf0df91580f78f5f04d5c50802780022d9e5124a7ef'
PATCH_PIN='13cbfc604bd9ebde790c50471e709651181dd415048b689bf8863fe0168409f3'
EXPECTED='e568bf8763102e8d9479fc75a295e4d6cdf709c1046d2fd33d0e716e2474ddf6'
if hashlib.sha256(I.read_bytes()).hexdigest()!=PIN:raise ValueError('Original v8 dependency hash mismatch')
if hashlib.sha256(P.read_bytes()).hexdigest()!=PATCH_PIN:raise ValueError('Supported-basal patch hash mismatch')
with np.load(I) as q:v=q['vertices'];f=q['triangles'];ids=q['object_ids']
w,g,oid,mapping,report,guide=compose(v,f,ids,build,nasal_insertion=lambda v,f,ids:integrate(v,f,ids,patch_path=P),coefficient_files={n:C/(n+'_cage.npz') for n in ['cheek','support','midface']})
with np.load(C/'oral_cage.npz') as q:oc=q['coefficients']
w=oral(w,g,oid,oc);w,_=midface(w,g,oid);w=rest(w,g,oid)
buf=io.BytesIO();np.savez_compressed(buf,vertices=w,triangles=g,object_ids=oid);data=buf.getvalue();sha=hashlib.sha256(data).hexdigest()
if sha!=EXPECTED:raise ValueError('Reconstruction differs from the pinned diagnostic; no mesh written')
target=O/'head.npz';cache=R/'output/nasal-cervical-v46-study/candidate_head.npz'
if target.exists():
 if hashlib.sha256(target.read_bytes()).hexdigest()!=sha:raise ValueError('Refusing to overwrite a different existing mesh')
elif cache.is_file() and hashlib.sha256(cache.read_bytes()).hexdigest()==sha:
 try:os.link(cache,target)
 except OSError:
  with target.open('xb') as q:q.write(data)
else:
 with target.open('xb') as q:q.write(data)
del data,buf,v,f,ids
np.savez_compressed(O/'source_vertex_map.npz',old_to_new=mapping)
with np.load(R/'output/coordinated-lid-body-v8/anatomical_semantics/authored_lip_lid_semantics.npz') as q:sem={k:q[k] for k in q.files}
for k in list(sem):
 if k.endswith('_vertex_ids'):
  sem[k]=mapping[sem[k]]
  if np.any(sem[k]<0):raise ValueError('Protected semantic vertex removed')
np.savez_compressed(O/'remapped_semantics_input.npz',**sem)
report.update({'input_sha256':PIN,'patch_sha256':PATCH_PIN,'mesh_sha256':sha,'vertices':len(w),'triangles':len(g),'method':'Fresh original cubic nasal control network with supported basal curves and finite inferior vestibules; curved cervical muscle support on the connected regional surface; retained v36 facial rest controls','globe_centers':[[31.3,12,94.6],[-31.3,12,94.6]],'globe_radius':12.3,'status':'Pinned rough original geometry checkpoint','limits':['Overall anatomy remains rough and stylized.','Basal nasal shadow, alar-to-philtrum transitions and lip shape remain simplified.','SCM-inspired surface curves are authored form controls, not validated musculoskeletal anatomy.','Directed edge incidence and local validity checks do not certify global self-intersections or physical transport.','Independent eyes and finite tissue retained; no complete blink/expression rig.','No imported head, scan, image-reference texture or unavailable r20 integration.']})
(O/'frozen_mesh.json').write_text(json.dumps(report,indent=2)+'\n')
refresh(target,O/'remapped_semantics_input.npz',O/'anatomical_semantics')
print(json.dumps({'mesh':str(target),'sha256':sha,'vertices':len(w),'triangles':len(g)}),flush=True)
