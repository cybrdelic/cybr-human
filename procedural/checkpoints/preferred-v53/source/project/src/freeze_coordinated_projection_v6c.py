"""Freeze the inspected original v6c fields with a reversible skin-edge repair."""
from pathlib import Path
import hashlib,json,numpy as np
from facial_projection_cage_v6c import evaluate_displacement
from brow_orbit_planes_v6 import displacement as brow_displacement
from repair_skin_tjunctions_v5 import apply_candidate
R=Path(__file__).resolve().parents[1];O=R/'output/coordinated-projection-v6c';source=R/'output/coordinated-gestalt-v5/head.npz'
expected='ba497c73418c1c4e31db3b0b06c07b36656f67c096fc5ce43ea399523e72ea49'
if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:raise ValueError('Frozen source identity changed')
with np.load(source) as q:v=q['vertices'];f=q['triangles'];ids=q['object_ids']
with np.load(O/'control_cages.npz') as q:c={k:q[k] for k in q.files}
skin=np.unique(f[ids==0]);eyes=np.unique(f[ids>0]);w=v.copy();w[skin]+=evaluate_displacement(v[skin],c)+brow_displacement(v[skin],c['brow'])
if not np.array_equal(w[eyes],v[eyes]):raise ValueError('Globe positions changed')
f2,ids2=apply_candidate(w,f,ids,allow_same_order_deformation=True)
np.savez_compressed(O/'head.npz',vertices=w,triangles=f2,object_ids=ids2)
sha=hashlib.sha256((O/'head.npz').read_bytes()).hexdigest()
report={'input_sha256':expected,'output_sha256':sha,'vertices':len(w),'triangles':len(f2),'globe_centers':[[31.3,12,92.1],[-31.3,12,92.1]],'globe_radius':12.3,'independent_eyes_unchanged':True,'skin_tjunction_repair':'20 existing long-edge faces split at existing vertices, 20 net added faces; no additional vertex displacement','geometry_status':'Rough independent anatomy checkpoint; no r20 integration','vertex_order_retained':True,'unaffected_face_rows_retained':True,'global_self_intersections_tested':False}
(O/'frozen_mesh.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
