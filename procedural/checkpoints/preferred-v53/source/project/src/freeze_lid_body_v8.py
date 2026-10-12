"""Reproduce the selected lid-body shape layer from the pinned original v6c mesh."""
from pathlib import Path
import hashlib,json,shutil,numpy as np
from lower_lid_support_v8 import deform
from refresh_anatomical_semantics import refresh
R=Path(__file__).resolve().parents[1];O=R/'output/coordinated-lid-body-v8';O.mkdir(exist_ok=True)
p=R/'output/coordinated-projection-v6c/head.npz';expected='f60e447413c791494a8856ee5571cda9574ecca600dfc05427f5c9f7182bfb27'
if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('Pinned v6c input mismatch')
with np.load(p) as q:v=q['vertices'];f=q['triangles'];ids=q['object_ids']
w=deform(v,f,ids);eyes=np.unique(f[ids>0]);assert np.array_equal(w[eyes],v[eyes])
np.savez_compressed(O/'head.npz',vertices=w,triangles=f,object_ids=ids)
sha=hashlib.sha256((O/'head.npz').read_bytes()).hexdigest()
report={'input_sha256':expected,'output_sha256':sha,'vertices':len(w),'triangles':len(f),'globe_centers':[[31.3,12,92.1],[-31.3,12,92.1]],'globe_radius':12.3,'independent_eyes_unchanged':True,'topology_unchanged':True,'vertex_order_retained':True,'selected_layers':['upper_lid_body_v7','lower_lid_support_v8'],'excluded_layers':['nasal v7 experiments','malar_buccal_planes_v9','medial_canthal_tissue_v10'],'status':'Inspected useful lid-body direction on rough independent v6c; incomplete anatomy and no r20 integration','limits':['Medial corner and free-edge differentiation remain unfinished','No full global self-intersection proof','Appearance optical interface must be refitted to changed lid bearing']}
(O/'frozen_mesh.json').write_text(json.dumps(report,indent=2)+'\n')
refresh(O/'head.npz',R/'output/coordinated-projection-v6c/anatomical_semantics/authored_lip_lid_semantics.npz',O/'anatomical_semantics')
snap=O/'source_snapshot';snap.mkdir(exist_ok=True)
for name in ['freeze_lid_body_v8.py','upper_lid_body_v7.py','lower_lid_support_v8.py','refresh_anatomical_semantics.py']:shutil.copyfile(R/'src'/name,snap/name)
(snap/'manifest.json').write_text(json.dumps({n.name:hashlib.sha256(n.read_bytes()).hexdigest() for n in snap.glob('*.py')},indent=2)+'\n');print(json.dumps(report),flush=True)
