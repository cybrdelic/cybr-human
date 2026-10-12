"""Reproduce the included original v6c geometry from pinned original v5 input."""
from pathlib import Path
import argparse,hashlib,json,sys,numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from facial_projection_cage_v6c import evaluate_displacement,deform
from brow_orbit_planes_v6 import displacement as brow_displacement,build_field
from repair_skin_tjunctions_v5 import apply_candidate

def reproduce(root=ROOT,output=None,rebuild_controls=False):
    root=Path(root);source=root/'output/coordinated-gestalt-v5/head.npz'
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    if sha!='ba497c73418c1c4e31db3b0b06c07b36656f67c096fc5ce43ea399523e72ea49':raise ValueError('Pinned original input changed')
    with np.load(source) as q:v=q['vertices'];f=q['triangles'];ids=q['object_ids']
    if rebuild_controls:
        _,_,c=deform(v,f,ids);c['brow']=build_field()[0]
    else:
        with np.load(root/'output/coordinated-projection-v6c/control_cages.npz') as q:c={k:q[k] for k in q.files}
    skin=np.unique(f[ids==0]);w=v.copy();w[skin]+=evaluate_displacement(v[skin],c)+brow_displacement(v[skin],c['brow'])
    ff,ii=apply_candidate(w,f,ids,allow_same_order_deformation=True,patch_path=root/'output/skin-boundary-v5-study/tjunction_face_repair.npz')
    if output:
        output=Path(output)
        if output.exists():raise FileExistsError('Refusing to overwrite an existing checkpoint')
        output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,vertices=w,triangles=ff,object_ids=ii)
        return {'output':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'vertices':len(w),'triangles':len(ff)}
    target=root/'output/coordinated-projection-v6c/head.npz'
    with np.load(target) as q:
        report={'vertex_arrays_exact':bool(np.array_equal(w,q['vertices'])),'maximum_vertex_difference':float(np.max(np.abs(w-q['vertices']))),'triangles_exact':bool(np.array_equal(ff,q['triangles'])),'object_ids_exact':bool(np.array_equal(ii,q['object_ids'])),'recomputed_controls':rebuild_controls}
    if report['maximum_vertex_difference']>1e-9 or not report['triangles_exact'] or not report['object_ids_exact']:raise ValueError(report)
    return report
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output');a.add_argument('--rebuild-controls',action='store_true');q=a.parse_args()
    print(json.dumps(reproduce(output=q.output,rebuild_controls=q.rebuild_controls),indent=2))
