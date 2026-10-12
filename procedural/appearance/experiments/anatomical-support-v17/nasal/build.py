"""Rebuild an original nasal section study on the pinned margin-v3 baseline."""
from pathlib import Path
import argparse,sys,json,hashlib,importlib.util,numpy as np
p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--version',type=int,choices=[22],default=22);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
C=Path(__file__).resolve().parent;meta=json.loads((C/'CHECKPOINT.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(a.baseline)==meta['baseline_head_sha256'],'Expected exact margin-v3 dependency'
a.output.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(C/'source'))
spec=importlib.util.spec_from_file_location('nasal_variant',C/'variants'/f'v{a.version}'/'nasal_surface.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
patch,report=module.build();np.savez_compressed(a.output/'nasal_patch.npz',**patch)
assert sha(a.output/'nasal_patch.npz')==meta['variants'][str(a.version)]['nasal_patch_sha256'],'Fresh source patch differs from frozen study'
b=np.load(a.baseline);contract=np.load(C/'assembly_contract.npz');v=b['vertices'];f=b['triangles'];oid=b['object_ids'];nb=len(patch['boundary_vertex_ids']);boundary=contract['boundary_base_vertex_ids'];assert np.array_equal(v[boundary],patch['vertices'][:nb])
keep=np.ones(len(f),bool);keep[contract['removed_base_face_ids']]=False
local=np.r_[boundary,np.arange(len(v),len(v)+len(patch['vertices'])-nb)]
vv=np.r_[v,patch['vertices'][nb:]];ff=np.r_[f[keep],local[patch['triangles']]];oo=np.r_[oid[keep],np.zeros(len(patch['triangles']),oid.dtype)]
used=np.unique(ff);compact=np.full(len(vv),-1,np.int64);compact[used]=np.arange(len(used));vv=vv[used];ff=compact[ff]
np.savez_compressed(a.output/'head.npz',vertices=vv,triangles=ff,object_ids=oo)
actual=sha(a.output/'head.npz');assert actual==meta['variants'][str(a.version)]['head_sha256'],'Assembled output differs from frozen study'
(a.output/'REBUILD_RECEIPT.json').write_text(json.dumps({'head_sha256':actual,'patch_sha256':sha(a.output/'nasal_patch.npz'),'fresh_procedural_rebuild':True,'selected':False},indent=2)+'\n')
print(actual)
