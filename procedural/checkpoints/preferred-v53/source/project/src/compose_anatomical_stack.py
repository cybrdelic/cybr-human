"""Compose original regional replacements before continuous facial controls.

The base is the pinned v8 original mesh. Optional insertions preserve original
vertex IDs; only then is the replaced head compacted with an explicit map.
This makes the facial fields reusable without relying on historical topology.
"""
from pathlib import Path
import numpy as np
from coordinated_facial_depth_v25 import build as facial_depth
from midface_cross_section_planes_v29 import deform as midface
R=Path(__file__).resolve().parents[1]

def compose(v,f,ids,cervical_builder,nasal_insertion=None,coefficient_files=None):
 original_count=len(v);reports={}
 if coefficient_files is None:coefficient_files={'cheek':R/'output/coordinated-facial-depth-v25/controls/cheek_cage.npz','support':R/'output/coordinated-facial-depth-v25/controls/support_cage.npz','midface':R/'output/coordinated-midface-v29/control_cage.npz'}
 if nasal_insertion is not None:
  v,f,ids,report=nasal_insertion(v,f,ids);reports['nose']=report
 v,f,ids,report,guide=cervical_builder(v,f,ids);reports['cervical']=report
 used=np.unique(f);mapping=np.full(len(v),-1,np.int64);mapping[used]=np.arange(len(used));v=v[used];f=mapping[f]
 with np.load(coefficient_files['cheek']) as q:cc=q['coefficients']
 with np.load(coefficient_files['support']) as q:sc=q['coefficients']
 v,report=facial_depth(v,f,ids,cc,sc);reports['facial_depth']=report
 with np.load(coefficient_files['midface']) as q:mc=q['coefficients']
 v=midface(v,f,ids,mc)
 return v,f,ids,mapping[:original_count],reports,guide
