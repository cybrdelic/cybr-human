"""Reproduce and freeze the inspected v53 orbital-proportion candidate.

Input is the exact separately delivered v46 original procedural mesh. This
script changes only the declared original similarity/return map; no imported
geometry or image assets are used. Existing mismatched outputs are refused.
"""
from pathlib import Path
from io import BytesIO
import hashlib,json,os,numpy as np
from coherent_orbital_scale_v53 import deform
from refresh_anatomical_semantics import refresh
R=Path(__file__).resolve().parents[1]
INPUT=R/'output/coordinated-nasal-cervical-v46/head.npz'
INPUT_SHA='e568bf8763102e8d9479fc75a295e4d6cdf709c1046d2fd33d0e716e2474ddf6'
EXPECTED='545b8ea2b292107dc2da9f1e90379989eb620c6ee5c2c50b3f9b5316cc77420e'
OUT=R/'output/coordinated-orbital-scale-v53'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze():
 if sha(INPUT)!=INPUT_SHA:raise ValueError('Exact delivered v46 input is required')
 with np.load(INPUT) as q:v=q['vertices'];f=q['triangles'];ids=q['object_ids']
 w,report=deform(v,f,ids,scale=1.07)
 buf=BytesIO();np.savez_compressed(buf,vertices=w,triangles=f,object_ids=ids);data=buf.getvalue();actual=hashlib.sha256(data).hexdigest()
 if actual!=EXPECTED:raise ValueError('Reproduced mesh hash differs from inspected candidate')
 OUT.mkdir(exist_ok=True);target=OUT/'head.npz'
 if target.exists():
  if sha(target)!=EXPECTED:raise ValueError('Refusing to overwrite a different canonical mesh')
 else:
  cached=R/'output/coherent-orbital-scale-v53-study/candidate_head.npz'
  if cached.is_file() and sha(cached)==EXPECTED:os.link(cached,target)
  else:
   with target.open('xb') as out:out.write(data)
 refresh(target,INPUT.parent/'anatomical_semantics/authored_lip_lid_semantics.npz',OUT/'anatomical_semantics')
 report.update({'input_sha256':INPUT_SHA,'mesh_sha256':EXPECTED,'vertices':len(w),'triangles':len(f),'reproduced_inspected_bytes':True,'source_modules':{p.name:sha(p) for p in [Path(__file__),R/'src/coherent_orbital_scale_v53.py',R/'src/refresh_anatomical_semantics.py']},'limits':'Rough original anatomy checkpoint. Independent eyes and finite tissue retained; optical assembly must inherit the 1.07 similarity and be freshly fit. No full blink rig or anatomical realism certificate.'})
 (OUT/'frozen_mesh.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':freeze()
