"""Rest-only triangle intersection check against the existing authored bone."""
import sys,json,argparse,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mandibular_collision_audit import broad_pairs,intersect_sat
p=argparse.ArgumentParser();p.add_argument('mesh',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();g=np.load(a.mesh);v=g['vertices'];t=g['tetrahedra'];faces=t[:,np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])].reshape(-1,3);keys=np.sort(faces,axis=1);_,ix,cnt=np.unique(keys,axis=0,return_index=True,return_counts=True);f=faces[ix[cnt==1]];bonepath=Path(__file__).resolve().parents[2]/'output/authored-bone-support-v1/bone_support.npz';b=np.load(bonepath);bs=b['vertices'][b['triangles']];results=[]
for side in [1,-1]:
 x=v.copy();x[:,0]*=side;sf=x[f];hits=[];nc=0
 for i,ids in broad_pairs(sf,bs):
  nc+=len(ids)
  if len(ids):hits.extend([[int(i),int(j)] for j in ids[intersect_sat(sf[i],bs[ids])]])
 print('side',side,'hits',len(hits),'pairs',nc,flush=True);results.append({'side':side,'aabb_candidate_pairs':nc,'intersecting_boundary_pairs':len(hits),'examples':[{'tissue_face':i,'bone_face':j,'tissue_nodes':f[i].tolist(),'location':v[f[i]].mean(0).tolist()} for i,j in hits[:25]]})
r={'mesh_sha256':hashlib.sha256(a.mesh.read_bytes()).hexdigest(),'bone_sha256':hashlib.sha256(bonepath.read_bytes()).hexdigest(),'status':'rest boundary intersection diagnostic; no containment or dynamics acceptance','method':'Complete AABB candidate pairs and float64 triangle SAT; contact counts as intersection. Existing right regional support is mirrored for left-side test only.','sides':results};a.output.write_text(json.dumps(r,indent=2)+'\n')
