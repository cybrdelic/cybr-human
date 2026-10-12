from pathlib import Path
import argparse
_ap=argparse.ArgumentParser();_ap.add_argument("--candidate",type=Path,required=True);_args=_ap.parse_args()
import numpy as np,json,hashlib
O=_args.candidate.resolve();h=np.load(O/'candidate_head.npz');f=h['triangles'];e=np.r_[f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]];_,inv,cnt=np.unique(np.sort(e,axis=1),axis=0,return_inverse=True,return_counts=True);w=np.bincount(inv,weights=np.where(e[:,0]<e[:,1],1,-1));r={'head_sha256':hashlib.sha256((O/'candidate_head.npz').read_bytes()).hexdigest(),'boundary_edges':int((cnt==1).sum()),'nonmanifold_edges':int((cnt>2).sum()),'inconsistent_winding':int(((cnt==2)&(w!=0)).sum())};assert r['boundary_edges']==r['nonmanifold_edges']==r['inconsistent_winding']==0,r;(O/'TOPOLOGY_REPORT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
