from pathlib import Path
import numpy as np,json
from mathutils import Vector
from mathutils.bvhtree import BVHTree
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parents[1];p=R/'user-human-preferred/recovery/restored-project/output/coordinated-orbital-scale-v53/head.npz';z=np.load(p);v=z['vertices'];f=z['triangles'][z['object_ids']==0];skin=BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True)
def hits(x,y):
 d=Vector((0,0,-1));o=Vector((x,y,200));out=[]
 for _ in range(12):
  h,n,i,dist=skin.ray_cast(o,d,400)
  if h is None:break
  out.append({'z':h.z,'normal':list(n)});o=h+d*1e-4
 return out
s=[]
for y in [-100,-90,-80,-70,-60,-50,-40,-20,0,12]:
 for x in [0,15,25,35,45,50,55,60,65]:s.append({'x':x,'y':y,'hits':hits(x,y)})
(B/'reports/selected_head_Z_intervals.json').write_text(json.dumps(s,indent=2))
for a in s:
 if a['hits']:print(a['x'],a['y'],[round(h['z'],2) for h in a['hits']],flush=True)
