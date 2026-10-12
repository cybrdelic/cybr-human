from pathlib import Path
import numpy as np,json,sys
from scipy.spatial import cKDTree
from mathutils import Vector
from mathutils.bvhtree import BVHTree
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'human-preferred-v53-mechanics-20261011/src'));from triangle_sat import sat_intersect
p=B/'mandible-v3/geometry.npz';g=np.load(p);v=g['vertices'];f=g['triangles'];tri=v[f];mid=tri.mean(1);rad=np.linalg.norm(tri-mid[:,None],axis=2).max(1);tree=cKDTree(mid);lo=tri.min(1);hi=tri.max(1);hit=[];ihit=[];nc=ic=0
for start in range(0,len(f),128):
 ids=np.arange(start,min(start+128,len(f)));ls=tree.query_ball_point(mid[ids],rad[ids]+rad.max()+1e-8);i=np.repeat(ids,[len(a) for a in ls]);j=np.concatenate(ls).astype(int);take=j>i;i=i[take];j=j[take];take=(hi[i]>=lo[j]-1e-10).all(1)&(hi[j]>=lo[i]-1e-10).all(1);i=i[take];j=j[take];inc=(f[i,:,None]==f[j,None,:]).any((1,2));nc+=(~inc).sum();ic+=inc.sum()
 if (~inc).any():a=i[~inc];b=j[~inc];hh=sat_intersect(tri[a],tri[b],eps=1e-10);hit.extend(np.c_[a[hh],b[hh]].tolist())
 if inc.any():
  a=i[inc];b=j[inc];ta=mid[a,None]+(1-1e-6)*(tri[a]-mid[a,None]);tb=mid[b,None]+(1-1e-6)*(tri[b]-mid[b,None]);hh=sat_intersect(ta,tb,eps=1e-12);ihit.extend(np.c_[a[hh],b[hh]].tolist())
h=np.load(R/'user-human-preferred/recovery/restored-project/output/coordinated-orbital-scale-v53/head.npz');hf=h['triangles'][h['object_ids']==0];skin=BVHTree.FromPolygons(h['vertices'].tolist(),hf.tolist(),all_triangles=True);mesh=BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True);pairs=np.array(mesh.overlap(skin),int);cross=[]
for st in range(0,len(pairs),3000):
 p=pairs[st:st+3000];hh=sat_intersect(v[f[p[:,0]]],h['vertices'][hf[p[:,1]]],eps=1e-9);cross.extend(p[hh].tolist())
r={'self_nonincident_candidates':int(nc),'self_nonincident_crossings':len(hit),'trimmed_incident_candidates':int(ic),'trimmed_incident_crossings':len(ihit),'self_examples':hit[:10],'trimmed_examples':ihit[:10],'skin_triangle_candidate_pairs':len(pairs),'skin_triangle_crossings':len(cross),'skin_examples':cross[:10],'limit':'Static triangle tests; incident strips exclude1e-6 centroid trim. No moving TMJ/contact dynamics.'};(B/'mandible-v3/surface_check.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
