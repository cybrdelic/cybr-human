"""Resolve a zero-volume internal flip tetrahedron by conforming both diagonals.
Its two crossing internal edges receive the same explicit intersection node; all
incident real tetrahedra are split. No exterior face or original point moves.
"""
from pathlib import Path
import numpy as np,json,hashlib
B=Path(__file__).resolve().parents[1];I=B/'shared-tet-v4';O=B/'shared-tet-v6';O.mkdir(exist_ok=False);q=np.load(I/'geometry.npz');v=q['vertices'].copy();t=q['tetrahedra'].copy();pat=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
def dets(v,t):
 p=v[t];return np.linalg.det(np.stack([p[:,1]-p[:,0],p[:,2]-p[:,0],p[:,3]-p[:,0]],-1))
def boundary(t):
 f=np.sort(t[:,pat].reshape(-1,3),1);u,c=np.unique(f,axis=0,return_counts=True);assert (c<=2).all();return {tuple(x) for x in u[c==1]}
oldb=boundary(t);oldvol=dets(v,t).sum()/6;det=dets(v,t);bad=np.flatnonzero(det<1e-11);assert len(bad)==1;index=int(bad[0]);a=t[index];choices=[]
for e1,e2 in [(a[[0,1]],a[[2,3]]),(a[[0,2]],a[[1,3]]),(a[[0,3]],a[[1,2]])]:
 x,y=v[e1];z,w=v[e2];ab=y-x;cd=w-z;st=np.linalg.lstsq(np.c_[ab,-cd],z-x,rcond=None)[0];p=x+st[0]*ab;pp=z+st[1]*cd
 if np.all(st>.01) and np.all(st<.99) and np.linalg.norm(p-pp)<1e-10:choices.append((e1,e2,(p+pp)/2,st,np.linalg.norm(p-pp)))
assert len(choices)==1;e1,e2,p,st,error=choices[0]
for edge in [e1,e2]:assert not any(set(edge).issubset(face) for face in oldb)
node=len(v);v=np.r_[v,p[None]];removed=set([index]);new=[]
for edge in [e1,e2]:
 ids=np.flatnonzero(np.isin(t,edge).sum(1)==2)
 for i in ids:
  if i==index:continue
  assert int(i) not in removed;removed.add(int(i));aa=t[i].copy();bb=t[i].copy();aa[aa==edge[0]]=node;bb[bb==edge[1]]=node;new.extend([aa,bb])
new=np.array(new);d=dets(v,new);assert abs(d).min()>1e-10;neg=d<0;new[neg]=new[neg][:,[0,2,1,3]];t=np.r_[np.delete(t,sorted(removed),axis=0),new];assert boundary(t)==oldb;det=dets(v,t);assert (det>0).all();assert abs(det.sum()/6-oldvol)<1e-9
out={k:q[k] for k in q.files};out.update(vertices=v,tetrahedra=t);np.savez_compressed(O/'geometry.npz',**out);r={'source_sha256':hashlib.sha256((I/'geometry.npz').read_bytes()).hexdigest(),'geometry_sha256':hashlib.sha256((O/'geometry.npz').read_bytes()).hexdigest(),'exact_original_vertices':True,'exact_boundary_faces':True,'exact_bone_footprints':True,'old_zero_volume_tet':index,'old_volume':float(dets(q['vertices'],q['tetrahedra'][index:index+1])[0]/6),'crossing_internal_edges':[e1.tolist(),e2.tolist()],'edge_intersection_parameters':st.tolist(),'edge_intersection_residual':float(error),'new_interior_node':p.tolist(),'old_real_tets_subdivided':len(removed)-1,'new_real_tets':len(new),'nodes':len(v),'tetrahedra':len(t),'min_determinant':float(det.min()),'total_volume':float(det.sum()/6),'volume_error':float(det.sum()/6-oldvol),'scope':'Exactly located numerical internal diagonal degeneracy resolved by common Steiner vertex and conforming subdivision of all incident cells. No exterior remesh, artificial material change, motion, load or rejected cell deletion.'};(O/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
