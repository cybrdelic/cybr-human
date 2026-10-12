"""Original short facial fibers rooted on literal skin triangles. No painted coverage."""
import numpy as np
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from rooted_groom import hair_material,create_curves

def smooth(x):
 q=np.clip(x,0,1);return q*q*(3-2*q)

def build(v,f,ids,semantics,transform,output,seed=74193,count=22000):
 rng=np.random.default_rng(seed);vv=np.asarray(v)
 vertex_ok=(vv[:,1]>-112)&(vv[:,1]<-17)&(np.abs(vv[:,0])<71)&(vv[:,2]>32)
 use=(ids==0)&np.all(vertex_ok[f],axis=1)
 exclude=np.zeros(len(v),bool)
 for key in ['vermilion_vertex_ids','inner_labial_return_vertex_ids']:
  if key in semantics:exclude[np.asarray(semantics[key]).ravel()]=True
 use&=~np.any(exclude[f],axis=1);face_ids=np.flatnonzero(use);sf=f[use];p=vv[sf]
 cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);area=np.linalg.norm(cross,axis=1)/2;n=cross/np.maximum(2*area[:,None],1e-15);c=p.mean(1);x,y,z=c.T;ax=np.abs(x)
 # Shared smooth beard/moustache growth fields, no per-pixel spots or arbitrary color mask.
 lower=smooth((-y-36)/19)*smooth((y+108)/19)*smooth((68-ax)/15)
 cheek=smooth((ax-15)/16)*smooth((-y-24)/25)*smooth((y+88)/15)*smooth((69-ax)/15)*.65
 moustache=smooth((y+52)/4)*smooth((-y-38)/5)*smooth((ax-3)/6)*smooth((28-ax)/7)*.48
 density=np.maximum(np.maximum(lower,cheek),moustache)*smooth((n[:,2]-.03)/.42)
 density*=np.clip((z-42)/20,0,1);w=area*density;assert w.sum()>0
 selected=rng.choice(len(sf),size=count,replace=True,p=w/w.sum());bary=rng.random((count,2));s=np.sqrt(bary[:,0]);bary=np.column_stack((1-s,s*(1-bary[:,1]),s*bary[:,1]));roots=np.einsum('ni,nij->nj',bary,p[selected]);normals=n[selected]
 # Local query mesh is used only for finite-radius clearance away from the root.
 used,ix=np.unique(sf,return_inverse=True);bvh=BVHTree.FromPolygons(vv[used].tolist(),ix.reshape(-1,3).tolist(),all_triangles=True);del p,c,cross
 paths=[];radii=[];gaps=[];corrections=[];lengths=[];root_r=[]
 for k,(root,normal)in enumerate(zip(roots,normals)):
  # Inferior/oblique growth is projected into the tangent plane, then emerges outward.
  side=1 if root[0]>0 else -1
  flow=np.array([side*.20,-1.,.10]);flow-=normal*np.dot(flow,normal);flow/=max(np.linalg.norm(flow),1e-10)
  length=rng.uniform(.30,.76);rad=rng.uniform(.022,.034);direction=normal*.78+flow*.63;direction/=np.linalg.norm(direction)
  path=[];rr=[]
  for u in np.linspace(0,1,5):
   point=root+normal*.003+direction*length*u+flow*.05*np.sin(np.pi*u);radius=rad*(1-u)**.7+.001
   before=point.copy()
   if u>0:
    q,no,fi,dist=bvh.find_nearest(Vector(point));no=np.asarray(no);signed=float(np.dot(point-np.asarray(q),no));needed=radius+.006
    if signed<needed:point+=no*(needed-signed)
    q,no,fi,dist=bvh.find_nearest(Vector(point));gaps.append(float(np.dot(point-np.asarray(q),np.asarray(no)))-radius)
   corrections.append(float(np.linalg.norm(point-before)));path.append(point);rr.append(radius)
  paths.append(np.asarray(path));radii.append(np.asarray(rr));lengths.append(length);root_r.append(rad)
 assert min(gaps)>0
 material=hair_material('Original short facial fibers | actual filtered curve coverage',.78,.47)
 create_curves('Original short beard and moustache | triangle-rooted',paths,radii,transform,material)
 out=Path(output)/'short_groom_binding.npz';np.savez_compressed(out,source_face_ids=face_ids[selected],source_triangle_vertex_ids=sf[selected],barycentric_weights=bary,root_positions=roots,paths=np.asarray(paths),radii=np.asarray(radii),seed=np.array(seed))
 return {'strands':count,'points_per_strand':5,'seed':seed,'root_method':'Literal original skin triangle area sampling with saved barycentric weights; no projected shadow/painted follicle field','length_authored_units_range':[min(lengths),max(lengths)],'root_radius_authored_units_range':[min(root_r),max(root_r)],'minimum_sampled_tube_skin_clearance_away_from_root':min(gaps),'maximum_sampled_collision_lift':max(corrections),'binding_file':str(out),'coverage':'Actual Cycles curve sampling/filtering; shafts and radii are not enlarged to occupy pixels','limits':['Authored uncalibrated grooming dimensions/density','Control-point clearance only; not a continuous swept-curve certificate','No expression/binding validation until a qualified pose is supplied']}
