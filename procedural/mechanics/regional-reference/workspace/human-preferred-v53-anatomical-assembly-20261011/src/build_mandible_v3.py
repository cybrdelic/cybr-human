"""Original mandibular contact-shape authoring; no scan or imported head/bone.
Positions are authored estimates constrained by the exact preferred head envelope.
Geometry/layout only: no jaw dynamics, teeth, TMJ, or clinical calibration.
"""
from pathlib import Path
import json,hashlib,numpy as np
from scipy.interpolate import PchipInterpolator,CubicSpline
from mathutils import Vector
from mathutils.bvhtree import BVHTree
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parents[1];O=B/'mandible-v3';O.mkdir(exist_ok=False)
HP=R/'user-human-preferred/recovery/restored-project/output/coordinated-orbital-scale-v53/head.npz';G=R/'human-preferred-v53-mechanics-20261011/cheek-perioral-insertion-v3/geometry.npz';h=np.load(HP);g=np.load(G);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
P=np.array([[0,-82,100],[20,-82,94],[32.5,-76,76],[41,-68,43],[44,-56,26],[49,-36,24],[51,-17,20],[52,-7,17],[52,0,17]],float)
rad=np.array([[12,5.5],[11.5,5],[9.5,4.5],[8.5,4.5],[9.5,4],[12,4],[8,4.5],[5.5,8],[.035,.035]],float)
left=P[1:][::-1].copy();left[:,0]*=-1;pp=np.r_[left,P];rr=np.r_[rad[1:][::-1],rad];ss=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pp,axis=0),axis=1))];s=np.linspace(ss[0],ss[-1],281);curve=CubicSpline(ss,pp,bc_type="natural")(s);r=PchipInterpolator(ss,rr)(s);tan=CubicSpline(ss,pp,bc_type="natural").derivative()(s);tan/=np.linalg.norm(tan,axis=1)[:,None]
# Parallel transport a plate axis along the entire continuous U body.
a=np.array([0.,0.,1.]);a-=tan[0]*(a@tan[0]);a/=np.linalg.norm(a);frames=[]
for t in tan:
 a-=t*(a@t);a/=np.linalg.norm(a);c=np.cross(t,a);c/=np.linalg.norm(c);frames.append([a.copy(),c])
frames=np.array(frames);th=np.arange(64)*2*np.pi/64;power=2/2.7;ca=np.sign(np.cos(th))*abs(np.cos(th))**power;si=np.sign(np.sin(th))*abs(np.sin(th))**power
v=curve[:,None]+r[:,None,0,None]*ca[None,:,None]*frames[:,0,None]+r[:,None,1,None]*si[None,:,None]*frames[:,1,None]
# An anterior ramal bearing gives an original coronoid-like crest without a separate union.
off=v-curve[:,None];anterior=np.clip(off[:,:,2]/np.maximum(r[:,None,0],1e-6),0,1)**3
v[:,:,1]+=9.*np.exp(-((curve[:,None,1]+23)/10)**2)*anterior*(np.abs(curve[:,None,0])>45)
v=v.reshape(-1,3);f=[]
for i in range(len(s)-1):
 for j in range(64):
  k=(j+1)%64;a=i*64+j;b=i*64+k;c=(i+1)*64+k;d=(i+1)*64+j;f.extend([[a,b,c],[a,c,d]])
for i in [0,len(s)-1]:
 center=len(v);v=np.r_[v,curve[i,None]]
 for j in range(64):
  k=(j+1)%64;f.append([center,i*64+k,i*64+j] if i==0 else [center,i*64+j,i*64+k])
f=np.array(f,int);tri=v[f];volume=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6
if volume<0:f=f[:,[0,2,1]];volume=-volume
skin=BVHTree.FromPolygons(h['vertices'].tolist(),h['triangles'][h['object_ids']==0].tolist(),all_triangles=True);bone=BVHTree.FromPolygons(g['bone_vertices'].tolist(),g['bone_triangles'].tolist(),all_triangles=True)
directions=[Vector((1,.173,.317)).normalized(),Vector((.193,1,.271)).normalized(),Vector((.127,.233,1)).normalized()]
def inside(p,d):
 o=Vector(p);n=0
 for _ in range(32):
  hit,_,_,_=skin.ray_cast(o,d,1000)
  if hit is None:return n%2
  o=hit+d*1e-4;n+=1
 return -1
par=np.array([[inside(p,d) for d in directions] for p in v]);bad=np.flatnonzero(~(par==1).all(1));dist=np.array([skin.find_nearest(Vector(p))[3] for p in v]);other=np.array([bone.find_nearest(Vector(p))[3] for p in v]);mesh=BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True);pairs=np.array(mesh.overlap(bone),int)
import sys;sys.path.insert(0,str(R/'human-preferred-v53-mechanics-20261011/src'));from triangle_sat import sat_intersect
cross=[]
for start in range(0,len(pairs),5000):
 p=pairs[start:start+5000];hit=sat_intersect(v[f[p[:,0]]],g['bone_vertices'][g['bone_triangles'][p[:,1]]],eps=1e-9);cross.extend(p[hit].tolist())
e=np.r_[f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]];_,ic,c=np.unique(np.sort(e,1),axis=0,return_inverse=True,return_counts=True);orient=np.bincount(ic,weights=np.where(e[:,0]<e[:,1],1,-1));assert (c==2).all() and (orient==0).all()
np.savez_compressed(O/'geometry.npz',vertices=v,triangles=f,curve=curve,frame_axes=frames,section_radii=r,source_stations=pp,source_station_radii=rr,skin_parity=par,skin_distance=dist,existing_support_distance=other)
report={'source_head_sha256':sha(HP),'existing_regional_geometry_sha256':sha(G),'geometry_sha256':sha(O/'geometry.npz'),'vertices':len(v),'triangles':len(f),'closed_oriented':True,'signed_volume':volume,'sampled_outside_vertices':len(bad),'outside_examples':bad[:30].tolist(),'minimum_sampled_skin_distance':float(dist.min()),'minimum_sampled_existing_support_distance':float(other.min()),'existing_support_triangle_crossings':len(cross),'crossing_examples':cross[:20],'authorship':'Original continuous transported-section U body, ramal bearing, condylar estimates and anterior ramal crests. No imported geometry.','scope':'Contact/attachment layout candidate only. No TMJ/hinge dynamics, dental occlusion, anatomical calibration, cortical thickness or marrow model. No exact human mandible claim.','admitted_layout':not len(bad) and not len(cross)};(O/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
