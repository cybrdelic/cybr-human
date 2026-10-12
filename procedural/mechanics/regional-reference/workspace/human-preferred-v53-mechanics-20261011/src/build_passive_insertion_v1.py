"""Original shared-node passive cheek insertion in the preferred-v53 domain.
A superficial passive patch is connected by tetrahedra to the actual existing
muscle terminal cap. Its peripheral support truncates the omitted face domain.
"""
from pathlib import Path
import json,hashlib,numpy as np
from scipy.optimize import linprog
from mathutils import Vector
from mathutils.bvhtree import BVHTree
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parents[1];IP=B/'cheek-remesh-v9/geometry.npz';G=np.load(IP);HP=R/'user-human-preferred/recovery/restored-project/output/coordinated-orbital-scale-v53/head.npz';H=np.load(HP);O=B/'cheek-loaded-insertion-v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(HP)=='545b8ea2b292107dc2da9f1e90379989eb620c6ee5c2c50b3f9b5316cc77420e'
v=G['vertices'];ids=G['body_ring_node_ids'][-1];bf=G['boundary_faces'];basefaces=bf[np.isin(bf,ids).all(1)];mp={int(k):i for i,k in enumerate(ids)};base=np.array([[mp[int(k)] for k in f] for f in basefaces]);bv=v[ids];skin=BVHTree.FromPolygons(H['vertices'].tolist(),H['triangles'][H['object_ids']==0].tolist(),all_triangles=True)
e=np.concatenate([base[:,[0,1]],base[:,[1,2]],base[:,[2,0]]]);_,ix,c=np.unique(np.sort(e,axis=1),axis=0,return_index=True,return_counts=True);ed=e[ix[c==1]];nx={int(a):int(b) for a,b in ed};loop=[int(ed[0,0])]
while nx[loop[-1]]!=loop[0]:loop.append(nx[loop[-1]])
a=(bv[loop]-bv[0])[:,:2];ang=np.unwrap(np.arctan2(a[:,1]/np.std(a[:,1]),a[:,0]/np.std(a[:,0])));direction=np.sign(np.median(np.diff(ang)));theta=ang[0]+direction*np.arange(12)*2*np.pi/12
cx,cy,rx,ry,depth=31.,-50.,3.5,4.,.6
pts=[];exact_surface=[]
for x,y in [(cx,cy)]+[(cx+r*rx*np.cos(t),cy+r*ry*np.sin(t)) for r in [.5,1.] for t in theta]:
 h,no,fi,dd=skin.ray_cast(Vector((x,y,200)),Vector((0,0,-1)),200)
 if h is None:raise ValueError('Missing exact selected skin')
 p=np.array(h);exact_surface.append(p.copy());p[2]-=depth;pts.append(p)
pts=np.array(pts);verts=np.r_[bv,pts];tf=[[13,14+i,14+(i+1)%12] for i in range(12)]
for i in range(12):
 j=(i+1)%12;tf.extend([[14+i,26+i,26+j],[14+i,26+j,14+j]])
side=[]
for i in range(12):
 a,b=loop[i],loop[(i+1)%12];A,BB=26+i,26+(i+1)%12;side.extend([[a,b,BB],[a,BB,A]])
faces=np.r_[base[:,[0,2,1]],np.array(tf),np.array(side)];edgefaces={}
for fi,f in enumerate(faces):
 for a,b in zip(f,np.roll(f,-1)):edgefaces.setdefault(tuple(sorted((int(a),int(b)))),[]).append((fi,1 if a<b else -1))
signs=np.zeros(len(faces),int);signs[0]=1;stack=[0]
while stack:
 fi=stack.pop()
 for a,b in zip(faces[fi],np.roll(faces[fi],-1)):
  pair=edgefaces[tuple(sorted((int(a),int(b))))];assert len(pair)==2;s=1 if a<b else -1
  for fj,sj in pair:
   if fj!=fi and not signs[fj]:signs[fj]=-signs[fi]*s*sj;stack.append(fj)
faces[signs<0]=faces[signs<0][:,[0,2,1]];tri=verts[faces];norm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);norm/=np.linalg.norm(norm,axis=1)[:,None];bc=np.cross(bv[base[0,1]]-bv[base[0,0]],bv[base[0,2]]-bv[base[0,0]])
if norm[0]@bc>0:faces=faces[:,[0,2,1]];norm=-norm
ans=linprog([0,0,0,-1],A_ub=np.c_[norm,np.ones(len(norm))],b_ub=np.sum(norm*verts[faces[:,0]],1),bounds=[(None,None)]*4,method='highs');assert ans.success and ans.x[3]>.1
newv=np.r_[v,pts,ans.x[None,:3]];index=np.r_[ids,np.arange(len(v),len(v)+len(pts))];newt=np.c_[np.full(len(faces),len(newv)-1),index[faces]];tets=np.r_[G['tetrahedra'],newt];p=newv[tets];D=np.stack([p[:,k]-p[:,0] for k in [1,2,3]],-1);det=np.linalg.det(D);assert det.min()>0
ff=np.concatenate([tets[:,[1,2,3]],tets[:,[0,3,2]],tets[:,[0,1,3]],tets[:,[0,2,1]]]);_,ix,count=np.unique(np.sort(ff,axis=1),axis=0,return_index=True,return_counts=True);assert count.max()==2;boundary=ff[ix[count==1]];topfaces=index[faces[len(base):len(base)+len(tf)]];topids=np.arange(len(v),len(v)+len(pts));rim=topids[-12:]
directions=[Vector((1,.173,.317)).normalized(),Vector((.193,1,.271)).normalized(),Vector((.127,.233,1)).normalized()]
def inside(point,direction):
 o=Vector(point);k=0
 for _ in range(32):
  h,no,fi,dd=skin.ray_cast(o,direction,1000)
  if h is None:return k%2
  k+=1;o=h+direction*1e-4
 return -1
par=np.array([[inside(p,d) for d in directions] for p in newv]);assert (par==1).all(),np.flatnonzero(~(par==1).all(1))
q={k:G[k] for k in G.files};q.update(vertices=newv,tetrahedra=tets,boundary_faces=boundary,bone_fixed_nodes=G['fixed_nodes'],fixed_nodes=np.r_[G['fixed_nodes'],rim],parity=par,skin_distance=np.array([skin.find_nearest(Vector(p))[3] for p in newv]),insertion_shared_nodes=ids,insertion_tetrahedra=np.arange(len(G['tetrahedra']),len(tets)),insertion_surface_node_ids=topids,insertion_surface_triangles=topfaces,insertion_fixed_nodes=rim,insertion_exact_skin_points=np.array(exact_surface),insertion_kernel=ans.x[:3]);O.mkdir(exist_ok=False);np.savez_compressed(O/'geometry.npz',**q)
report={'head_sha256':sha(HP),'source_geometry_sha256':sha(IP),'geometry_sha256':sha(O/'geometry.npz'),'nodes':len(newv),'tetrahedra':len(tets),'new_passive_tetrahedra':len(newt),'shared_terminal_nodes':len(ids),'superficial_proxy_nodes':len(topids),'fixed_truncation_rim_nodes':len(rim),'preferred_skin_modified':False,'all_sampled_nodes_inside':True,'minimum_signed_six_volume':float(det.min()),'insertion_Dm_condition_quantiles':np.percentile(np.linalg.cond(D[-len(newt):]),[0,50,95,100]).tolist(),'star_kernel_inradius':float(ans.x[3]),'skin_proxy_Z_offset':depth,'patch_center_xy':[cx,cy],'patch_radii_xy':[rx,ry],'scope':'New passive shared-node insertion with a superficially placed coarse skin proxy; rim fixed as a regional truncation. Actual reciprocal constitutive loading must be verified by the coupled solve.','limits':['Rigid root/first collar retained; no resolved bone-interface traction.','Passive layer and materials authored, uncalibrated.','Full surrounding face, jaw and oral/lid mechanics absent.','No skin display pose until coupled checks and correspondence pass.']};(O/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
