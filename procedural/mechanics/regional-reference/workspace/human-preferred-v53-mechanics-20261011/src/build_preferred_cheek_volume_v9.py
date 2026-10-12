"""New original one-sided cheek volume in the exact preferred-v53 domain.
A swept tetrahedral bundle has an exact recovered bone-root footprint, new
rest cells, and a tapered subcutaneous path. This is an uncalibrated regional
construction, not a segmented whole facial muscle system.
"""
from pathlib import Path
import sys,json,hashlib,numpy as np
from scipy.interpolate import CubicSpline,PchipInterpolator
from scipy.spatial.transform import Rotation
from mathutils import Vector
from mathutils.bvhtree import BVHTree
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parents[1];O=B/'cheek-remesh-v9';C=R/'human-mechanics-20261010/output/bilateral-distributed-mechanics-v1';h=R/'user-human-preferred/recovery/restored-project/output/coordinated-orbital-scale-v53/head.npz'
d=json.loads((C/'domain.json').read_text());a=d['distributed_origin_attachments'][0];an=np.load(B/'reference-anatomy/anatomy.npz');old=an['tissue_vertices'];rootids=np.array(a['root_nodes']);root=old[rootids];n=len(root);cx=root.mean(0);lookup={int(k):i for i,k in enumerate(rootids)};rf=np.array([[lookup[int(k)] for k in f] for f in a['root_face_nodes']]);norm=np.cross(root[rf[:,1]]-root[rf[:,0]],root[rf[:,2]]-root[rf[:,0]]).sum(0);norm/=np.linalg.norm(norm)
z=np.load(h);v=z['vertices'];f=z['triangles'][z['object_ids']==0];tree=BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True)
def front(x,y):
 hit,no,fi,dd=tree.ray_cast(Vector((x,y,200)),Vector((0,0,-1)),200)
 if hit is None:raise ValueError('No preferred frontal support')
 return hit.z
bone_input=np.load(B/'rest-fit-v4/candidate_geometry.npz');bone_tree=BVHTree.FromPolygons(bone_input['bone_vertices'].tolist(),bone_input['bone_triangles'].tolist(),all_triangles=True)
def corridor(x,y,depth):
 skin=front(x,y);hit,no,fi,dd=bone_tree.ray_cast(Vector((x,y,200)),Vector((0,0,-1)),200);base=skin-depth
 if hit is not None:base=max(base,hit.z+1.4)
 if base>skin-.9:raise ValueError('Insufficient bone/skin centerline corridor')
 return base
# Artist-authored centerline follows the actual selected cheek to the lateral
# upper-mouth support. It deliberately excludes the old oral cavity topology.
xy=[(51.,-8.,2.8),(48.,-16.,4.3),(42.,-28.,3.8),(34.,-41.,3.3),(28.5,-49.,2.8)]
e1=np.cross(norm,[0,0,1]);e1/=np.linalg.norm(e1);e2=np.cross(norm,e1);uv=(root-cx)@np.array([e1,e2]).T
collar_center=cx+1.2*norm;collar=collar_center+.4*(root-cx)
coarse_center=cx+2.*norm
path=np.array([coarse_center]+[[x,y,corridor(x,y,depth)] for x,y,depth in xy]);length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))];total=length[-1];u=length/total
base_curve=CubicSpline(u,path,axis=0,bc_type=((1,norm*total),(1,(path[-1]-path[-2])/np.linalg.norm(path[-1]-path[-2])*total)))
def curve(u,derivative=0):
 point=base_curve(u);a=np.clip((-point[...,1]-8)/3,0,1);b=np.clip((-point[...,1]-17)/4,0,1);S=lambda v:v**3*(10-15*v+6*v*v);Sp=lambda v:30*v*v*(1-v)**2;wa=S(a);wb=1-S(b)
 if derivative:
  out=base_curve(u,derivative);out[...,2]+=.45*(-Sp(a)*wb/3+wa*Sp(b)/4)*out[...,1];return out
 point[...,2]+=.45*wa*wb;return point

rings=np.unique(np.r_[0,np.linspace(.008,.10,15),np.linspace(.13,1,34)]);centers=curve(rings);tangent=curve(rings,1);tangent/=np.linalg.norm(tangent,axis=1)[:,None]
# Healthy body sections have 12 rim vertices plus one center, instead of
# carrying 176 highly uneven bone-patch vertices through the entire muscle.
angles=np.arange(12)*2*np.pi/12;scale=PchipInterpolator([0,.08,.25,.55,.8,1],[.4,.65,1.2,1.5,1.2,.8])(rings);thickness=scale*.62
vertices=[*root,*collar];body_ids=[]
for u0,center,tan,width,depth in zip(rings,centers,tangent,scale,thickness):
 axis=np.cross(norm,tan);si=np.linalg.norm(axis);co=float(norm@tan);rot=np.eye(3) if si<1e-10 else Rotation.from_rotvec(axis/si*np.arctan2(si,co)).as_matrix();basis=np.array([e1,e2])@rot.T
 hit,no,fi,dd=tree.ray_cast(Vector((float(center[0]),float(center[1]),200)),Vector((0,0,-1)),200);outward=np.array(no);wide=np.cross(tan,outward);wide/=np.linalg.norm(wide);wide*=1 if wide@basis[0]>=0 else -1;phi=np.arctan2(tan@np.cross(basis[0],wide),basis[0]@wide);fade=float(np.clip(u0/.12,0,1));fade=fade**3*(10-15*fade+6*fade*fade);basis=basis@Rotation.from_rotvec(tan*phi*fade).as_matrix().T
 pp=np.array([center]+[center+width*np.cos(t)*basis[0]+depth*np.sin(t)*basis[1] for t in angles]);body_ids.append(np.arange(len(vertices),len(vertices)+13));vertices.extend(pp)
vertices=np.array(vertices);body_ids=np.array(body_ids)
# Root collar uses the exact footprint and matching globally consistent prisms.
def prism_rows(bottom,top,faces):
 out=[]
 for a,b,c in np.sort(faces,axis=1):out.extend([[bottom[a],bottom[b],bottom[c],top[c]],[bottom[a],bottom[b],top[b],top[c]],[bottom[a],top[a],top[b],top[c]]])
 return np.array(out)
first=prism_rows(np.arange(n),np.arange(n,2*n),rf)
# New adapter joins the fine cap to the coarse body. The source footprint is
# star-shaped about its declared center; all fan tetrahedra are checked below.
edges=np.concatenate([rf[:,[0,1]],rf[:,[1,2]],rf[:,[2,0]]]);_,ei,ec=np.unique(np.sort(edges,axis=1),axis=0,return_index=True,return_counts=True);edge=edges[ei[ec==1]];nxt={int(a):int(b) for a,b in edge};loop=[int(edge[0,0])]
while nxt[loop[-1]]!=loop[0]:loop.append(nxt[loop[-1]])
loop=np.array(loop);ang=np.mod(np.arctan2(uv[loop,1],uv[loop,0]),2*np.pi);order=np.argsort(ang);loop=loop[order];ang=ang[order];low=(loop+n).tolist();high=body_ids[0,1:].tolist();side=[];i=j=0
while i<len(low) or j<len(high):
 ni=ang[(i+1)%len(low)]+(2*np.pi if i+1>=len(low) else 0);nj=angles[(j+1)%len(high)]+(2*np.pi if j+1>=len(high) else 0)
 if ni<nj and i<len(low) or j>=len(high):side.append([low[i%len(low)],low[(i+1)%len(low)],high[j%len(high)]]);i+=1
 else:side.append([low[i%len(low)],high[(j+1)%len(high)],high[j%len(high)]]);j+=1
cap=np.array([[0,j+1,(j+1)%12+1] for j in range(12)])
adapter_boundary=np.r_[rf+n,body_ids[0][cap],np.array(side)];center_id=len(vertices);vertices=np.r_[vertices,((collar_center+coarse_center)/2)[None]];adapter=np.c_[adapter_boundary,np.full(len(adapter_boundary),center_id)]
body=np.concatenate([prism_rows(body_ids[k],body_ids[k+1],cap) for k in range(len(rings)-1)])
# Orient root/adapter cells from the literal polyhedron. Body orientation is
# set once against a straight reference prism, then retained through the sweep.
for part in [first,adapter]:
 pp=vertices[part];sgn=np.linalg.det(np.stack([pp[:,k]-pp[:,0] for k in [1,2,3]],-1));neg=sgn<0;part[neg,1:3]=part[neg,2:0:-1]
base=body[:36].copy();ref=np.array([[0,0,0]]+[[np.cos(t),np.sin(t),0] for t in angles]+[[0,0,1]]+[[np.cos(t),np.sin(t),1] for t in angles]);test=prism_rows(np.arange(13),np.arange(13,26),cap);pp=ref[test];neg=np.linalg.det(np.stack([pp[:,k]-pp[:,0] for k in [1,2,3]],-1))<0
for start in range(0,len(body),36):tmp=body[start:start+36];tmp[neg,1:3]=tmp[neg,2:0:-1]
tets=np.r_[first,adapter,body];p=vertices[tets];dm=np.stack([p[:,j]-p[:,0] for j in [1,2,3]],-1);det=np.linalg.det(dm);active_start=len(first)+len(adapter)
# Complete boundary with exact source-to-bone footprint mapping.
faces=np.concatenate([tets[:,[1,2,3]],tets[:,[0,3,2]],tets[:,[0,1,3]],tets[:,[0,2,1]]]);keys=np.sort(faces,axis=1);_,first,count=np.unique(keys,axis=0,return_index=True,return_counts=True);boundary=faces[first[count==1]]
par=[];dist=[];directions=[Vector((1,.173,.317)).normalized(),Vector((.193,1,.271)).normalized(),Vector((.127,.233,1)).normalized()]
def inside(p,di):
 origin=Vector(p);k=0
 for _ in range(32):
  hit,no,ix,dd=tree.ray_cast(origin,di,1000)
  if hit is None:return k%2
  k+=1;origin=hit+di*1e-4
 return -1
for point in vertices:
 par.append([inside(point,di) for di in directions]);dist.append(tree.find_nearest(Vector(point))[3])
par=np.array(par);dist=np.array(dist);out=(par==0).all(1);bone=np.load(B/'rest-fit-v4/candidate_geometry.npz')
O.mkdir(exist_ok=False);np.savez_compressed(O/'geometry.npz',vertices=vertices,tetrahedra=tets,boundary_faces=boundary,bone_vertices=bone['bone_vertices'],bone_triangles=bone['bone_triangles'],root_node_ids=np.arange(n),root_bone_vertex_ids=np.array(a['root_combined_bone_vertex_ids']),root_faces=rf,root_bone_face_ids=np.array(a['root_combined_bone_triangle_ids']),fixed_nodes=np.arange(2*n),path_controls=path,ring_parameters=rings,ring_centers=centers,ring_tangents=tangent,ring_scale=scale,active_tetrahedra=np.arange(active_start,len(tets)),body_ring_node_ids=body_ids,parity=par,skin_distance=dist,original_root_node_ids=rootids)
report={'status':'New regional cheek rest construction, unaccepted until validity/contact checks','preferred_head_sha256':hashlib.sha256(h.read_bytes()).hexdigest(),'nodes':len(vertices),'tetrahedra':len(tets),'rings':len(rings),'active_body_tetrahedra':len(body),'passive_root_adapter_tetrahedra':active_start,'root_nodes':n,'root_triangles':len(rf),'bone_footprint_exact':np.array_equal(vertices[:n],bone['bone_vertices'][a['root_combined_bone_vertex_ids']]),'minimum_signed_six_volume':float(det.min()),'nonpositive_tets':int((det<=0).sum()),'Dm_condition_quantiles':np.percentile(np.linalg.cond(dm),[0,50,95,99,100]).tolist(),'sampled_outside_skin_nodes':int(out.sum()),'outside_distance_quantiles':np.percentile(dist[out],[0,50,95,100]).tolist() if out.any() else [],'ambiguous_nodes':int((~out&~(par==1).all(1)).sum()),'path_length':total,'selected_skin_modified':False,'scope':'New right-cheek approximate active bundle and passive root transition. Oral, eyelid, jaw and neck mechanics are absent.'};(O/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
