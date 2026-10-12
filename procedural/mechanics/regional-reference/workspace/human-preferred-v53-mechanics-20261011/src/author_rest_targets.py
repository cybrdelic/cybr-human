"""Original regional rest-fit targets against immutable preferred-v53 skin.
Targets are authored geometric correspondence, not a physical trajectory.
"""
from pathlib import Path
import sys,numpy as np,json,hashlib,gc
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from scipy.spatial import cKDTree
R=Path(__file__).resolve().parents[2];BASE=Path(__file__).resolve().parents[1];OUT=BASE/'rest-targets-v1'
case=R/'human-mechanics-20261010/output/bilateral-distributed-mechanics-v1';head=R/'user-human-preferred/recovery/restored-project/output/coordinated-orbital-scale-v53/head.npz';sem=head.parent/'anatomical_semantics/authored_lip_lid_semantics.npz';bp=R/'human-mechanics-20261010/output/bilateral-cavity-annulus-v1/native-binding-v1/native_surface_binding.npz';op=R/'human-mechanics-20261010/inputs/native_oral_charts_v16_mucosa24.npz';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(head)=='545b8ea2b292107dc2da9f1e90379989eb620c6ee5c2c50b3f9b5316cc77420e'
d=json.loads((case/'domain.json').read_text());raw=(case/d['binary']).read_bytes();q=d['arrays']['rest'];rest=np.frombuffer(raw,dtype=q['dtype'],count=np.prod(q['shape']),offset=q['byte_offset']).reshape(q['shape'])*1000
with np.load(head) as q:v=q['vertices'];f=q['triangles'];oid=q['object_ids']
ff=f[oid==0];edges=np.concatenate([ff[:,[0,1]],ff[:,[1,2]],ff[:,[2,0]]]);_,inv,count=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True);wind=np.bincount(inv,weights=np.where(edges[:,0]<edges[:,1],1,-1));closure={'skin_faces':len(ff),'boundary_edges':int((count==1).sum()),'nonmanifold_edges':int((count>2).sum()),'inconsistent_winding':int(((count==2)&(wind!=0)).sum())};print('closure',closure,flush=True);assert closure['boundary_edges']==closure['nonmanifold_edges']==closure['inconsistent_winding']==0
del edges,inv,count,wind;gc.collect()
tree=BVHTree.FromPolygons(v.tolist(),ff.tolist(),all_triangles=True)
with np.load(bp) as z:b={k:z[k] for k in z.files}
with np.load(op) as z:o={k:z[k] for k in z.files}
with np.load(sem) as z:s={k:z[k] for k in z.files}
rows=np.flatnonzero((b['weights']==1).any(1)&((b['weights']>0).sum(1)==1));nodes=b['indices'][rows,np.argmax(b['weights'][rows],axis=1)].astype(int);native=b['vertex_ids'][rows]
assert len(np.unique(nodes))==len(nodes)==2737
lookup={int(node):(i,j,region) for region,k in [(1,'external'),(2,'mucosal')] for i,row in enumerate(o[k+'_vertex_ids']) for j,node in enumerate(row)}
# Shared aperture belongs to both charts and has identical target coordinates.
def angles(points):
 c=(points[:,:2].min(0)+points[:,:2].max(0))/2;r=np.ptp(points[:,:2],axis=0)/2
 theta=np.unwrap(np.arctan2((points[:,1]-c[1])/r[1],(points[:,0]-c[0])/r[0]));assert (np.diff(theta)>0).all();return theta
oldtheta=angles(o['external_positions'][16]);newtheta=angles(s['vermilion_points'][0]);col=np.interp((oldtheta-newtheta[0])%(2*np.pi)+newtheta[0],np.r_[newtheta,newtheta[0]+2*np.pi],np.arange(385))
mask=np.zeros(len(v),bool);mask[s['vermilion_vertex_ids']]=True;mask[s['inner_labial_return_vertex_ids']]=True
oral_faces=np.flatnonzero(mask[f].all(1));face_lookup={tuple(sorted(map(int,f[fi]))):int(fi) for fi in oral_faces}
def on_chart(grid,row,column):
 row=float(np.clip(row,0,len(grid)-1));i=min(int(row),len(grid)-2);r=row-i;j=int(column)%384;c=column-int(column);a,bb,cc,dd=map(int,[grid[i,j],grid[i,(j+1)%384],grid[i+1,(j+1)%384],grid[i+1,j]])
 if tuple(sorted((a,bb,cc))) in face_lookup:
  ids,ws=((a,bb,cc),(1-c,c-r,r)) if c>=r else ((a,cc,dd),(1-r,c,r-c))
 else:
  ids,ws=((a,bb,dd),(1-r-c,c,r)) if r+c<=1 else ((bb,cc,dd),(1-r,r+c-1,1-c))
 fi=face_lookup[tuple(sorted(ids))];weight_by_id=dict(zip(ids,ws));ws=np.array([weight_by_id[int(ii)] for ii in f[fi]])
 return ws@v[f[fi]],fi,ws
positions=[];face_ids=[];weights=[];regions=[];dist=[]
for node,ni in zip(nodes,native):
 if int(ni) in lookup:
  row,angle,region=lookup[int(ni)]
  grid=s['vermilion_vertex_ids'] if region==1 else s['inner_labial_return_vertex_ids']
  rr=30*(1-row/16) if region==1 else row*.5
  target,fi,ww=on_chart(grid,rr,col[angle])
 else:
  region=0;hit,no,local_face,dd=tree.find_nearest(Vector(rest[node]));fi=int(np.flatnonzero(oid==0)[local_face]);target=np.array(hit)
  tri=v[f[fi]];e1=tri[1]-tri[0];e2=tri[2]-tri[0];ee=target-tri[0];D=np.array([[e1@e1,e1@e2],[e1@e2,e2@e2]]);ab=np.linalg.solve(D,np.array([ee@e1,ee@e2]));ww=np.r_[1-ab.sum(),ab];ww=np.maximum(ww,0);ww/=ww.sum();target=ww@tri
 positions.append(target);face_ids.append(fi);weights.append(ww);regions.append(region);dist.append(np.linalg.norm(target-rest[node]))
positions=np.array(positions);regions=np.array(regions);weights=np.array(weights)
rigid=np.unique(np.concatenate([q['fixed_collar_nodes'] for q in d['distributed_origin_attachments']]));assert not np.intersect1d(nodes,rigid).size
targets=rest.copy();targets[nodes]=positions
# Bone support: a C2 inward clearance map near the preferred skin. Actual roots
# and their adjacent bone face vertices remain exact and are independently checked.
br=(case/'bones.bin').read_bytes();bv=np.frombuffer(br,dtype='<f8',count=d['bones']['vertex_count']*3).reshape(-1,3)*1000;bt=np.frombuffer(br,dtype='<u4',count=d['bones']['triangle_count']*3,offset=d['bones']['vertex_count']*24).reshape(-1,3)
root_vertices=np.unique(np.concatenate([a['root_combined_bone_vertex_ids'] for a in d['distributed_origin_attachments']]));root_distance=cKDTree(bv[root_vertices]).query(bv)[0];z=np.clip(root_distance/3,0,1);root_fade=z*z*z*(10-15*z+6*z*z)
new_bone=bv.copy();signed=[]
for i,point in enumerate(bv):
 hit,no,fi,dd=tree.find_nearest(Vector(point));no=np.array(no);ss=float((point-np.array(hit))@no);signed.append(ss);z=np.clip((ss+3)/3,0,1);blend=z*z*z*(10-15*z+6*z*z);new_bone[i]-=root_fade[i]*(ss+3)*blend*no if ss>-3 else 0
root_vertices=np.unique(np.concatenate([a['root_combined_bone_vertex_ids'] for a in d['distributed_origin_attachments']]));root_change=float(np.linalg.norm(new_bone[root_vertices]-bv[root_vertices],axis=1).max());assert root_change==0,root_change
p0=bv[bt];p1=new_bone[bt];n0=np.cross(p0[:,1]-p0[:,0],p0[:,2]-p0[:,0]);n1=np.cross(p1[:,1]-p1[:,0],p1[:,2]-p1[:,0]);lens=np.linalg.norm(n1,axis=1);cos=np.sum(n0*n1,axis=1)/np.maximum(np.linalg.norm(n0,axis=1)*lens,1e-30)
OUT.mkdir(exist_ok=False);np.savez_compressed(OUT/'targets.npz',original_rest=rest,targets=targets,surface_nodes=nodes,rigid_nodes=rigid,target_v53_face_ids=np.array(face_ids),target_v53_weights=weights,target_regions=regions,original_native_anchor_ids=native,preferred_head_sha256=np.array(sha(head)),bone_vertices=new_bone,bone_triangles=bt,original_bone_vertices=bv,root_bone_vertex_ids=root_vertices)
report={'status':'Authored rest-fit target only; not an accepted volume or pose','head_sha256':sha(head),'skin_closure':closure,'source_case':str(case.relative_to(R)),'source_domain_sha256':sha(case/'domain.json'),'surface_nodes':len(nodes),'rigid_root_collar_nodes':len(rigid),'coordinate_frame':'X lateral,Y up,Z front; authored units, common with original FEM after solver scaling0.001','target_regions':{str(k):{'count':int((regions==k).sum()),'motion_quantiles':np.percentile(np.array(dist)[regions==k],[0,50,95,100]).tolist()} for k in np.unique(regions)},'oral_mapping':'Monotone normalized outer-border angle correspondence; original preferred exact triangle barycentric samples. Native outer row0..16 maps preferred vermilion row30..0. Native mucosa row0..24 maps preferred finite inner return row0..12. Explicit authored correspondence, not same material IDs.','midface_mapping':'Nearest exact preferred skin triangle, independently stored face/barycentric coordinates. Subject to fold/containment review.','bone_target':{'changed_vertices':int(np.any(new_bone!=bv,axis=1).sum()),'maximum_motion':float(np.linalg.norm(new_bone-bv,axis=1).max()),'root_motion_exact':root_change,'minimum_face_normal_cosine':float(cos.min()),'minimum_triangle_area':float(lens.min()/2),'nonpositive_normal_count':int((cos<=0).sum()),'method':'C2 signed-nearest-skin inward target with3-unit near-surface range and C2 three-unit spatial holdout at actual root vertices; retained exact bone-origin footprint'} ,'limits':['No old equilibrium state can be reused after a reference change.','Bone target geometry and full tissue reference require independent overlap/contact/quality admission.','No selected skin coordinate has been modified.']};(OUT/'target_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
