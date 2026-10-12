"""Explicit local one-way preferred-v53 skin correspondence for a new FEM case.
The exact selected skin remains the neutral output, including every uncovered
vertex. Skin is a render surface here, not a mass/contact/force boundary.
"""
from pathlib import Path
import sys,json,hashlib,numpy as np
from scipy.spatial import cKDTree
from mathutils import Vector
from mathutils.bvhtree import BVHTree
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parents[1];O=B/'cheek-skin-binding-v5';G=B/'cheek-remesh-v9/geometry.npz';C=B/'cheek-case-v3';H=R/'user-human-preferred/recovery/restored-project/output/coordinated-orbital-scale-v53/head.npz';S=H.parent/'anatomical_semantics/authored_lip_lid_semantics.npz';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(H)=='545b8ea2b292107dc2da9f1e90379989eb620c6ee5c2c50b3f9b5316cc77420e'
g=np.load(G);d=json.loads((C/'domain.json').read_text());raw=(C/'arrays.bin').read_bytes();a=d['arrays']['rest'];x=np.frombuffer(raw,dtype=a['dtype'],count=np.prod(a['shape']),offset=a['byte_offset']).reshape(a['shape'])*1000;bf=g['boundary_faces'];carrier=BVHTree.FromPolygons(x.tolist(),bf.tolist(),all_triangles=True)
z=np.load(H);v=z['vertices'];f=z['triangles'];skin=f[z['object_ids']==0];skin_ids=np.unique(skin)
# Area-weighted C2 spatial normal field, independent of the irregular original
# vertex tessellation. Near-coincident skin points receive a common direction.
# This is a display correspondence field, not a change to the rest geometry.
cent=v[skin].mean(1);pg=v[skin];fn=np.cross(pg[:,1]-pg[:,0],pg[:,2]-pg[:,0])
regional=(cent[:,0]>12)&(cent[:,0]<66)&(cent[:,1]>-66)&(cent[:,1]<15)&(cent[:,2]>75)
cent=cent[regional];fn=fn[regional];ntree=cKDTree(cent)
def spatial_normals(points):
 out=np.empty_like(points)
 for k,p in enumerate(points):
  ids=ntree.query_ball_point(p,3.5);r=np.linalg.norm(cent[ids]-p,axis=1)/3.5;w=(1-r)**4*(4*r+1);nn=np.sum(w[:,None]*fn[ids],axis=0);out[k]=nn/max(np.linalg.norm(nn),1e-30)
 return out
norm=np.zeros_like(v)
s=np.load(S);oralids=np.unique(np.r_[s['vermilion_vertex_ids'].ravel(),s['inner_labial_return_vertex_ids'].ravel()]);oral=cKDTree(v[oralids]);margin=8;lo=x.min(0)-margin;hi=x.max(0)+margin;eligible=skin_ids[((v[skin_ids]>=lo)&(v[skin_ids]<=hi)).all(1)&(v[skin_ids,0]>20)&(v[skin_ids,1]<2)]
S5=lambda u:np.clip(u,0,1)**3*(10-15*np.clip(u,0,1)+6*np.clip(u,0,1)**2)
norm[eligible]=spatial_normals(v[eligible])
rows=[];faces=[];points=[];dd=[];dots=[]
for vi in eligible:
 hit,no,fi,dist=carrier.find_nearest(Vector(v[vi]),margin)
 if hit is None:continue
 rows.append(vi);faces.append(fi);points.append(hit[:]);dd.append(dist);dots.append(float(norm[vi]@np.array(no)))
vid=np.array(rows);fi=np.array(faces);cp=np.array(points);dist=np.array(dd)
ix=bf[fi];tri=x[ix];A=tri[:,1]-tri[:,0];D=tri[:,2]-tri[:,0];P=cp-tri[:,0];a=np.sum(A*A,1);b=np.sum(A*D,1);c=np.sum(D*D,1);e=np.sum(P*A,1);f0=np.sum(P*D,1);den=a*c-b*b;u=(c*e-b*f0)/den;w=(a*f0-b*e)/den;weights=np.c_[1-u-w,u,w];assert weights.min()>-1e-4;weights=np.maximum(weights,0);weights/=weights.sum(1)[:,None];cp=np.einsum('ni,nij->nj',weights,tri)
# A shared C0 vertex-normal field replaces BVH per-triangle facet normals.
# Thus influence does not jump when nearest support crosses a carrier edge.
cn=np.zeros_like(x);bp=x[bf];fn=np.cross(bp[:,1]-bp[:,0],bp[:,2]-bp[:,0])
for k in range(3):np.add.at(cn,bf[:,k],fn)
cn/=np.maximum(np.linalg.norm(cn,axis=1)[:,None],1e-30);cn=np.einsum('ni,nij->nj',weights,cn[ix]);cn/=np.maximum(np.linalg.norm(cn,axis=1)[:,None],1e-30);dots=np.sum(norm[vid]*cn,axis=1)
oral_d=oral.query(v[vid])[0];normal_offset=np.sum((v[vid]-cp)*norm[vid],axis=1);gain=(1-S5((dist-3)/5))*S5((oral_d-1)/4)*S5(normal_offset/.75)*S5((v[vid,0]-20)/4)*S5((2-v[vid,1])/4);accept=gain>0
vid=vid[accept];fi=fi[accept];cp=cp[accept];dist=dist[accept];dots=dots[accept];gain=gain[accept];ix=ix[accept];weights=weights[accept]
assert not np.intersect1d(vid,oralids).size
O.mkdir(exist_ok=False);np.savez_compressed(O/'binding.npz',vertex_ids=vid.astype(np.int32),points=v[vid],indices=ix.astype(np.uint32),weights=weights,gain=gain,carrier_face_ids=fi,closest_points=cp,offset=v[vid]-cp,normal_alignment=dots,distance=dist,source_head_sha256=np.array(sha(H)),source_case_sha256=np.array(sha(C/'domain.json')),source_geometry_sha256=np.array(sha(G)))
report={'scope':'Local one-way skin rendering transfer from actual new right-cheek FEM. No skin force/contact feedback.','head_sha256':sha(H),'case_sha256':sha(C/'domain.json'),'binding_sha256':sha(O/'binding.npz'),'preferred_skin_vertices':len(skin_ids),'covered_vertices':len(vid),'uncovered_vertices':len(skin_ids)-len(vid),'covered_XYZ_bounds':[v[vid].min(0).tolist(),v[vid].max(0).tolist()],'distance_quantiles':np.percentile(dist,[0,50,95,100]).tolist(),'signed_normal_offset_quantiles':np.percentile(np.sum((v[vid]-cp)*norm[vid],1),[0,50,95,100]).tolist(),'gain_quantiles':np.percentile(gain,[0,50,95,100]).tolist(),'full_gain_vertices':int((gain>.99999).sum()),'mapping':'Exact closest triangle and nonnegative barycentric FEM supports; shared area-weighted C2 spatial skin-normal field (radius3.5); carrier orientation retained only as a diagnostic, avoiding radial angular gain stripes; neutral residual transported by volume-weighted nodal deformation gradients. C2 gain distance3..8, orientation measured but not used as a gain, positive exterior-normal offset0..0.75, X20..24 and Y2..minus2; finite oral geometry excluded with1..5 distance fade.','neutral_identity':'All selected skin points remain exact by construction; explicit zero-displacement path at exact neutral.','limits':['This is a regional cheek response, not smile, jaw, blink or neck animation.','All finite lips, inner oral return, globes, other cheek and uncovered skin remain fixed.','No anatomical or clinical calibration.']};(O/'binding_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
