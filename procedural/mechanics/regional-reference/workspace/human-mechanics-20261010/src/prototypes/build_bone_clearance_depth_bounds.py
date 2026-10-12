"""Bone-derived conservative rest-thickness bounds for complete coarse prisms.
If every column of an outward triangle has displacement norm below d-clearance,
its full linearly interpolated volume stays in that triangle's bone-free tube.
This is conservative authoring, not contact mechanics or measured anatomy.
"""
from pathlib import Path
import sys,argparse,json,hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mandibular_collision_audit import broad_pairs,intersect_sat
from zygomatic_support_build import closest_on_triangle

def edge_distance(a,b,c,d):
 # Segment–segment constrained convex quadratic: interior stationary point,
 # then all four segment boundaries. Parallel cases use boundaries only.
 u=b-a;v=d-c;w=a-c;aa=np.dot(u,u);bb=v@u;cc=np.einsum('ij,ij->i',v,v);dd=w@u;ee=np.einsum('ij,ij->i',v,w);den=aa*cc-bb*bb
 s=np.divide(bb*ee-cc*dd,den,out=np.zeros_like(den),where=den>1e-20);t=np.divide(aa*ee-bb*dd,den,out=np.zeros_like(den),where=den>1e-20)
 delta=w+s[:,None]*u-t[:,None]*v;best=np.einsum('ij,ij->i',delta,delta);best[(den<=1e-20)|(s<0)|(s>1)|(t<0)|(t>1)]=np.inf
 for point in [a,b]:
  alpha=np.clip(np.einsum('ij,ij->i',point-c,v)/cc,0,1);r=point-(c+alpha[:,None]*v);best=np.minimum(best,np.einsum('ij,ij->i',r,r))
 for point in [c,d]:
  alpha=np.clip((point-a)@u/aa,0,1);r=point-(a+alpha[:,None]*u);best=np.minimum(best,np.einsum('ij,ij->i',r,r))
 return best

def distances(a,b):
 best=np.full(len(b),np.inf)
 for point in a:best=np.minimum(best,closest_on_triangle(point,b)[1])
 aa=np.broadcast_to(a,b.shape)
 for j in range(3):best=np.minimum(best,closest_on_triangle(b[:,j],aa)[1])
 for j in range(3):
  for k in range(3):best=np.minimum(best,edge_distance(a[j],a[(j+1)%3],b[:,k],b[:,(k+1)%3]))
 return np.sqrt(best)
p=argparse.ArgumentParser();p.add_argument('mesh',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args();g=np.load(args.mesh);N=len(g['midface_column_nodes']);v=g['vertices'];tri=g['outer_triangles'];parent=g['surface_triangle_parent_faces'];mask=g['face_parts'][parent]!=20;tri=tri[mask];top=v[tri];radius=16.;bounds=np.stack([top.min(1)-radius,top.max(1)+radius],1);bpath=Path(__file__).resolve().parents[2]/'output/authored-bone-support-v1/bone_support.npz';b=np.load(bpath);right=b['vertices'][b['triangles']];left=right.copy();left[:,:,0]*=-1;left=left[:,[0,2,1]];bone=np.concatenate([right,left]);limits=np.full(len(tri),radius);nearest=np.full(len(tri),-1,int);narrow=0
for i,ids in broad_pairs(bounds,bone,cell=8.):
 if len(ids):
  narrow+=len(ids)
  if intersect_sat(top[i],bone[ids]).any():raise ValueError('Actual bone intersects fixed coarse outer skin triangle '+str(i))
  ds=distances(top[i],bone[ids]);j=ds.argmin()
  if ds[j]<radius:limits[i]=ds[j];nearest[i]=ids[j]
cap=np.full(N,radius)
for i,t in enumerate(tri):np.minimum.at(cap,t,limits[i]-.20)
fixed=np.unique(np.r_[g['protected_boundary_columns'],g['oral_interface_columns']]);depth=np.linalg.norm(g['inner_carrier_displacement'],axis=1);viol=fixed[depth[fixed]>=cap[fixed]];args.output.mkdir(parents=True,exist_ok=False);np.savez_compressed(args.output/'bone_depth_bounds.npz',column_norm_caps=cap,triangle_clearance=limits,nearest_bone_triangle=nearest,outer_triangles=tri)
r={'schema':'conservative-whole-prism-bone-tube-bound-v1','source_mesh_sha256':hashlib.sha256(args.mesh.read_bytes()).hexdigest(),'bone_sha256':hashlib.sha256(bpath.read_bytes()).hexdigest(),'outer_triangles':len(tri),'candidate_triangle_pairs':narrow,'minimum_exact_triangle_distance':float(limits.min()),'rest_margin_design_units':.20,'column_cap_percentiles':np.percentile(cap,[0,25,50,75,100]).tolist(),'fixed_column_bound_violations':[{'column':int(i),'fixed_depth':float(depth[i]),'cap':float(cap[i])} for i in viol],'limits':['Floating-point geometric distance bounds, not exact arithmetic','Caps are conservative Euclidean displacement bounds; no claim of measured tissue thickness','Only original authored bilateral regional bones covered; central bone/cartilage support incomplete','This does not implement dynamic contact or certify self-intersection absence']};(args.output/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)
