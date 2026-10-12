#!/usr/bin/env python3
"""Original editable regional bones: authored fields, no imported anatomy meshes.
Uses a common rigid regional surface with semantic zygoma/maxilla labels.
Spatial values are design choices; literature guides organization, not scale.
"""
from pathlib import Path
import argparse, ctypes, hashlib, json, shutil, subprocess, sys
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
ROOT=Path(__file__).resolve().parents[1]
HEAD=ROOT.parent/'human-reference-pipeline-20261009/output/native-connected-head-study/shared-head-v16/head.npz'
HEAD_SHA='f4fa73042fab1534b45c0982b47ae668b16eb990c2029e6a06cfc9b94b01df14'
PARAMS=Path(__file__).with_name('zygomatic_support_parameters.json')
REGIONS={0:'zygomatic_body',1:'zygomatic_arch',2:'maxillary_facial_shell',3:'inferior_orbital_rim',4:'lateral_orbital_rim'}

def smooth_min(a,b,k):
 h=np.maximum(k-abs(a-b),0)/k
 return np.minimum(a,b)-h*h*k*.25

def capsule_field(q,desc):
 cps=np.array(desc['points']);r=np.array(desc['radii']);t=np.r_[0,np.linalg.norm(np.diff(cps,axis=0),axis=1).cumsum()];curve=PchipInterpolator(t,cps)(np.linspace(0,t[-1],desc['interpolation_samples']))
 x=q/r;curve=curve/r;field=np.full(len(q),np.inf)
 for a,b in zip(curve[:-1],curve[1:]):
  e=b-a;u=np.clip(((x-a)*e).sum(1)/(e*e).sum(),0,1);field=np.minimum(field,np.linalg.norm(x-(a+u[:,None]*e),axis=1))
 return (field-1)*r.min()

def polygon_sd(q,poly):
 d=np.full(len(q),np.inf);inside=np.zeros(len(q),bool)
 for a,b in zip(poly,np.roll(poly,-1,axis=0)):
  e=b-a;t=np.clip(((q-a)*e).sum(1)/(e*e).sum(),0,1);d=np.minimum(d,np.linalg.norm(q-(a+t[:,None]*e),axis=1))
  crosses=(a[1]>q[:,1])!=(b[1]>q[:,1]);den=b[1]-a[1]
  if abs(den)>1e-14:inside^=crosses&(q[:,0]<(b[0]-a[0])*(q[:,1]-a[1])/den+a[0])
 return np.where(inside,-d,d)

def fields(q,p):
 body=p['zygomatic_body'];r=np.array(body['radii']);s=q-np.array(body['center']);s[:,2]-=body['z_shear_x']*s[:,0]+body['z_shear_y']*s[:,1];fb=(np.linalg.norm(s/r,axis=1)-1)*r.min()
 maxilla=p['maxillary_face'];x,y,z=q.T
 front=maxilla['base_z']+maxilla['x_quadratic']*(x-15)**2+maxilla['y_linear']*(y+20)-maxilla['upper_recess_depth']*np.exp(-((y+6)/9)**2-((x-20)/12)**2)-maxilla['canine_fossa_depth']*np.exp(-((x-23)/6)**2-((y+23)/8)**2)
 fs=np.maximum(polygon_sd(q[:,:2],np.array(maxilla['outline_xy'])),np.maximum(z-front,front-maxilla['thickness']-z))
 values=np.stack([fb,capsule_field(q,p['zygomatic_arch']),fs,capsule_field(q,p['inferior_orbital_rim']),capsule_field(q,p['lateral_orbital_rim'])],axis=1)
 result=values[:,0].copy()
 for j in range(1,values.shape[1]):result=smooth_min(result,values[:,j],p['union_blend'])
 globe=p['globe'];result=np.maximum(result,globe['radius']+globe['reserved_clearance']-np.linalg.norm(q-np.array(globe['center']),axis=1))
 return result,values.argmin(1).astype(np.uint8)

def marching_tets(axes,values):
 # Freudenthal tetrahedra share every voxel face diagonal. Intersections use
 # canonical grid-edge identities, so neighboring cells share mesh vertices.
 shape=tuple(map(len,axes));index=np.arange(np.prod(shape)).reshape(shape);corner=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
 templates=np.array([[0,1,2,6],[0,2,3,6],[0,3,7,6],[0,7,4,6],[0,4,5,6],[0,5,1,6]])
 corners=np.stack([index[a:a+shape[0]-1,b:b+shape[1]-1,c:c+shape[2]-1].ravel() for a,b,c in corner],axis=1)
 vals=values.ravel();active=(vals[corners].min(1)<0)&(vals[corners].max(1)>0);ts=corners[active][:,templates].reshape(-1,4)
 points=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3);verts=[];faces=[];lookup={}
 def intersect(i,j):
  key=(min(i,j),max(i,j))
  if key not in lookup:
   lookup[key]=len(verts);u=vals[i]/(vals[i]-vals[j]);verts.append(points[i]+u*(points[j]-points[i]))
  return lookup[key]
 for tet in ts:
  ins=tet[vals[tet]<0];outs=tet[vals[tet]>=0]
  if len(ins) in (0,4):continue
  start=len(faces)
  if len(ins)==1:faces.append([intersect(ins[0],o) for o in outs])
  elif len(ins)==3:faces.append([intersect(outs[0],i) for i in ins])
  else:
   a,b=ins;c,d=outs;ac=intersect(a,c);ad=intersect(a,d);bc=intersect(b,c);bd=intersect(b,d);faces.extend([[ac,ad,bc],[bc,ad,bd]])
  outward=points[outs].mean(0)-points[ins].mean(0)
  for fi in range(start,len(faces)):
   a,b,c=(verts[k] for k in faces[fi])
   if np.dot(np.cross(b-a,c-a),outward)<0:faces[fi][1],faces[fi][2]=faces[fi][2],faces[fi][1]
 return np.array(verts),np.array(faces,dtype=np.int32)

def closest_on_triangle(q,tris):
 a,b,c=np.moveaxis(tris,1,0);u=b-a;v=c-a;r=q-a;uu=(u*u).sum(1);uv=(u*v).sum(1);vv=(v*v).sum(1);ru=(r*u).sum(1);rv=(r*v).sum(1);den=uu*vv-uv*uv
 s=(ru*vv-rv*uv)/np.maximum(den,1e-30);t=(rv*uu-ru*uv)/np.maximum(den,1e-30);point=a+s[:,None]*u+t[:,None]*v;ds=((point-q)**2).sum(1);ds[(s<0)|(t<0)|(s+t>1)|(den<1e-24)]=np.inf
 for x,y in [(a,b),(b,c),(c,a)]:
  edge=y-x;w=np.clip(((q-x)*edge).sum(1)/np.maximum((edge*edge).sum(1),1e-30),0,1);p=x+w[:,None]*edge;d=((p-q)**2).sum(1);m=d<ds;point[m]=p[m];ds[m]=d[m]
 return point,ds

def native_audit(skinv,skinf,points,out):
 source=Path(__file__).with_name('zygomatic_support_distance.cpp');libpath=out/'support_distance.so';subprocess.run(['g++','-O2','-shared','-fPIC','-std=c++17',str(source),'-o',str(libpath)],check=True)
 lib=ctypes.CDLL(str(libpath));dp=ctypes.POINTER(ctypes.c_double);ip=ctypes.POINTER(ctypes.c_int64);bp=ctypes.POINTER(ctypes.c_uint8)
 lib.clay_contact_create.argtypes=[dp,ip,ctypes.c_uint32];lib.clay_contact_create.restype=ctypes.c_void_p;lib.clay_contact_destroy.argtypes=[ctypes.c_void_p];lib.support_audit.argtypes=[ctypes.c_void_p,dp,ctypes.c_uint32,dp,bp]
 v=np.ascontiguousarray(skinv,dtype=np.float64);f=np.ascontiguousarray(skinf,dtype=np.int64);q=np.ascontiguousarray(points,dtype=np.float64);d=np.empty(len(q));inside=np.empty((len(q),3),np.uint8);ptr=lib.clay_contact_create(v.ctypes.data_as(dp),f.ctypes.data_as(ip),len(f))
 if not ptr:raise RuntimeError('Native BVH allocation failed')
 try:lib.support_audit(ptr,q.ctypes.data_as(dp),len(q),d.ctypes.data_as(dp),inside.ctypes.data_as(bp))
 finally:lib.clay_contact_destroy(ptr)
 libpath.unlink()
 return d,inside

def build(out,parameters):
 if shutil.disk_usage(out.parent).free<125*1024**2:raise RuntimeError('Disk reserve below 125 MiB; refusing build')
 if hashlib.sha256(HEAD.read_bytes()).hexdigest()!=HEAD_SHA:raise RuntimeError('Source identity mismatch')
 p=json.loads(parameters.read_text());lo,hi=np.array(p['grid_bounds']);axes=[np.arange(a,b+p['grid_spacing']*.5,p['grid_spacing']) for a,b in zip(lo,hi)]
 q=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1);shape=q.shape[:-1];field,_=fields(q.reshape(-1,3),p);v,f=marching_tets(axes,field.reshape(shape));c=v[f].mean(1)
 n=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]);_,labels=fields(c,p)
 edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1);unique,counts=np.unique(edges,axis=0,return_counts=True);graph=coo_matrix((np.ones(len(unique)),(unique[:,0],unique[:,1])),shape=(len(v),len(v)));components,_=connected_components(graph,directed=False);area=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)*.5
 directed=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);_,ei=np.unique(np.sort(directed,axis=1),axis=0,return_inverse=True);winding=int((np.bincount(ei,weights=np.where(directed[:,0]<directed[:,1],1,-1))!=0).sum());signed_volume=float(np.einsum('ij,ij->i',v[f[:,0]],np.cross(v[f[:,1]],v[f[:,2]])).sum()/6)
 out.mkdir(parents=True,exist_ok=True);head=np.load(HEAD);distance,parity=native_audit(head['vertices'],head['triangles'][head['object_ids']==0],c,out);radii=np.linalg.norm(v[f]-c[:,None,:],axis=2).max(1);lower=distance-radii
 _,globe_ds=closest_on_triangle(np.array(p['globe']['center']),v[f]);globe_clear=np.sqrt(globe_ds)-p['globe']['radius']
 # Every triangle lies inside a skin-free ball around a centroid proven interior
 # by three independent non-axis-aligned parity rays. This bounds its WHOLE
 # area, rather than only checking vertices or centroid samples.
 certified=(parity==1).all(1)&(lower>0)
 records={}
 for name,seed in p['landmark_seeds'].items():
  nearest,ds=closest_on_triangle(np.array(seed),v[f]);k=int(ds.argmin());point=nearest[k];normal=np.cross(v[f[k,1]]-v[f[k,0]],v[f[k,2]]-v[f[k,0]]);normal/=np.linalg.norm(normal)
  records[name]={'seed':seed,'point':point.tolist(),'triangle':k,'region':REGIONS[int(labels[k])],'normal':normal.tolist(),'seed_projection_distance':float(np.sqrt(ds[k])),'status':'authored landmark, not measured anatomy'}
 origin=records['zygomaticus_major_origin_patch_center'];originpoint=np.array(origin['point']);patch=np.flatnonzero((labels==0)&(np.linalg.norm(c-originpoint,axis=1)<4.0)&(n[:,2]>0));origin['attachment_patch_triangles']=patch.tolist();origin['patch_radius_design_units']=4.0
 report={'schema':'human-original-regional-bone-audit-v1','source_head_sha256':HEAD_SHA,'parameters_sha256':hashlib.sha256(parameters.read_bytes()).hexdigest(),'vertices':len(v),'triangles':len(f),'components':int(components),'open_edges':int((counts==1).sum()),'nonmanifold_edges':int((counts>2).sum()),'inconsistent_winding_edges':winding,'signed_volume_design_units_cubed':signed_volume,'minimum_triangle_area':float(area.min()),'bbox':[v.min(0).tolist(),v.max(0).tolist()],'skin_audit':{'tested_triangle_centroids':len(c),'three_ray_parity_outside_or_ambiguous':int((~(parity==1).all(1)).sum()),'centroid_min_exact_surface_distance':float(distance.min()),'triangles_wholly_contained_by_distance_ball':int(certified.sum()),'uncertified_triangles':int((~certified).sum()),'minimum_whole_triangle_clearance_lower_bound':float(lower.min()),'method':'Exact nearest native-skin triangle distance with BVH; three oblique parity rays per centroid; Lipschitz distance bound d(centroid)-max(vertex radius) certifies entire triangle. Floating-point, source mesh assumed closed.'},'globe_audit':{'minimum_exact_triangle_clearance':float(globe_clear.min()),'native_globe_center':p['globe']['center'],'native_globe_radius':p['globe']['radius']},'regions':{REGIONS[k]:int((labels==k).sum()) for k in REGIONS},'anatomical_status':'Original anatomically organized regional approximation. No measurements, scans, segmented or atlas geometry imported; no calibrated biological dimensions or mechanical material claim.','mechanics_status':'Rigid support surface and attachment landmarks prepared; no solver integration or contact forces claimed.'}
 (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');(out/'landmarks.json').write_text(json.dumps(records,indent=2)+'\n');(out/'parameters.json').write_text(parameters.read_text());np.savez_compressed(out/'bone_support.npz',vertices=v,triangles=f,region_ids=labels,component_ids=np.zeros(len(f),np.uint8),bone_ids=np.where((labels==2)|((labels==3)&(c[:,0]<35)),1,0).astype(np.uint8),certified_clearance_lower_bound=lower)
 with (out/'bone_support.obj').open('w') as h:
  h.write('# Original bounded regional bones; design units, not calibrated mm\n');h.writelines('v %.7f %.7f %.7f\n'%tuple(x) for x in v)
  for k in REGIONS:
   h.write('g '+REGIONS[k]+'\n');h.writelines('f %d %d %d\n'%tuple(x+1) for x in f[labels==k])
 print(json.dumps(report),flush=True)
 if (counts!=2).any() or components!=1 or winding or signed_volume<=0 or not certified.all() or globe_clear.min()<1.5:raise RuntimeError('Support audit did not pass; retained outputs for inspection only')
 return v,f,labels

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,default=ROOT/'output/authored-bone-support-v1');a.add_argument('--parameters',type=Path,default=PARAMS);args=a.parse_args();build(args.output,args.parameters)
