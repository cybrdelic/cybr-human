#!/usr/bin/env python3
"""Original distributed-origin rest geometry feasibility. No main writes/solve.
Exact authored bone disk -> fixed translated collar -> strict-kernel fan ->
unaltered multi-triangle existing muscle boundary. All coordinates are design units.
"""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,itertools,json,resource,shutil,sys,time
import numpy as np
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'src/prototypes'))
from design_finite_origin_attachments import boundary,determinants,strict_tet_overlap
from mandibular_collision_audit import intersect_sat
from audit_connected_midface_boundary import shared_intersection
START=time.monotonic();TOL=1e-9
OUT=ROOT/'output/distributed-origin-attachment-feasibility-v1'
def guard():
 if time.monotonic()-START>110:raise RuntimeError('110 second budget')
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>300*1024:raise RuntimeError('300 MiB RSS budget')
 if shutil.disk_usage(ROOT).free<120*1024**2:raise RuntimeError('120 MiB disk reserve')
def area(tri):return np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
def ring(faces):
 e=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);u,ix,c=np.unique(np.sort(e,axis=1),axis=0,return_index=True,return_counts=True)
 if c.max()>2 or len(np.unique(faces))-len(u)+len(faces)!=1:return None
 b=e[ix[c==1]];succ={int(a):int(c) for a,c in b}
 if len(succ)!=len(b) or set(succ)!=set(succ.values()):return None
 first=int(b[0,0]);r=[first]
 while succ[r[-1]]!=first:
  r.append(succ[r[-1]])
  if len(r)>len(b):return None
 return np.array(r) if len(r)==len(b) else None
def rotate_by_angle(ids,x,u,w):
 center=x[ids].mean(0);angle=np.arctan2((x[ids]-center)@w,(x[ids]-center)@u);idx=np.argmin(abs(angle));return np.roll(ids,-idx)
def zipper(bottom,top,x):
 # Monotone perimeter correspondence allows unrelated loop tessellations.
 def parameter(ids):
  lengths=np.linalg.norm(x[np.roll(ids,-1)]-x[ids],axis=1);return np.r_[0,np.cumsum(lengths)/lengths.sum()]
 rb=parameter(bottom);rt=parameter(top);i=j=0;side=[]
 while i<len(bottom) or j<len(top):
  a=bottom[i%len(bottom)];b=top[j%len(top)]
  if i<len(bottom) and (j==len(top) or rb[i+1]<=rt[j+1]):side.append([a,bottom[(i+1)%len(bottom)],b]);i+=1
  else:side.append([a,top[(j+1)%len(top)],b]);j+=1
 return np.array(side)
def kernel(x,f):
 tri=x[f];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);nn=np.linalg.norm(normal,axis=1)
 if nn.min()<1e-12:return None
 normal/=nn[:,None];off=np.einsum('ij,ij->i',normal,tri[:,0]);res=linprog([0,0,0,-1],A_ub=np.c_[normal,np.ones(len(f))],b_ub=off,bounds=[(None,None)]*3+[(0,None)],method='highs',options={'time_limit':3,'presolve':True,'threads':1})
 if not res.success:return None
 actual=np.min(off-normal@res.x[:3]);return res.x[:3],float(actual)
def meanratio(x,t):
 q=x[t];vol=determinants(x,t)/6;ed=np.array(list(itertools.combinations(range(4),2)));sq=np.sum((q[:,ed[:,0]]-q[:,ed[:,1]])**2,axis=(1,2));return 12*(3*vol)**(2/3)/sq

LAST_SHELL=None
def construct(side,patch,tipfaces,bv,bf,v):
 global LAST_SHELL
 LAST_SHELL=None
 bverts=np.unique(bf[patch]);nr=len(bverts);bfmap={int(q):i for i,q in enumerate(bverts)};rootfaces=np.array([[bfmap[int(q)] for q in f] for f in bf[patch]])
 root=bv[bverts].copy();root[:,0]*=side
 if side<0:rootfaces=rootfaces[:,[0,2,1]]
 tids=np.unique(tipfaces);nt=len(tids);mapping={int(q):2*nr+i for i,q in enumerate(tids)};tip=np.array([[mapping[int(q)] for q in f[::-1]] for f in tipfaces]);rootloop=ring(rootfaces);tiploop=ring(tip)
 if rootloop is None or tiploop is None:return None,'not_disk'
 direction=np.array([.6*side,0.,.8]);direction/=np.linalg.norm(direction)
 normal=np.cross(root[rootfaces[:,1]]-root[rootfaces[:,0]],root[rootfaces[:,2]]-root[rootfaces[:,0]]);normal/=np.linalg.norm(normal,axis=1)[:,None]
 if min(normal@direction)<.1:return None,'wrong_collar_direction'
 # Fixed collar thickness is explicit original design, not anatomy calibration.
 collar_length=.4;x=np.r_[root,root+collar_length*direction,v[tids]]
 u=np.array([0.,1.,0.]);u-=direction*(u@direction);u/=np.linalg.norm(u);w=np.cross(direction,u)
 rl=rotate_by_angle(rootloop+nr,x,u,w);tl=rotate_by_angle(tiploop,x,u,w)
 sidefaces=zipper(rl,tl,x);cap=np.r_[rootfaces[:,[0,2,1]]+nr,tip,sidefaces]
 # All cap edges must pair in opposite orientations.
 ce=np.concatenate([cap[:,[0,1]],cap[:,[1,2]],cap[:,[2,0]]]);_,ci,cc=np.unique(np.sort(ce,axis=1),axis=0,return_inverse=True,return_counts=True);co=np.bincount(ci,weights=np.where(ce[:,0]<ce[:,1],1,-1))
 if np.any(cc!=2) or np.any(co):return None,'inconsistent_cap'
 LAST_SHELL=dict(x=x,cap=cap,rootfaces=rootfaces,tip=tip,sidefaces=sidefaces,rootloop=rl,tiploop=tl,nr=nr,direction=direction)
 k=kernel(x,cap)
 if k is None:return None,'empty_kernel'
 center,radius=k
 if radius<1e-5:return None,'non_strict_kernel'
 x=np.r_[x,center[None]];tt=[]
 for f in rootfaces:
  a,b,c=sorted(f);A,B,C=np.array([a,b,c])+nr;small=np.array([[a,b,c,A],[b,c,A,B],[c,B,C,A]])
  if determinants(x,small)[0]<0:small[:,[1,2]]=small[:,[2,1]]
  tt.extend(small.tolist())
 collar_count=len(tt);tt=np.r_[np.array(tt),np.c_[np.full(len(cap),len(x)-1),cap]];det=determinants(x,tt)
 if det.min()<=0:return None,'nonpositive_tets'
 return dict(x=x,t=tt,root_faces=rootfaces,first_faces=rootfaces+nr,transition_shell=cap,source_bone_vertices=bverts,source_bone_triangles=patch,source_tissue_vertices=tids,tip_local_faces=tip,nr=nr,collar_tets=collar_count,kernel=center,kernel_radius=radius,direction=direction,collar_length=collar_length,quality=meanratio(x,tt)),None

def main():
 guard();OUT.mkdir(exist_ok=True);a=np.load(ROOT/'output/bilateral-midface-bone-aware-v2/rest_geometry.npz');v=a['vertices'];t=a['tetrahedra'];f,owner=boundary(t);b=np.load(ROOT/'output/authored-bone-support-v1/bone_support.npz');d=json.loads((ROOT/'output/bilateral-mechanics-v1/domain.json').read_text());land=json.loads((ROOT/'output/authored-bone-support-v1/landmarks.json').read_text())['zygomaticus_major_origin_patch_center'];patch=np.array(land['attachment_patch_triangles']);deep=np.isin(f,a['midface_column_nodes'][:,-1]).all(1);trials=[];success=[]
 for si,side in enumerate([1,-1]):
  ids=np.flatnonzero(deep&np.isin(owner,d['muscles'][si]['tetrahedra']));ids=ids[np.argsort(np.linalg.norm(v[f[ids]].mean(1)-np.array(land['point'])*[side,1,1],axis=1))]
  for n in range(3,13):
   guard();chosen=ids[:n];mesh,error=construct(side,patch,f[chosen],b['vertices'],b['triangles'],v);rec=dict(side=side,tissue_faces=chosen.tolist(),tissue_patch_area=float(area(v[f[chosen]]).sum()),error=error)
   if mesh is not None:
    rec.update(kernel_radius=mesh['kernel_radius'],minimum_quality=float(mesh['quality'].min()),quality_1pct=float(np.percentile(mesh['quality'],1)),quality_10pct=float(np.percentile(mesh['quality'],10)),nodes=len(mesh['x']),tets=len(mesh['t']),minimum_tet_volume=float(determinants(mesh['x'],mesh['t']).min()/6));success.append((rec,mesh))
   trials.append(rec);print(json.dumps(rec),flush=True)
  # Store the strongest visibility radius first; detailed overlap audit is separate.
  cand=[q for q in success if q[0]['side']==side]
  if cand:
   rec,mesh=max(cand,key=lambda q:q[0]['kernel_radius']);name='right' if side>0 else 'left';np.savez_compressed(OUT/(name+'_kernel_candidate.npz'),**{k:q for k,q in mesh.items() if isinstance(q,np.ndarray)},tissue_source_face_ids=np.array(rec['tissue_faces']),collar_tet_count=mesh['collar_tets'],root_node_count=mesh['nr']);print('BEST '+json.dumps(rec),flush=True)
 (OUT/'kernel_search.json').write_text(json.dumps(dict(status='Construction candidates only; overlap and quality acceptance pending',root_triangles=len(patch),root_area=float(area(b['vertices'][b['triangles'][patch]]).sum()),trials=trials,elapsed_seconds=time.monotonic()-START,rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024),indent=2)+'\n')
 assert sum(p.stat().st_size for p in OUT.iterdir())<3*1024**2
if __name__=='__main__':main()
