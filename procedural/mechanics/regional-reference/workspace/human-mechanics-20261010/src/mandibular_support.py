#!/usr/bin/env python3
"""Original closed editable mandibular support. No scanned/atlas geometry.
Run under shared heavy lock. Native head is read-only and SHA-pinned.
"""
from pathlib import Path
import argparse, hashlib, json, shutil, sys
import numpy as np
from scipy.interpolate import PchipInterpolator
ROOT=Path(__file__).resolve().parents[1]
HEAD=ROOT.parent/'human-reference-pipeline-20261009/output/native-connected-head-study/shared-head-v16/head.npz'
HEAD_SHA='f4fa73042fab1534b45c0982b47ae668b16eb990c2029e6a06cfc9b94b01df14'
# s, x, z, inferior y, superior y, radial half-thickness. Entirely authored.
CONTROLS=[
 [0.,0,103,-96,-70,5.7], [.17,18,100,-94,-68,5.5],
 [.34,32,84,-89,-66,5.0], [.50,40,62,-83,-62,4.4],
 [.62,44,47,-80,-43,4.2], [.70,45.5,39,-78,-16,3.6],
 [.75,47,33,-75,-12,3.3], [.82,49,26,-67,-31,3.0],
 [.88,51,21,-53,-32,3.0], [.94,52,16,-32,-19,3.8],
 [1.,53,11,-16,-6,5.6]]
SOURCES=[
 dict(title='Ahn et al. (2019), Relationship between the maxillofacial skeletal pattern and the morphology of the mandibular symphysis',url='https://pmc.ncbi.nlm.nih.gov/articles/PMC6533179/',doi='10.4041/kjod.2019.49.3.170',use='Structural distinction of midline symphysis, mandibular body, gonion at ramus/body junction and condylion. No measurements, scans, figures or geometry copied.'),
 dict(title='Harn and Shackelford (1982), Further evaluation of the superficial and deep tendons of the human temporalis muscle',url='https://pubmed.ncbi.nlm.nih.gov/7072995/',doi='10.1002/ar.1092020413',use='Primary dissection evidence for complexity of temporalis/fascial attachments along ramus; named attachment patches here are intentionally coarse engineering simplifications.'),
 dict(title='Mezey et al. (2022), The human masseter muscle revisited',url='https://pubmed.ncbi.nlm.nih.gov/34863910/',doi='10.1016/j.aanat.2021.151879',use='Primary anatomical evidence for zygomatic-to-coronoid muscle attachments and elevating/retracting action. We do not model or claim all masseter layers.'),
 dict(title='Throckmorton and Dechow (1994), In vitro strain measurements in the condylar process of the human mandible',url='https://pubmed.ncbi.nlm.nih.gov/7741655/',doi='10.1016/0003-9969(94)90017-5',use='Primary loading experiment motivates distinct muscle and joint supports. Neither reported material properties nor forces calibrate this demo.')]

SOURCES.append(dict(title='De-Ary-Pires et al. (2003), The human digastric muscle: patterns and variations with clinical and surgical correlations',url='https://pubmed.ncbi.nlm.nih.gov/14575275/',doi='10.1016/S0940-9602(03)80110-3',use='Primary bilateral cadaver study documents paired digastric architecture and mandibular depression/hyoid elevation. Two cables here simplify only the anterior-belly direction; no subject morphometry copied.'))
FIXED_ORIGINS={
 'right_masseter_origin':dict(position=[61.9456997,-4.31080084,46.86545234],source='../authored-bone-support-v1/landmarks.json',kind='Surface-projected inferior arch landmark from coordinated original regional support'),
 'left_masseter_origin':dict(position=[-61.9456997,-4.31080084,46.86545234],source='Mirror of right arch landmark; left arch mesh not yet authored',kind='Symmetry proposal'),
 'right_digastric_hyoid_side_support':dict(position=[10,-114,49],kind='Authored fixed tendon-side engineering support; no hyoid mesh'),
 'left_digastric_hyoid_side_support':dict(position=[-10,-114,49],kind='Authored fixed tendon-side engineering support; no hyoid mesh')}

def reserve():
 free=shutil.disk_usage(ROOT).free
 if free<120*1024**2:raise RuntimeError(f'Disk reserve below 120 MiB: {free}')

def build_mesh(controls=CONTROLS,rings_per_side=54,cross_samples=20):
 c=np.array(controls,float);s=np.linspace(-1,1,2*rings_per_side+1);a=abs(s)
 data=PchipInterpolator(c[:,0],c[:,1:],axis=0)(a)
 x,z,lo,hi,w=data.T;x*=np.sign(s)
 # Tangent along the complete arch and horizontal outward normal.
 tangent=np.c_[np.gradient(x,s),np.zeros(len(s)),np.gradient(z,s)]
 radial=np.c_[-tangent[:,2],np.zeros(len(s)),tangent[:,0]]
 radial/=np.linalg.norm(radial,axis=1)[:,None]
 phi=np.linspace(0,2*np.pi,cross_samples,endpoint=False)
 center=np.c_[x,(lo+hi)/2,z]
 vv=center[:,None,:]+radial[:,None,:]*(w[:,None,None]*np.cos(phi)[None,:,None])
 vv[:,:,1]+=((hi-lo)/2)[:,None]*np.sin(phi)[None,:]
 v=vv.reshape(-1,3);f=[];regions=[]
 for i in range(len(s)-1):
  q=(abs(s[i])+abs(s[i+1]))/2
  region=0 if q<.2 else 1 if q<.60 else 2 if q<.89 else 3
  for j in range(cross_samples):
   k=(j+1)%cross_samples;a0=i*cross_samples+j;b=i*cross_samples+k;cc=(i+1)*cross_samples+k;d=(i+1)*cross_samples+j
   f.extend([[a0,b,cc],[a0,cc,d]]);regions.extend([region,region])
 # Shared end ring center fans make one connected, closed two-manifold.
 v=np.vstack([v,center[0],center[-1]])
 for end,idx in [(0,len(v)-2),(len(s)-1,len(v)-1)]:
  for j in range(cross_samples):
   k=(j+1)%cross_samples
   face=[idx,end*cross_samples+k,end*cross_samples+j] if end==0 else [idx,end*cross_samples+j,end*cross_samples+k]
   f.append(face);regions.append(3)
 f=np.asarray(f,np.int32)
 if np.einsum('ij,ij->i',v[f[:,0]],np.cross(v[f[:,1]],v[f[:,2]])).sum()<0:f=f[:,[0,2,1]]
 return v,f,np.asarray(regions,np.uint8)

def topology(v,f):
 directed=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);edges=np.sort(directed,axis=1)
 unique,inv,count=np.unique(edges,axis=0,return_inverse=True,return_counts=True)
 orientation=np.bincount(inv,weights=np.where(directed[:,0]<directed[:,1],1,-1))
 area=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)/2
 adj=[[] for _ in v]
 for a,b in unique:adj[a].append(b);adj[b].append(a)
 visited=set();components=0
 for i in range(len(v)):
  if i in visited:continue
  components+=1;todo=[i]
  while todo:
   j=todo.pop()
   if j in visited:continue
   visited.add(j);todo.extend(k for k in adj[j] if k not in visited)
 return dict(vertices=len(v),triangles=len(f),connected_components=components,open_edges=int((count==1).sum()),nonmanifold_edges=int((count>2).sum()),inconsistent_winding_edges=int((orientation!=0).sum()),minimum_triangle_area=float(area.min()),signed_volume=float(np.einsum('ij,ij->i',v[f[:,0]],np.cross(v[f[:,1]],v[f[:,2]])).sum()/6),euler_characteristic=int(len(v)-len(unique)+len(f)))

class NativeSkinRay:
 """Exact +Z ray intersections with original triangles, XY uniform-bin broad phase."""
 def __init__(self,v,f,cell=3.):
  self.p=v[f];self.cell=cell;self.bins={}
  mn=np.floor(self.p[:,:,:2].min(1)/cell).astype(int);mx=np.floor(self.p[:,:,:2].max(1)/cell).astype(int)
  # Only candidate bins in mandibular region are required, skin remains unmodified.
  use=(mx[:,0]>=-22)&(mn[:,0]<=22)&(mx[:,1]>=-37)&(mn[:,1]<=3)
  for i in np.where(use)[0]:
   for x in range(max(-22,mn[i,0]),min(22,mx[i,0])+1):
    for y in range(max(-37,mn[i,1]),min(3,mx[i,1])+1):self.bins.setdefault((x,y),[]).append(i)
 def check(self,points):
  inside=[];gaps=[];counts=[]
  for q0 in points:
   q=q0.copy();q[:2]+=[1.123e-8,2.739e-8]
   ids=self.bins.get(tuple(np.floor(q[:2]/self.cell).astype(int)),[])
   p=self.p[ids];aa=p[:,1,:2]-p[:,0,:2];bb=p[:,2,:2]-p[:,0,:2];r=q[:2]-p[:,0,:2]
   det=aa[:,0]*bb[:,1]-aa[:,1]*bb[:,0];good=abs(det)>1e-14
   aa=aa[good];bb=bb[good];r=r[good];p=p[good];det=det[good]
   u=(r[:,0]*bb[:,1]-r[:,1]*bb[:,0])/det;w=(aa[:,0]*r[:,1]-aa[:,1]*r[:,0])/det
   ok=(u>=0)&(w>=0)&(u+w<=1)
   zz=(p[:,0,2]+u*(p[:,1,2]-p[:,0,2])+w*(p[:,2,2]-p[:,0,2]))[ok]
   zz=np.sort(zz);zz=zz[np.r_[True,np.diff(zz)>1e-7]] if len(zz) else zz
   count=int((zz>q[2]).sum());inside.append(count%2==1);counts.append(count)
   gaps.append(float(np.min(abs(zz-q[2]))) if len(zz) else float('inf'))
  return np.array(inside),np.array(gaps),np.array(counts)

def make_landmarks(v):
 targets={
  'symphysis_anterior':[0,-85,108.7],'menton':[0,-96,103],
  'geniohyoid_origin':[0,-88,97.7],
  'right_gonion':[44.5,-79,40],'right_masseter_insertion':[47,-67,42],
  'right_medial_pterygoid_insertion':[41,-68,42],
  'right_coronoid_tip':[47,-12,33],
  'right_temporalis_insertion':[46,-23,36],
  'right_condylar_center':[53,-11,11],
  'right_digastric_fossa':[7,-91,98.5],
  'right_lateral_pterygoid_insertion':[49,-24,18]}
 for key,p in list(targets.items()):
  if key.startswith('right_'):targets['left_'+key[6:]]=[-p[0],p[1],p[2]]
 out={}
 for k,q in targets.items():
  idx=int(np.argmin(np.sum((v-q)**2,axis=1)))
  # Joint centers are interior rigid-body landmarks rather than surface vertices.
  p=np.array(q) if 'condylar_center' in k else v[idx]
  out[k]=dict(position=p.tolist(),nearest_vertex_id=idx,kind='interior_hinge_center' if 'condylar_center' in k else 'authored_surface_landmark',interpretation='Anatomically named engineering landmark, not measured subject anatomy.')
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'output/authored-mandible-v1');ap.add_argument('--head',type=Path,default=HEAD);ap.add_argument('--controls',type=Path);a=ap.parse_args();reserve()
 if hashlib.sha256(a.head.read_bytes()).hexdigest()!=HEAD_SHA:raise ValueError('Source head hash differs')
 controls=json.loads(a.controls.read_text())['control_rows'] if a.controls else CONTROLS
 v,f,regions=build_mesh(controls);report=topology(v,f)
 assert report['open_edges']==report['nonmanifold_edges']==report['inconsistent_winding_edges']==0
 assert report['connected_components']==1 and report['signed_volume']>0
 a.output.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(a.output/'mandible_rest.npz',vertices=v,triangles=f,region_ids=regions)
 with (a.output/'mandible_rest.obj').open('w') as fp:
  fp.write('# Original authored mandibular support. X lateral Y up Z anterior. Design units.\n')
  for p in v:fp.write('v %.8f %.8f %.8f\n'%tuple(p))
  for t in f:fp.write('f %d %d %d\n'%tuple(t+1))
 lm=make_landmarks(v)
 landmarks=dict(schema='cybr-authored-mandible-landmarks-v1',coordinate_frame='X lateral (right positive), Y superior, Z anterior',units='authored design units; no physical measurement calibration',source_head_sha256=HEAD_SHA,hinge=dict(pivot=[0,-11,11],axis=[1,0,0],positive_rotation='right-hand +X rotation opens jaw: anterior points travel downward',constrained_dofs=5,active_dofs=1,limit_degrees=[0,18],approximation='Fixed bilateral hinge; no condylar translation, articular disc or calibrated TMJ mechanics'),landmarks=lm,fixed_attachment_origins=FIXED_ORIGINS)
 (a.output/'landmarks.json').write_text(json.dumps(landmarks,indent=2)+'\n')
 (a.output/'authoring_controls.json').write_text(json.dumps(dict(columns=['arch_fraction','x','z','inferior_y','superior_y','radial_half_thickness'],control_rows=controls,rings_per_side=54,cross_samples=20,regions=['symphysis','body','angle_ramus_coronoid','condylar_neck_support'],method='Original editable PCHIP loft with variable vertical extent and thickness; connected U-shaped body, separated coronoid/notch/condylar profiles. No external geometry.',units='authored design units'),indent=2)+'\n')
 native=np.load(a.head);hv=native['vertices'];hf=native['triangles'][native['object_ids']==0]
 ray=NativeSkinRay(hv,hf)
 # Every bone vertex, face centroid, and edge midpoint; sampling is not a continuous collision proof.
 edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
 q=np.vstack([v,v[f].mean(1),v[edges].mean(1)]);inside,gap,count=ray.check(q)
 report['native_skin_sample_check']=dict(method='Exact +Z parity against original full native skin triangles at all bone vertices, face centroids, unique-edge midpoints',samples=len(q),inside_count=int(inside.sum()),outside_count=int((~inside).sum()),minimum_z_ray_clearance=float(gap.min()),clearance_is_euclidean=False,continuous_triangle_collision_certified=False,outside_positions=q[~inside][:40].tolist(),limits='Sample containment only; no complete skin/bone intersection or skin coupling claimed.')
 report['source_head_sha256']=HEAD_SHA;report['source_head_unchanged']=hashlib.sha256(a.head.read_bytes()).hexdigest()==HEAD_SHA
 report['bounds']=[v.min(0).tolist(),v.max(0).tolist()]
 (a.output/'geometry_report.json').write_text(json.dumps(report,indent=2)+'\n')
 (a.output/'sources.json').write_text(json.dumps(dict(geometry_provenance='All mandible vertices generated here from authored control rows. The native skin is used only for placement/containment tests. No scan, atlas, external mesh or paper figure imported.',primary_sources=SOURCES),indent=2)+'\n')
 print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
