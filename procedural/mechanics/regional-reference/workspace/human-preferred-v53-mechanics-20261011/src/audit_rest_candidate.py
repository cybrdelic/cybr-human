"""Literal mesh sections and sampled containment of an unaccepted rest candidate."""
from pathlib import Path
import sys,json,hashlib,gc
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parents[1];folder=sys.argv[sys.argv.index('--')+1] if '--' in sys.argv else 'rest-fit-v4';O=B/folder
h=R/'user-human-preferred/recovery/restored-project/output/coordinated-orbital-scale-v53/head.npz';z=np.load(h);v=z['vertices'];f=z['triangles'][z['object_ids']==0];q=np.load(O/'candidate_geometry.npz');x=q['vertices'];b=q['boundary_faces'];bv=q['bone_vertices'];bt=q['bone_triangles'];old=q['original_rest']
def sections(v,f,axis,level):
 allseg=[]
 for start in range(0,len(f),20000):
  tri=v[f[start:start+20000]];dd=tri[:,:,axis]-level;sel=(dd.min(1)<0)&(dd.max(1)>0);tri=tri[sel];dd=dd[sel];cuts=np.full((len(tri),3,3),np.nan)
  for j,(a,c) in enumerate([(0,1),(1,2),(2,0)]):
   active=dd[:,a]*dd[:,c]<0;u=dd[active,a]/(dd[active,a]-dd[active,c]);cuts[active,j]=tri[active,a]*(1-u[:,None])+tri[active,c]*u[:,None]
  good=np.isfinite(cuts[:,:,0]);ok=good.sum(1)==2;allseg.append(cuts[ok][good[ok]].reshape(-1,2,3))
 return np.concatenate(allseg)
sec={}
for label,ax,val in [('X25',0,25.),('X43',0,43.),('Yminus25',1,-25.),('Yminus54',1,-54.)]:
 for name,p,ff in [('preferred',v,f),('original',old,b),('fitted',x,b),('bone',bv,bt)]:sec[label+'_'+name]=sections(p,ff,ax,val)
np.savez_compressed(O/'literal_sections.npz',**sec)
tree=BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True);del v,f,z;gc.collect()
dirs=[Vector((1.,.173,.317)).normalized(),Vector((.193,1.,.271)).normalized(),Vector((.127,.233,1.)).normalized()]
def inside(p,di):
 origin=Vector(p);n=0
 for _ in range(40):
  hit,no,ix,dd=tree.ray_cast(origin,di,1000)
  if hit is None:return n%2
  n+=1;origin=hit+di*1e-4
 return -1
report={}
for name,points in [('tissue_nodes',x),('bone_samples',np.r_[bv[::8],bv[bt[::8]].mean(1)])]:
 par=[];dist=[];signed=[];closest=[]
 for p in points:
  hit,no,ix,dd=tree.find_nearest(Vector(p));dist.append(dd);signed.append(float((Vector(p)-hit).dot(no)));closest.append(hit[:]);par.append([inside(p,di) for di in dirs])
 par=np.array(par);dist=np.array(dist);signed=np.array(signed);out=(par==0).all(1);on=dist<1e-4
 report[name]={'count':len(points),'all_three_inside':int((par==1).all(1).sum()),'all_three_outside':int(out.sum()),'near_boundary':int(on.sum()),'outside_over_point1':int((out&(dist>.1)).sum()),'outside_distance_quantiles':np.percentile(dist[out],[0,50,95,100]).tolist() if out.any() else [],'ambiguous':int((~(par==1).all(1)&~out).sum())}
 np.savez_compressed(O/(name+'_containment.npz'),points=points,parity=par,distance=dist,signed_nearest_normal=signed,closest_points=np.array(closest));print(name,report[name],flush=True)
report['status']='Sampled rest-fit audit; does not accept inverted or overlapping volume';report['head_sha256']=hashlib.sha256(h.read_bytes()).hexdigest();report['candidate_sha256']=hashlib.sha256((O/'candidate_geometry.npz').read_bytes()).hexdigest();(O/'containment_report.json').write_text(json.dumps(report,indent=2)+'\n')
