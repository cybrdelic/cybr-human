"""Read-only local skin-crossing and tissue/skin separation gate for sparse poses."""
from pathlib import Path
import sys,json,hashlib,gc,numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
sys.path.insert(0,str(Path(__file__).resolve().parent));from triangle_sat import sat_intersect
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parents[1];D=B/'cheek-loaded-skin-states-v1';O=B/'cheek-loaded-skin-validity-v1';H=R/'user-human-preferred/recovery/restored-project/output/coordinated-orbital-scale-v53/head.npz';G=B/'cheek-loaded-insertion-v1/geometry.npz';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
z=np.load(H);v=z['vertices'];f=z['triangles'];skin_ids=np.flatnonzero(z['object_ids']==0);skin_f=f[skin_ids];g=np.load(G);bf=g['boundary_faces'];bind=np.load(B/'cheek-loaded-skin-binding-v1/binding.npz');vid=bind['vertex_ids'];changed=np.zeros(len(v),bool);changed[vid]=True;affected=np.flatnonzero(changed[f].any(1));lo=v[vid].min(0)-2;hi=v[vid].max(0)+2;tri=v[skin_f];ri=skin_ids[((tri.max(1)>=lo)&(tri.min(1)<=hi)).all(1)];del tri

def tree(points,faces):
 ids,inv=np.unique(faces,return_inverse=True);return BVHTree.FromPolygons(points[ids].tolist(),inv.reshape(-1,3).tolist(),all_triangles=True)
def crossings(points):
 a=tree(points,f[affected]);b=tree(points,f[ri]);pairs=np.array(a.overlap(b),dtype=np.int64);del a,b;gc.collect();i=affected[pairs[:,0]];j=ri[pairs[:,1]];pairs=np.unique(np.sort(np.c_[i,j],axis=1),axis=0);pairs=pairs[pairs[:,0]!=pairs[:,1]];pairs=pairs[(changed[f[pairs[:,0]]].any(1)|changed[f[pairs[:,1]]].any(1))];inc=(f[pairs[:,0],:,None]==f[pairs[:,1],None,:]).any((1,2));hits=[];ihits=[]
 for start in range(0,len(pairs),5000):
  p=pairs[start:start+5000];ic=inc[start:start+5000];aa=points[f[p[:,0]]];bb=points[f[p[:,1]]]
  if (~ic).any():hits.extend(p[~ic][sat_intersect(aa[~ic],bb[~ic],eps=1e-10)].tolist())
  if ic.any():
   aa=aa[ic];bb=bb[ic];ma=aa.mean(1);mb=bb.mean(1);aa=ma[:,None]+(1-1e-6)*(aa-ma[:,None]);bb=mb[:,None]+(1-1e-6)*(bb-mb[:,None]);ihits.extend(p[ic][sat_intersect(aa,bb,eps=1e-12)].tolist())
 return {'nonincident_candidates':int((~inc).sum()),'trimmed_incident_candidates':int(inc.sum()),'nonincident':hits,'trimmed_incident':ihits}
def separation(points,x):
 skin=tree(points,skin_f);todo=x[bf];bounds=[];counts=[]
 for depth in range(9):
  center=todo.mean(1);rad=np.linalg.norm(todo-center[:,None],axis=2).max(1);dist=np.array([skin.find_nearest(Vector(p))[3] for p in center]);bound=dist-rad-2e-5;passed=bound>0;bounds.extend(bound[passed]);counts.append({'depth':depth,'tested':len(todo),'admitted':int(passed.sum())});bad=todo[~passed]
  if not len(bad):return {'complete':True,'lower_distance_bound':float(min(bounds)),'levels':counts}
  ab=(bad[:,0]+bad[:,1])/2;bc=(bad[:,1]+bad[:,2])/2;ca=(bad[:,2]+bad[:,0])/2;todo=np.r_[np.stack([bad[:,0],ab,ca],1),np.stack([ab,bad[:,1],bc],1),np.stack([ca,bc,bad[:,2]],1),np.stack([ab,bc,ca],1)]
  if len(todo)>50000:break
 return {'complete':False,'unresolved_triangles':len(todo),'levels':counts}
O.mkdir(exist_ok=False);base=crossings(v);report={'head_sha256':sha(H),'binding_sha256':sha(B/'cheek-loaded-skin-binding-v1/binding.npz'),'covered_vertices':len(vid),'affected_faces':len(affected),'regional_faces':len(ri),'baseline_crossings':{k:len(base[k]) if isinstance(base[k],list) else base[k] for k in base},'states':{},'limits':['Static state tests, not a skin-contact force model or between-state swept proof.','Shared triangle strips excluded by1e-6 centroid trim.','Baseline defects, if present, remain declared rather than silently repaired.']};print('baseline',report['baseline_crossings'],flush=True)
for name in ['right-02-secanted']:
 q=np.load(D/f'{name}.npz');assert str(q['head_sha256'])==sha(H);p=v.copy();p[q['vertex_ids']]+=q['displacement'];cc=crossings(p);sr={'state_sha256':sha(D/f'{name}.npz'),'crossings':{k:len(cc[k]) if isinstance(cc[k],list) else cc[k] for k in cc}}
 for k in ['nonincident','trimmed_incident']:
  old=set(map(tuple,base[k]));new=set(map(tuple,cc[k]));sr['introduced_'+k+'_count']=len(new-old);sr['introduced_'+k+'_examples']=list(new-old)[:10]
 sr['continuous_tissue_skin_separation']=separation(p,q['fem_vertices']);report['states'][name]=sr;print(name,json.dumps(sr),flush=True);gc.collect()
(O/'surface_checks.json').write_text(json.dumps(report,indent=2))
