from pathlib import Path
import sys,json,hashlib,numpy as np
from scipy.spatial import cKDTree
from mathutils import Vector
from mathutils.bvhtree import BVHTree
sys.path.insert(0,str(Path(__file__).resolve().parent));from triangle_sat import sat_intersect
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parents[1];G=B/'cheek-perioral-insertion-v3/geometry.npz';C=B/'cheek-perioral-case-v3';O=B/'cheek-perioral-loaded-validity-v3';H=R/'user-human-preferred/recovery/restored-project/output/coordinated-orbital-scale-v53/head.npz';g=np.load(G);bf=g['boundary_faces'];statefiles=[B/'cheek-perioral-equilibrium-v3'/f'{s}-state.json' for s in ['right-02-initial']];statefiles=[p for p in statefiles if p.exists()]
z=np.load(H);v=z['vertices'];f=z['triangles'][z['object_ids']==0];skin=BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True)
def self_intersections(x):
 tri=x[bf];mid=tri.mean(1);rad=np.linalg.norm(tri-mid[:,None],axis=2).max(1);tr=cKDTree(mid);lo=tri.min(1);hi=tri.max(1);total=incident=hits=ihits=0;examples=[]
 for start in range(0,len(bf),128):
  ids=np.arange(start,min(start+128,len(bf)));ls=tr.query_ball_point(mid[ids],rad[ids]+rad.max()+1e-8);ii=np.repeat(ids,[len(l) for l in ls]);jj=np.concatenate(ls).astype(int);take=jj>ii;ii=ii[take];jj=jj[take];take=(hi[ii]>=lo[jj]-1e-10).all(1)&(hi[jj]>=lo[ii]-1e-10).all(1);ii=ii[take];jj=jj[take];inc=(bf[ii,:,None]==bf[jj,None,:]).any((1,2));total+=(~inc).sum();incident+=inc.sum()
  if (~inc).any():
   hh=sat_intersect(tri[ii[~inc]],tri[jj[~inc]],eps=1e-10);hits+=hh.sum();examples.extend(zip(ii[~inc][hh].tolist(),jj[~inc][hh].tolist()))
  if inc.any():
   a=mid[ii[inc],None]+(1-1e-6)*(tri[ii[inc]]-mid[ii[inc],None]);b=mid[jj[inc],None]+(1-1e-6)*(tri[jj[inc]]-mid[jj[inc],None]);ihits+=sat_intersect(a,b,eps=1e-12).sum()
 return {'nonincident_candidates':int(total),'nonincident_crossings':int(hits),'incident_candidates':int(incident),'trimmed_incident_crossings':int(ihits),'examples':examples[:10],'incident_limit':'Shared strips excluded by1e-6 centroid trim; no exact crossing claim inside that strip.'}
def head_cover(x):
 todo=x[bf];bounds=[];counts=[]
 for depth in range(9):
  center=todo.mean(1);rad=np.linalg.norm(todo-center[:,None],axis=2).max(1);dist=np.array([skin.find_nearest(Vector(p))[3] for p in center]);bound=dist-rad-2e-5;passmask=bound>0;bounds.extend(bound[passmask]);counts.append({'depth':depth,'tested':len(todo),'admitted':int(passmask.sum())});bad=todo[~passmask]
  if not len(bad):return {'complete':True,'lower_distance_bound':float(min(bounds)),'levels':counts,'side':'Interior side follows the separately verified inside-node rest component; this clearance test itself is unsigned.'}
  ab=(bad[:,0]+bad[:,1])/2;bc=(bad[:,1]+bad[:,2])/2;ca=(bad[:,2]+bad[:,0])/2;todo=np.r_[np.stack([bad[:,0],ab,ca],1),np.stack([ab,bad[:,1],bc],1),np.stack([ca,bc,bad[:,2]],1),np.stack([ab,bc,ca],1)]
  if len(todo)>50000:break
 return {'complete':False,'unresolved_triangles':len(todo),'levels':counts}
O.mkdir(exist_ok=False);report={}
for p in statefiles:
 state=json.loads(p.read_text());assert state['converged'];x=np.array(state['positions']).reshape(-1,3)*1000;selfcheck=self_intersections(x);cover=head_cover(x);report[p.stem]={'state_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'self_surface':selfcheck,'unchanged_v53_skin_boundary_clearance':cover};print(p.stem,json.dumps(report[p.stem]),flush=True)
(O/'surface_checks.json').write_text(json.dumps(report,indent=2))
