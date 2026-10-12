from pathlib import Path
import argparse
_ap=argparse.ArgumentParser();_ap.add_argument("--candidate",type=Path,required=True);_args=_ap.parse_args()
import json,hashlib
import numpy as np
from scipy.spatial import cKDTree
from triangle_checks import sat_intersect,nearest_points_batch
O=_args.candidate.resolve();z=np.load(O/'candidate_head.npz');v=z['vertices'];f=z['triangles'];oid=z['object_ids'];maps=np.load(O/'retained_source_maps.npz');changed_ids=maps['modified_face_ids'];lowall=v[f[changed_ids]].min((0,1));highall=v[f[changed_ids]].max((0,1));selected=[];minarea=np.inf
for start in range(0,len(f),30000):
    p=v[f[start:start+30000]];area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)/2;minarea=min(minarea,float(area.min()));mask=(p.max(1)>=lowall-1e-9).all(1)&(p.min(1)<=highall+1e-9).all(1)&(oid[start:start+30000]==0);selected.extend((np.flatnonzero(mask)+start).tolist())
selected=np.array(selected);ff=f[selected];tri=v[ff];mid=tri.mean(1);rad=np.linalg.norm(tri-mid[:,None],axis=2).max(1);buckets=[]
for bucket in np.unique(np.floor(np.log2(np.maximum(rad,1e-8)))):
    members=np.flatnonzero(np.floor(np.log2(np.maximum(rad,1e-8)))==bucket);buckets.append((members,cKDTree(mid[members]),rad[members].max()))
lo=tri.min(1);hi=tri.max(1);modified=np.isin(selected,changed_ids);ci=np.flatnonzero(modified);tested=hits=0;examples=[]
for start in range(0,len(ci),128):
    ii0=ci[start:start+128];pairs=[]
    for members,tree,rmax in buckets:
        ls=tree.query_ball_point(mid[ii0],rad[ii0]+rmax+1e-9);pairs.append((np.repeat(ii0,[len(x) for x in ls]),members[np.concatenate(ls).astype(int)]))
    ii=np.concatenate([x[0] for x in pairs]);jj=np.concatenate([x[1] for x in pairs]);take=(ii!=jj)&((~modified[jj])|(jj>ii));ii=ii[take];jj=jj[take];take=(hi[ii]>=lo[jj]-1e-10).all(1)&(hi[jj]>=lo[ii]-1e-10).all(1);ii=ii[take];jj=jj[take];adj=(ff[ii,:,None]==ff[jj,None,:]).any((1,2));ii=ii[~adj];jj=jj[~adj];tested+=len(ii)
    if len(ii):
        hh=sat_intersect(tri[ii],tri[jj]);hits+=int(hh.sum());examples.extend(np.c_[selected[ii[hh]],selected[jj[hh]]].tolist())
    if start%20000<128:print('AUDIT_PROGRESS',start,len(ci),hits,flush=True)
optics={}
for side,sign in [('right',1),('left',-1)]:
    c=np.array([sign*31.3,12,94.6]);near=np.linalg.norm(mid-c,axis=1)-rad<17;clear=nearest_points_batch(c,tri[near])[0]-13.161;optics[side]={'tested_skin_triangles':int(near.sum()),'minimum_continuous_triangle_sphere_clearance':float(clear.min()),'sphere_crossings':int((clear<=0).sum())}
r={'head_sha256':hashlib.sha256((O/'candidate_head.npz').read_bytes()).hexdigest(),'minimum_all_head_triangle_area':minarea,'modified_skin_faces_tested':len(ci),'relevant_full_skin_faces':len(ff),'nonincident_pair_tests':tested,'nonincident_intersections':hits,'examples':examples[:30],'ocular_sphere':optics,'passed':hits==0 and minarea>1e-12 and all(x['sphere_crossings']==0 for x in optics.values()),'limits':['Modified-versus-entire-relevant-skin triangle SAT. Incident pairs sharing any vertex are excluded.','No claim for unchanged-versus-unchanged or exact full optical cap/contact/groom validation.']}
(O/'SURFACE_AUDIT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True);assert r['passed']
