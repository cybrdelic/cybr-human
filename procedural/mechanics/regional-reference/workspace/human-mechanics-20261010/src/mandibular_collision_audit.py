#!/usr/bin/env python3
"""Finite triangle SAT check for rest mandible versus actual native skin.
Broad phase uses complete triangle AABBs, never triangle centroids alone.
"""
from pathlib import Path
import argparse,json,itertools
import numpy as np
from mandibular_support import HEAD,HEAD_SHA,ROOT,reserve

def intersect_sat(a,bb,tol=1e-9):
 """Triangle-vs-many, including coplanar separating axes. Touch counts as hit."""
 if not len(bb):return np.zeros(0,bool)
 b=np.asarray(bb);ea=np.roll(a,-1,axis=0)-a;eb=np.roll(b,-1,axis=1)-b
 na=np.cross(ea[0],ea[1]);nb=np.cross(eb[:,0],eb[:,1])
 axes=[np.broadcast_to(na,(len(b),3)),nb]
 for e in ea:
  for j in range(3):axes.append(np.cross(np.broadcast_to(e,(len(b),3)),eb[:,j]))
 # The extra in-plane axes resolve coplanar overlap/near-coplanarity robustly.
 for e in ea:axes.append(np.broadcast_to(np.cross(na,e),(len(b),3)))
 for j in range(3):axes.append(np.cross(nb,eb[:,j]))
 hit=np.ones(len(b),bool)
 for axis in axes:
  norm=np.linalg.norm(axis,axis=1);valid=norm>1e-13
  axis=np.divide(axis,norm[:,None],out=np.zeros_like(axis),where=valid[:,None])
  pa=axis@a.T;pb=np.einsum('ni,nji->nj',axis,b)
  sep=(pa.max(1)<pb.min(1)-tol)|(pb.max(1)<pa.min(1)-tol)
  hit&=~(valid&sep)
 return hit

def broad_pairs(a,b,cell=4.):
 """For each A triangle, yield all B indices with overlapping AABBs."""
 amin=a.min(1);amax=a.max(1);bmin=b.min(1);bmax=b.max(1)
 global_min=amin.min(0);global_max=amax.max(0)
 relevant=np.where(np.all(bmax>=global_min-1e-9,1)&np.all(bmin<=global_max+1e-9,1))[0]
 bins={}
 for i in relevant:
  lo=np.floor(np.maximum(bmin[i],global_min)/cell).astype(int);hi=np.floor(np.minimum(bmax[i],global_max)/cell).astype(int)
  for key in itertools.product(*(range(lo[j],hi[j]+1) for j in range(3))):bins.setdefault(key,[]).append(i)
 for i in range(len(a)):
  lo=np.floor(amin[i]/cell).astype(int);hi=np.floor(amax[i]/cell).astype(int);ids=set()
  for key in itertools.product(*(range(lo[j],hi[j]+1) for j in range(3))):ids.update(bins.get(key,[]))
  ids=np.fromiter(ids,dtype=np.int64)
  if len(ids):ids=ids[np.all(bmax[ids]>=amin[i]-1e-9,1)&np.all(bmin[ids]<=amax[i]+1e-9,1)]
  yield i,ids

def main():
 p=argparse.ArgumentParser();p.add_argument('output',nargs='?',type=Path,default=ROOT/'output/authored-mandible-v1');p.add_argument('--head',type=Path,default=HEAD);a=p.parse_args();reserve()
 test=np.array([[0,0,0],[1,0,0],[0,1,0]],float)
 fixtures=np.array([test+np.array([0,0,1]),test+np.array([2,0,0]),test+np.array([.1,.1,0]),[[.2,.2,-1],[.2,.2,1],[.8,.2,0]]])
 assert np.array_equal(intersect_sat(test,fixtures),[False,False,True,True])
 jaw=np.load(a.output/'mandible_rest.npz');v=jaw['vertices'];f=jaw['triangles'];j=v[f]
 skin=np.load(a.head);sv=skin['vertices'];sf=skin['triangles'][skin['object_ids']==0];s=sv[sf]
 count=0;hits=[]
 for i,ids in broad_pairs(j,s):
  count+=len(ids);hh=ids[intersect_sat(j[i],s[ids])]
  hits.extend([[int(i),int(k)] for k in hh])
 self_count=0;self_hits=[]
 for i,ids in broad_pairs(j,j):
  ids=ids[ids>i]
  if len(ids):ids=ids[~np.any(f[ids,:,None]==f[i][None,None,:],axis=(1,2))]
  self_count+=len(ids);hh=ids[intersect_sat(j[i],j[ids])]
  self_hits.extend([[int(i),int(k)] for k in hh])
 report=dict(method='Complete triangle-AABB broad phase followed by triangle SAT with face normals, 9 edge cross products, and 6 coplanar in-plane axes. Float64, 1e-9 design-unit separation tolerance; contact counted as overlap.',rest_bone_triangles=len(f),native_skin_triangles=len(sf),rest_skin_candidate_pairs=count,rest_skin_intersecting_pairs=len(hits),rest_skin_intersections=hits[:20],self_nonadjacent_candidate_pairs=self_count,self_nonadjacent_intersecting_pairs=len(self_hits),self_nonadjacent_intersections=self_hits[:20],fixture_tests_passed=True,limits='Floating-point rest-mesh intersection audit. Shared-vertex self-triangle pairs excluded; not an exact-arithmetic proof. Active bone intentionally moves without deforming rest skin, so active-state containment is not claimed. Combined with inside samples and closed skin topology this supports rest containment, not anatomy accuracy or contact dynamics.',native_skin_sha256=HEAD_SHA)
 (a.output/'rest_collision_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
 assert not hits and not self_hits
if __name__=='__main__':main()
