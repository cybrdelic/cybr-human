"""Conforming passive cheek annulus around the proved shared-node insertion.
Same active bundle, material constants, and neutral selected-v53 surface.
The old inner truncation becomes interior; only the new farther boundary is fixed.
"""
from pathlib import Path
import json,hashlib,numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parents[1]
IP=B/'cheek-loaded-insertion-v1/geometry.npz';G=np.load(IP);O=B/'cheek-compliant-insertion-v2'
HP=R/'user-human-preferred/recovery/restored-project/output/coordinated-orbital-scale-v53/head.npz';H=np.load(HP)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(HP)=='545b8ea2b292107dc2da9f1e90379989eb620c6ee5c2c50b3f9b5316cc77420e'
skin=BVHTree.FromPolygons(H['vertices'].tolist(),H['triangles'][H['object_ids']==0].tolist(),all_triangles=True)
v=G['vertices'];top0=G['insertion_surface_node_ids'][-12:];cap=G['insertion_shared_nodes'];baseT=G['tetrahedra'][:3041]
ff=np.concatenate([baseT[:,[1,2,3]],baseT[:,[0,3,2]],baseT[:,[0,1,3]],baseT[:,[0,2,1]]]);_,ix,c=np.unique(np.sort(ff,1),axis=0,return_index=True,return_counts=True);capF=ff[ix[c==1]];capF=capF[np.isin(capF,cap).all(1)]
e=np.concatenate([capF[:,[0,1]],capF[:,[1,2]],capF[:,[2,0]]]);_,ix,c=np.unique(np.sort(e,1),axis=0,return_index=True,return_counts=True);ed=e[ix[c==1]];nx={int(a):int(b) for a,b in ed};bottom0=[int(ed[0,0])]
while nx[bottom0[-1]]!=bottom0[0]:bottom0.append(nx[bottom0[-1]])
bottom0=np.array(bottom0);assert len(bottom0)==12
# Preserve the exact existing side diagonals; no proximity weld or fresh cap.
oldSide=np.array([[bottom0[i],bottom0[(i+1)%12],top0[(i+1)%12]] for i in range(12)]+[[bottom0[i],top0[(i+1)%12],top0[i]] for i in range(12)])
boundarySet={tuple(sorted(x)) for x in G['boundary_faces']};assert all(tuple(sorted(x)) in boundarySet for x in oldSide)
def front(xy):
 h,_,_,_=skin.ray_cast(Vector((float(xy[0]),float(xy[1]),200)),Vector((0,0,-1)),400)
 if h is None:raise RuntimeError('Missing skin at '+str(xy))
 return np.array(h)
theta=np.arctan2((v[top0,1]+50)/4,(v[top0,0]-31)/3.5)
outer=np.c_[36.1+8.9*np.cos(theta),-49.5+12.5*np.sin(theta)]
oldBottomDepth=np.array([front(p[:2])[2]-p[2] for p in v[bottom0]])
verts=list(v.copy());rows=[np.stack([bottom0,top0])];exactTop=list(G['insertion_exact_skin_points']);newTop=[]
for s in [.25,.5,.75,1.]:
 ids=np.zeros((2,12),int)
 for layer in range(2):
  for i in range(12):
   old=v[(bottom0 if layer==0 else top0)[i]];xy=(1-s)*old[:2]+s*outer[i];p=front(xy)
   # A positive-thickness bottom sheet follows the surface, matching old oblique cap.
   depth=.6 if layer else (1-s)*oldBottomDepth[i]+s*2.0
   if layer:exactTop.append(p.copy())
   p[2]-=depth;ids[layer,i]=len(verts);verts.append(p)
   if layer:newTop.append(ids[layer,i])
 rows.append(ids)
verts=np.array(verts);rows=np.array(rows)
pat=np.array([[0,1,2,6],[0,2,3,6],[0,3,7,6],[0,7,4,6],[0,4,5,6],[0,5,1,6]])
new=[];topfaces=[];hexSigns=[]
for k in range(4):
 for i in range(12):
  j=(i+1)%12;he=np.array([rows[k,0,i],rows[k,0,j],rows[k+1,0,j],rows[k+1,0,i],rows[k,1,i],rows[k,1,j],rows[k+1,1,j],rows[k+1,1,i]])
  tt=he[pat];p=verts[tt];d=np.linalg.det(np.stack([p[:,a]-p[:,0] for a in [1,2,3]],-1));hexSigns.append(d.tolist())
  assert (d>0).all() or (d<0).all(),('twisted hex',k,i,d)
  if (d<0).all():tt=tt[:,[0,2,1,3]]
  new.extend(tt);topfaces.extend([[he[4],he[5],he[6]],[he[4],he[6],he[7]]])
tets=np.r_[G['tetrahedra'],new];p=verts[tets];Dm=np.stack([p[:,k]-p[:,0] for k in [1,2,3]],-1);d=np.linalg.det(Dm);assert d.min()>0
ff=np.concatenate([tets[:,[1,2,3]],tets[:,[0,3,2]],tets[:,[0,1,3]],tets[:,[0,2,1]]]);_,ix,iv,c=np.unique(np.sort(ff,1),axis=0,return_index=True,return_inverse=True,return_counts=True);assert c.max()==2
sign=np.where(((ff[:,0]>ff[:,1]).astype(int)+(ff[:,0]>ff[:,2])+(ff[:,1]>ff[:,2]))%2,-1,1);assert not ((c==2)&(np.bincount(iv,weights=sign)!=0)).any()
boundary=ff[ix[c==1]];boundarySet={tuple(sorted(x)) for x in boundary};assert not any(tuple(sorted(x)) in boundarySet for x in oldSide)
# Keep literal outward top winding from the union boundary.
topSet={tuple(sorted(x)) for x in np.r_[G['insertion_surface_triangles'],topfaces]};topfaces=np.array([x for x in boundary if tuple(sorted(x)) in topSet]);assert len(topfaces)==132
DIR=[Vector((1,.173,.317)).normalized(),Vector((.193,1,.271)).normalized(),Vector((.127,.233,1)).normalized()]
def inside(p,d):
 o=Vector(p);n=0
 for _ in range(40):
  h,_,_,_=skin.ray_cast(o,d,1000)
  if h is None:return n%2
  n+=1;o=h+d*1e-4
 return -1
par=np.array([[inside(p,d) for d in DIR] for p in verts]);centers=verts[np.array(new)].mean(1);cp=np.array([[inside(p,d) for d in DIR] for p in centers]);bad=np.flatnonzero(~(par==1).all(1));badc=np.flatnonzero(~(cp==1).all(1))
q={k:G[k] for k in G.files};topids=np.r_[G['insertion_surface_node_ids'],newTop];rim=rows[-1].ravel();q.update(vertices=verts,tetrahedra=tets,boundary_faces=boundary,fixed_nodes=np.r_[G['bone_fixed_nodes'],rim],parity=par,skin_distance=np.array([skin.find_nearest(Vector(p))[3] for p in verts]),insertion_tetrahedra=np.arange(3041,len(tets)),insertion_surface_node_ids=topids,insertion_surface_triangles=topfaces,insertion_fixed_nodes=rim,insertion_exact_skin_points=np.array(exactTop),compliant_extension_tetrahedra=np.arange(len(G['tetrahedra']),len(tets)),compliant_extension_ring_ids=rows,released_inner_perimeter_nodes=top0)
O.mkdir(exist_ok=False);np.savez_compressed(O/'geometry.npz',**q)
report={'head_sha256':sha(HP),'source_geometry_sha256':sha(IP),'geometry_sha256':sha(O/'geometry.npz'),'nodes':len(verts),'tetrahedra':len(tets),'new_passive_tetrahedra':len(new),'shared_inner_side_faces':len(oldSide),'released_old_perimeter_nodes':len(top0),'new_far_perimeter_nodes':len(rim),'superficial_proxy_nodes':len(topids),'top_boundary_triangles':len(topfaces),'old_positions_exact':np.array_equal(verts[:len(v)],v),'old_connectivity_exact':np.array_equal(tets[:len(G['tetrahedra'])],G['tetrahedra']),'minimum_signed_six_volume':float(d.min()),'new_Dm_condition_quantiles':np.percentile(np.linalg.cond(Dm[-len(new):]),[0,50,95,100]).tolist(),'node_outside_ids':bad.tolist(),'cell_center_outside_ids':badc.tolist(),'outer_ellipse_center_radii':[36.1,-49.5,8.9,12.5],'superficial_depth':.6,'outer_bottom_depth':2.,'material_change':False,'scope':'Conforming passive volume annulus around the exact earlier insertion. Same bundle/material; supported farther boundary replaces the tiny fixed rim. Original skin unchanged at rest.','admitted':len(bad)==0 and len(badc)==0}
(O/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
assert report['admitted'],'Outside selected head'
