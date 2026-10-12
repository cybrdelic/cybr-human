"""Resolve an exact-Boolean root sliver without changing the bone footprint.
A root-perimeter chord is split at its existing curved bone vertex. Nonbonded
near-root vertices receive a tiny outward numerical clearance; anchors stay exact.
"""
from pathlib import Path
import json,hashlib,numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parents[1];I=B/'route-layout-v6';O=B/'route-layout-v7';O.mkdir(exist_ok=False);z=np.load(I/'parts.npz');q={k:z[k] for k in z.files};v=q['mandibular_depressor_vertices'].copy();f=q['mandibular_depressor_triangles'].copy();root=q['mandibular_origin_local_vertex_ids'];cap=q['mandibular_origin_local_face_ids'];oldv=v.copy();oldf=f.copy();candidates=np.setdiff1d(np.flatnonzero(np.isin(f,root).all(1)),cap);flips=[]
for i in candidates:
 p=v[f[i]];area=np.linalg.norm(np.cross(p[1]-p[0],p[2]-p[0]))/2
 if area>1e-3:raise ValueError('Unresolved larger all-root body face')
 lens=np.linalg.norm(p-np.roll(p,-1,axis=0),axis=1);k=int(np.argmax(lens));u=int(f[i,k]);w=int(f[i,(k+1)%3]);mid=int(f[i,(k+2)%3]);adj=np.flatnonzero(np.isin(f,[u,w]).sum(1)==2);adj=adj[adj!=i];assert len(adj)==1;j=int(adj[0]);free=int(f[j][~np.isin(f[j],[u,w])][0]);assert free not in root
 # Preserve the directed edge from the adjacent free-body triangle.
 fj=f[j].tolist();a=fj.index(u);forward=fj[(a+1)%3]==w
 if not forward:u,w=w,u
 f[j]=[u,mid,free];f[i]=[mid,w,free];flips.append({'old_root_sliver_face':int(i),'old_free_face':j,'split_vertex':mid,'old_area':float(area)})
M=np.load(B/'mandible-v3/geometry.npz');tree=BVHTree.FromPolygons(M['vertices'].tolist(),M['triangles'].tolist(),all_triangles=True);free=np.setdiff1d(np.arange(len(v)),root);moved=[]
for i in free:
 h,n,fi,d=tree.find_nearest(Vector(v[i]))
 if d<.5:
  amount=.005*(1-(d/.5)**2)**2;v[i]+=np.array(n)*amount;moved.append([int(i),float(amount)])
assert np.array_equal(v[root],oldv[root]);q['mandibular_depressor_vertices']=v;q['mandibular_depressor_triangles']=f
np.savez_compressed(O/'parts.npz',**q);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();report=json.loads((I/'report.json').read_text());report.update(parts_sha256=sha(O/'parts.npz'),source_parts_sha256=sha(I/'parts.npz'),root_interface_repair={'edge_flips':flips,'nonbonded_near_root_nodes_lifted':len(moved),'maximum_outward_lift':max([x[1] for x in moved],default=0),'footprint_positions_and_weights_exact':True,'scope':'Numerical geometry repair at the Boolean/barycentric contact perimeter. No bone coordinate, origin correspondence, route control, material or physical solve changes.'});(O/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['root_interface_repair'],indent=2));print(sha(O/'parts.npz'))
