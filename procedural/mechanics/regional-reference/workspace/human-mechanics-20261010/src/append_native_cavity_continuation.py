"""Append a finite cavity-side annulus behind the valid original oral volume.

The existing exterior, oral wedge, tissue nodes and tetrahedra are immutable.
An existing inner-carrier annulus supplies the full shared attachment surface.
Its inner edge follows exact native mucosal rings 16..24; its outer edge uses
the same authored posterior translation. No determinant-dependent winding.
This is a rest-construction candidate, not a mechanical/contact acceptance.
"""
from pathlib import Path
import argparse,json,hashlib
from collections import defaultdict
import numpy as np
from build_bilateral_mechanics_case import boundary,surface_audit
R=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=R/'output/bilateral-midface-bone-aware-v2/rest_geometry.npz');p.add_argument('--output',type=Path,required=True);p.add_argument('--outer-flow',type=float,default=1.);a=p.parse_args()
g=np.load(a.base);oral_path=R/'inputs/native_oral_charts_v16_mucosa24.npz';o=np.load(oral_path)
v=g['vertices'];t=g['tetrahedra'];cols=g['surface_column_nodes'];tri=g['all_surface_triangles'];parents=g['surface_triangle_parent_faces'];ring=set(map(int,g['oral_interface_columns']))
faces=np.unique(parents[(np.isin(tri,list(ring)).sum(1)>=2)&(g['face_parts'][parents]!=20)])
assert len(faces)==60 and np.all(g['face_parts'][faces]==15)
selected=np.flatnonzero(np.isin(parents,faces)&(g['face_parts'][parents]!=20));source=cols[tri[selected],-1]
actual_boundary=boundary(t);by_key={tuple(sorted(map(int,f))):f for f in actual_boundary}
patch=np.array([by_key[tuple(sorted(map(int,f)))] for f in source]);assert len(patch)==180
nodes=np.unique(patch);local={int(n):i for i,n in enumerate(nodes)}
native_angle={int(n):i for i,n in enumerate(o['external_vertex_ids'][16])}
angle_by_column={int(c):native_angle[int(g['outer_column_native_ids'][c])] for c in ring}
for fi in faces:
 q=g['all_surface_quads'][fi]
 for k in range(4):
  aa,bb,cc,dd=map(int,[q[k],q[(k+1)%4],q[(k+2)%4],q[(k+3)%4]])
  if aa in ring and bb in ring:
   for node,angle in [(cc,angle_by_column[bb]),(dd,angle_by_column[aa])]:
    if node in angle_by_column:assert angle_by_column[node]==angle
    angle_by_column[node]=angle
node_angle={int(cols[c,-1]):angle for c,angle in angle_by_column.items()}
assert set(nodes)==set(node_angle)
inner_nodes={int(cols[c,-1]) for c in ring}
rows=[16,18,20,22,24];levels=[nodes.copy()];points=[v.copy()];param={int(n):(node_angle[int(n)],16) for n in inner_nodes};cursor=len(v)
for row in rows[1:]:
 layer=v[nodes].copy()
 for j,node in enumerate(nodes):
  angle=node_angle[int(node)];delta=o['mucosal_positions'][row,angle]-o['mucosal_positions'][16,angle]
  if int(node) in inner_nodes:layer[j]=o['mucosal_positions'][row,angle]
  else:layer[j]+=a.outer_flow*delta
 ids=np.arange(cursor,cursor+len(nodes));cursor+=len(nodes);levels.append(ids);points.append(layer)
 for j,node in enumerate(nodes):
  if int(node) in inner_nodes:param[int(ids[j])]=(node_angle[int(node)],row)
levels=np.array(levels);x=np.concatenate(points);new_tets=[];new_params=[];new_parent=[];new_layer=[]
for pi,face in enumerate(patch):
 order=np.argsort(face);ss=face[order];odd=sum(order[i]>order[j] for i in range(3) for j in range(i+1,3))%2
 for layer in range(len(rows)-1):
  aa,bb,cc=[levels[layer,local[int(n)]] for n in ss];ad,bd,cd=[levels[layer+1,local[int(n)]] for n in ss]
  for q in [[aa,bb,cc,cd],[aa,bb,cd,bd],[aa,bd,cd,ad]]:
   if odd:q[1],q[2]=q[2],q[1]
   new_tets.append(q);new_params.append([int(parents[selected[pi]]),4+layer]);new_parent.append(pi);new_layer.append(layer)
nt=np.array(new_tets);all_tets=np.concatenate([t,nt]);dm=x[all_tets[:,1:]]-x[all_tets[:,0,None]];det=np.linalg.det(dm);cond=np.linalg.cond(dm)
new_boundary=boundary(all_tets);new_muc=[];new_uv=[]
for f in new_boundary:
 if all(int(n) in param for n in f) and any(param[int(n)][1]>16 for n in f):
  uv=np.array([param[int(n)] for n in f],float)
  if np.ptp(uv[:,0])>240:uv[uv[:,0]<240,0]+=480
  new_muc.append(f);new_uv.append(uv)
assert len(new_muc)==960
data={k:g[k] for k in g.files};data.update(vertices=x,tetrahedra=all_tets,cell_parameters=np.concatenate([g['cell_parameters'],np.array(new_params)]),cavity_extension_tetrahedra=np.arange(len(t),len(all_tets)),cavity_extension_layers=np.array(new_layer),cavity_extension_parent_inner_faces=np.array(new_parent),cavity_extension_attachment_faces=patch,cavity_extension_layer_nodes=levels,cavity_extension_base_node_ids=nodes,cavity_extension_native_mucosal_rows=np.array(rows),cavity_extension_mucosal_support_triangles=np.array(new_muc),cavity_extension_mucosal_support_uv=np.array(new_uv),cavity_extension_native_cut_nodes=np.array([int(levels[-1,local[n]]) for n in sorted(inner_nodes)]),cavity_extension_inner_base_nodes=np.array(sorted(inner_nodes)))
audit=surface_audit(new_boundary);report={'schema':'appended-native-cavity-annulus-rest-v1','status':'Construction only; reject if any negative cell or overlap','source_rest_sha256':sha(a.base),'source_oral_chart_sha256':sha(oral_path),'old_nodes_unchanged':bool(np.array_equal(x[:len(v)],v)),'old_tetrahedra_unchanged':bool(np.array_equal(all_tets[:len(t)],t)),'old_nodes':len(v),'nodes':len(x),'old_tetrahedra':len(t),'tetrahedra':len(all_tets),'shared_attachment_triangles':len(patch),'new_native_mucosal_support_faces':len(new_muc),'negative_tetrahedra':int((det<=0).sum()),'negative_new_tetrahedra':int((det[len(t):]<=0).sum()),'minimum_signed_tet_volume':float(det.min()/6),'maximum_condition':float(cond.max()),'boundary_audit':audit,'outer_flow':a.outer_flow,'limits':['Posterior annular support is authored, not measured anatomy','No overlap or contact acceptance','New boundary constraints and exact material binding must be rebuilt before a solve']}
a.output.mkdir(parents=True,exist_ok=False);np.savez_compressed(a.output/'rest_geometry.npz',**data);(a.output/'construction_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
