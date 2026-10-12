"""One conforming native midface/oral volume; first rest-geometry study.
No native surface edit or simulated result. Internal anatomical depths are
original authored approximations and must pass geometric/support audits.
"""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,json,hashlib
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import dijkstra
from prototypes.refine_oral_midface_interface import refine_oral_midface_interface
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build(out,depth_scale=1.):
 p=R/'inputs/native_midface_patch_v16/native_midface_patch_v16.npz';a=np.load(p);o=np.load(R/'inputs/native_oral_charts_v16.npz');v=a['surface_sample_positions_design_units'];quads=a['patch_quads_local'];N=len(v);fractions=np.array([0.,.25,.5,.75,1.]);global_ids=a['cage_vertex_ids'];lookup={int(g):i for i,g in enumerate(global_ids)}
 tri=np.concatenate([quads[:,[0,1,2]],quads[:,[0,2,3]]]);parents=np.tile(np.arange(len(quads)),2)
 coarse_angles=8*a['oral_angle_from_cage_border'];coarse_border=np.array([lookup[int(g)] for g in a['oral_interface_cage_ids']])
 refined=refine_oral_midface_interface(v,a['surface_sample_native_vertex_ids'],tri,parents,coarse_angles,coarse_border,external_positions=o['external_positions'],external_vertex_ids=o['external_vertex_ids'],mucosal_positions=o['mucosal_positions'],mucosal_vertex_ids=o['mucosal_vertex_ids'])
 v=refined['vertices'];native_ids=refined['native_vertex_ids'];tri=refined['triangles'];parents=refined['triangle_parent_faces'];N=len(v);normal=np.zeros_like(v)
 fn=np.cross(v[tri[:,1]]-v[tri[:,0]],v[tri[:,2]]-v[tri[:,0]])
 for k in range(3):np.add.at(normal,tri[:,k],fn)
 normal/=np.linalg.norm(normal,axis=1)[:,None]
 # Broad editable depth fields, without any isolated spherical surface bulge.
 x,y,z=v.T;depth=np.interp(y,[-83,-65,-45,-20,10,53],[5.,9.,13.,12.,7.,6.])
 depth*=np.interp(abs(x),[0,15,35,60,70],[.7,.85,1.,.65,.5])
 central=(abs(x)<17)&(y>-40)&(y<35);depth[central]=np.minimum(depth[central],np.interp(y[central],[-40,-28,-5,35],[3.,2.5,4.,5.]))
 depth*=depth_scale;target=-normal*depth[:,None]
 edges=np.unique(np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1),axis=0)
 lengths=np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1);w=1/np.maximum(lengths,.15)
 adj=sparse.coo_matrix((np.r_[w,w],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(N,N)).tocsr();L=sparse.diags(np.asarray(adj.sum(1)).ravel())-adj
 fixed={};protected=[]
 for name in ['right_eye_cut','left_eye_cut','right_nostril_cut','left_nostril_cut']:
  ids=np.array([lookup[int(g)] for g in a[name+'_cage_ids']]);protected.extend(ids.tolist())
  for n in ids:fixed[int(n)]=-.30*normal[n]
 # A thin omitted-structure collar must widen over real surface distance,
 # not jump from0.3 to several units across one tight orbital control row.
 distance_graph=sparse.coo_matrix((np.r_[lengths,lengths],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(N,N)).tocsr()
 collar_distance=dijkstra(distance_graph,directed=False,indices=np.array(protected),min_only=True)
 depth=np.minimum(depth,.30+.60*collar_distance);target=-normal*depth[:,None]
 angles=refined['oral_angles'];border=refined['oral_columns'];oral_ext=o['external_positions'][16,angles];oral_inner=o['mucosal_positions'][16,angles]
 if not np.array_equal(v[border],oral_ext):raise ValueError('Oral interface exterior is not byte-exact')
 for n,q in zip(border,oral_inner-oral_ext):fixed[int(n)]=q
 # One continuous vector carrier. Dirichlet data are shared tissue interfaces;
 # the remaining field balances authored depth and graph smoothness.
 M=sparse.eye(N,format='csr')+.65*L;fix=np.array(sorted(fixed));free=np.setdiff1d(np.arange(N),fix);u=np.zeros_like(v);u[fix]=np.array([fixed[int(i)] for i in fix]);u[free]=spsolve(M[free][:,free],target[free]-M[free][:,fix]@u[fix])
 nodes=np.concatenate([v+f*u for f in fractions]);grid=np.arange(5*N).reshape(5,N).T.copy();keys=list(native_ids.astype(int));columns=[grid[i].copy() for i in range(N)];outer_positions=list(v.copy());column_provenance=list(native_ids.astype(int));node_provenance=np.full(5*N,-1,np.int64);node_provenance[:N]=native_ids;node_list=list(nodes);source_parts=list(a['source_face_parts'].astype(int));all_quads=list(quads.copy());chart_tags=list(a['material_chart_tags'].copy());oral_ids=np.empty((120,5,5),int);oral_cols=np.empty((120,5),int);angle_to_border={int(angle):int(n) for angle,n in zip(angles,border)}
 for ii,angle in enumerate(range(0,480,4)):
  for jj,row in enumerate([0,4,8,12,16]):
   if row==16:
    n=angle_to_border[angle];oral_ids[ii,jj]=grid[n];oral_cols[ii,jj]=n;continue
   ep=o['external_positions'][row,angle];ip=o['mucosal_positions'][row,angle];column=len(columns);oral_cols[ii,jj]=column;ids=[]
   if row==0:
    ids=[len(node_list)]*5;node_list.append(ep.copy());node_provenance=np.r_[node_provenance,int(o['external_vertex_ids'][row,angle])]
   else:
    for k,f in enumerate(fractions):
     ids.append(len(node_list));node_list.append((1-f)*ep+f*ip);node_provenance=np.r_[node_provenance,int(o['external_vertex_ids'][row,angle]) if k==0 else int(o['mucosal_vertex_ids'][row,angle]) if k==4 else -1]
   oral_ids[ii,jj]=ids;columns.append(np.array(ids));keys.append(int(o['external_vertex_ids'][row,angle]));outer_positions.append(ep.copy());column_provenance.append(int(o['external_vertex_ids'][row,angle]))
 # Normalize oral orientation from the exact shared outer border, not normals
 # guessed from a frontal projection.
 boundary_dir={}
 for q in tri:
  for aa,bb in zip(q,np.roll(q,-1)):
   key=tuple(sorted((int(aa),int(bb))))
   if key in boundary_dir:boundary_dir[key]=None
   else:boundary_dir[key]=(int(aa),int(bb))
 oral_q=[]
 for i in range(120):
  for j in range(4):oral_q.append([oral_cols[i,j],oral_cols[(i+1)%120,j],oral_cols[(i+1)%120,j+1],oral_cols[i,j+1]])
 border_edges=[]
 for i in range(120):
  aa,bb=oral_cols[(i+1)%120,4],oral_cols[i,4];direction=boundary_dir[tuple(sorted((aa,bb)))];border_edges.append(direction==(int(aa),int(bb)))
 if len(set(border_edges))!=1:raise ValueError('Oral interface winding is inconsistent')
 if border_edges[0]:oral_q=[q[::-1] for q in oral_q]
 all_quads.extend(oral_q);source_parts.extend([20]*len(oral_q));chart_tags.extend([[5,i,j] for i in range(120) for j in range(4)])
 nodes=np.array(node_list);columns=np.array(columns);all_quads=np.array(all_quads,int);keys=np.array(keys);tets=[];params=[];outer_tri=[]
 surface_triangles=list(tri);triangle_parents=list(parents)
 for fi,q in enumerate(oral_q,len(quads)):
  surface_triangles.extend([q[:3],[q[0],q[2],q[3]]]);triangle_parents.extend([fi,fi])
 for original,fi in zip(np.array(surface_triangles),triangle_parents):
  outer_tri.append(columns[original,0]);order=np.argsort(keys[original]);ss=original[order];parity=(-1)**sum(order[i]>order[j] for i in range(3) for j in range(i+1,3))
  for k in range(4):
   aa,bb,cc=columns[ss,k];ad,bd,cd=columns[ss,k+1]
   for tet in [[aa,bb,cc,cd],[aa,bb,cd,bd],[aa,bd,cd,ad]]:
    if len(set(tet))<4:continue
    if parity==1:tet[1],tet[2]=tet[2],tet[1]
    tets.append(tet);params.append([fi,k])
 t=np.array(tets,int);params=np.array(params);dm=np.stack([nodes[t[:,k]]-nodes[t[:,0]] for k in [1,2,3]],-1);det=np.linalg.det(dm);cond=np.linalg.cond(dm)
 report={'schema':'connected-bilateral-midface-rest-study-v1','status':'geometry only; no support/contact/solve acceptance','source_patch_sha256':sha(p),'source_head_sha256':str(a['source_head_sha256']),'nodes':len(nodes),'tetrahedra':len(t),'depth_scale':depth_scale,'positive_tets':int((det>1e-10).sum()),'negative_tets':int((det<-1e-10).sum()),'degenerate_tets':int((abs(det)<=1e-10).sum()),'min_signed_volume_design_units_cubed':float(det.min()/6),'max_condition':float(cond.max()),'condition_percentiles':np.percentile(cond,[50,90,99]).tolist(),'shared_oral_interface_nodes':int(len(np.unique(grid[border]))),'maximum_oral_interface_error':float(np.linalg.norm(nodes[grid[border,-1]]-oral_inner,axis=1).max()),'min_column_length':float(np.linalg.norm(u,axis=1).min()),'bad_cell_examples':[{'tet':int(i),'face':int(params[i,0]),'part':int(source_parts[params[i,0]]),'depth_band':int(params[i,1]),'det':float(det[i])} for i in np.flatnonzero(det<=1e-10)[:30]],'limits':['Authored depth guide is not measured anatomy','No attached skull/cartilage, contact certification or mechanical solve yet','Coarse outer facets chord exact native skin; native head unchanged','Positive tets alone do not prove nonoverlap or protected-cavity clearance']}
 out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'rest_geometry.npz',vertices=nodes,tetrahedra=t,cell_parameters=params,outer_triangles=np.array(outer_tri),surface_column_nodes=columns,midface_column_nodes=grid,oral_node_grid=oral_ids,all_surface_quads=all_quads,all_surface_triangles=np.array(surface_triangles),surface_triangle_parent_faces=np.array(triangle_parents),face_parts=np.array(source_parts),chart_tags=np.array(chart_tags),native_node_provenance=node_provenance,outer_column_native_ids=np.array(column_provenance),inner_carrier_displacement=u,protected_boundary_columns=np.array(protected),oral_interface_columns=border)
 (out/'geometry_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--depth-scale',type=float,default=1.);a=p.parse_args();build(a.output,a.depth_scale)
