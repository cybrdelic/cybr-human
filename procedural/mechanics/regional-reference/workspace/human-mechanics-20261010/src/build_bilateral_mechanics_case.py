"""Original connected regional mechanics inputs on the native-v16 facial chart.
Authoring choices remain uncalibrated. Existing real authored regional bones
supply attachment coordinates; omitted tissue is an explicitly named support.
No simulation is run by this builder.
"""
from pathlib import Path
from collections import defaultdict
import argparse,json,hashlib
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from zygomatic_support_build import closest_on_triangle
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def boundary(t):
 f=t[:,np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])].reshape(-1,3);key=np.sort(f,axis=1);_,i,c=np.unique(key,axis=0,return_index=True,return_counts=True)
 if c.max()>2:raise ValueError('Nonmanifold volume faces')
 return f[i[c==1]]
def surface_audit(f):
 e=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);ee=np.sort(e,axis=1);u,i,c=np.unique(ee,axis=0,return_inverse=True,return_counts=True);orient=np.bincount(i,weights=np.where(e[:,0]<e[:,1],1,-1));links=defaultdict(list)
 for tri in f:
  for k,n in enumerate(tri):links[int(n)].append((int(tri[(k+1)%3]),int(tri[(k+2)%3])))
 bad=[]
 for n,edges in links.items():
  ad=defaultdict(set)
  for a,b in edges:ad[a].add(b);ad[b].add(a)
  seen=set();stack=[next(iter(ad))]
  while stack:
   k=stack.pop()
   if k not in seen:seen.add(k);stack.extend(ad[k]-seen)
  if any(len(z)!=2 for z in ad.values()) or len(seen)!=len(ad):bad.append(n)
 return {'boundary_faces':len(f),'open_or_nonmanifold_edges':int((c!=2).sum()),'inconsistent_winding_edges':int((orient!=0).sum()),'nonmanifold_vertex_links':bad[:30],'nonmanifold_vertex_count':len(bad)}
def build(mesh,out):
 a=np.load(mesh);v=a['vertices'];t=a['tetrahedra'];grid=a['midface_column_nodes'];N=len(grid);columns=a['surface_column_nodes'];surface=a['all_surface_triangles'];parent=a['surface_triangle_parent_faces'];faces=a['all_surface_quads'];parts=a['face_parts'];params=a['cell_parameters'];patch=np.load(R/'inputs/native_midface_patch_v16/native_midface_patch_v16.npz');native_to_col={int(g):i for i,g in enumerate(a['outer_column_native_ids'])};fixed_cut=[]
 for name in ['outer_cut','right_eye_cut','left_eye_cut','right_nostril_cut','left_nostril_cut']:
  fixed_cut.extend(native_to_col[int(g)] for g in patch[name+'_sample_native_ids'])
 exterior_cut_nodes=np.unique(columns[np.array(fixed_cut)].ravel());mucosal_cut=np.unique(a['oral_node_grid'][:,-1,-1]);cut_nodes=np.unique(np.r_[exterior_cut_nodes,mucosal_cut]);touch=np.isin(t,cut_nodes).any(1);fixed=np.unique(t[touch]);inner=grid[:,-1];free_inner=np.setdiff1d(inner,fixed)
 # A single passive matrix and a dermal proxy in the outer depth interval.
 x=v*.001;dm=np.stack([x[t[:,i]]-x[t[:,0]] for i in [1,2,3]],-1);det=np.linalg.det(dm)
 if det.min()<=0:raise ValueError('Nonpositive rest tet')
 vol=det/6;grad=np.zeros((len(t),4,3));grad[:,1:]=np.linalg.inv(dm);grad[:,0]=-grad[:,1:].sum(1);young=np.where(params[:,1]==0,24000.,3200.);nu=.43;mu=young/(2*(1+nu));lam=young*nu/((1+nu)*(1-2*nu))+mu;mat=np.c_[vol,mu,lam];mass=np.zeros(len(v))
 for j in range(4):np.add.at(mass,t[:,j],1000.*vol/4)
 # Authored surface ribbons approximate two zygomaticus-like bodies. They stop
 # at the perioral bulk: the entire actual oral wedge is passive in this test.
 centers=np.array([v[columns[q,0]].mean(0) for q in faces]);face_adj=defaultdict(set);edgeowners=defaultdict(list)
 for fi,q in enumerate(faces):
  for e in zip(q,np.roll(q,-1)):edgeowners[tuple(sorted(map(int,e)))].append(fi)
 for owners in edgeowners.values():
  if len(owners)==2:aa,bb=owners;face_adj[aa].add(bb);face_adj[bb].add(aa)
 bone=np.load(R/'output/authored-bone-support-v1/bone_support.npz');land=json.loads((R/'output/authored-bone-support-v1/landmarks.json').read_text());origin=np.array(land['zygomaticus_major_origin_patch_center']['point']);regions=[];origins=[];active_report=[]
 for side in [1,-1]:
  xx=side*centers[:,0];yy=centers[:,1];cx=np.interp(yy,[-61,-52,-38,-22,-5],[28,29,36,43,52]);mask=(abs(xx-cx)<6.5)&(yy>=-61)&(yy<=-3)&(parts!=20);candidate=set(np.flatnonzero(mask).tolist());seed=min(candidate,key=lambda k:np.linalg.norm(centers[k]-origin*np.array([side,1,1])));connected=set();stack=[seed]
  while stack:
   f=stack.pop()
   if f in candidate and f not in connected:connected.add(f);stack.extend(face_adj[f]-connected)
  # Fill only vertex-fan gaps within the same candidate corridor when possible.
  active=np.flatnonzero(np.isin(params[:,0],list(connected))&(params[:,1]>=1));af=boundary(t[active]);audit=surface_audit(af)
  if audit['open_or_nonmanifold_edges'] or audit['inconsistent_winding_edges'] or audit['nonmanifold_vertex_count']:
   raise ValueError('Authored muscle ribbon is not a manifold finite region: '+json.dumps(audit))
  c=v[t[active]].mean(1);direction=origin*np.array([side,1,1])-c;direction/=np.linalg.norm(direction,axis=1)[:,None];regions.append({'id':'right-zygomatic-like' if side==1 else 'left-zygomatic-like','tetrahedra':active.tolist(),'fibreDirections':direction.tolist(),'activation':0.,'stiffness':24000.,'scope':'Authored approximate incoming cheek body; oral wedge passive; not measured muscle segmentation'})
  # The deep patch follows a rigid bone landmark through its actual rest offset.
  candidate_nodes=np.intersect1d(np.unique(t[active]),free_inner);target=origin*np.array([side,1,1]);order=np.argsort(np.linalg.norm(v[candidate_nodes]-target,axis=1));on=candidate_nodes[order[:6]];fixed=np.unique(np.r_[fixed,on]);ties=[]
  btri=bone['vertices'][bone['triangles']]
  for node in on:
   query=v[node]*np.array([side,1,1]);points,ds=closest_on_triangle(query,btri);bi=int(ds.argmin());bp=points[bi];bb=btri[bi];bary=np.r_[0.,np.linalg.lstsq((bb[1:]-bb[0]).T,bp-bb[0],rcond=None)[0]];bary[0]=1-bary[1:].sum();bp=bp*np.array([side,1,1]);ties.append({'node':int(node),'bone_triangle':bi,'bone_weights':bary.tolist(),'bone_point_design_units':bp.tolist(),'rest_offset_design_units':(v[node]-bp).tolist(),'rest_offset_length_design_units':float(np.linalg.norm(v[node]-bp))})
  origins.append({'side':side,'nodes':on.tolist(),'authored_patch_landmark':target.tolist(),'ties':ties,'scope':'Distributed rigid bone-origin support with explicit periosteal/omitted-attachment rest offsets; not a zero-gap anatomical enthesis'})
  active_report.append({'side':side,'tetrahedra':len(active),'rest_volume_solver_units_cubed':float(vol[active].sum()),'boundary':audit,'nearest_origin_offset':float(np.linalg.norm(v[on]-target,axis=1).min()),'maximum_origin_offset':float(np.linalg.norm(v[on]-target,axis=1).max()),'actual_bone_surface_offset_range':[min(q['rest_offset_length_design_units'] for q in ties),max(q['rest_offset_length_design_units'] for q in ties)]})
 out.mkdir(parents=True,exist_ok=False);arrays={'rest':x.astype('<f8'),'nodes':t.astype('<u4'),'gradients':grad.astype('<f8'),'material':mat.astype('<f8'),'mass':mass.astype('<f8')};contract={};offset=0
 with (out/'arrays.bin').open('wb') as h:
  for name,z in arrays.items():
   if offset%8:h.write(b'\0'*(8-offset%8));offset+=(8-offset%8)
   raw=z.tobytes();contract[name]={'byte_offset':offset,'dtype':z.dtype.str,'shape':list(z.shape),'bytes':len(raw)};h.write(raw);offset+=len(raw)
 bf=bone['triangles'];bv=bone['vertices'];left=bv.copy();left[:,0]*=-1;bonev=np.r_[bv,left]*.001;bonef=np.r_[bf,bf[:,[0,2,1]]+len(bv)].astype('<u4')
 with (out/'bones.bin').open('wb') as h:h.write(bonev.astype('<f8').tobytes());h.write(bonef.tobytes())
 data={'schema':'connected-bilateral-native-mechanics-v1','status':'Input construction only; no activated or accepted mechanical state','source_geometry_sha256':sha(mesh),'coordinate_scale_solver_per_design_unit':.001,'units_note':'Authored design scale, not calibrated human millimetres','model':{'schema':'cybr-layered-fem-v1','nodes':len(v),'tetrahedra':len(t)},'binary':'arrays.bin','arrays':contract,'fixed_nodes':fixed.tolist(),'frozen_transfer_gradient_nodes':cut_nodes.tolist(),'omitted_deep_tissue_spring_nodes':np.setdiff1d(inner,fixed).tolist(),'omitted_deep_tissue_spring_stiffness':.5,'boundary_triangles':boundary(t).tolist(),'muscles':regions,'bone_origin_supports':origins,'bones':{'binary':'bones.bin','vertex_count':len(bonev),'triangle_count':len(bonef),'source_sha256':sha(R/'output/authored-bone-support-v1/bone_support.npz'),'scope':'Original bilateral mirrored regional zygomatic/maxillary support; central skull/cartilage incomplete'},'limits':['No solved state or dynamic contact acceptance','Materials and muscle geometry authored, uncalibrated','Omitted lid, nostril and deep-cavity interfaces fixed with one-ring rest collar','Only incoming cheek volumes can activate; oral tissue stays passive','Rigid origin supports retain explicit finite rest offsets','Central bone/cartilage and full-head mechanics remain incomplete']}
 (out/'domain.json').write_text(json.dumps(data,separators=(',',':'))+'\n');report={'source_geometry_sha256':sha(mesh),'nodes':len(v),'tetrahedra':len(t),'fixed_nodes':len(fixed),'active_regions':active_report,'mass_positive':bool((mass>0).all()),'minimum_rest_tet_volume':float(vol.min()),'binary_bytes':offset,'domain_sha256':sha(out/'domain.json'),'binary_sha256':sha(out/'arrays.bin')};(out/'construction_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mesh',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.mesh,a.output)
