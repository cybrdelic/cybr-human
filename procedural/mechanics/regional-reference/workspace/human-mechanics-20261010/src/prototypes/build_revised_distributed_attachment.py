#!/usr/bin/env python3
"""Source-only distributed origin with unchanged bone/tissue cap coordinates.
Four-face footprint fill; fixed exact-bone collar; two cap projection layers;
strict-kernel transition between differently triangulated planar sections.
"""
import sys,json,numpy as np,warnings,hashlib
from pathlib import Path
from scipy.optimize import linprog,OptimizeWarning
warnings.filterwarnings('ignore',category=OptimizeWarning)
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/prototypes'));import design_distributed_origin_attachment as d
from build_distributed_fixed_collar import build
from visibility_annulus import visibility_zipper
from design_finite_origin_attachments import boundary,determinants
r=ROOT/'output';OUT=r/'distributed-origin-attachment-feasibility-v1/revised';OUT.mkdir(exist_ok=True)
def layer_tets(x,lowfaces,upper_map):
 """Use the declared lower-face winding, directed toward its upper cap.

 Sorting vertex IDs fixes shared prism diagonals. Permutation parity restores
 the declared face orientation; determinants only reject invalid geometry.
 """
 out=[]
 for q in lowfaces:
  a,b,c=sorted(q);A,B,C=[upper_map[z] for z in [a,b,c]];t=np.array([[a,b,c,A],[b,c,A,B],[c,B,C,A]])
  odd=sum(int(q[i])>int(q[j]) for i,j in ((0,1),(0,2),(1,2)))%2
  if odd:t=t[:,[0,2,1,3]]
  out.extend(t)
 out=np.array(out)
 assert determinants(x,out).min()>0, 'Declared prism winding or geometry is invalid'
 return out
def normalized_normals(tr):
 n=np.cross(tr[:,1]-tr[:,0],tr[:,2]-tr[:,0]);return n/np.linalg.norm(n,axis=1)[:,None]
def main():
 a=np.load(r/'bilateral-midface-bone-aware-v2/rest_geometry.npz');v=a['vertices'];f,_=boundary(a['tetrahedra']);b=np.load(r/'authored-bone-support-v1/bone_support.npz');patch=np.load(r/'distributed-origin-attachment-feasibility-v1/revised_footprint.npz')['source_triangle_ids'];records=[]
 for side,name,chosen in [(1,'right',[5644,5641,5578,5580,5577]),(-1,'left',[6066,6063,6002,6001])]:
  d.guard();x,collar,rf,ids=build(side,b['vertices'],b['triangles'],patch);nr=len(ids);root=x[nr:];tipfaces=f[chosen][:,::-1];tids=np.unique(tipfaces);tip=v[tids];mp={int(g):i for i,g in enumerate(tids)};tf=np.array([[mp[int(q)] for q in z] for z in tipfaces]);norm=np.r_[normalized_normals(root[rf]),normalized_normals(tip[tf])];delta=(tip[:,None,:]-root[None,:,:]).reshape(-1,3);lp=linprog([0,0,0,-1],A_ub=np.r_[np.c_[-delta,np.ones(len(delta))],np.c_[-norm,np.zeros(len(norm))]],b_ub=np.r_[np.zeros(len(delta)),np.full(len(norm),-.01)],bounds=[(-1,1)]*3+[(None,None)],method='highs',options={'threads':1,'time_limit':5});assert lp.success;axis=lp.x[:3];axis/=np.linalg.norm(axis);gap=float(min(tip@axis)-max(root@axis));assert gap>.15;sb=max(root@axis)+.25*gap;st=min(tip@axis)-.25*gap;rp=root+(sb-root@axis)[:,None]*axis;tp=tip+(st-tip@axis)[:,None]*axis
  rootplane_ids=np.arange(2*nr,3*nr);tissueplane_ids=np.arange(3*nr,3*nr+len(tids));tip_ids=np.arange(3*nr+len(tids),3*nr+2*len(tids));x=np.r_[x,rp,tp,tip];rpcap=rf+2*nr;tpcap=tf+3*nr;tipcap=tf+3*nr+len(tids);rl=d.ring(rpcap);tl=d.ring(tpcap);u=np.array([0.,1.,0.]);u-=axis*(u@axis);u/=np.linalg.norm(u);w=np.cross(axis,u);rl=d.rotate_by_angle(rl,x,u,w);tl=d.rotate_by_angle(tl,x,u,w);sf=d.zipper(rl,tl,x);shell=np.r_[rpcap[:,[0,2,1]],tpcap,sf];k=d.kernel(x,shell)
  if k is None or k[1]<=1e-5:
   points=[]
   for loop in [rl,tl]:
    q=x[loop];ed=np.roll(q,-1,axis=0)-q;nn=np.cross(ed,axis);nn/=np.linalg.norm(nn,axis=1)[:,None];off=np.einsum('ij,ij->i',nn,q);kk=linprog([0,0,0,-1],A_ub=np.c_[nn,np.ones(len(nn))],b_ub=off,A_eq=np.array([[*axis,0.]]),b_eq=[float(q[0]@axis)],bounds=[(None,None)]*4,method='highs',options={'threads':1});assert kk.success and kk.x[3]>0;points.append(kk.x[:3])
   center=np.mean(points,axis=0);vs=visibility_zipper(rl,tl,x,center)
   if vs:
    sf=vs[1];shell=np.r_[rpcap[:,[0,2,1]],tpcap,sf];k=d.kernel(x,shell)
  rec=dict(side=side,root_triangles=len(patch),tissue_triangles=len(chosen),axis=axis.tolist(),separating_cap_gap=gap,minimum_cap_normal_dot=float(min(norm@axis)),planar_bridge_gap=float(st-sb),kernel_pass=bool(k and k[1]>1e-5))
  if k and k[1]>1e-5:
   center,radius=k;x=np.r_[x,center[None]];root_layer=layer_tets(x,rf+nr,{int(a):int(z) for a,z in zip(np.arange(nr,2*nr),rootplane_ids)});tip_layer=layer_tets(x,tpcap,{int(a):int(z) for a,z in zip(tissueplane_ids,tip_ids)});transition=np.c_[np.full(len(shell),len(x)-1),shell];t=np.r_[collar,root_layer,tip_layer,transition];dt=determinants(x,t);qual=d.meanratio(x,t);assert dt.min()>0;bf,_=boundary(t)
   rec.update(nodes=len(x),new_nodes=len(x)-len(tids),tets=len(t),fixed_collar_nodes=2*nr,kernel_radius=radius,minimum_tet_volume=float(dt.min()/6),minimum_quality=float(qual.min()),quality_1pct=float(np.percentile(qual,1)),quality_10pct=float(np.percentile(qual,10)),quality_by_region={tag:dict(minimum=float(qual[start:end].min()),median=float(np.median(qual[start:end]))) for tag,start,end in [('fixed_collar',0,len(collar)),('root_projection',len(collar),len(collar)+len(root_layer)),('tissue_projection',len(collar)+len(root_layer),len(collar)+len(root_layer)+len(tip_layer)),('bridge',len(t)-len(transition),len(t))]},status='Positive source candidate only; complete overlap audit pending')
   np.savez_compressed(OUT/(name+'_attachment_candidate.npz'),vertices=x,tetrahedra=t,boundary_faces=bf,root_faces=rf,root_source_bone_vertex_ids=ids,root_source_bone_triangle_ids=patch,shared_tissue_local_node_ids=tip_ids,shared_tissue_source_node_ids=tids,shared_tissue_source_face_ids=chosen,fixed_local_nodes=np.arange(2*nr),region_tet_offsets=np.array([0,len(collar),len(collar)+len(root_layer),len(collar)+len(root_layer)+len(tip_layer),len(t)]),axis=axis,kernel=center)
  else:rec['status']='Planar side annulus rejected; no positive kernel'
  records.append(rec);print(json.dumps(rec),flush=True)
 (OUT/'construction_report.json').write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':main()
