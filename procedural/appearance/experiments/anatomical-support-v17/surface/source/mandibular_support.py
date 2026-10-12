"""Original 2-D curved mandibular Loop-subdivision control cage.
Independent landmark controls, no horizontal station profiles or neck carve.
The absolute limit surface replaces low-frequency lower-face geometry inside a broad curved attachment; a shared volumetric carrier preserves tissue order.
"""
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay,cKDTree
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import lsq_linear,minimize,LinearConstraint,Bounds

def smooth(t):
 t=np.clip(t,0,1);return t*t*t*(10+t*(-15+6*t))

def subdivide(v,f):
 edges=np.sort(np.r_[f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]],axis=1);edges,inv,cnt=np.unique(edges,axis=0,return_inverse=True,return_counts=True);n=len(v)
 neighbors=[set() for _ in range(n)];bn=[[] for _ in range(n)]
 for (a,b),c in zip(edges,cnt):
  neighbors[a].add(b);neighbors[b].add(a)
  if c==1:bn[a].append(b);bn[b].append(a)
 updated=v.copy()
 for i,ns in enumerate(neighbors):
  if bn[i]:updated[i]=.75*v[i]+.125*v[bn[i]].sum(0)
  else:
   nn=len(ns);beta=(.625-(.375+.25*np.cos(2*np.pi/nn))**2)/nn;updated[i]=(1-nn*beta)*v[i]+beta*v[list(ns)].sum(0)
 opp=np.r_[f[:,2],f[:,0],f[:,1]];oppsum=np.zeros((len(edges),v.shape[1]));np.add.at(oppsum,inv,v[opp]);ev=.375*v[edges].sum(1)+.125*oppsum;ev[cnt==1]=.5*v[edges[cnt==1]].sum(1)
 a,b,c=f.T;ab,bc,ca=inv.reshape(3,-1)+n;nf=np.r_[np.c_[a,ab,ca],np.c_[b,bc,ab],np.c_[c,ca,bc],np.c_[ab,bc,ca]]
 return np.r_[updated,ev],nf

def rebuild_mandibular_support(p,output_dir):
 # Curved perimeter travels from lower-lip bearing, across the mandibular
 # angles, around soft submental/neck support and returns symmetrically.
 perimeter=np.array([[0,-65],[20,-64],[37,-68],[47,-78],[49,-96],[45,-120],[40,-146],[20,-162],[0,-168]],float)
 # x,y, dx,dy,dz. Each anatomical handle is independent in three dimensions.
 handles=np.array([
 [0,-76,0,0,-2.0], [12,-79,-.2,0,-1.1], [26,-79,-.5,.5,.1],
 [0,-91,0,-.6,6.2], [13,-91,-.9,-.2,5.3], [24,-95,-2.2,1.2,.4],
 [0,-105,0,.5,2.0], [13,-106,-.6,1.2,1.8], [28,-112,-1.2,2.8,1.2],
 [40,-98,-1.8,4.5,1.0], [42,-82,-1.2,2.0,.5], [39,-121,-.4,1.8,.8],
 [0,-122,0,0,3.5], [18,-128,0,.4,2.8], [0,-146,0,0,1.2], [30,-148,0,0,.3]
 ],float)
 uv=np.r_[perimeter,handles[:,:2]];d=np.r_[np.zeros((len(perimeter),3)),handles[:,2:]];half=Delaunay(uv);faces=half.simplices.copy()
 pos=np.flatnonzero(uv[:,0]>0);mirror=np.arange(len(uv));mirror[pos]=len(uv)+np.arange(len(pos));uv=np.r_[uv,uv[pos]*[-1,1]];d=np.r_[d,d[pos]*[-1,1,1]];faces=np.r_[faces,mirror[faces][:,::-1]]
 # Source depth is sampled only from our existing procedural anterior skin.
 use=(p[:,1]<-62)&(p[:,1]>-171)&(p[:,2]>30);sp=p[use];tree=cKDTree(sp[:,:2]);dist,ii=tree.query(uv,k=32);w=1/np.maximum(dist,.10)**2;depth=(sp[ii,2]*w).sum(1)/w.sum(1)
 source=np.c_[uv,depth];target=source+d
 data=np.c_[uv,source,target,np.eye(len(uv))]
 coarse=data.copy();coarsef=faces.copy()
 for _ in range(5):data,faces=subdivide(data,faces)
 # Fit the absolute subdivision cage to the original anterior skin before
 # artistic handle changes. This compensates subdivision shrinkage rather
 # than retaining a high-frequency residual containing the old jaw ledge.
 weights=data[:,8:];carrier=LinearNDInterpolator(data[:,:2],weights,fill_value=0)(sp[:,:2]);valid=carrier.sum(1)>.99;A=carrier[valid];b=sp[valid,2]
 # Bounded bending-energy fit: avoid compensating subdivision shrinkage
 # with off-axis overshoots and large alternating control values.
 adjacency=[set() for _ in range(len(uv))]
 for tri in coarsef:
  for i,j in [(0,1),(1,2),(2,0)]:adjacency[tri[i]].add(tri[j]);adjacency[tri[j]].add(tri[i])
 L=np.eye(len(uv))
 for i,ns in enumerate(adjacency):L[i,list(ns)]=-1/len(ns)
 fit_result=lsq_linear(np.r_[A,np.sqrt(20)*L,np.sqrt(.1)*np.eye(len(uv))],np.r_[b,np.zeros(len(uv)),np.sqrt(.1)*depth],bounds=(depth-8,depth+12),tol=1e-10,max_iter=100)
 assert fit_result.success,fit_result.message
 fit=fit_result.x
 # A shared support constraint on the limit cage prevents the rounded chin
 # from buying its silhouette by excavating the existing lateral jaw plane.
 sample=np.array([[sg*x,y] for sg in [-1,1] for x in [30,34,38,42] for y in [-116,-110,-104,-98]],float)
 C=LinearNDInterpolator(data[:,:2],weights,fill_value=0)(sample);support_z=LinearNDInterpolator(sp[:,:2],sp[:,2],fill_value=np.nan)(sample);valid_support=(C.sum(1)>.99)&np.isfinite(support_z);C=C[valid_support];support_z=support_z[valid_support];sample=sample[valid_support]
 ce=np.sort(np.r_[faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]],axis=1);ce,cc=np.unique(ce,axis=0,return_counts=True);segs=data[ce[cc==1],:2];stree=cKDTree(segs.mean(1));_,si=stree.query(sample,k=4);sb=segs[si,0];ee=segs[si,1]-sb;off=sample[:,None,:]-sb;tt=np.clip((off*ee).sum(2)/(ee*ee).sum(2),0,1);dd=np.linalg.norm(off-tt[:,:,None]*ee,axis=2).min(1);support_gate=smooth(dd/20)*smooth((-66-sample[:,1])/5)*smooth((support_z-35)/45)
 minimum=support_z-1.25/np.maximum(support_gate,1e-6)-C@d[:,2]
 assert np.min(C@(depth+16)-minimum)>0,('infeasible support bounds',float(np.min(C@(depth+16)-minimum)))
 Q=A.T@A+20*L.T@L+.1*np.eye(len(uv));rhs=A.T@b+.1*depth
 refined=minimize(lambda u:(.5*u@Q@u-u@rhs)/len(A),fit,jac=lambda u:(Q@u-rhs)/len(A),method='SLSQP',bounds=Bounds(depth-8,depth+16),constraints=[LinearConstraint(C,minimum,np.inf)],options={'maxiter':200,'ftol':1e-7})
 assert refined.success,(refined.message,float(np.min(C@refined.x-minimum)))
 fit=refined.x
 assert np.min(C@fit-minimum)>-1e-5

 for i in range(len(perimeter)+len(handles)):
  if mirror[i]!=i:fit[i]=fit[mirror[i]]=(fit[i]+fit[mirror[i]])/2
 coarse[:,4]=fit;coarse[:,7]=fit+d[:,2];data[:,4]=weights@fit;data[:,7]=weights@(fit+d[:,2])
 delta=data[:,5:8]-data[:,2:5];field=LinearNDInterpolator(data[:,:2],delta,fill_value=0)
 target_depth=LinearNDInterpolator(data[:,:2],data[:,7],fill_value=np.nan)
 source_depth=LinearNDInterpolator(sp[:,:2],sp[:,2],fill_value=np.nan)
 # Pin a finite band around the truly curved perimeter, not a rectangular
 # X/Y chart. C2 volumetric release also protects posterior neck tissue.
 edges=np.sort(np.r_[faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]],axis=1);edges,count=np.unique(edges,axis=0,return_counts=True);boundary=data[edges[count==1],:2];btree=cKDTree(boundary.mean(1))
 def map_points(points):
  q=points.copy();ids=np.flatnonzero((points[:,1]<-66)&(points[:,1]>-169)&(abs(points[:,0])<64)&(points[:,2]>35))
  for start in range(0,len(ids),16384):
   ix=ids[start:start+16384];a=points[ix];delta=field(a[:,:2]);absolute=target_depth(a[:,:2]);original=source_depth(a[:,:2]);delta[:,2]=np.nan_to_num(absolute-original);_,bi=btree.query(a[:,:2],k=4);b=boundary[bi,0];e=boundary[bi,1]-b;offset=a[:,None,:2]-b;t=np.clip((offset*e).sum(2)/(e*e).sum(2),0,1);bd=np.linalg.norm(offset-t[:,:,None]*e,axis=2).min(1)
   gate=smooth(bd/20)*smooth((-66-a[:,1])/5)*smooth((a[:,2]-35)/45);q[ix]+=gate[:,None]*delta
  return q
 q=map_points(p);assert np.array_equal(q[p[:,1]>=-66],p[p[:,1]>=-66]);assert np.array_equal(q[p[:,2]<=35],p[p[:,2]<=35])
 grid=np.stack(np.meshgrid(np.linspace(-62,62,25),np.linspace(-167,-67,31),np.linspace(36,129,25),indexing='ij'),-1).reshape(-1,3);jac=np.empty((len(grid),3,3));eps=.01
 for axis in range(3):
  h=np.zeros(3);h[axis]=eps;jac[:,:,axis]=(map_points(grid+h)-map_points(grid-h))/(2*eps)
 det=np.linalg.det(jac);sv=np.linalg.svd(jac,compute_uv=False)[:,-1];assert det.min()>.20 and sv.min()>.20,(det.min(),sv.min(),grid[det.argmin()].tolist(),(map_points(grid[det.argmin():det.argmin()+1])-grid[det.argmin()]).tolist(),float(np.linalg.norm(q-p,axis=1).max()),grid[sv.argmin()].tolist())
 np.savez_compressed(Path(output_dir)/'MANDIBULAR_CAGE.npz',coarse_source=coarse[:,2:5],coarse_target=coarse[:,5:8],coarse_triangles=coarsef,source=data[:,2:5],target=data[:,5:8],uv=data[:,:2],triangles=faces,boundary=boundary)
 return q,{'representation':'Absolute two-dimensional triangular Loop-subdivision lower-face surface, with independent anatomical XYZ handles and a broad curved pinned perimeter','fit_regularization':{'bending_weight':20,'source_depth_bounds':[-8,12],'support_refinement_bounds':[-8,16],'minimum_lateral_support_allowance':-1.25,'support_constraints':len(C),'converged':bool(fit_result.success)},'coarse_vertices':len(coarse),'coarse_triangles':len(coarsef),'subdivision_levels':5,'limit_approximation_triangles':len(faces),'positive_neck_support':'No authored neck-carve handle; positive soft submental and throat bearing controls','maximum_motion':float(np.linalg.norm(q-p,axis=1).max()),'modified_vertices':int(np.any(q!=p,axis=1).sum()),'finite_lips_and_upper_face_exact_above_y':-66.,'sampled_xyz_jacobian':{'points':len(grid),'minimum_determinant':float(det.min()),'minimum_singular_value':float(sv.min())},'limits':['Original procedural source only; generated target is qualitative','Finite subdivision approximation and sampled Jacobians do not prove global injectivity','The dense attachment topology samples the new absolute limit surface; no mechanical rest replacement']}
