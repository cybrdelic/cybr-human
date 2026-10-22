"""Reciprocal finite triangle-pair barrier and conservative separating-plane paths.

All eligible pairs are dynamically broadphased. Nonpenetration is for the two
provided finite triangle sets, not unrelated surfaces or tetrahedra. Closest
feature envelope gradients are exact away from ties; tangent is PSD GN only.
"""
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix,coo_matrix,diags
from moving_triangle_contact import (_carrier, _map_interval, _iadd, _imul,
    _down,_up,point_triangle_geometry,activated_log_barrier,InfeasibleContactError)
from robust_separating_plane import plane_certificate
EDGES=((0,1),(1,2),(2,0))

def segment_segment(a,b,c,d):
 u=b-a;v=d-c;w=a-c;aa=np.sum(u*u,1);bb=np.sum(u*v,1);cc=np.sum(v*v,1);dd=np.sum(u*w,1);ee=np.sum(v*w,1);den=aa*cc-bb*bb
 ss=np.divide(bb*ee-cc*dd,den,out=np.zeros_like(den),where=den>1e-14*aa*cc);tt=np.divide(aa*ee-bb*dd,den,out=np.zeros_like(den),where=den>1e-14*aa*cc)
 valid=(den>1e-14*aa*cc)&(ss>=0)&(ss<=1)&(tt>=0)&(tt<=1)
 ds=np.sum((w+ss[:,None]*u-tt[:,None]*v)**2,1);ds[~valid]=np.inf
 for s,t in [(np.zeros(len(a)),np.clip(ee/cc,0,1)),(np.ones(len(a)),np.clip((ee+bb)/cc,0,1)),(np.clip(-dd/aa,0,1),np.zeros(len(a))),(np.clip((bb-dd)/aa,0,1),np.ones(len(a)))]:
  val=np.sum((w+s[:,None]*u-t[:,None]*v)**2,1);k=val<ds;ds[k]=val[k];ss[k]=s[k];tt[k]=t[k]
 return ds,ss,tt

def edge_face_hits(A,B,tol):
 """Finite segment/face crossing, including touching. Coplanar handled by distances."""
 n=np.cross(B[:,1]-B[:,0],B[:,2]-B[:,0]);nl=np.linalg.norm(n,axis=1);n=n/nl[:,None];hit=np.zeros(len(A),bool)
 for i,j in EDGES:
  h0=np.sum((A[:,i]-B[:,0])*n,1);h1=np.sum((A[:,j]-B[:,0])*n,1);den=h0-h1
  t=np.divide(h0,den,out=np.full(len(A),-1.),where=np.abs(den)>tol)
  relevant=(t>=-1e-12)&(t<=1+1e-12)&((h0*h1)<=tol*tol)
  if not relevant.any():continue
  ids=np.flatnonzero(relevant);p=A[ids,i]+t[ids,None]*(A[ids,j]-A[ids,i]);geo=point_triangle_geometry(p,B[ids]);hit[ids]|=geo['gap']<=tol[ids]
 return hit

def triangle_geometry(A,B):
 A=np.asarray(A,float);B=np.asarray(B,float);n=len(A)
 if not n:return dict(gap=np.empty(0),normal=np.empty((0,3)),bary_a=np.empty((0,3)),bary_b=np.empty((0,3)),intersect=np.empty(0,bool),support_gap=np.empty(0),separating_normal=np.empty((0,3)),separation_fallback=np.empty(0,bool),separation_lower=np.empty(0))
 scale=np.maximum(1.,np.maximum(abs(A).max((1,2)),abs(B).max((1,2))));tol=256*np.finfo(float).eps*scale
 best=np.full(n,np.inf);ba=np.zeros((n,3));bb=np.zeros((n,3))
 for i in range(3):
  g=point_triangle_geometry(A[:,i],B);d=g['gap']**2;k=d<best;best[k]=d[k];ba[k]=np.eye(3)[i];bb[k]=g['barycentric'][k]
  g=point_triangle_geometry(B[:,i],A);d=g['gap']**2;k=d<best;best[k]=d[k];ba[k]=g['barycentric'][k];bb[k]=np.eye(3)[i]
 for i,j in EDGES:
  for k,l in EDGES:
   d,s,t=segment_segment(A[:,i],A[:,j],B[:,k],B[:,l]);take=d<best;best[take]=d[take];ba[take]=0;bb[take]=0;ba[take,i]=1-s[take];ba[take,j]=s[take];bb[take,k]=1-t[take];bb[take,l]=t[take]
 pa=np.einsum('ni,nij->nj',ba,A);pb=np.einsum('ni,nij->nj',bb,B);delta=pa-pb;gap=np.linalg.norm(delta,axis=1);normal=np.divide(delta,gap[:,None],out=np.zeros_like(delta),where=gap[:,None]>0)
 hit=(gap<=tol)|edge_face_hits(A,B,tol)|edge_face_hits(B,A,tol)
 # Convex distance normal must certify a separating plane. Reject unresolved
 # roundoff/near-degenerate cases rather than silently grant feasibility.
 support=np.min(np.einsum('nij,nj->ni',A,normal),1)-np.max(np.einsum('nij,nj->ni',B,normal),1)
 # The energy's closest-feature normal and its support gap remain unchanged.
 # Only an unresolved support plane can use the independent proof below.
 # Gap-floor hits and edge/face hits are NEVER cleared by this fallback.
 geometric_hit=hit.copy();unresolved=(support<=tol)|~np.isfinite(support)
 separating_normal=normal.copy();fallback=np.zeros(n,bool);lower=np.zeros(n)
 hit|=unresolved
 for i in np.flatnonzero(unresolved&~geometric_hit):
  proof=plane_certificate(A[i],B[i],tol=tol[i])
  if proof is not None:
   separating_normal[i]=proof['normal'];lower[i]=proof['lower'];fallback[i]=True;hit[i]=False
 return dict(gap=gap,normal=normal,bary_a=ba,bary_b=bb,intersect=hit,support_gap=support,separating_normal=separating_normal,separation_fallback=fallback,separation_lower=lower)

class DynamicTriangleContact:
 def __init__(self,carrier_a,faces_a,carrier_b,faces_b,*,face_weights_a=1.,face_weights_b=1.,clearance=.02,eligibility=None,chunk_size=128):
  self.C=_carrier(carrier_a,'triangle carrier A');self.M=_carrier(carrier_b,'triangle carrier B');self.faces_a=np.asarray(faces_a,int);self.faces_b=np.asarray(faces_b,int)
  if self.C.shape[1]!=self.M.shape[1]:raise ValueError('global DOFs differ')
  for f,C in [(self.faces_a,self.C),(self.faces_b,self.M)]:
   if f.ndim!=2 or f.shape[1]!=3 or not len(f) or f.min()<0 or f.max()>=C.shape[0] or (np.diff(np.sort(f,axis=1),axis=1)==0).any():raise ValueError('invalid faces')
  self.wa=np.broadcast_to(np.asarray(face_weights_a,float),(len(self.faces_a),)).copy();self.wb=np.broadcast_to(np.asarray(face_weights_b,float),(len(self.faces_b),)).copy();self.clearance=float(clearance);self.chunk_size=chunk_size
  if self.clearance<=0 or not np.isfinite(self.clearance) or np.any(self.wa<=0) or np.any(self.wb<=0) or not np.isfinite(self.wa).all() or not np.isfinite(self.wb).all():raise ValueError('positive weights/clearance')
  self.eligibility=None if eligibility is None else np.asarray(eligibility,bool)
  if self.eligibility is not None and self.eligibility.shape!=(len(self.faces_a),len(self.faces_b)):raise ValueError('eligibility shape')
 def candidates(self,q,direction=None,max_step=1.,clearance=None):
  h=self.clearance if clearance is None else clearance;A=_map_interval(self.C,q);B=_map_interval(self.M,q)
  if direction is not None:
   for name,C,P in [('A',self.C,A),('B',self.M,B)]:
    D=_map_interval(C,direction);end=_iadd(P,_imul((max_step,max_step),D));swept=(np.minimum(P[0],end[0]),np.maximum(P[1],end[1]))
    if name=='A':A=swept
    else:B=swept
  alo=_down(A[0][self.faces_a].min(1)-h);ahi=_up(A[1][self.faces_a].max(1)+h);blo=B[0][self.faces_b].min(1);bhi=B[1][self.faces_b].max(1);pairs=[]
  for start in range(0,len(alo),self.chunk_size):
   end=min(start+self.chunk_size,len(alo));hit=np.ones((end-start,len(blo)),bool)
   for j in range(3):hit&=(ahi[start:end,j,None]>=blo[None,:,j])&(alo[start:end,j,None]<=bhi[None,:,j])
   if self.eligibility is not None:hit&=self.eligibility[start:end]
   p=np.argwhere(hit);p[:,0]+=start;pairs.append(p)
  return np.concatenate(pairs)
 def evaluate(self,q):
  try:
   va=np.asarray(self.C@q);vb=np.asarray(self.M@q);A=va[self.faces_a];B=vb[self.faces_b]
   # Validate all triangles, including those excluded by current broadphase.
   for t in [A,B]:point_triangle_geometry(t[:,0],t)
   pairs=self.candidates(q);geo=triangle_geometry(A[pairs[:,0]],B[pairs[:,1]])
   good=not geo['intersect'].any();s=SimpleNamespace(feasible=good,reason='' if good else 'finite triangle intersection or unresolved separation',geometry=geo,pairs=pairs)
  except (InfeasibleContactError,ValueError,FloatingPointError) as exc:s=SimpleNamespace(feasible=False,reason=str(exc),geometry={'gap':np.empty(0)},pairs=np.empty((0,2),int))
  self.last_state=s;return s
 def require_feasible(self,q):
  s=self.evaluate(q)
  if not s.feasible:raise InfeasibleContactError(s.reason)
  return s
 def jacobian(self,s):
  p=s.pairs;g=s.geometry;n=len(p);rows=np.repeat(np.arange(n),3)
  SA=csr_matrix((g['bary_a'].ravel(),(rows,self.faces_a[p[:,0]].ravel())),shape=(n,self.C.shape[0]));SB=csr_matrix((g['bary_b'].ravel(),(rows,self.faces_b[p[:,1]].ravel())),shape=(n,self.M.shape[0]));D=(SA@self.C-SB@self.M).tocoo();rr=np.repeat(D.row,3);cc=(D.col[:,None]*3+np.arange(3)).ravel();data=(D.data[:,None]*g['normal'][D.row]).ravel()
  return csr_matrix((data,(rr,cc)),shape=(n,3*self.C.shape[1]))
 def __call__(self,q):
  s=self.require_feasible(q);p=s.pairs
  if not len(p):return 0.,np.zeros_like(q)
  b,db,_=activated_log_barrier(s.geometry['gap'],self.clearance);w=self.wa[p[:,0]]*self.wb[p[:,1]];J=self.jacobian(s);return float(w@b),np.asarray(J.T@(w*db)).reshape(q.shape)
 def normal_tangent(self,q):
  s=self.require_feasible(q);p=s.pairs;J=self.jacobian(s);w=self.wa[p[:,0]]*self.wb[p[:,1]];dd=activated_log_barrier(s.geometry['gap'],self.clearance)[2];return (J.T@diags(w*dd)@J).tocsr()
 def safe_step_bound(self,q,direction,*,safety=.9,max_step=1.):
  if not 0<safety<1 or max_step<0:raise ValueError('invalid step')
  self.require_feasible(q);p=self.candidates(q,direction,max_step,clearance=0.)
  if not len(p):return float(max_step)
  A=np.asarray(self.C@q)[self.faces_a[p[:,0]]];B=np.asarray(self.M@q)[self.faces_b[p[:,1]]];g=triangle_geometry(A,B)
  if g['intersect'].any():raise InfeasibleContactError('path starts intersecting')
  n=g['separating_normal']
  def projected_difference(P,Q):
   lo=_down(P[0][:,:,None,:]-Q[1][:,None,:,:]);hi=_up(P[1][:,:,None,:]-Q[0][:,None,:,:]);prod=_imul((lo,hi),(n[:,None,None,:],n[:,None,None,:]))
   total=(np.zeros(prod[0].shape[:-1]),np.zeros(prod[0].shape[:-1]))
   for j in range(3):total=_iadd(total,(prod[0][...,j],prod[1][...,j]))
   return total
  def indexed(C,values,faces,which):
   lo,hi=_map_interval(C,values);idx=faces[p[:,which]];return lo[idx],hi[idx]
  G=projected_difference(indexed(self.C,q,self.faces_a,0),indexed(self.M,q,self.faces_b,1))
  V=projected_difference(indexed(self.C,direction,self.faces_a,0),indexed(self.M,direction,self.faces_b,1))
  low=G[0];closing=V[0]<0
  if np.any(low<=0):return 0.
  bounds=np.divide(low,-V[0],out=np.full_like(low,np.inf),where=closing);bound=min(float(max_step),float(safety*np.min(bounds,initial=np.inf)))
  return max(0.,bound)

 def check_trial(self,q,direction,alpha,*,safety=.9):
  if not np.isfinite(alpha) or alpha<0:return False,'nonnegative finite step required'
  try:
   bound=self.safe_step_bound(q,direction,safety=safety,max_step=alpha)
   if bound<alpha:return False,'whole-triangle separating-plane path bound requires shorter step'
   state=self.evaluate(q+alpha*direction)
   return state.feasible,state.reason
  except (ValueError,InfeasibleContactError) as exc:return False,str(exc)
