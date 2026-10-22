"""Small explicit-reference finite-strain tissue mechanics, N/mm/assumed s.
No desired exterior coordinates or expression targets. Material/reference
anatomy, attachments and activation remain explicit input assumptions.
"""
import numpy as np
from scipy.optimize import minimize
class Solid:
 def __init__(self,X,T,mu,lam,active,fibre):
  self.X=np.asarray(X);self.T=np.asarray(T);self.mu=np.broadcast_to(mu,len(T));self.lam=np.broadcast_to(lam,len(T));self.active=np.asarray(active,bool);self.fibre=np.asarray(fibre)
  dm=np.stack([X[T[:,j]]-X[T[:,0]]for j in [1,2,3]],axis=2);det=np.linalg.det(dm)
  if np.any(det<=0):raise ValueError('Reference must have finite positive tetrahedra')
  self.volume=det/6;self.inv=np.linalg.inv(dm);self.mass=np.zeros(len(X));np.add.at(self.mass,T.ravel(),np.repeat(self.volume*1e-9/4,4))
 def energy(self,q,activation=0.):
  T=self.T;ds=np.stack([q[T[:,j]]-q[T[:,0]]for j in [1,2,3]],axis=2);F=ds@self.inv;A=np.broadcast_to(np.eye(3),(len(T),3,3)).copy();gamma=1-.2*activation;f=self.fibre[self.active]
  A[self.active]=np.sqrt(gamma)*np.eye(3)+(1/gamma-np.sqrt(gamma))*f[:,:,None]*f[:,None,:]
  Fe=F@A;J=np.linalg.det(Fe)
  if not np.isfinite(J).all() or np.any(J<=0):return 1e30,np.zeros_like(q),float(np.nanmin(J))
  l=np.log(J);it=np.linalg.inv(Fe).transpose(0,2,1);P=(self.mu[:,None,None]*(Fe-it)+self.lam[:,None,None]*l[:,None,None]*it)@A.transpose(0,2,1)
  W=.5*self.mu*((Fe*Fe).sum((1,2))-3)-self.mu*l+.5*self.lam*l*l;G=self.volume[:,None,None]*(P@self.inv.transpose(0,2,1));grad=np.zeros_like(q)
  for j in range(3):np.add.at(grad,T[:,j+1],G[:,:,j]);np.add.at(grad,T[:,0],-G[:,:,j])
  return float(self.volume@W),grad,float(np.linalg.det(F).min())

class Joint:
 def __init__(self,solid,fixed,terms=(),gravity=(0.,-9810.,0.)):
  self.solid=solid;self.fixed=np.asarray(fixed,int);self.free=np.setdiff1d(np.arange(len(solid.X)),fixed);self.terms=list(terms);self.gravity=np.asarray(gravity)
 def energy(self,q,activation=0):
  e,g,j=self.solid.energy(q,activation)
  if e>=1e29:return e,g,j
  e-=np.sum(self.solid.mass[:,None]*self.gravity*q);g-=self.solid.mass[:,None]*self.gravity
  for fn in self.terms:
   ee,gg=fn(q);e+=ee;g+=gg
  return float(e),g,j
 def step(self,q,activation=0,pred=None,dt=.01,maxiter=300):
  q=q.copy();m=self.solid.mass
  def obj(z):
   q[self.free]=z.reshape(-1,3);e,g,j=self.energy(q,activation)
   if pred is not None:e+=.5*np.sum(m[:,None]*(q-pred)**2)/dt**2;g+=m[:,None]*(q-pred)/dt**2
   return e,g[self.free].ravel()
  r=minimize(obj,q[self.free].ravel(),jac=True,method='L-BFGS-B',options=dict(maxiter=maxiter,ftol=1e-12,gtol=1e-8,maxcor=15,maxls=30));q[self.free]=r.x.reshape(-1,3);e,g,j=self.energy(q,activation);res=g.copy()
  if pred is not None:res+=m[:,None]*(q-pred)/dt**2
  return q,dict(optimizer_success=bool(r.success),message=str(r.message),iterations=r.nit,energy=e,Jmin=j,physical_force_max=float(np.linalg.norm(g[self.free],axis=1).max()),implicit_residual_max=float(np.linalg.norm(res[self.free],axis=1).max()))

class AffineBond:
 """Reciprocal finite area averaged attachment; row-rest vectors explicit."""
 def __init__(self,matrix,offset,weights):self.B=matrix;self.offset=offset;self.weights=np.asarray(weights)
 def __call__(self,q):
  r=self.B@q-self.offset;return .5*float(np.sum(self.weights[:,None]*r*r)),self.B.T@(self.weights[:,None]*r)

class MovingFacetContact:
 """Barycentric local normal contact with exact orientation derivatives.
 Valid only while tracked material points stay within their initial face chart;
 callers must check tangential footprint and full source contact independently.
 Not a global collision detector.
 """
 def __init__(self,receiver,face_nodes,barycentric,weights,clearance=0):self.receiver=np.asarray(receiver);self.face=np.asarray(face_nodes);self.bary=np.asarray(barycentric);self.weights=np.asarray(weights);self.clearance=clearance
 def __call__(self,q):
  t=q[self.face];e1=t[:,1]-t[:,0];e2=t[:,2]-t[:,0];c=np.cross(e1,e2);cl=np.linalg.norm(c,axis=1);n=c/np.maximum(cl[:,None],1e-12);p=np.einsum('ni,nij->nj',self.bary,t);delta=q[self.receiver]-p;gap=np.sum(delta*n,axis=1);r=np.minimum(gap-self.clearance,0);force=self.weights*r;g=np.zeros_like(q);np.add.at(g,self.receiver,force[:,None]*n)
  hn=force[:,None]*(delta-gap[:,None]*n)/np.maximum(cl[:,None],1e-12);a=np.cross(e2,hn);b=np.cross(hn,e1)
  extra=np.stack([-a-b,a,b],axis=1)-self.bary[:,:,None]*force[:,None,None]*n[:,None,:]
  for j in range(3):np.add.at(g,self.face[:,j],extra[:,j])
  return .5*float(np.sum(self.weights*r*r)),g

class FrozenGapContact:
 """Piecewise-linear local source-contact approximation, audited externally."""
 def __init__(self,matrix,offset,reference_points,gap,gradient,valid,area,density=.2,clearance=.04):
  self.C=matrix;self.offset=offset;self.points=reference_points;self.gap=np.asarray(gap);self.gradient=np.asarray(gradient);self.weights=np.where(valid,area*density,0);self.clearance=clearance
 def __call__(self,q):
  s=self.C@q+self.offset;r=np.minimum(self.gap+np.sum((s-self.points)*self.gradient,axis=1)-self.clearance,0);force=self.weights*r;return .5*float(np.sum(self.weights*r*r)),self.C.T@(force[:,None]*self.gradient)

class EmbeddedSolid:
 """Objective finite-volume connective element with reciprocal affine carriers."""
 def __init__(self,mapping,solid):self.M=mapping;self.solid=solid;self.last_Jmin=1.
 def __call__(self,q):
  e,g,j=self.solid.energy(self.M@q);self.last_Jmin=j;return e,self.M.T@g

def linear_stiffness(solid):
 from scipy.sparse import coo_matrix
 grad=np.concatenate([-solid.inv.sum(1)[:,None,:],solid.inv],axis=1);dot=np.einsum('tai,tbi->tab',grad,grad);eye=np.eye(3)
 # Direct form of mu * gradNb_i gradNa_j, avoiding index ambiguity.
 H=solid.volume[:,None,None,None,None]*(solid.mu[:,None,None,None,None]*np.einsum('tab,ij->taibj',dot,eye)+solid.mu[:,None,None,None,None]*np.einsum('tbi,taj->taibj',grad,grad)+solid.lam[:,None,None,None,None]*np.einsum('tai,tbj->taibj',grad,grad))
 dof=(solid.T[:,:,None]*3+np.arange(3)).reshape(-1,12);return coo_matrix((H.reshape(-1),(np.repeat(dof,12,axis=1).ravel(),np.tile(dof,(1,12)).ravel())),shape=(len(solid.X)*3,)*2).tocsr()

def bounded_equilibrate(model,start,iterations=50,tolerance=1e-6):
 from scipy.sparse import eye,kron,coo_matrix,diags
 from scipy.sparse.linalg import factorized
 K=linear_stiffness(model.solid)
 for term in model.terms:
  if isinstance(term,EmbeddedSolid):
   M=kron(term.M,eye(3),format='csr');K+=M.T@linear_stiffness(term.solid)@M
 # Include conservative all-sample contact metric, not contact energy for inactive samples.
 for term in model.terms:
  if isinstance(term,FrozenGapContact):
   cc=term.C.tocoo(); rows=np.repeat(cc.row,3); cols=(cc.col[:,None]*3+np.arange(3)).ravel(); vals=(cc.data[:,None]*term.gradient[cc.row]).ravel()
   D=coo_matrix((vals,(rows,cols)),shape=(term.C.shape[0],len(start)*3)).tocsr();K+=D.T@diags(term.weights)@D
 free=(model.free[:,None]*3+np.arange(3)).ravel();Kf=K[free][:,free];reg=max(float(Kf.diagonal().max())*1e-12,1e-9);inv=factorized((Kf+reg*eye(len(free))).tocsc());q=start.copy();history=[]
 for it in range(iterations):
  e,g,j=model.energy(q);res=float(np.linalg.norm(g[model.free],axis=1).max());history.append(dict(iteration=it,energy=e,residual=res,Jmin=j))
  if it%5==0:print('equilibrium',history[-1],flush=True)
  if res<tolerance:break
  d=np.zeros_like(q);d.ravel()[free]=-inv(g.ravel()[free]);slope=float(np.sum(d*g))
  if slope>=0:raise RuntimeError('Non-descent reference metric')
  accepted=False
  for ls in range(30):
   trial=q+(0.5**ls)*d;ee,_,jj=model.energy(trial)
   if jj>.1 and ee<e+1e-4*(0.5**ls)*slope:q=trial;accepted=True;break
  if not accepted:break
 return q,history
