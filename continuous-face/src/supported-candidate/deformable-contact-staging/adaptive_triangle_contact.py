"""Isolated enhancement: recursively certified changing separating planes.

A midpoint plane is accepted only if it separates ALL vertices at BOTH ends
of the subinterval using outward-rounded carrier arithmetic. Linear paths
then remain separated for the entire interval, not just sampled times.
"""
import numpy as np
from triangle_interface_contact import DynamicTriangleContact,triangle_geometry
from moving_triangle_contact import _map_interval,_down,_up,_imul,_iadd
from robust_separating_plane import plane_certificate

class AdaptiveTriangleContact(DynamicTriangleContact):
 def certify_segment(self,q,direction,alpha,*,max_depth=8):
  pairs=self.candidates(q,direction,max_step=alpha,clearance=0.)
  A=_map_interval(self.C,q);B=_map_interval(self.M,q);DA=_map_interval(self.C,direction);DB=_map_interval(self.M,direction)
  va=np.asarray(self.C@q);vb=np.asarray(self.M@q);da=np.asarray(self.C@direction);db=np.asarray(self.M@direction);visits=0;alternate_planes=0
  def endpoints(C,D,faces,ids,t):
   p=_iadd(C,_imul((t,t),D));return p[0][faces[ids]],p[1][faces[ids]]
  def separated(P,Q,n):
   lo=_down(P[0][:,:,None,:]-Q[1][:,None,:,:]);hi=_up(P[1][:,:,None,:]-Q[0][:,None,:,:]);prod=_imul((lo,hi),(n[:,None,None,:],n[:,None,None,:]));total=(np.zeros(prod[0].shape[:-1]),np.zeros(prod[0].shape[:-1]))
   for j in range(3):total=_iadd(total,(prod[0][...,j],prod[1][...,j]))
   return total[0].min((1,2))>0
  def certify(ids,l,r,depth):
   nonlocal visits,alternate_planes
   visits+=1
   if not len(ids):return True
   p=pairs[ids];mid=(l+r)/2;tA=(va+mid*da)[self.faces_a[p[:,0]]];tB=(vb+mid*db)[self.faces_b[p[:,1]]];g=triangle_geometry(tA,tB)
   if g['intersect'].any():return False
   n=g['separating_normal']
   left_a=endpoints(A,DA,self.faces_a,p[:,0],l);left_b=endpoints(B,DB,self.faces_b,p[:,1],l)
   right_a=endpoints(A,DA,self.faces_a,p[:,0],r);right_b=endpoints(B,DB,self.faces_b,p[:,1],r)
   clear=separated(left_a,left_b,n)&separated(right_a,right_b,n)
   # Try a distinct geometric plane before subdividing. All previous midpoint
   # feasibility checks and the exact same outward endpoint conditions remain.
   for k in np.flatnonzero(~clear):
    scale=max(1.,np.max(abs(tA[k])),np.max(abs(tB[k])));tol=256*np.finfo(float).eps*scale
    endpoint_pairs=[((X[0][k],X[1][k]),(Y[0][k],Y[1][k])) for X,Y in ((left_a,left_b),(right_a,right_b))]
    proof=plane_certificate(tA[k],tB[k],tol=tol,endpoints=endpoint_pairs)
    if proof is not None:clear[k]=True;alternate_planes+=1
   unresolved=ids[~clear]
   if not len(unresolved):return True
   if depth>=max_depth:return False
   return certify(unresolved,l,mid,depth+1) and certify(unresolved,mid,r,depth+1)
  valid=certify(np.arange(len(pairs)),0.,float(alpha),0)
  return dict(certified=bool(valid),subinterval_visits=visits,alternate_planes=alternate_planes,swept_pairs=len(pairs),max_depth=max_depth,alpha=float(alpha),guarantee='whole finite triangles separated throughout affine path by interval-verified subinterval endpoint planes')
 def safe_step_bound(self,q,direction,*,safety=.9,max_step=1.):
  base=super().safe_step_bound(q,direction,safety=safety,max_step=max_step)
  if base>=max_step or base<=0:return base
  reports=[]
  if max_step<=8*base:
   direct=self.certify_segment(q,direction,max_step);reports.append(direct)
   if direct['certified']:
    self.last_path_report=dict(original_bound=base,enhanced_bound=float(max_step),certificates=reports)
    return float(max_step)
  lo=base;hi=min(max_step,2*lo)
  # Bounded search: up to six geometric expansions and three refinements.
  for _ in range(6):
   r=self.certify_segment(q,direction,hi);reports.append(r)
   if not r['certified']:break
   lo=hi
   if lo>=max_step:break
   hi=min(max_step,2*hi)
  if lo<hi:
   for _ in range(3):
    mid=(lo+hi)/2;r=self.certify_segment(q,direction,mid);reports.append(r)
    if r['certified']:lo=mid
    else:hi=mid
  self.last_path_report=dict(original_bound=base,enhanced_bound=lo,certificates=reports)
  return lo
