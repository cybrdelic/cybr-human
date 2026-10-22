"""Existing cached finite-triangle law, accelerated conservative broadphase only."""
from pathlib import Path
import sys,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'deformable-contact-staging'))
from cached_adaptive_triangle_contact import CachedAdaptiveTriangleContact
from moving_triangle_contact import _map_interval,_iadd,_imul
from conservative_broadphase import candidates
from contextvars import ContextVar
from contained_pair_cache import ContainedPairCache
_bound_pair_scope=ContextVar("accelerated_triangle_bound_pairs",default=None)
class AcceleratedTriangleContact(CachedAdaptiveTriangleContact):
 def safe_step_bound(self,q,direction,*,safety=.9,max_step=1.):
  # New local cache per bound; nested calls and exceptions restore prior scope.
  token=_bound_pair_scope.set((self,ContainedPairCache(candidates)))
  try:return super().safe_step_bound(q,direction,safety=safety,max_step=max_step)
  finally:_bound_pair_scope.reset(token)
 def candidates(self,q,direction=None,max_step=1.,clearance=None):
  if not np.isfinite(max_step) or max_step<0:raise ValueError('nonnegative finite step required')
  h=self.clearance if clearance is None else clearance
  A=_map_interval(self.C,q);B=_map_interval(self.M,q)
  if direction is not None:
   out=[]
   for C,P in [(self.C,A),(self.M,B)]:
    E=_iadd(P,_imul((max_step,max_step),_map_interval(C,direction)))
    out.append((np.minimum(P[0],E[0]),np.maximum(P[1],E[1])))
   A,B=out
  scope=_bound_pair_scope.get()
  broadphase=scope[1] if scope is not None and scope[0] is self else candidates
  p=broadphase(A[0][self.faces_a].min(1),A[1][self.faces_a].max(1),B[0][self.faces_b].min(1),B[1][self.faces_b].max(1),h)
  if self.eligibility is not None:p=p[self.eligibility[p[:,0],p[:,1]]]
  return p
