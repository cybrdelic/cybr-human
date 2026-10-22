"""Unified moving exterior contact, with sparse material-adjacency partition.
No pair/body suppression except exact common material features. Internal bond
and reference mixture faces must already be removed by the scene constructor.
"""
from types import SimpleNamespace
from itertools import combinations
import numpy as np
from scipy.sparse import csr_matrix
from accelerated_triangle_contact import AcceleratedTriangleContact
from shared_feature_contact import SharedFeatureContact
from moving_triangle_contact import _carrier,InfeasibleContactError

def material_ids(C):
 keys={};ids=[]
 for i in range(C.shape[0]):
  l,h=C.indptr[i:i+2];key=(tuple(C.indices[l:h]),tuple(C.data[l:h]));ids.append(keys.setdefault(key,len(keys)))
 return np.asarray(ids)

def adjacency_pairs(faces,ids):
 incidence={}
 for i,face in enumerate(ids[faces]):
  for v in set(face):incidence.setdefault(int(v),[]).append(i)
 pairs=set()
 for entries in incidence.values():pairs.update(combinations(entries,2))
 return np.asarray(sorted(pairs),int).reshape(-1,2)

class NonsharedSelfContact(AcceleratedTriangleContact):
 def __init__(self,C,faces,ids,**kwargs):
  self._face_material_ids=np.asarray(ids)[faces].copy();self._face_material_ids.flags.writeable=False
  super().__init__(C,faces,C,faces,**kwargs)
 def candidates(self,q,direction=None,max_step=1.,clearance=None):
  p=super().candidates(q,direction,max_step,clearance);p=p[p[:,0]<p[:,1]]
  if len(p):
   f=self._face_material_ids;shared=(f[p[:,0],:,None]==f[p[:,1],None,:]).any((1,2));p=p[~shared]
  return p

class CachedSharedFeatureContact(SharedFeatureContact):
 def evaluate(self,q):
  q=np.asarray(q)
  if hasattr(self,'_cached_q') and np.array_equal(q,self._cached_q):
   self.last_state=self._cached_state;return self.last_state
  state=super().evaluate(q)
  for value in state.geometry.values():
   if isinstance(value,np.ndarray):value.flags.writeable=False
  self._cached_q=np.array(q,copy=True);self._cached_q.flags.writeable=False;self._cached_state=state
  return state

class UnifiedMovingSurfaceContact:
 def __init__(self,carrier,faces,*,clearance=.02,face_weights=1.,shared_reference_q=None,shared_local_pair_mask=None):
  self.C=_carrier(carrier,'unified exterior');self.faces=np.asarray(faces,int).copy()
  if self.faces.ndim!=2 or self.faces.shape[1]!=3 or self.faces.min()<0 or self.faces.max()>=self.C.shape[0]:raise ValueError('invalid faces')
  self.ids=material_ids(self.C);pairs=adjacency_pairs(self.faces,self.ids)
  duplicate=[]
  for a,b in pairs:
   if len(np.intersect1d(self.ids[self.faces[a]],self.ids[self.faces[b]]))==3:duplicate.append([int(a),int(b)])
  if duplicate:raise ValueError(f'Coincident material facets must be removed by ownership, examples {duplicate[:5]}')
  w=np.broadcast_to(np.asarray(face_weights,float),(len(self.faces),)).copy()
  self.nonshared=NonsharedSelfContact(self.C,self.faces,self.ids,clearance=clearance,face_weights_a=w,face_weights_b=w)
  self.shared=CachedSharedFeatureContact(self.C,self.faces,self.faces,pairs,clearance=clearance,pair_weights=w[pairs[:,0]]*w[pairs[:,1]])
  if shared_local_pair_mask is not None:
   if shared_reference_q is None:raise ValueError('Explicit reference q required for local activation')
   self.shared.set_reference_local_clearance(shared_reference_q,shared_local_pair_mask)
  self.last_state=None
 def evaluate(self,q):
  a=self.nonshared.evaluate(q);b=self.shared.evaluate(q)
  self.last_state=SimpleNamespace(feasible=a.feasible and b.feasible,reason='; '.join(s.reason for s in (a,b) if not s.feasible),nonshared=a,shared=b)
  return self.last_state
 def require_feasible(self,q):
  s=self.evaluate(q)
  if not s.feasible:raise InfeasibleContactError(s.reason)
  return s
 def __call__(self,q):
  e,g=self.nonshared(q);f,h=self.shared(q);self.last_state=SimpleNamespace(feasible=True,reason='',nonshared=self.nonshared.last_state,shared=self.shared.last_state)
  return e+f,g+h
 def normal_tangent(self,q):return (self.nonshared.normal_tangent(q)+self.shared.normal_tangent(q)).tocsr()
 def safe_step_bound(self,q,direction,*,safety=.9,max_step=1.):
  bound=self.nonshared.safe_step_bound(q,direction,safety=safety,max_step=max_step)
  return self.shared.safe_step_bound(q,direction,safety=safety,max_step=bound)
 def diagnostics(self,q):
  s=self.evaluate(q);g=s.shared.geometry['gap'];h=getattr(self.shared,'feature_clearances',self.shared.clearance)
  out=dict(feasible=s.feasible,reason=s.reason,exterior_faces=len(self.faces),shared_facet_pairs=len(self.shared.pairs),shared_features=len(g),shared_minimum_gap=float(np.min(g,initial=np.inf)),shared_active_features=int(np.sum(g<h)),nonshared_candidate_pairs=len(s.nonshared.pairs))
  if s.shared.feasible:
   e,grad=self.shared(q);out.update(shared_barrier_energy=float(e),shared_barrier_max_node_force=float(np.linalg.norm(grad,axis=1).max()))
  return out


def cross_adjacency_pairs(faces_a,faces_b,ids):
 A={};B={}
 for i,f in enumerate(ids[faces_a]):
  for v in set(f):A.setdefault(int(v),[]).append(i)
 for i,f in enumerate(ids[faces_b]):
  for v in set(f):B.setdefault(int(v),[]).append(i)
 pairs=set()
 for v in A.keys()&B.keys():
  pairs.update((a,b) for a in A[v] for b in B[v])
 return np.asarray(sorted(pairs),int).reshape(-1,2)

class NonsharedCrossContact(AcceleratedTriangleContact):
 def __init__(self,C,faces_a,faces_b,ids,**kwargs):
  self._a_ids=np.asarray(ids)[faces_a].copy();self._b_ids=np.asarray(ids)[faces_b].copy()
  self._a_ids.flags.writeable=False;self._b_ids.flags.writeable=False
  super().__init__(C,faces_a,C,faces_b,**kwargs)
 def candidates(self,q,direction=None,max_step=1.,clearance=None):
  p=super().candidates(q,direction,max_step,clearance)
  if len(p):p=p[~(self._a_ids[p[:,0],:,None]==self._b_ids[p[:,1],None,:]).any((1,2))]
  return p

class UnifiedTwoSurfaceContact(UnifiedMovingSurfaceContact):
 """Same carrier namespace, distinct moving/static surface sets.
 Default clearance unchanged for BOTH shared and nonshared bone features.
 FixedGeometryContact can eliminate static coordinates outside this term.
 """
 def __init__(self,carrier,faces_a,faces_b,*,clearance=.02,face_weights_a=1.,face_weights_b=1.):
  self.C=_carrier(carrier,'two-surface exterior');self.faces_a=np.asarray(faces_a,int).copy();self.faces_b=np.asarray(faces_b,int).copy();self.faces=np.r_[self.faces_a,self.faces_b]
  for f in (self.faces_a,self.faces_b):
   if f.ndim!=2 or f.shape[1]!=3 or f.min()<0 or f.max()>=self.C.shape[0]:raise ValueError('invalid faces')
  self.ids=material_ids(self.C);pairs=cross_adjacency_pairs(self.faces_a,self.faces_b,self.ids)
  wa=np.broadcast_to(np.asarray(face_weights_a,float),(len(self.faces_a),));wb=np.broadcast_to(np.asarray(face_weights_b,float),(len(self.faces_b),))
  self.nonshared=NonsharedCrossContact(self.C,self.faces_a,self.faces_b,self.ids,clearance=clearance,face_weights_a=wa,face_weights_b=wb)
  self.shared=CachedSharedFeatureContact(self.C,self.faces_a,self.faces_b,pairs,clearance=clearance,pair_weights=wa[pairs[:,0]]*wb[pairs[:,1]])
  self.last_state=None
