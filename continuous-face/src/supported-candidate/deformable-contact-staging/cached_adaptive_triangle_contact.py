"""Exact-last-state geometry cache; no approximate or stale-coordinate reuse.

Constructor inputs are copied and owned arrays are read-only. Public geometric
configuration cannot be rebound. Do not structurally modify the owned scipy
sparse matrices; reconstruct a contact object to change material carriers.
"""
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np
from adaptive_triangle_contact import AdaptiveTriangleContact

@dataclass(frozen=True)
class FrozenTriangleState:
 feasible: bool
 reason: str
 geometry: object
 pairs: np.ndarray

class CachedAdaptiveTriangleContact(AdaptiveTriangleContact):
 _configuration=frozenset(('C','M','faces_a','faces_b','wa','wb','clearance','eligibility','chunk_size'))
 def __setattr__(self,name,value):
  if getattr(self,'_configuration_locked',False) and name in self._configuration:raise AttributeError('Contact configuration is immutable; construct a new instance')
  super().__setattr__(name,value)
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  for name in ('faces_a','faces_b','wa','wb','eligibility'):
   a=getattr(self,name)
   if a is not None:
    a=np.array(a,copy=True);a.flags.writeable=False;setattr(self,name,a)
  # _carrier already copied its sparse matrices and froze their CSR arrays.
  self._configuration_locked=True;self._cached_q=None;self._cached_state=None;self.cache_hits=0;self.cache_misses=0
 def evaluate(self,q):
  q=np.asarray(q,float)
  if self._cached_q is not None and np.array_equal(q,self._cached_q):
   self.cache_hits+=1;self.last_state=self._cached_state;return self._cached_state
  self.cache_misses+=1
  state=super().evaluate(q);geometry={}
  for k,v in state.geometry.items():
   if isinstance(v,np.ndarray):v=v.copy();v.flags.writeable=False
   geometry[k]=v
  pairs=state.pairs.copy();pairs.flags.writeable=False
  frozen=FrozenTriangleState(state.feasible,state.reason,MappingProxyType(geometry),pairs)
  self._cached_q=q.copy();self._cached_q.flags.writeable=False;self._cached_state=frozen;self.last_state=frozen;return frozen
