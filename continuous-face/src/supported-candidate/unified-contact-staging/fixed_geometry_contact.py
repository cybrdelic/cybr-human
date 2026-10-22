"""Exact elimination of static contact coordinates, without adding model DOFs.

The wrapped finite-surface term sees [q; fixed_vertices]. No contact law,
feature eligibility, tolerances or path certificates are changed here.
"""
import numpy as np

class FixedGeometryContact:
    def __init__(self, term, physical_node_count, fixed_vertices):
        self.term=term; self.n=int(physical_node_count)
        if self.n<=0: raise ValueError('Positive physical node count required')
        self.fixed=np.array(fixed_vertices,dtype=float,copy=True)
        if self.fixed.ndim!=2 or self.fixed.shape[1]!=3 or not np.isfinite(self.fixed).all():
            raise ValueError('Finite fixed vertices with shape (n,3) required')
        if term.C.shape[1]!=self.n+len(self.fixed): raise ValueError('Carrier namespace mismatch')
        self.fixed.setflags(write=False)
        self.C=term.C[:,:self.n].copy() # Physical namespace descriptor, not a new carrier map.
        self.last_state=None; self.fixed_energy_gradient=None
    def _q(self,q):
        q=np.asarray(q,dtype=float)
        if q.shape!=(self.n,3) or not np.isfinite(q).all(): raise ValueError('Invalid physical coordinates')
        return np.concatenate((q,self.fixed))
    def _direction(self,d):
        d=np.asarray(d,dtype=float)
        if d.shape!=(self.n,3) or not np.isfinite(d).all(): raise ValueError('Invalid physical direction')
        return np.concatenate((d,np.zeros_like(self.fixed)))
    def evaluate(self,q):
        self.last_state=self.term.evaluate(self._q(q)); return self.last_state
    def require_feasible(self,q):
        self.last_state=self.term.require_feasible(self._q(q)); return self.last_state
    def __call__(self,q):
        e,g=self.term(self._q(q)); self.last_state=self.term.last_state
        self.fixed_energy_gradient=np.array(g[self.n:],copy=True)
        return e,np.array(g[:self.n],copy=True)
    def normal_tangent(self,q):
        k=self.term.normal_tangent(self._q(q)); return k[:3*self.n,:3*self.n].tocsr()
    def safe_step_bound(self,q,direction,**kwargs):
        return self.term.safe_step_bound(self._q(q),self._direction(direction),**kwargs)
    def candidates(self,q,direction=None,**kwargs):
        return self.term.candidates(self._q(q),None if direction is None else self._direction(direction),**kwargs)
