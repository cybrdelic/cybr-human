"""Sufficient global injectivity proof on the full regular Freudenthal box.

Unused background-node displacement is zero ONLY in this proof extension.
No mechanical DOFs, constraints, mass, energy, or forces are created. Failure
is inconclusive and requires the unchanged finite self-contact fallback.
"""
from itertools import permutations
from fractions import Fraction
import numpy as np


def upper(x):return np.nextafter(x,np.inf)
def lower(x):return np.nextafter(x,-np.inf)


class RegularGridInjectivity:
    def __init__(self,solid,spacing_mm,*,max_background_tets=2_000_000):
        X=np.asarray(solid.X,float);h=float(spacing_mm)
        if not np.isfinite(h) or h<=0:raise ValueError('explicit positive grid spacing required')
        self.X=X.copy();self.h=h;self.origin=X.min(0)
        index=np.rint((X-self.origin)/h).astype(np.int64)
        if not np.array_equal(self.origin+h*index,X):
            raise ValueError('carrier nodes are not exactly on the declared regular grid')
        # The preceding floating equality is only a fast rejection. Verify
        # the mathematical binary-rational grid as well; rounded origin+h*i
        # equality alone does not prove identical cell widths.
        hf=Fraction.from_float(h)
        for axis in range(3):
            origin=Fraction.from_float(float(self.origin[axis]))
            for value in np.unique(X[:,axis]):
                position=int(np.rint((value-self.origin[axis])/h))
                if Fraction.from_float(float(value))!=origin+position*hf:
                    raise ValueError('rounded grid coordinates are not an exact affine regular grid')
        if len(np.unique(index,axis=0))!=len(X):raise ValueError('duplicate carrier grid nodes')
        self.shape=index.max(0)+1
        if np.any(self.shape<2):raise ValueError('three-dimensional convex background box required')
        count=int(np.prod(self.shape-1))*6
        if count>max_background_tets:raise ValueError('proof extension exceeds bounded tetrahedron budget')
        I=index[np.asarray(solid.T,int)];relative=I-I.min(1)[:,None,:]
        if np.any((relative<0)|(relative>1)):raise ValueError('carrier tet is not in a single regular cube')
        order=np.argsort(relative.sum(2),axis=1)
        chain=np.take_along_axis(relative,order[:,:,None],axis=1)
        delta=np.diff(chain,axis=1)
        if not np.all(chain[:,0]==0) or not np.all(chain[:,-1]==1) or not np.all(delta.sum(2)==1) or np.any(delta<0):
            raise ValueError('carrier tetrahedra do not match the full Freudenthal triangulation')
        if len(np.unique(np.sort(np.ravel_multi_index(I.reshape(-1,3).T,tuple(self.shape)).reshape(-1,4),axis=1),axis=0))!=len(solid.T):
            raise ValueError('duplicate carrier tetrahedra')
        self.physical_to_background=np.ravel_multi_index(index.T,tuple(self.shape))
        cells=np.indices(tuple(self.shape-1)).reshape(3,-1).T
        chains=[];directions=[]
        for permutation in permutations(range(3)):
            offsets=np.zeros((4,3),int)
            for k,axis in enumerate(permutation):offsets[k+1]=offsets[k];offsets[k+1,axis]+=1
            nodes=cells[:,None,:]+offsets
            chains.append(np.ravel_multi_index(nodes.reshape(-1,3).T,tuple(self.shape)).reshape(-1,4))
            directions.append(permutation)
        self.chains=np.concatenate(chains)
        self.background_nodes=int(np.prod(self.shape))
        self.report=dict(enabled=True,physical_nodes=len(X),physical_carrier_tets=len(solid.T),
            background_nodes=self.background_nodes,background_tets=len(self.chains),
            proof_only_zero_displacement_nodes=self.background_nodes-len(X),spacing_mm=h,
            mechanics_dofs_added=0,mechanics_constraints_added=0,
            geometry='verified common regular Freudenthal triangulation on its complete convex bounding box',
            guarantee='if every outward Frobenius upper bound of grad(u) is below1, X+u is globally injective by the Lipschitz inequality')

    def bound(self,q):
        q=np.asarray(q,float)
        return self._interval_bound(q,q)

    def _interval_bound(self,qlo,qhi):
        if qlo.shape!=self.X.shape or qhi.shape!=self.X.shape or not np.isfinite(qlo).all() or not np.isfinite(qhi).all() or np.any(qlo>qhi):
            return dict(certified=False,reason='finite shape-matched coordinates required')
        lo=np.zeros((self.background_nodes,3));hi=np.zeros_like(lo)
        lo[self.physical_to_background]=lower(qlo-self.X)
        hi[self.physical_to_background]=upper(qhi-self.X)
        # Along each Freudenthal chain, successive edges point along the three
        # coordinate axes. Their displacement difference/h is exactly the
        # corresponding gradient column. Frobenius norm ignores column order.
        low=lower(lo[self.chains[:,1:]]-hi[self.chains[:,:-1]])
        high=upper(hi[self.chains[:,1:]]-lo[self.chains[:,:-1]])
        low=lower(low/self.h);high=upper(high/self.h)
        magnitude=np.maximum(abs(low),abs(high))
        squared=upper(magnitude*magnitude)
        total=np.zeros(len(self.chains))
        for a in range(3):
            for b in range(3):total=upper(total+squared[:,a,b])
        norm=upper(np.sqrt(total));maximum=float(norm.max(initial=0))
        return dict(certified=bool(np.isfinite(maximum) and maximum<1),
            maximum_gradient_Frobenius_upper=maximum,
            separation_scale_lower=float(lower(1-maximum)),background_tets=len(norm),
            reason='' if maximum<1 else 'sufficient Lipschitz bound inconclusive; retain ordinary finite self checks')

    def path(self,q0,q1):
        a,b=self.bound(q0),self.bound(q1)
        good=a['certified'] and b['certified']
        return dict(certified=bool(good),start=a,end=b,
            whole_segment_certified=bool(good),
            guarantee='convexity bounds every intermediate piecewise-affine displacement gradient; complete convex-box Lipschitz bound proves global injectivity',
            self_contact_energy_zero_proved=False)

    def direction_path(self,q,direction):
        q=np.asarray(q,float);direction=np.asarray(direction,float)
        if q.shape!=self.X.shape or direction.shape!=q.shape:
            return dict(certified=False,whole_segment_certified=False,reason='shape mismatch')
        a=self.bound(q);end=q+direction
        b=self._interval_bound(lower(end),upper(end))
        good=a['certified'] and b['certified']
        return dict(certified=bool(good),start=a,end=b,whole_segment_certified=bool(good),
            guarantee='outward q+direction endpoint enclosure plus convexity and full-box Lipschitz injectivity',
            self_contact_energy_zero_proved=False)
