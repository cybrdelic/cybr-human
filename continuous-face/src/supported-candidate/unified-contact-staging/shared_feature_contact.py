"""Adjacent-material finite feature barrier and conservative affine-path gate.
Only explicitly supplied shared-feature facet pairs are owned by this term.
Nonshared facet pairs must remain in the ordinary finite-triangle contact term.
Feature weights are a discrete barrier assumption, not calibrated surface law.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent.parent/'deformable-contact-staging'))
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix,diags
from moving_triangle_contact import (_carrier,point_triangle_geometry,activated_log_barrier,InfeasibleContactError,_map_interval,_iadd,_imul,_down,_up)
from triangle_interface_contact import segment_segment
from adaptive_shared_feature_geometry import pack_shared_indices,any_forbidden_intersection
from robust_feature_plane import feature_plane_certificate
E=((0,1),(1,2),(2,0))

class SharedFeatureContact:
 def __init__(self,carrier,faces_a,faces_b,pairs,*,clearance=.02,pair_weights=1.):
  self.C=_carrier(carrier,'shared-feature carrier');self.fa=np.asarray(faces_a,int).copy();self.fb=np.asarray(faces_b,int).copy();self.pairs=np.asarray(pairs,int).reshape(-1,2).copy();self.clearance=float(clearance)
  if self.clearance<=0:raise ValueError('positive clearance required')
  C=self.C;C.sum_duplicates();C.sort_indices();keys={};ids=[]
  for i in range(C.shape[0]):
   lo,hi=C.indptr[i:i+2];key=(tuple(C.indices[lo:hi]),tuple(C.data[lo:hi]));ids.append(keys.setdefault(key,len(keys)))
  self.material_ids=np.array(ids);self.shared=[];vf=[];ee=[];weights=np.broadcast_to(np.asarray(pair_weights,float),(len(self.pairs),))
  if np.any(weights<=0):raise ValueError('positive weights required')
  wv=[];we=[]
  for pair,w in zip(self.pairs,weights):
   a,b=self.fa[pair[0]],self.fb[pair[1]];ia,ib=self.material_ids[a],self.material_ids[b];shared=np.intersect1d(ia,ib)
   if len(shared) not in (1,2):raise ValueError('Each pair must share one or two exact material vertices; full cap duplicates must be removed')
   self.shared.append(np.array([a[np.flatnonzero(ia==s)[0]] for s in shared]))
   for v in a:
    if self.material_ids[v] not in ib:vf.append([v,*b]);wv.append(w)
   for v in b:
    if self.material_ids[v] not in ia:vf.append([v,*a]);wv.append(w)
   for u,v in E:
    for s,t in E:
     if not np.intersect1d(ia[[u,v]],ib[[s,t]]).size:ee.append([a[u],a[v],b[s],b[t]]);we.append(w)
  self.vf=np.asarray(vf,int).reshape(-1,4);self.ee=np.asarray(ee,int).reshape(-1,4);self.weights=np.r_[wv,we]
  self.shared_indices,self.shared_counts=pack_shared_indices(self.shared)
  for x in (self.fa,self.fb,self.pairs,self.material_ids,self.vf,self.ee,self.weights,*self.shared):x.flags.writeable=False
  self.last_state=None
 def candidates(self,q,direction=None,max_step=1.,clearance=None):
  # Topological adjacency is fixed; no proximity-based pair exclusions.
  return self.pairs
 def geometry(self,V):
  n=len(self.vf);m=len(self.ee);normal=np.empty((n+m,3));gap=np.empty(n+m);bary=np.empty((n+m,4))
  if n:
   g=point_triangle_geometry(V[self.vf[:,0]],V[self.vf[:,1:]]);gap[:n]=g['gap'];normal[:n]=g['normal'];bary[:n,0]=1;bary[:n,1:]=-g['barycentric']
  if m:
   x=V[self.ee];ds,s,t=segment_segment(x[:,0],x[:,1],x[:,2],x[:,3]);delta=x[:,0]+s[:,None]*(x[:,1]-x[:,0])-x[:,2]-t[:,None]*(x[:,3]-x[:,2]);gg=np.linalg.norm(delta,axis=1);gap[n:]=gg;normal[n:]=np.divide(delta,gg[:,None],out=np.zeros_like(delta),where=gg[:,None]>0);bary[n:]=np.c_[1-s,s,-(1-t),-t]
  return dict(gap=gap,normal=normal,barycentric=bary)
 def geometry_subset(self, V, ids):
     """Return original geometry, bit-for-bit, in the requested feature order.

     No new geometry formula: the existing implementation operates on a proxy
     containing the selected immutable VF/EE rows.  Splitting/reassembling the
     arrays preserves unsorted/repeated IDs, too; the recursion uses sorted IDs.
     """
     ids = np.asarray(ids, dtype=np.intp)
     if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(self.weights)):
         raise ValueError('one-dimensional in-range feature IDs required')
     is_vf = ids < len(self.vf)
     proxy = SimpleNamespace(vf=self.vf[ids[is_vf]],
                             ee=self.ee[ids[~is_vf] - len(self.vf)])
     g = SharedFeatureContact.geometry(proxy, V)
     if np.all(is_vf[:-1] >= is_vf[1:]):
         return g
     order = np.r_[np.flatnonzero(is_vf), np.flatnonzero(~is_vf)]
     out = {}
     for key, value in g.items():
         out[key] = np.empty_like(value)
         out[key][order] = value
     return out

 def _validate_all_vf_sources(self, V):
     """Keep global midpoint finiteness/rank rejection, not only active IDs.

     Full geometry rejects a degenerate source triangle even when that feature
     was already separated at an ancestor.  This repeats its exact arithmetic
     and thresholds on each distinct source-triangle row (not a spatial merge).
     All original VF point/source vertices are still checked for finiteness.
     """
     cache = getattr(self, '_finite_path_vf_validation', None)
     if cache is None or cache[0] is not self.vf:
         vertices = np.unique(self.vf)
         triangles = np.unique(self.vf[:, 1:], axis=0)
         vertices.flags.writeable = False
         triangles.flags.writeable = False
         cache = (self.vf, vertices, triangles)
         self._finite_path_vf_validation = cache
     _, vertices, triangles = cache
     if not len(triangles):
         return
     if not np.isfinite(V[vertices]).all():
         raise ValueError('finite paired N x 3 points and N x 3 x 3 triangles required')
     tri = V[triangles]
     a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
     ab, ac = b-a, c-a
     scale = np.maximum(np.linalg.norm(ab, axis=1), np.linalg.norm(ac, axis=1))
     if np.any(~np.isfinite(scale) | (scale <= 0)):
         raise InfeasibleContactError('degenerate or unrepresentable source triangle')
     uedge, vedge = ab/scale[:, None], ac/scale[:, None]
     face_cross = np.cross(uedge, vedge)
     rank = np.linalg.norm(face_cross, axis=1)
     if np.any(~np.isfinite(rank) | (rank <= 1e-12)):
         raise InfeasibleContactError('unresolved or degenerate source triangle')

 def evaluate(self,q):
  try:
   V=np.asarray(self.C@q);g=self.geometry(V)
   bad=not np.isfinite(g['gap']).all() or np.any(g['gap']<=256*np.finfo(float).eps*max(1.,abs(V).max()))
   if not bad:
    bad=any_forbidden_intersection(V,self.fa,self.fb,self.pairs,self.shared_indices,self.shared_counts)
   self.last_state=SimpleNamespace(feasible=not bad,reason='forbidden shared-feature intersection' if bad else '',geometry=g);return self.last_state
  except (ValueError,FloatingPointError) as exc:
   self.last_state=SimpleNamespace(feasible=False,reason=str(exc),geometry={'gap':np.empty(0)});return self.last_state
 def require_feasible(self,q):
  s=self.evaluate(q)
  if not s.feasible:raise InfeasibleContactError(s.reason)
  return s
 def jacobian(self,s):
  f=np.r_[self.vf,self.ee];g=s.geometry;n=len(f);D=csr_matrix((g['barycentric'].ravel(),(np.repeat(np.arange(n),4),f.ravel())),shape=(n,self.C.shape[0]))@self.C;D=D.tocoo()
  return csr_matrix(((D.data[:,None]*g['normal'][D.row]).ravel(),(np.repeat(D.row,3),(D.col[:,None]*3+np.arange(3)).ravel())),shape=(n,3*self.C.shape[1]))
 def barrier(self,gap):
  if not hasattr(self,'feature_clearances'):return activated_log_barrier(gap,self.clearance)
  g=np.asarray(gap,float);h=self.feature_clearances;out=[np.zeros_like(g) for _ in range(3)];bad=~np.isfinite(g)|(g<=0);out[0][bad]=np.inf;out[1][bad]=np.nan;out[2][bad]=np.nan;k=(~bad)&(g<h)
  x=g[k];u=h[k]-x;lr=np.log(x/h[k]);ratio=u/x
  out[0][k]=-u*u*lr;out[1][k]=2*u*lr-u*ratio;out[2][k]=-2*lr+4*ratio+ratio*ratio
  return tuple(out)
 def set_reference_local_clearance(self,reference_q,pair_mask):
  if hasattr(self,'feature_clearances'):raise ValueError('Reference activation is immutable once installed')
  mask=np.asarray(pair_mask,bool)
  if mask.shape!=(len(self.pairs),):raise ValueError('Explicit pair ownership mask required')
  counts=np.array([len(x) for x in self.shared]);feature_mask=np.r_[np.repeat(mask,6-2*counts),np.repeat(mask,np.where(counts==1,5,2))]
  g=self.geometry(np.asarray(self.C@reference_q))['gap']
  floor=256*np.finfo(float).eps*max(1.,float(np.max(abs(self.C@reference_q))))
  if np.any(g[feature_mask]<=floor) or not np.isfinite(g).all():raise ValueError(f'Reference feature gap is within numerical ambiguity floor {floor}; resolve ownership, never suppress with h≈0')
  self.reference_gap_ambiguity_floor=floor
  h=np.full(len(g),self.clearance);h[feature_mask]=np.minimum(h[feature_mask],.5*g[feature_mask]);h.flags.writeable=False;self.feature_clearances=h
  self.reference_local_pair_mask=mask.copy();self.reference_local_pair_mask.flags.writeable=False
 def __call__(self,q):
  s=self.require_feasible(q);b,db,_=self.barrier(s.geometry['gap'])
  if not np.any(db):return float(self.weights@b),np.zeros(q.shape)
  J=self.jacobian(s);return float(self.weights@b),np.asarray(J.T@(self.weights*db)).reshape(q.shape)
 def normal_tangent(self,q):
  s=self.require_feasible(q);dd=self.barrier(s.geometry['gap'])[2]
  if not np.any(dd):return csr_matrix((3*self.C.shape[1],3*self.C.shape[1]))
  J=self.jacobian(s);return (J.T@diags(self.weights*dd)@J).tocsr()
 def certify_segment(self, q, direction, alpha, *, max_depth=10):
     if not np.isfinite(alpha) or alpha < 0:
         raise ValueError('nonnegative finite alpha required')
     self.require_feasible(q)
     P = _map_interval(self.C, q)
     D = _map_interval(self.C, direction)
     V = self.C @ q
     dV = self.C @ direction
     visits = 0; alternate_planes = 0
     A = np.r_[np.repeat(self.vf[:, :1], 2, axis=1), self.ee[:, :2]]
     B = np.r_[self.vf[:, 1:], self.ee[:, [2, 3, 3]]]

     def separated(ids, time, n):
         Z = _iadd(P, _imul((time, time), D))
         aa, bb = A[ids], B[ids]
         lo = _down(Z[0][aa][:, :, None, :] - Z[1][bb][:, None, :, :])
         hi = _up(Z[1][aa][:, :, None, :] - Z[0][bb][:, None, :, :])
         pr = _imul((lo, hi), (n[:, None, None, :], n[:, None, None, :]))
         total = (np.zeros(pr[0].shape[:-1]), np.zeros(pr[0].shape[:-1]))
         for j in range(3):
             total = _iadd(total, (pr[0][..., j], pr[1][..., j]))
         return np.min(total[0], axis=(1, 2)) > 0

     def rec(ids, l, r, depth):
         nonlocal visits, alternate_planes
         visits += 1
         if not len(ids):
             return True
         mid = (l+r)/2
         W = V + mid*dV
         if depth == 0:
             g = self.geometry(W)
         else:
             self._validate_all_vf_sources(W)
             g = self.geometry_subset(W, ids)
         n = g['normal']
         if np.any(g['gap'] <= 0):
             return False
         ok = separated(ids, l, n) & separated(ids, r, n)
         # Only alternate proof proposals. Use original global ambiguity scale,
         # with no smaller near-feature floor and no shared-feature exemption.
         if not ok.all():
             Zl = _iadd(P, _imul((l, l), D))
             Zr = _iadd(P, _imul((r, r), D))
             floor = 256*np.finfo(float).eps*max(1.,abs(W).max())
             for k in np.flatnonzero(~ok):
                 i = ids[k]
                 aa = A[i,:1] if i < len(self.vf) else A[i]
                 bb = B[i] if i < len(self.vf) else B[i,:2]
                 endpoints = [((Z[0][aa],Z[1][aa]),(Z[0][bb],Z[1][bb])) for Z in (Zl,Zr)]
                 proof = feature_plane_certificate(W[aa],W[bb],tol=floor,endpoints=endpoints)
                 if proof is not None:
                     ok[k] = True; alternate_planes += 1
         bad = ids[~ok]
         if not len(bad):
             return True
         if depth >= max_depth:
             return False
         return rec(bad, l, mid, depth+1) and rec(bad, mid, r, depth+1)

     good = rec(np.arange(len(self.weights)), 0., float(alpha), 0)
     good = good and self.evaluate(q + alpha*direction).feasible
     return dict(certified=bool(good), subinterval_visits=visits,
                 alternate_planes=alternate_planes,features=len(self.weights),alpha=float(alpha),max_depth=max_depth,
                 guarantee='all nonincident VF/disjoint EE convex features strictly separated throughout affine path by outward interval planes; exact shared features excluded only')


 def safe_step_bound(self,q,direction,*,safety=.9,max_step=1.):
  if not 0<safety<1 or max_step<0:raise ValueError('invalid bound parameters')
  if self.certify_segment(q,direction,max_step)['certified']:return float(max_step)
  h=float(max_step)
  for _ in range(32):
   h*=.5
   if self.certify_segment(q,direction,h)['certified']:return float(h*safety)
  return 0.
