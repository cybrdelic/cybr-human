"""Bounded fixtures for explicitly sampled physical-site contact only."""
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
from coarse_physical_site_contact import CoarsePhysicalSiteContact
from coarse_rigid_query import NativeRigidQuery
class CoarseTests(unittest.TestCase):
    def setUp(self):
        X=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float);T=np.array([[0,1,2,3]])
        self.s=SimpleNamespace(X=X,T=T,inv=np.eye(3)[None],mm_per_L0=1.)
        self.s.deformation=lambda q:np.stack([q[T[:,k]]-q[T[:,0]] for k in (1,2,3)],axis=2)
        self.C=csr_matrix([[.25,.25,.25,.25]])
        V=np.array([[-3,-3,.2],[4,-3,.2],[-3,4,.2]])
        self.r=NativeRigidQuery(V,[[0,1,2]],[1]);self.g=CoarsePhysicalSiteContact(self.s,self.C,[1.],self.r,initial_occupancy_verified=True,attachment_qualified=True)
        self.assertEqual(self.g.provenance['contact_enforcement'],'compact_cap_barrier_v1')
        self.assertEqual(len(self.g.provenance['compact_barrier_source_sha256']),64)
    def test_gradient_and_psd(self):
        q=self.s.X.copy();e,g=self.g.energy(q);d=np.random.default_rng(7).normal(size=q.shape);eps=1e-6
        fd=(self.g.energy(q+eps*d)[0]-self.g.energy(q-eps*d)[0])/(2*eps)
        self.assertLess(abs(fd-np.sum(g*d)),1e-9)
        self.assertGreaterEqual(np.linalg.eigvalsh(self.g.tangent(q).toarray()).min(),-1e-12)
        self.assertGreater(e,0)
    def test_no_contact_and_finite_sheet(self):
        q=self.s.X+[5.,0,0];self.assertEqual(self.g.energy(q)[0],0.)
        self.assertTrue(self.g.endpoint(q)['feasible'])
        self.assertFalse(self.g.provenance['global_surface_collision_certified'])
    def test_positive_volume_and_increment(self):
        q=self.s.X.copy();q[3,2]=-1
        self.assertFalse(self.g.endpoint(q)['feasible'])
        d=np.tile([0.,0,2.],(4,1));a=self.g.safe_step_bound(self.s.X,d)
        self.assertLessEqual(a*2,.5)
        p=self.g.path(self.s.X,self.s.X+[0,0,.1]);self.assertTrue(p['coarse_whole_segment_checked']);self.assertNotIn('whole_segment_certified',p)
    def test_exact_state_cache(self):
        self.g.endpoint(self.s.X);n=self.g.query_count
        self.g.endpoint(self.s.X+[0,0,.1]);self.g.endpoint(self.s.X)
        self.assertEqual(self.g.query_count,n+1)
if __name__=='__main__':unittest.main()
