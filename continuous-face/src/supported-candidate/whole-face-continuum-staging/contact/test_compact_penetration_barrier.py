"""Focused primal interior-point contact tests; no actual-model dynamics."""
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.optimize import brentq
from scipy.sparse import csr_matrix
from compact_penetration_barrier import compact_barrier_terms
from coarse_rigid_query import NativeRigidQuery
from coarse_physical_site_contact import CoarsePhysicalSiteContact


def fixture():
    X=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float);T=np.array([[0,1,2,3]])
    solid=SimpleNamespace(X=X,T=T,inv=np.eye(3)[None],mm_per_L0=1.)
    solid.deformation=lambda q:np.stack([q[T[:,k]]-q[T[:,0]] for k in (1,2,3)],axis=2)
    tet=np.array([[-2,-2,.295],[4,-2,.295],[-2,4,.295],[-2,-2,-4.]])
    faces=np.array([[0,1,2],[0,3,1],[0,2,3],[1,3,2]])
    second=tet.copy();second[:,2]-=.002
    sheet=np.array([[-2,-2,.251],[4,-2,.251],[-2,4,.251]])
    rigid=NativeRigidQuery(np.r_[tet,second,sheet],np.r_[faces,faces+4,[[8,9,10]]],[1]*4+[2]*4+[3])
    guard=CoarsePhysicalSiteContact(solid,csr_matrix([[.25]*4]),[1.],rigid,initial_occupancy_verified=True,attachment_qualified=True)
    return solid,guard


class CompactBarrierTests(unittest.TestCase):
    def test_analytic_loaded_stop(self):
        F=.0015;s0=.01;previous=np.inf
        for mu in [1.5e-5,1.5e-6,1.5e-7]:
            slack=mu/(F+mu/s0);b=compact_barrier_terms(np.array([slack-.05]),mu)
            self.assertAlmostEqual(b['reaction'][0],F,places=13)
            self.assertLess(slack,previous);previous=slack
            self.assertLessEqual(b['complementarity'][0],mu*(1+1e-12))
    def test_loaded_stop_retains_quadratic_force(self):
        k=.01;h=.1;p=.05;load=.01;rows=[]
        for scale in [1.,.1,.01]:
            mu=k*(h+p)*.01*scale
            residual=lambda slack:k*(h+p-slack)+compact_barrier_terms(np.array([slack-p]),mu)['reaction'][0]-load
            slack=brentq(residual,1e-10,.01,xtol=1e-14)
            b=compact_barrier_terms(np.array([slack-p]),mu)
            self.assertLess(abs(residual(slack)),1e-11)
            self.assertGreater(slack,0);self.assertGreater(b['reaction'][0],0)
            rows.append(slack)
        self.assertTrue(np.all(np.diff(rows)<0))
    def test_activation_gradient_and_strict_guard(self):
        mu=1.5e-5;eps=1e-7
        for slack in [.001,.005,.01,.010001,.05]:
            g=slack-.05;b=compact_barrier_terms([g],mu)
            fd=(compact_barrier_terms([g+eps],mu)['energy'][0]-compact_barrier_terms([g-eps],mu)['energy'][0])/(2*eps)
            self.assertLess(abs(fd+b['reaction'][0]),5e-8)
        below=compact_barrier_terms([.01-1e-10-.05],mu)
        above=compact_barrier_terms([.01+1e-10-.05],mu)
        self.assertLess(abs(below['reaction'][0]-above['reaction'][0]),2e-11)
        self.assertFalse(compact_barrier_terms([-.05],mu)['valid'][0])
        self.assertTrue(np.isinf(compact_barrier_terms([-.051],mu)['energy'][0]))
    def test_owner_accounting_and_open_sheet(self):
        solid,guard=fixture();a=guard.contact_accounting(solid.X,include_terms=True)
        terms=dict(zip(a['owner_ids'],a['lambda_N']))
        self.assertAlmostEqual(terms[1],.0015,places=12)
        self.assertAlmostEqual(terms[2],1.5e-5*(1/.007-1/.01),places=12)
        self.assertEqual(terms[3],0.)
        self.assertEqual(a['active_barrier_owner_terms'],2)
        e,g=guard.energy(solid.X)
        self.assertAlmostEqual(e,a['penalty_energy_N_mm']+a['barrier_energy_N_mm'])
        self.assertTrue(np.allclose(g,a['penalty_gradient_N']+a['barrier_gradient_N']))
        self.assertAlmostEqual(-a['barrier_gradient_N'][:,2].sum(),sum(terms.values()),places=12)
        self.assertTrue(a['complementarity_bound_satisfied'])
    def test_adapter_gradient_and_psd_tangent(self):
        solid,guard=fixture();q=solid.X.copy();e,g=guard.energy(q)
        d=np.random.default_rng(11).normal(size=q.shape);eps=1e-7
        fd=(guard.energy(q+eps*d)[0]-guard.energy(q-eps*d)[0])/(2*eps)
        self.assertLess(abs(fd-np.sum(g*d)),1e-9)
        H=guard.tangent(q);self.assertGreater(np.linalg.eigvalsh(H.toarray()).min(),-1e-12)
        fdg=(guard.energy(q+eps*d)[1]-guard.energy(q-eps*d)[1])/(2*eps)
        self.assertLess(np.max(abs(fdg.ravel()-H@d.ravel())),1e-7)
    def test_mu_cache_invalidation_and_site_difference(self):
        solid,guard=fixture();a=guard.contact_accounting(solid.X);old_count=guard.query_count
        guard.set_barrier_scale(.1);b=guard.contact_accounting(solid.X)
        self.assertEqual(guard.query_count,old_count+1)
        self.assertAlmostEqual(b['barrier_energy_N_mm']/a['barrier_energy_N_mm'],.1)
        self.assertTrue(np.allclose(b['barrier_gradient_N'],.1*a['barrier_gradient_N']))
        self.assertTrue(np.array_equal(b['penalty_gradient_N'],a['penalty_gradient_N']))
        self.assertAlmostEqual(guard.max_physical_site_difference(solid.X,solid.X+[0,0,.003]),.003)
    def test_fraction_to_boundary_and_crossing_rejection(self):
        solid,guard=fixture();direction=np.tile([0.,0.,-1.],(4,1));alpha=guard.safe_step_bound(solid.X,direction)
        self.assertAlmostEqual(alpha,.0045,places=12)
        self.assertTrue(guard.endpoint(solid.X+alpha*direction)['feasible'])
        crossing=solid.X+.006*direction
        self.assertFalse(guard.endpoint(crossing)['feasible'])
        self.assertFalse(guard.path(solid.X,crossing)['feasible'])
        self.assertTrue(np.isinf(guard.energy(crossing)[0]))

if __name__=='__main__':unittest.main()
