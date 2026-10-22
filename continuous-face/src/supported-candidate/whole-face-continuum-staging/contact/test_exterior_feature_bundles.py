import unittest
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
from coarse_physical_site_contact import CoarsePhysicalSiteContact
from coarse_physical_site_contact_epigraph import ExteriorBranchPhysicalContact
from coarse_rigid_query import NativeRigidQuery
from test_compact_penetration_barrier import fixture as interior_fixture


def fixture(owners=(9,9,9)):
    X=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)-.25
    T=np.array([[0,1,2,3]])
    solid=SimpleNamespace(X=X,T=T,inv=np.eye(3)[None],mm_per_L0=1.)
    solid.deformation=lambda q:np.stack([q[T[:,k]]-q[T[:,0]] for k in (1,2,3)],axis=2)
    C=csr_matrix([[.25]*4])
    V=np.array([[-3,-3,-.05],[4,-3,-.05],[-3,4,-.05],[-3,-3,.05],[4,-3,.05],[-3,4,.05]])
    rigid=NativeRigidQuery(V,[[0,1,2],[3,4,5],[0,1,2]],owners)
    base=CoarsePhysicalSiteContact(solid,C,[1.],rigid,initial_occupancy_verified=True,attachment_qualified=True)
    return solid,ExteriorBranchPhysicalContact.from_base(base)


class BranchTests(unittest.TestCase):
    def test_complete_symmetric_branches_and_exact_duplicates(self):
        solid,guard=fixture();b=guard.contact_branches(solid.X)
        self.assertEqual(len(b['group_base_branch']),1)
        self.assertEqual(len(b['branch_energy_N_mm']),3)
        self.assertEqual(b['provenance']['exact_value_gradient_duplicates_removed'],1)
        np.testing.assert_allclose(b['branch_energy_N_mm'],[.0000125,.0000125,0],atol=1e-19)
        np.testing.assert_allclose((b['branch_gradient_N'][0]+b['branch_gradient_N'][1]).toarray(),0,atol=1e-18)
        self.assertEqual(b['branch_equivalent_native_faces'][0],[0,2])
        reconstructed=b['smooth_contact_gradient_N']+np.asarray(b['branch_gradient_N'][b['group_base_branch']].sum(0)).reshape(solid.X.shape)
        np.testing.assert_array_equal(reconstructed,guard.energy(solid.X)[1])
        d=np.tile([0,0,1.],(4,1));eps=1e-7
        fd=(guard.energy(solid.X+eps*d)[0]-guard.energy(solid.X)[0])/eps
        self.assertLess(abs(fd-np.max(b['branch_gradient_N']@d.ravel())),1e-9)

    def test_finite_branch_derivatives_and_near_duplicates_retained(self):
        solid,guard=fixture();q=solid.X+np.array([0.,0.,.001]);b=guard.contact_branches(q)
        d=np.tile([.1,.2,.3],(4,1));eps=1e-6
        plus=guard.contact_branches(q+eps*d);minus=guard.contact_branches(q-eps*d)
        for j,face in enumerate(b['branch_triangle']):
            a=np.flatnonzero(plus['branch_triangle']==face)[0];c=np.flatnonzero(minus['branch_triangle']==face)[0]
            fd=(plus['branch_energy_N_mm'][a]-minus['branch_energy_N_mm'][c])/(2*eps)
            self.assertLess(abs(fd-float(b['branch_gradient_N'][j]@d.ravel())),1e-12)

    def test_owner_sum_not_triangle_sum_or_cross_owner_bundle(self):
        solid,guard=fixture((9,10,9));b=guard.contact_branches(solid.X)
        self.assertEqual(len(b['group_base_branch']),0)
        self.assertAlmostEqual(guard.energy(solid.X)[0],.000025)
        np.testing.assert_array_equal(b['smooth_contact_gradient_N'],guard.energy(solid.X)[1])

    def test_interior_barrier_untouched_and_crossing_rejected(self):
        solid,base=interior_fixture();guard=ExteriorBranchPhysicalContact.from_base(base)
        b=guard.contact_branches(solid.X)
        self.assertEqual(len(b['group_base_branch']),0)
        np.testing.assert_array_equal(b['smooth_contact_gradient_N'],base.energy(solid.X)[1])
        old=b;guard.set_barrier_scale(.1);b=guard.contact_branches(solid.X)
        self.assertIsNot(old,b)
        np.testing.assert_array_equal(b['smooth_contact_gradient_N'],guard.energy(solid.X)[1])
        with self.assertRaises(ValueError):guard.contact_branches(solid.X-[0,0,.006])


if __name__=='__main__':unittest.main()
