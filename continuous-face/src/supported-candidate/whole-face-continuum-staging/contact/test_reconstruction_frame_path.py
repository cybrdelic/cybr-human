import unittest
import numpy as np
from scipy.sparse import csr_matrix
from coarse_physical_site_contact import CoarsePhysicalSiteContact
from reconstruction_frame_path import reconstruction_frame_path
from test_compact_penetration_barrier import fixture
from test_reconstruction_allowance import slab


class FinalReconstructionTests(unittest.TestCase):
    def guard(self,depth=.053):
        solid,_=fixture()
        guard=CoarsePhysicalSiteContact(solid,csr_matrix([[.25]*4]),[1.],slab(depth).rigid,
            initial_occupancy_verified=True,attachment_qualified=True,reference_lower_bound=np.zeros(1))
        q0=solid.X+[-.12,-.18,-.14];q1=solid.X+[-.02,-.18,-.36]
        return solid,guard,q0,q1

    def test_final_only_declared_reconstruction_and_positive_j(self):
        _,guard,q0,q1=self.guard()
        guard.path=lambda *args:(_ for _ in ()).throw(AssertionError('Final callback must not change/reuse strict Newton path'))
        r=reconstruction_frame_path(guard,q0,q1)
        self.assertTrue(r['feasible']);self.assertTrue(r['coarse_whole_segment_checked'])
        self.assertTrue(r['volume']['certified']);self.assertEqual(r['certified_interframe_penetration_cap_mm'],.055)
        self.assertTrue(all(x['feasible'] for x in r['endpoint_checks']))

    def test_endpoint_and_deeper_chord_fail(self):
        solid,guard,q0,q1=self.guard()
        self.assertFalse(reconstruction_frame_path(guard,q0,solid.X+[-.12,-.18,-.25])['feasible'])
        _,guard,q0,q1=self.guard(.056)
        self.assertFalse(reconstruction_frame_path(guard,q0,q1)['feasible'])

    def test_original_resolution_budget_and_nonfinite_fail(self):
        _,guard,q0,q1=self.guard()
        r=reconstruction_frame_path(guard,q0,q0+[33.,0.,0.])
        self.assertFalse(r['feasible']);self.assertIn('budget',r['reason'])
        with self.assertRaises(ValueError):reconstruction_frame_path(guard,q0,q1*np.nan)

    def test_positive_j_cannot_be_skipped(self):
        _,guard,q0,q1=self.guard();q1=q0.copy();q1[[1,2]]=q1[[2,1]]
        r=reconstruction_frame_path(guard,q0,q1)
        self.assertFalse(r['feasible']);self.assertEqual(r['reason'],'positive-J path')


if __name__=='__main__':unittest.main()
