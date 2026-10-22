import unittest
import numpy as np
from scipy.sparse import csr_matrix
from exterior_epigraph import epigraph_direction


class LiftedCandidateTests(unittest.TestCase):
    def data(self, tied=False):
        return dict(branch_energy_N_mm=np.array([1.,1. if tied else .99,0.]),
            branch_gradient_N=csr_matrix([[1.],[-1.],[0.]]), branch_group=np.array([0,0,0]),
            group_base_branch=np.array([0]),branch_site=np.array([0,0,0]),
            branch_owner=np.array([1,1,1]),branch_triangle=np.array([1,2,-1]))

    def test_useful_lifted_candidate_skips_first_blocker_without_mixture(self):
        g=np.array([4.])
        d,returned,slope,linear,e=epigraph_direction(csr_matrix([[10.]]),g,None,self.data(),
            inexact_search_after_operations=16)
        self.assertFalse(e['qp_solved']);self.assertFalse(e['usable_for_generalized_stationarity'])
        self.assertEqual(e['search_candidate_kind'],'lifted_equality_candidate')
        self.assertTrue(e['candidate_lift_uses_all_retained_branches'])
        self.assertEqual(linear['exterior_active_set_iterations'],1)
        np.testing.assert_allclose(d,[-.4]);np.testing.assert_array_equal(returned,g)
        self.assertAlmostEqual(e['qp_model_decrease_N_mm'],.01)
        self.assertLess(slope,0)

    def test_rising_lifted_model_is_rejected_and_original_qp_continues(self):
        d,_,_,_,e=epigraph_direction(csr_matrix([[10.]]),[2.],None,self.data(),
            inexact_search_after_operations=16)
        self.assertTrue(e['qp_solved']);self.assertFalse(e['search_only'])
        self.assertFalse(e['active_set_trace'][0]['lifted_candidate_search_qualified'])
        np.testing.assert_allclose(d,[-.005],atol=1e-14)

    def test_exact_loaded_cusp_falls_back_to_full_balance(self):
        d,mixed,_,_,e=epigraph_direction(csr_matrix([[10.]]),[.8],None,self.data(tied=True),
            inexact_search_after_operations=16)
        self.assertTrue(e['qp_solved']);self.assertTrue(e['actual_comaximality_satisfied'])
        np.testing.assert_allclose(d,0,atol=1e-14);np.testing.assert_allclose(mixed,0,atol=1e-14)


if __name__=='__main__':
    unittest.main()
