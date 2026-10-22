import unittest
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
from force_state_accuracy import accuracy_fallback

class ForceStateAccuracyTests(unittest.TestCase):
    def setUp(self):
        self.q=np.zeros((2,3));self.d=np.ones((2,3))*1e-8;self.g=np.ones((2,3))*1e-7
        self.solver=SimpleNamespace(constraints=None,solid=SimpleNamespace(mass=np.array([1.,2.])),
            guard=SimpleNamespace(max_physical_site_difference=lambda a,b:np.linalg.norm(a-b,axis=1).max()))
        self.e=dict(qp_solved=True,search_only=False,usable_for_generalized_stationarity=True,
            energy_law_unchanged=True,seam_rows_are_source_support_pins=False,actual_comaximality_satisfied=True,
            branch_weights_nonnegative=True,group_weight_sum_error=0.)
    def check(self,**kw):
        return accuracy_fallback(kw.get('solver',self.solver),self.q,kw.get('d',self.d),self.g,.00125,
            {'relative_dynamic_balance_max':kw.get('relative',4.4e-6)},kw.get('e',self.e),kw.get('reg',0.))
    def test_qualified_local_indicator_and_mass_history_units(self):
        r=self.check();self.assertTrue(r['qualified']);self.assertAlmostEqual(r['maximum_occupied_site_velocity_correction_mm_s'],np.sqrt(3)*1e-8/.00125)
        self.assertGreater(r['source_constrained_mass_only_site_correction_mm'],0)
    def test_regularized_or_unfinished_reactions_cannot_qualify(self):
        self.assertFalse(self.check(reg=.001)['qualified'])
        self.assertFalse(self.check(e=dict(self.e,search_only=True))['qualified'])
        self.assertFalse(self.check(e=dict(self.e,qp_solved=False))['qualified'])
        self.assertFalse(self.check(e=dict(self.e,actual_comaximality_satisfied=False))['qualified'])
    def test_relative_balance_and_actual_correction_both_required(self):
        self.assertFalse(self.check(relative=1.1e-5)['qualified'])
        self.assertFalse(self.check(d=np.ones((2,3))*.001)['qualified'])
        self.assertFalse(self.check(d=np.ones((2,3))*float('nan'))['qualified'])
    def test_source_compatible_mass_only_direction(self):
        C=csr_matrix([[1.,0,0,1.,0,0]])
        self.solver.constraints=SimpleNamespace(C3=C)
        d=self.d.copy();d[1,0]=-d[0,0]
        r=self.check(d=d);self.assertTrue(r['qualified']);self.assertEqual(r['source_constraint_direction_error_mm'],0.)
        self.assertFalse(self.check()['qualified'])

if __name__=='__main__':unittest.main()
