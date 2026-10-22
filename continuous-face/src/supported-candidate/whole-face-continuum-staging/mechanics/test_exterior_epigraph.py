import unittest
import numpy as np
from scipy.sparse import csr_matrix
from exterior_epigraph import epigraph_direction, EpigraphOperationBudget
from implicit_dynamics import ImplicitDynamics, REQUIRED_GUARDS


def branches(x, ndof=1):
    k, b = 10., .1
    E = np.array([.5*k*(b-x)**2, .5*k*(b+x)**2, 0.])
    G = np.zeros((3, ndof)); G[:, 0] = [-k*(b-x), k*(b+x), 0.]
    base = int(np.argmax(E))
    return dict(branch_energy_N_mm=E, branch_gradient_N=csr_matrix(G),
        branch_group=np.zeros(3, int), group_base_branch=np.array([base]))


class EpigraphTests(unittest.TestCase):
    def test_loaded_cusp_nonzero_selected_force_zero_generalized_force(self):
        b = branches(0.)
        g = np.array([-1.-.3])
        d, mixed, slope, linear, report = epigraph_direction(csr_matrix([[12.]]), g, None, b)
        self.assertGreater(abs(g[0]), 1.)
        np.testing.assert_allclose(d, 0., atol=1e-14)
        np.testing.assert_allclose(mixed, 0., atol=1e-14)
        np.testing.assert_allclose(report['mixed_groups'][0]['weights'], [.35, .65], atol=1e-14)
        self.assertTrue(report['actual_comaximality_satisfied'])
        self.assertLess(abs(slope), 1e-14)

    def test_release_beyond_reaction_bound(self):
        b = branches(0.)
        d, mixed, slope, linear, report = epigraph_direction(csr_matrix([[12.]]), [-1.-1.5], None, b)
        np.testing.assert_allclose(d, [(1.5-1.)/12.], atol=1e-14)
        self.assertFalse(report['mixed_groups'])
        self.assertLess(slope, 0.)
        self.assertTrue(report['branch_weights_nonnegative'])

    def test_off_seam_mixture_cannot_claim_current_stationarity(self):
        x = .001; b = branches(x)
        g = np.array([2*x-.3+10*(.1+x)])
        d, mixed, slope, linear, report = epigraph_direction(csr_matrix([[12.]]), g, None, b)
        self.assertFalse(report['actual_comaximality_satisfied'])
        self.assertLess(slope, 0.)
        self.assertGreater(report['maximum_used_branch_energy_gap_N_mm'], 1e-12)

    def test_source_constraint_not_contact_projection(self):
        b = branches(0., 2)
        C = csr_matrix([[0., 1.]])
        d, mixed, slope, linear, report = epigraph_direction(csr_matrix(np.eye(2)*12), [-1.3, 8.], C, b)
        np.testing.assert_allclose(d, 0., atol=1e-14)
        np.testing.assert_allclose(mixed, [0., 8.], atol=1e-14)
        self.assertFalse(report['seam_rows_are_source_support_pins'])

    def test_wrong_selected_branch_is_rejected(self):
        b = branches(.01); b['group_base_branch'][0] = 0
        with self.assertRaises(ValueError):
            epigraph_direction(csr_matrix([[12.]]), [0.], None, b)

    def test_dependent_different_offset_seam_releases_old_branch(self):
        energy, gradient = [], []
        for k, a in [(100., .002), (10., .001)]:
            energy.extend([.5*k*(.1+a)**2, .5*k*(.1-a)**2, 0.])
            gradient.extend([-k*(.1+a), k*(.1-a), 0.])
        b = dict(branch_energy_N_mm=np.array(energy),
            branch_gradient_N=csr_matrix(np.array(gradient)[:, None]),
            branch_group=np.repeat([0, 1], 3), group_base_branch=np.array([0, 3]))
        g = np.array([-2.+gradient[0]+gradient[3]])
        d, mixed, slope, linear, report = epigraph_direction(csr_matrix([[112.]]), g, None, b)
        np.testing.assert_allclose(d, .002, atol=1e-14)
        self.assertTrue(any('released_branch' in t for t in report['active_set_trace']))
        self.assertEqual(len(report['mixed_groups']), 1)
        self.assertEqual(report['mixed_groups'][0]['group'], 0)
        self.assertFalse(report['actual_comaximality_satisfied'])

    def test_budget_checkpoint_resumes_same_feasible_qp(self):
        b = branches(.001); K = csr_matrix([[12.]])
        g = np.array([.812])
        expected = epigraph_direction(K, g, None, b)
        with self.assertRaises(EpigraphOperationBudget) as caught:
            epigraph_direction(K, g, None, b, maximum_active_set_iterations=1)
        cp = caught.exception.checkpoint
        self.assertTrue(cp['numerical_QP_only'])
        self.assertFalse(cp['usable_as_accepted_frame'])
        resumed = epigraph_direction(K, g, None, b, resume_checkpoint=cp)
        np.testing.assert_allclose(resumed[0], expected[0], atol=1e-14)
        np.testing.assert_allclose(resumed[1], expected[1], atol=1e-14)
        self.assertGreater(len(resumed[-1]['active_set_trace']), len(cp['active_set_trace']))
        with self.assertRaises(ValueError):
            epigraph_direction(K, g+.1, None, b, resume_checkpoint=cp)

    def test_many_independent_cusps_need_more_than_old64_cap_but_one_factor(self):
        n = 75
        offsets = np.linspace(.0001, .009, n)
        E = np.zeros(3*n); G = np.zeros((3*n, n))
        for i, a in enumerate(offsets):
            E[3*i:3*i+2] = [.5*100*(.1+a)**2, .5*100*(.1-a)**2]
            G[3*i:3*i+2, i] = [-100*(.1+a), 100*(.1-a)]
        b = dict(branch_energy_N_mm=E, branch_gradient_N=csr_matrix(G),
            branch_group=np.repeat(np.arange(n), 3), group_base_branch=3*np.arange(n))
        g = G[3*np.arange(n)].sum(axis=0)-2.
        d, _, _, linear, report = epigraph_direction(csr_matrix(np.eye(n)*112.), g, None, b)
        np.testing.assert_allclose(d, offsets, atol=1e-13)
        self.assertGreater(linear['exterior_active_set_iterations'], 64)
        self.assertEqual(linear['exterior_sparse_factorizations'], 1)
        self.assertLess(report['maximum_predicted_branch_violation_N_mm'], 1e-12)

    def test_unfinished_search_never_supplies_generalized_force(self):
        b = branches(.001); g = np.array([.812])
        d, returned_gradient, slope, linear, evidence = epigraph_direction(
            csr_matrix([[12.]]), g, None, b, inexact_search_after_operations=1)
        self.assertTrue(evidence['search_only'])
        self.assertFalse(evidence['qp_solved'])
        self.assertFalse(evidence['usable_for_generalized_stationarity'])
        self.assertFalse(evidence['actual_comaximality_satisfied'])
        np.testing.assert_array_equal(returned_gradient, g)
        self.assertLess(slope, 0.)
        self.assertGreater(evidence['qp_model_decrease_N_mm'], 0.)
        self.assertLess(evidence['maximum_predicted_branch_violation_N_mm'], 1e-12)

    def test_tied_loaded_cusp_cannot_exit_as_zero_search_direction(self):
        b = branches(0.)
        result = epigraph_direction(csr_matrix([[12.]]), [-1.3], None, b,
            inexact_search_after_operations=1)
        self.assertTrue(result[-1]['qp_solved'])
        self.assertFalse(result[-1]['search_only'])
        np.testing.assert_allclose(result[1], 0., atol=1e-14)

    def test_actual_driver_accepts_only_explicit_cusp_reaction(self):
        class Solid:
            X = np.zeros((1, 3))
            mass = np.array([.1])
            provenance = dict(kind='synthetic_loaded_cusp')
            def activation(self, a): return {}
            def deformation(self, q): return np.eye(3)[None]
            def energy(self, q, a): return 0., np.zeros_like(q), 1.
            def tangent(self, q, a): return csr_matrix((3, 3))
        class Guard:
            certification = dict.fromkeys(REQUIRED_GUARDS, True)
            fixture_only = True
            def energy(self, q):
                b = branches(q[0, 0], 3); j = b['group_base_branch'][0]
                return b['branch_energy_N_mm'][j], b['branch_gradient_N'][j].toarray().reshape(q.shape)
            def tangent(self, q): return csr_matrix(np.diag([10., 0., 0.]))
            def contact_branches(self, q): return branches(q[0, 0], 3)
            def endpoint(self, q): return dict(feasible=abs(q[0, 0]) < .09)
            def path(self, a, b): return dict(feasible=True, whole_segment_certified=True)
            def safe_step_bound(self, q, d, safety): return 1.
        s = Solid(); solver = ImplicitDynamics(s, Guard(), gravity=[3., 0., 0.])
        q, v, r = solver.step(s.X, np.zeros_like(s.X), dt=.01)
        self.assertTrue(r['accepted'])
        self.assertTrue(r['contact_generalized_stationarity_qualified'])
        self.assertGreater(r['smooth_selected_branch_dynamic_residual_max_N'], 1.)
        self.assertLess(r['dynamic_residual_max_N'], 1e-12)
        self.assertEqual(r['support_reaction_max_N'], 0.)
        np.testing.assert_array_equal(q, s.X)
        np.testing.assert_array_equal(v, 0.)


if __name__ == '__main__':
    unittest.main()
