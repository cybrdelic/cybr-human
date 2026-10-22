import unittest
import numpy as np
from scipy.sparse import csr_matrix, vstack
from linear_equilibration import solve_kkt
from factored_seam_kkt import FactoredSeamKKT


class FactoredTests(unittest.TestCase):
    def test_source_and_changing_seams_match_full_kkt(self):
        rng = np.random.default_rng(632)
        M = rng.normal(size=(15, 15))
        K = csr_matrix(M.T @ M + np.eye(15))
        C = csr_matrix(rng.normal(size=(3, 15)))
        seams = csr_matrix(rng.normal(size=(4, 15)))
        rhs = rng.normal(size=15)
        solver = FactoredSeamKKT(K, C)
        for ids in ([], [0], [0, 1], [1, 2, 3], [0, 1]):
            A = seams[ids]
            expected, lm, _ = solve_kkt(K, rhs, vstack([C, A]))
            actual, mu, report = solver.solve(rhs, A)
            np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)
            np.testing.assert_allclose(mu, lm, rtol=2e-11, atol=2e-12)
            self.assertLess(report['original_primal_residual_max_N'], 2e-12)
            self.assertLess(report['original_source_constraint_residual_max_mm'], 2e-12)
            self.assertLess(report['original_seam_constraint_residual_max_N_mm'], 2e-12)
        self.assertEqual(solver.inverse_column_solves, 4)
        self.assertEqual(report['factorization_count'], 0)

    def test_equilibration_and_unconstrained_system(self):
        K = csr_matrix(np.diag([1e-8, .03, 8e4]))
        rhs = np.array([1e-6, -.03, 8.])
        A = csr_matrix([[1., 0., 1.]])
        for eq in (False, True):
            solver = FactoredSeamKKT(K, equilibrate=eq)
            x, lm, report = solver.solve(rhs, A)
            expected, em, _ = solve_kkt(K, rhs, A, equilibrate=eq)
            np.testing.assert_allclose(x, expected, atol=1e-10)
            np.testing.assert_allclose(lm, em, atol=1e-12)
            self.assertLess(report['original_system_componentwise_backward_error'], 1e-6)

    def test_fixed_arrays_are_copied(self):
        K = csr_matrix(np.eye(3)); C = csr_matrix([[1., 0., 0.]])
        solver = FactoredSeamKKT(K, C)
        K.data *= 4; C.data *= 9
        x, _, _ = solver.solve(np.ones(3))
        np.testing.assert_allclose(x, [0, 1, 1])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            FactoredSeamKKT(csr_matrix([[np.nan]]))
        solver = FactoredSeamKKT(csr_matrix(np.eye(2)))
        with self.assertRaises(ValueError):
            solver.solve(np.array([1., np.inf]))

if __name__ == '__main__':
    unittest.main()
