"""Small algebra tests only: diagonal scaling never changes the physical model."""
import unittest
import numpy as np
from scipy.sparse import csr_matrix, diags
from linear_equilibration import solve_kkt
from test_mechanics import fixture, ObstacleFreeFixture
from implicit_dynamics import ImplicitDynamics, OriginConstraints


class LinearEquilibrationTests(unittest.TestCase):
    def test_weak_diagonal_primal_equations_are_reversible(self):
        diagonal = np.logspace(-20, 2, 15)
        exact = np.linspace(-.01, .03, 15)
        K = diags(diagonal)
        x, multipliers, report = solve_kkt(K, diagonal*exact)
        np.testing.assert_allclose(x, exact, rtol=1e-14, atol=1e-17)
        self.assertEqual(len(multipliers), 0)
        self.assertLess(report['original_system_componentwise_backward_error'], 1e-14)
        np.testing.assert_array_equal(K.diagonal(), diagonal)

    def test_kkt_constraint_and_multiplier_recovery(self):
        rng = np.random.default_rng(41)
        A = rng.normal(size=(8, 8))
        K = csr_matrix(A.T@A+np.eye(8))
        C = csr_matrix([[1., 0, .3, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, .25, 0, .75]])
        x_exact, lambda_exact = rng.normal(size=8), rng.normal(size=2)
        rhs, target = K@x_exact+C.T@lambda_exact, C@x_exact
        x, multipliers, report = solve_kkt(K, rhs, C, target)
        np.testing.assert_allclose(x, x_exact, atol=2e-14, rtol=2e-14)
        np.testing.assert_allclose(multipliers, lambda_exact, atol=2e-14, rtol=2e-14)
        raw, raw_l, _ = solve_kkt(K, rhs, C, target, equilibrate=False)
        np.testing.assert_allclose(x, raw, atol=2e-14, rtol=2e-14)
        np.testing.assert_allclose(multipliers, raw_l, atol=2e-14, rtol=2e-14)
        self.assertLess(report['original_constraint_residual_max_mm'], 1e-14)

    def test_guarded_synthetic_transient_is_unchanged(self):
        solid = fixture()
        supports = OriginConstraints(csr_matrix([[1., 0, 0, 0], [0, 1., 0, 0]]), solid.X,
            provenance={'source_origin_only': True, 'fixture_only': True})
        outputs = []
        for scaled in (False, True):
            dynamics = ImplicitDynamics(solid, ObstacleFreeFixture(), constraints=supports,
                equilibrate_linear_system=scaled)
            q, v, report = dynamics.step(solid.X, np.zeros_like(solid.X), dt=.001,
                activation={'zygomaticus': .05}, max_iterations=12)
            self.assertTrue(report['accepted'])
            outputs.append((q, v, report))
        np.testing.assert_allclose(outputs[0][0], outputs[1][0], atol=1e-13, rtol=0)
        np.testing.assert_allclose(outputs[0][1], outputs[1][1], atol=1e-10, rtol=0)
        self.assertAlmostEqual(outputs[0][2]['physical_energy_N_mm'], outputs[1][2]['physical_energy_N_mm'], places=13)


if __name__ == '__main__':
    unittest.main(verbosity=2)
