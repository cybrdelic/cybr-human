"""Small local verification only. Synthetic fixtures are not facial frames."""
import json
from pathlib import Path
import unittest
import numpy as np
from scipy.sparse import csr_matrix

from continuum import CutContinuum, MUSCLE_COLUMN
from implicit_dynamics import (ImplicitDynamics, MissingGeometryCertification,
    OriginConstraints, REQUIRED_GUARDS, source_activation)


def fixture():
    X = np.array([[0., 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4]])
    T = np.array([[0, 1, 2, 3]])
    a, b = .5854101966249685, .1381966011250105
    bary = np.full((4, 4), b)
    np.fill_diagonal(bary, a)
    fibre = np.array([[1., 0, 0], [0, 1., 0], [1., 0, 0], [0, 1., 0]])
    return CutContinuum(nodes=X, tets=T, quadrature_points=bary@X,
        quadrature_weights=np.full(4, 64/24), quadrature_tet=np.zeros(4, int),
        quadrature_barycentric=bary, material_fractions=np.tile([.2, .3, .1, .2, .2], (4, 1)),
        quadrature_muscle_family=np.array([0, 1, 0, 1]),
        family_names=["zygomaticus", "frontalis"], quadrature_fibre=fibre,
        provenance={"kind": "synthetic four-point integration fixture, no face geometry"})


class ObstacleFreeFixture:
    fixture_only = True
    certification = dict.fromkeys(REQUIRED_GUARDS, True)
    def endpoint(self, q):
        return {"feasible": True, "fixture_only": True, "reason": "single isolated tetrahedron, no obstacles or separated surfaces"}
    def path(self, a, b):
        return {"feasible": True, "whole_segment_certified": True, "fixture_only": True}
    def energy(self, q):
        return 0., np.zeros_like(q)
    def tangent(self, q):
        return csr_matrix((q.size, q.size))
    def safe_step_bound(self, q, d, **kwargs):
        return 1.


RESULTS = {}


class MechanicsTests(unittest.TestCase):
    def test_derivatives(self):
        s = fixture()
        rng = np.random.default_rng(71021)
        q = s.X @ np.array([[1.03, .04, -.02], [.01, .96, .03], [.02, -.01, 1.02]]).T
        q += rng.normal(size=q.shape)*.03
        d = rng.normal(size=q.shape)
        d /= np.linalg.norm(d)
        activation = {"zygomaticus": .35, "frontalis": .2}
        eps = 2e-6
        e, g, _ = s.energy(q, activation)
        ep, gp, _ = s.energy(q+eps*d, activation)
        em, gm, _ = s.energy(q-eps*d, activation)
        K = s.tangent(q, activation)
        gradient_error = abs((ep-em)/(2*eps)-np.sum(g*d))
        tangent_error = np.linalg.norm((gp-gm)/(2*eps)-(K@d.ravel()).reshape(q.shape))
        symmetry = np.linalg.norm((K-K.T).toarray())
        RESULTS["derivative"] = dict(gradient_abs_error=gradient_error, exact_tangent_abs_error=tangent_error, symmetry_abs_error=symmetry)
        self.assertLess(gradient_error, 1e-9)
        self.assertLess(tangent_error, 1e-8)
        self.assertLess(symmetry, 1e-12)

    def test_rigid_objectivity(self):
        s = fixture()
        q = s.X @ np.diag([1.03, .97, 1.01])
        theta = .47
        R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        act = {"zygomaticus": .4, "frontalis": .25}
        e, g, _ = s.energy(q, act)
        er, gr, _ = s.energy(q@R.T+np.array([20, -35, 11]), act)
        resultant = np.linalg.norm(g.sum(0))
        moment = np.linalg.norm(np.cross(q, g).sum(0))
        RESULTS["objectivity"] = dict(energy_abs_error=abs(er-e), force_rotation_error=float(np.linalg.norm(gr-g@R.T)), resultant_N=float(resultant), moment_N_mm=float(moment))
        self.assertAlmostEqual(e, er, places=12)
        np.testing.assert_allclose(gr, g@R.T, atol=1e-12, rtol=0)
        self.assertLess(resultant, 1e-12)
        self.assertLess(moment, 1e-12)

    def test_mass_and_volume(self):
        s = fixture()
        audit = s.audit()
        self.assertAlmostEqual(audit["physical_volume_mm3"], 64/6)
        self.assertAlmostEqual(sum(v for v in audit["phase_volume_mm3"].values()), 64/6)
        self.assertLess(abs(audit["mass_conservation_error"]), 1e-22)
        self.assertLess(audit["consistent_mass_row_sum_error"], 1e-22)
        self.assertAlmostEqual(sum(s.volume.sum() for _, s in s.batches), 64/6)
        self.assertAlmostEqual(sum(audit["active_source_volume_mm3"].values()), .3*64/6)
        for activation in (0., .1, .5, 1.):
            gamma = 1-.2*activation
            for f in s.fibre:
                A = np.sqrt(gamma)*np.eye(3)+(1/gamma-np.sqrt(gamma))*np.outer(f, f)
                self.assertAlmostEqual(np.linalg.det(A), 1., places=14)
        RESULTS["conservation"] = audit

    def test_guard_fail_closed(self):
        s = fixture()
        solver = ImplicitDynamics(s, None)
        with self.assertRaises(MissingGeometryCertification):
            solver.step(s.X, np.zeros_like(s.X), dt=.01)
        RESULTS["missing_geometry_guard"] = "rejects motion before producing a frame"

    def test_positive_J_path(self):
        s = fixture()
        solver = ImplicitDynamics(s, ObstacleFreeFixture())
        d = -2*s.X
        alpha = solver.safe_volume_step_bound(s.X, d)
        minimum = min(np.linalg.det(s.deformation(s.X+t*d)).min() for t in np.linspace(0, alpha, 51))
        self.assertGreater(minimum, solver.jacobian_floor)
        RESULTS["positive_J_path"] = dict(bound=alpha, sampled_minimum_J=float(minimum))

    def test_constraints_and_transient(self):
        s = fixture()
        C = csr_matrix(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]], float))
        supports = OriginConstraints(C, s.X, provenance={"source_origin_only": True, "fixture_only": True})
        self.assertEqual(supports.report["rank_scalar"], 2)
        solver = ImplicitDynamics(s, ObstacleFreeFixture(), constraints=supports, damping_per_second=0.)
        q, v, report = solver.step(s.X, np.zeros_like(s.X), dt=.001, activation={"zygomaticus": .05}, max_iterations=12)
        self.assertTrue(report["accepted"], report)
        self.assertTrue(report["fixture_only"])
        self.assertLess(supports.error(q), 1e-10)
        self.assertGreater(report["physical_force_max_N"], report["dynamic_residual_max_N"])
        accepted = [r for r in report["history"] if "step" in r]
        for before, after in zip(report["history"], report["history"][1:]):
            self.assertLess(after["implicit_objective_N_mm"], before["implicit_objective_N_mm"])
        RESULTS["synthetic_transient_only"] = dict(status=report["status"], nonlinear_iterations=len(report["history"])-1,
            physical_force_max_N=report["physical_force_max_N"], dynamic_residual_max_N=report["dynamic_residual_max_N"],
            support_reaction_max_N=report["support_reaction_max_N"], displacement_max_mm=float(np.linalg.norm(q-s.X, axis=1).max()),
            support=supports.report, accepted_for_facial_motion=False)

    def test_failed_step_never_advances(self):
        s = fixture()
        solver = ImplicitDynamics(s, ObstacleFreeFixture())
        q, v, report = solver.step(s.X, np.zeros_like(s.X), dt=.01, max_iterations=0)
        self.assertFalse(report["accepted"])
        np.testing.assert_array_equal(q, s.X)
        np.testing.assert_array_equal(v, np.zeros_like(s.X))
        RESULTS["failed_step"] = "returns unchanged physical state, not pending trial coordinates"


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MechanicsTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    RESULTS["summary"] = dict(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        passed=result.wasSuccessful(), scope="small synthetic mechanics only; no accepted facial motion")
    Path(__file__).with_name("mechanics-tests.json").write_text(json.dumps(RESULTS, indent=2)+"\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
