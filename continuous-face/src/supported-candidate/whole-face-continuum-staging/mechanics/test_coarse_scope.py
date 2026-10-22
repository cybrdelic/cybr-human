"""The deliberately coarse guard scope cannot masquerade as boundary CCD."""
import unittest
import numpy as np
from test_mechanics import fixture, ObstacleFreeFixture
from implicit_dynamics import COARSE_SITE_GUARDS, ImplicitDynamics, MissingGeometryCertification


class CoarseFixture(ObstacleFreeFixture):
    certification = dict.fromkeys(COARSE_SITE_GUARDS, True)
    def path(self, a, b):
        return dict(feasible=True, coarse_whole_segment_checked=True, fixture_only=True)


class CoarseScopeTests(unittest.TestCase):
    def test_opt_in_and_labels(self):
        s = fixture()
        with self.assertRaises(MissingGeometryCertification):
            ImplicitDynamics(s, CoarseFixture()).require_guards()
        with self.assertRaises(MissingGeometryCertification):
            ImplicitDynamics(s, ObstacleFreeFixture(), guard_scope='coarse_physical_sites').require_guards()
        solver = ImplicitDynamics(s, CoarseFixture(), guard_scope='coarse_physical_sites')
        q, v, r = solver.step(s.X, np.zeros_like(s.X), dt=.001)
        self.assertTrue(r['accepted'])
        self.assertFalse(r['exact_whole_boundary_collision_claim'])
        self.assertNotIn('whole_segment_certified', r['frame_path'])
        self.assertEqual(r['guard_scope'], 'coarse_physical_sites')
        self.assertGreater(np.linalg.norm(q-s.X), 0.)

    def test_no_fixture_on_face(self):
        s = fixture()
        s.provenance['kind'] = 'actual registered source anatomy'
        with self.assertRaises(MissingGeometryCertification):
            ImplicitDynamics(s, CoarseFixture(), guard_scope='coarse_physical_sites').require_guards()


if __name__ == '__main__':
    unittest.main()
