import unittest
import numpy as np
from test_compact_penetration_barrier import fixture
from coarse_physical_site_contact_rolling import RollingPhysicalSiteContact


class RollingAdapterTests(unittest.TestCase):
    def test_same_energy_gradient_tangent_and_cap(self):
        solid, base = fixture()
        rolling = RollingPhysicalSiteContact.from_base(base)
        for displacement in ([0, 0, .55], [.01, 0, .56], [0, 0, 0], [0, 0, -.006]):
            q = solid.X+displacement
            eb, gb = base.energy(q)
            er, gr = rolling.energy(q)
            self.assertEqual(eb, er)
            self.assertTrue(np.array_equal(gb, gr, equal_nan=True))
            self.assertEqual(base.endpoint(q)['feasible'], rolling.endpoint(q)['feasible'])
            if np.isfinite(eb):
                self.assertTrue(np.array_equal(base.tangent(q).toarray(), rolling.tangent(q).toarray()))
        self.assertTrue(rolling.rolling.eligible.any())
        self.assertEqual(rolling.provenance['contact_enforcement'], 'compact_cap_barrier_v1')
        self.assertEqual(len(rolling.provenance['rolling_screen_source_sha256']), 64)

    def test_scale_change_preserves_geometry_anchors_but_updates_force(self):
        solid, base = fixture()
        rolling = RollingPhysicalSiteContact.from_base(base)
        rolling.endpoint(solid.X+[0, 0, .55])
        anchors = rolling.rolling.anchors.copy()
        base.set_barrier_scale(.1)
        rolling.set_barrier_scale(.1)
        self.assertTrue(np.array_equal(anchors, rolling.rolling.anchors))
        eb, gb = base.energy(solid.X)
        er, gr = rolling.energy(solid.X)
        self.assertEqual(eb, er)
        self.assertTrue(np.array_equal(gb, gr))

    def test_path_and_fraction_bound_unchanged(self):
        solid, base = fixture()
        rolling = RollingPhysicalSiteContact.from_base(base)
        q0 = solid.X+[0, 0, .55]
        q1 = solid.X-[0, 0, .007]
        rolling.endpoint(q0)
        self.assertFalse(rolling.path(q0, q1)['feasible'])
        self.assertEqual(base.path(q0, q1)['feasible'], rolling.path(q0, q1)['feasible'])
        direction = np.tile([0., 0., -1.], (4, 1))
        self.assertEqual(base.safe_step_bound(solid.X, direction), rolling.safe_step_bound(solid.X, direction))


if __name__ == '__main__':
    unittest.main()
