"""Bounded anchor membership, finite-band and existing cap/path regressions."""
import unittest
import numpy as np
from coarse_rigid_query import NativeRigidQuery
from rolling_exterior_bounds import RollingExteriorBounds
from test_compact_penetration_barrier import fixture


class RollingBoundsTests(unittest.TestCase):
    def test_valid_shallow_inside_is_not_an_exterior_anchor(self):
        solid, guard = fixture()
        p = guard.C@solid.X
        r = guard.rigid.query_band(p, owner_terms=True)
        self.assertTrue(r['valid'][0])
        self.assertTrue(r['inside_closed_owner'][0])
        screen = RollingExteriorBounds(guard.rigid, p, [0.])
        report = screen.refresh(p, [0], r)
        self.assertEqual(report['rejected_inside_sites'], 1)
        self.assertFalse(screen.eligible[0])
        self.assertLessEqual(screen.lower_bound(p)[0], .1)

    def test_exterior_anchor_releases_only_inactive_finite_sites(self):
        solid, guard = fixture()
        p = guard.C@solid.X+[0., 0., .55]
        screen = RollingExteriorBounds(guard.rigid, p, [0.])
        r = guard.rigid.query_band(p, owner_terms=True)
        self.assertTrue(r['valid'][0])
        self.assertFalse(r['inside_closed_owner'][0])
        self.assertTrue(r['distance_capped'][0])
        screen.refresh(p, [0], r)
        self.assertTrue(screen.eligible[0])
        for delta in ([.1, 0., 0.], [0., 0., -.2]):
            moved = p+delta
            self.assertGreater(screen.lower_bound(moved)[0], .1)
            exact = guard.rigid.query_band(moved, owner_terms=True)
            self.assertEqual(len(exact['contact_site_index']), 0)
            self.assertFalse(exact['inside_closed_owner'][0])

    def test_invalid_exterior_query_cannot_create_anchor(self):
        solid, guard = fixture()
        p = guard.C@solid.X+[0., 0., .55]
        r = guard.rigid.query_band(p, owner_terms=True)
        r['valid'][:] = False
        screen = RollingExteriorBounds(guard.rigid, p, [0.])
        screen.refresh(p, [0], r)
        self.assertFalse(screen.eligible.any())

    def test_cap_and_sampled_path_still_reject_crossing(self):
        solid, guard = fixture()
        q0 = solid.X+[0., 0., .55]
        q1 = solid.X-[0., 0., .007]
        p0, p1 = guard.C@q0, guard.C@q1
        screen = RollingExteriorBounds(guard.rigid, p0, [0.])
        screen.refresh(p0, [0], guard.rigid.query_band(p0, owner_terms=True))
        self.assertLessEqual(screen.lower_bound(p1)[0], .1)
        count = int(np.ceil(np.linalg.norm(p1-p0)/.25))
        feasibility = []
        for j in range(count+1):
            p = p0+(j/count)*(p1-p0)
            baseline = guard.rigid.query_band(p, owner_terms=True)
            if screen.lower_bound(p)[0] > .1:
                self.assertTrue(baseline['valid'][0])
                self.assertFalse(baseline['inside_closed_owner'][0])
                self.assertEqual(len(baseline['contact_site_index']), 0)
                feasibility.append(True)
            else:
                feasibility.append(bool(baseline['valid'][0] and baseline['signed_gap'][0] > -.05))
        self.assertFalse(all(feasibility))
        self.assertFalse(guard.endpoint(q1)['feasible'])
        self.assertFalse(guard.path(q0, q1)['feasible'])

    def test_open_sheet_remains_active_without_solid_invention(self):
        rigid = NativeRigidQuery([[-3., -3., 0.], [4., -3., 0.], [-3., 4., 0.]], [[0, 1, 2]], [9])
        p = np.array([[.1, .1, .5]])
        screen = RollingExteriorBounds(rigid, p, [0.])
        screen.refresh(p, [0], rigid.query_band(p, owner_terms=True))
        for z in (.025, -.025):
            near = np.array([[.1, .1, z]])
            self.assertLessEqual(screen.lower_bound(near)[0], .1)
            r = rigid.query_band(near, owner_terms=True)
            self.assertFalse(r['inside_closed_owner'][0])
            self.assertEqual(r['contact_owner_id'].tolist(), [9])
            self.assertAlmostEqual(r['contact_signed_gap'][0], abs(z))

    def test_geometry_or_scale_change_invalidates_anchors(self):
        solid, guard = fixture()
        p = guard.C@solid.X+[0., 0., .55]
        screen = RollingExteriorBounds(guard.rigid, p, [0.])
        guard.rigid.vertices[0, 0] += .001
        with self.assertRaises(ValueError):
            screen.lower_bound(p)

    def test_nonfinite_or_misaligned_queries_fail_closed(self):
        solid, guard = fixture()
        p = guard.C@solid.X+[0., 0., .55]
        screen = RollingExteriorBounds(guard.rigid, p, [0.])
        r = guard.rigid.query_band(p, owner_terms=True)
        for bad in (p*np.nan, p*np.inf, np.r_[p, p]):
            with self.assertRaises(ValueError):
                screen.lower_bound(bad)
            with self.assertRaises(ValueError):
                screen.refresh(bad, [0], r)
        bad_evidence = dict(r, valid=np.array([np.nan]))
        with self.assertRaises(ValueError):
            screen.refresh(p, [0], bad_evidence)


if __name__ == '__main__':
    unittest.main()
