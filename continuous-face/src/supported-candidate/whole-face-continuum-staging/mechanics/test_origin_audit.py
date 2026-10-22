"""Deterministic finite-clipping checks, independent of anatomical claims."""
import json
from pathlib import Path
import unittest
import tempfile
import numpy as np

from audit_origin_occupancy import (VoxelDomain, clip_triangle_cube,
                                    polygon_area, closest_triangle_distance)


class AuditTests(unittest.TestCase):
    def test_finite_triangle_clip(self):
        tri = np.array([[-1., .2, .5], [2., .2, .5], [.5, 2., .5]])
        self.assertAlmostEqual(clip_triangle_cube(tri, np.zeros(3), np.ones(3)), .8, places=12)
        self.assertAlmostEqual(clip_triangle_cube(tri[::-1], np.zeros(3), np.ones(3)), .8, places=12)
        inside = np.array([[.1, .1, .5], [.9, .1, .5], [.1, .9, .5]])
        self.assertAlmostEqual(clip_triangle_cube(inside, np.zeros(3), np.ones(3)), polygon_area(inside), places=12)
        self.assertEqual(clip_triangle_cube(inside+3, np.zeros(3), np.ones(3)), 0.)

    def test_union_boundary_and_segment_gap(self):
        domain = VoxelDomain.__new__(VoxelDomain)
        domain.origin, domain.h, domain.shape = np.zeros(3), 1., np.array([3, 1, 1])
        domain.occ = np.array([True, False, True]).reshape(3, 1, 1)
        status = domain.classify_points([[.5, .5, .5], [1., .5, .5], [1.5, .5, .5]])
        np.testing.assert_array_equal(status, [1, 2, 0])
        self.assertAlmostEqual(domain.segment_coverage(np.array([.25, .5, .5]), np.array([2.75, .5, .5])), .6, places=12)

    def test_finite_native_triangle_distance(self):
        triangle = np.array([[[0., 0, 0], [1., 0, 0], [0., 1, 0]]])
        self.assertAlmostEqual(closest_triangle_distance([3., 0, 0], triangle)[0], 2.)
        self.assertAlmostEqual(closest_triangle_distance([.2, .2, .7], triangle)[0], .7)

    def test_repaired_linear_cut_source_patch(self):
        from clipped_origin_support import CutFineDomain, triangle_barycentric
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"fixture.npz"
            np.savez(path, fine_cell_ijk=np.array([[0, 0, 0]]), fine_permutation=np.array([0]),
                fine_node_phi=np.array([[-.5, .5, .5, .5]]), parent_tet=np.array([0]),
                geometry_origin=np.zeros(3), geometry_spacing=np.array(1.))
            domain = CutFineDomain(path)
            triangle = np.array([[.3, .2, .2], [.8, .2, .2], [.8, .7, .2]])
            pieces = list(domain.pieces(triangle))
            self.assertEqual(len(pieces), 1)
            _, _, polygon, area, centroid = pieces[0]
            self.assertAlmostEqual(area, .105, places=12)
            self.assertGreaterEqual(polygon[:, 0].min(), .5)
            self.assertGreaterEqual(triangle_barycentric(centroid, triangle).min(), 0.)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AuditTests))
    Path(__file__).with_name("origin-audit-tests.json").write_text(json.dumps(dict(
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        passed=result.wasSuccessful(), scope="finite clipping, union boundary, segment gaps and finite native-triangle distance"), indent=2)+"\n")
    raise SystemExit(not result.wasSuccessful())
