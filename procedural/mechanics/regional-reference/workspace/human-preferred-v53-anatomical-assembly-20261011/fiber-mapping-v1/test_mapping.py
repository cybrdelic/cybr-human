"""Small deterministic source-only tests. Run under the shared HUMAN guard."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_layered_fibres_v1 import (AuthoredCurve, ClosedVolume, QUADRATURE,
                                   assign_fractions, check_closed, closed_boundary)


class MappingTests(unittest.TestCase):
    def test_quadrature_is_interior_and_normalized(self):
        self.assertTrue((QUADRATURE > 0).all())
        np.testing.assert_allclose(QUADRATURE.sum(axis=1), 1)
        np.testing.assert_allclose(QUADRATURE.mean(axis=0), .25)

    def test_independent_families_and_passive_precedence(self):
        m = np.zeros((7, 4, 7), dtype=np.int8)
        m[0, :, 0] = 1
        m[1, :, :2] = 1
        m[2, :, :2] = 1
        m[2, :, 5] = 1
        m[4, :, 0] = 1
        m[4, 0, 1] = -1
        m[5, :, 0] = 1
        m[5, :, 6] = 1
        m[6, :2, 0] = 1
        m[6, 2:, 1] = 1
        result = assign_fractions(m)
        np.testing.assert_array_equal(result["family_weights"][0], [1, 0, 0, 0, 0])
        np.testing.assert_array_equal(result["family_weights"][1], [.5, .5, 0, 0, 0])
        self.assertEqual(result["passive_fraction"][2], 1)
        self.assertEqual(result["family_weights"][2].sum(), 0)
        self.assertEqual(result["unresolved_fraction"][3], 1)
        self.assertEqual(result["family_weights"][3].sum(), 0)
        self.assertEqual(result["unresolved_fraction"][4], .25)
        self.assertEqual(result["family_weights"][4, 0], .75)
        self.assertEqual(result["passive_fraction"][5], 1)
        np.testing.assert_array_equal(result["family_weights"][6], [.5, .5, 0, 0, 0])
        np.testing.assert_allclose(result["family_weights"].sum(axis=1)
                                   + result["passive_fraction"] + result["unresolved_fraction"], 1)

    @staticmethod
    def tetrahedron():
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        faces = closed_boundary(np.array([[0, 1, 2, 3]]))
        return vertices, faces

    def test_actual_containment_rejects_outside_near_route(self):
        vertices, faces = self.tetrahedron()
        source = ClosedVolume("synthetic_tetrahedron", vertices, faces)
        points = np.array([[.1, .1, .1], [.4, .4, .4], [-.01, .1, .1], [2, 2, 2],
                           [1e-8, .1, .1], [-1e-8, .1, .1]])
        np.testing.assert_array_equal(source.classify(points), [1, 0, 0, 0, 1, 0])
        self.assertGreaterEqual(source.stats["winding_fallbacks"], 2)

    def test_exact_source_boundary_is_ambiguous(self):
        vertices, faces = self.tetrahedron()
        source = ClosedVolume("synthetic_tetrahedron", vertices, faces)
        self.assertEqual(source.exact_winding(np.array([0., .2, .2]) - source.offset), -1)

    def test_open_surface_is_rejected(self):
        vertices, faces = self.tetrahedron()
        with self.assertRaises(ValueError):
            check_closed(vertices, faces[:-1])

    def test_straight_curve_unit_tangents(self):
        curve = AuthoredCurve(np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]], dtype=float))
        directions, parameters = curve.directions(np.array([[.5, .2, 0], [2, -.2, 0], [4, 0, 0.]]))
        np.testing.assert_allclose(directions, [[1, 0, 0]] * 3, atol=1e-12)
        np.testing.assert_allclose(parameters, [1 / 6, 2 / 3, 1], atol=1e-6)

    def test_hermite_preserves_authored_ring_directions(self):
        points = np.array([[0., 0., 0.], [1., .2, 0.], [2., 1., 0.]])
        tangents = np.array([[1., 0., 0.], [1., .5, 0.], [1., 1., 0.]])
        tangents /= np.linalg.norm(tangents, axis=1)[:, None]
        curve = AuthoredCurve(points, tangents)
        np.testing.assert_allclose(curve.derivative(curve.s), tangents, atol=1e-12)
        for knot in curve.s[1:-1]:
            np.testing.assert_allclose(curve.derivative(knot - 1e-8),
                                       curve.derivative(knot + 1e-8), atol=1e-6)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MappingTests))
    print(json.dumps({"tests": result.testsRun, "failures": len(result.failures),
                      "errors": len(result.errors), "passed": result.wasSuccessful()}), flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)
