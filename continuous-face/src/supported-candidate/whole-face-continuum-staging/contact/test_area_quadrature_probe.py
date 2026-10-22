"""Bounded synthetic qualification of the diagnostic reference-area proposal.

This is not an adapter, source-geometry qualification, collision certificate,
or authorization to change a force law. No actual anatomy or dynamics runs.
Run directly to refresh area-quadrature-synthetic.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.sparse import eye

from area_quadrature_probe import reference_vertex_areas, evaluate_area_rule


HERE = Path(__file__).resolve().parent
CLEARANCE = .02
COEFFICIENT = .001
GAP = .01


def opposed_patches(subdivisions=1, gap=GAP, x_shift=0.):
    """Two finite, opposed 1 mm x 1 mm squares with disjoint vertices."""
    n = int(subdivisions)
    xy = np.array([(i/n, j/n) for j in range(n+1) for i in range(n+1)])
    lower = np.column_stack((xy, np.zeros(len(xy))))
    upper = lower + [x_shift, 0., gap]
    one = []
    for j in range(n):
        for i in range(n):
            a = j*(n+1)+i
            b, c, d = a+1, a+n+1, a+n+2
            one.extend(((a, b, d), (a, d, c)))
    one = np.asarray(one, dtype=np.int64)
    faces = np.concatenate((one, one[:, ::-1]+len(lower)))
    q = np.concatenate((lower, upper))
    pairs = np.array([(i, j) for i in range(len(one))
                      for j in range(len(one), len(faces))], dtype=np.int64)
    C = eye(len(q), format='csr')
    areas = reference_vertex_areas(C, faces, q)
    return C, faces, q, areas, pairs, len(lower)


def analytic_barrier(gap, clearance=CLEARANCE):
    if gap >= clearance:
        return 0., 0.
    delta = gap-clearance
    logarithm = np.log(gap/clearance)
    return -delta*delta*logarithm, -2*delta*logarithm-delta*delta/gap


def evaluate(fixture, q=None, pairs=None):
    C, faces, reference, areas, eligible, _ = fixture
    energy, gradient, report = evaluate_area_rule(
        C, faces, reference if q is None else q, areas,
        eligible if pairs is None else pairs,
        clearance=CLEARANCE, coefficient=COEFFICIENT)
    gradient = np.asarray(gradient)
    assert np.isfinite(energy)
    assert gradient.shape == reference.shape
    assert np.isfinite(gradient).all()
    return float(energy), gradient, report


def close(actual, expected, *, rtol=2e-12, atol=1e-16):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def unit_square_energy_and_reciprocity():
    fixture = opposed_patches()
    _, _, q, areas, _, split = fixture
    assert np.all(areas > 0)
    close(areas[:split].sum(), 1.)
    close(areas[split:].sum(), 1.)
    energy, gradient, report = evaluate(fixture)
    barrier, derivative = analytic_barrier(GAP)
    expected_energy = COEFFICIENT*barrier
    expected_force = COEFFICIENT*(-derivative)
    forces = -gradient
    lower, upper = forces[:split].sum(axis=0), forces[split:].sum(axis=0)
    close(energy, expected_energy)
    close(lower, [0., 0., -expected_force])
    close(upper, [0., 0., expected_force])
    close(forces.sum(axis=0), np.zeros(3))
    close(np.cross(q, forces).sum(axis=0), np.zeros(3))
    close(report['active_gap_mm'], GAP)
    assert report['active_vertices'] == len(q)
    return dict(gap_mm=GAP, reference_area_per_patch_mm2=1.,
                expected_energy_N_mm=expected_energy, energy_N_mm=energy,
                expected_traction_N_per_mm2=expected_force,
                lower_total_force_N=lower.tolist(), upper_total_force_N=upper.tolist(),
                net_force_N=forces.sum(axis=0).tolist(),
                net_torque_N_mm=np.cross(q, forces).sum(axis=0).tolist())


def fixed_reference_area_under_current_stretch():
    fixture = opposed_patches(2)
    q = fixture[2].copy()
    q[:, :2] *= [1.7, .8]
    energy, gradient, _ = evaluate(fixture, q=q)
    original_energy, original_gradient, _ = evaluate(fixture)
    close(energy, original_energy)
    close(gradient, original_gradient)
    return dict(current_area_per_patch_mm2=1.7*.8,
                reference_area_per_patch_mm2=1., energy_N_mm=energy,
                reference_weights_reused_without_current_area_rescaling=True)


def constant_gap_refinement_invariance():
    barrier, derivative = analytic_barrier(GAP)
    expected_energy, expected_force = COEFFICIENT*barrier, -COEFFICIENT*derivative
    rows = []
    for n in (1, 2, 4):
        fixture = opposed_patches(n)
        energy, gradient, _ = evaluate(fixture)
        split = fixture[-1]
        lower = -gradient[:split].sum(axis=0)
        upper = -gradient[split:].sum(axis=0)
        close(energy, expected_energy)
        close(lower, [0., 0., -expected_force])
        close(upper, [0., 0., expected_force])
        rows.append(dict(subdivisions_per_side=n, vertices_per_patch=split,
                         facets_per_patch=2*n*n, supplied_facet_pairs=len(fixture[4]),
                         energy_N_mm=energy, upper_total_force_N=upper.tolist(),
                         upper_traction_N_per_mm2=float(upper[2]),
                         energy_relative_error=abs(energy/expected_energy-1),
                         force_relative_error=abs(upper[2]/expected_force-1)))
    return dict(cases=rows,
                limitation='Constant-gap flat-patch identity, not general curved-surface convergence.')


def finite_difference_gradient():
    fixture = opposed_patches(2)
    q, split = fixture[2], fixture[-1]
    _, gradient, _ = evaluate(fixture)
    directions = []
    separation = np.zeros_like(q)
    separation[split:, 2] = 1.
    directions.append(('upper_patch_normal_translation', separation))
    rng = np.random.default_rng(59311)
    for i in range(6):
        direction = rng.normal(size=q.shape)
        direction /= np.linalg.norm(direction)
        directions.append((f'seeded_direction_{i}', direction))
    rows = []
    epsilon = 2e-7
    for label, direction in directions:
        plus = evaluate(fixture, q=q+epsilon*direction)[0]
        minus = evaluate(fixture, q=q-epsilon*direction)[0]
        finite_difference = (plus-minus)/(2*epsilon)
        analytic = float(np.sum(gradient*direction))
        error = abs(finite_difference-analytic)/max(abs(analytic), 1e-10)
        assert error < 2e-5, (label, error, finite_difference, analytic)
        rows.append(dict(direction=label, finite_difference=finite_difference,
                         analytic_directional_derivative=analytic, relative_error=error))
    return dict(epsilon_mm=epsilon, cases=rows,
                max_relative_error=max(row['relative_error'] for row in rows))


def eligible_pair_multiplicity_and_order():
    fixture = opposed_patches(2)
    pairs = fixture[4]
    energy, gradient, report = evaluate(fixture)
    repeated = np.concatenate((pairs, pairs, pairs, pairs[:7]))
    permuted = repeated[np.random.default_rng(781).permutation(len(repeated))]
    for supplied in (repeated, permuted, pairs[::-1]):
        other_energy, other_gradient, other_report = evaluate(fixture, pairs=supplied)
        close(other_energy, energy)
        close(other_gradient, gradient)
        for key in ('active_vertex_ids', 'active_target_faces', 'active_gap_mm'):
            assert np.array_equal(other_report[key], report[key]), key
        assert other_report['unique_vertex_target_queries'] == report['unique_vertex_target_queries']
    again_energy, again_gradient, _ = evaluate(fixture)
    assert again_energy == energy
    assert np.array_equal(again_gradient, gradient)
    return dict(unique_supplied_facet_pairs=len(pairs), duplicated_facet_pairs=len(repeated),
                energy_N_mm=energy, duplication_and_order_invariant=True,
                identical_input_bitwise_repeatable=True,
                tied_target_selection_and_reported_distances_deterministic=True,
                limitation='Eligible-pair list duplication only; geometric duplicate surfaces are not tested.')


def inactive_gap_and_empty_candidates():
    rows = []
    for gap in (CLEARANCE, .03):
        fixture = opposed_patches(2, gap=gap)
        energy, gradient, _ = evaluate(fixture)
        assert energy == 0.
        assert np.count_nonzero(gradient) == 0
        rows.append(dict(gap_mm=gap, energy_N_mm=energy, nonzero_gradient_entries=0))
    fixture = opposed_patches()
    energy, gradient, _ = evaluate(fixture, pairs=np.empty((0, 2), dtype=np.int64))
    assert energy == 0. and np.count_nonzero(gradient) == 0
    return dict(cases=rows, empty_candidate_energy_N_mm=energy,
                empty_candidates_are_not_a_collision_certificate=True)


def finite_edges_not_infinite_planes():
    lateral_gap = .006
    fixture = opposed_patches(x_shift=1.+lateral_gap)
    distance = float(np.hypot(lateral_gap, GAP))
    barrier, derivative = analytic_barrier(distance)
    expected_energy = .5*COEFFICIENT*barrier
    expected_upper = -.5*COEFFICIENT*derivative*np.array([lateral_gap, 0., GAP])/distance
    energy, gradient, report = evaluate(fixture)
    split = fixture[-1]
    upper, lower = -gradient[split:].sum(axis=0), -gradient[:split].sum(axis=0)
    close(energy, expected_energy)
    close(upper, expected_upper)
    close(lower, -expected_upper)
    close(report['active_gap_mm'], distance)
    assert report['active_vertices'] == 4
    for _ in range(3):
        again_energy, again_gradient, _ = evaluate(fixture)
        assert again_energy == energy and np.array_equal(again_gradient, gradient)
    far = opposed_patches(x_shift=2.)
    far_energy, far_gradient, _ = evaluate(far)
    assert far_energy == 0. and np.count_nonzero(far_gradient) == 0
    return dict(normal_plane_gap_mm=GAP, lateral_edge_gap_mm=lateral_gap,
                finite_active_distance_mm=distance,
                expected_energy_N_mm=expected_energy, energy_N_mm=energy,
                upper_total_force_N=upper.tolist(), far_disjoint_square_energy_N_mm=far_energy,
                finite_feature_distance_distinguished_from_plane_gap=True,
                finite_triangle_collision_guards_still_required=True)


def intersecting_faces_require_independent_guard():
    # No vertex is near the opposite triangle, yet an edge crosses its interior.
    # The energy proposal may return zero, so finite-face guards are essential.
    from triangle_interface_contact import triangle_geometry
    q = np.array([[-1., -1., 0.], [1., -1., 0.], [0., 1., 0.],
                  [0., -.5, -1.], [0., -.5, 1.], [0., .5, 1.]])
    faces = np.arange(6, dtype=np.int64).reshape(2, 3)
    C = eye(len(q), format='csr')
    areas = reference_vertex_areas(C, faces, q)
    pairs = np.array([[0, 1]], dtype=np.int64)
    fixture = C, faces, q, areas, pairs, 3
    energy, gradient, report = evaluate(fixture)
    assert energy == 0. and np.count_nonzero(gradient) == 0
    geometry = triangle_geometry(q[faces[:1]], q[faces[1:]])
    assert bool(geometry['intersect'][0])
    assert report['geometric_guards_replaced'] is False
    return dict(area_quadrature_energy_N_mm=energy,
                finite_triangle_guard_rejects_intersection=True,
                zero_quadrature_energy_does_not_certify_separation=True,
                guard_function='unchanged triangle_interface_contact.triangle_geometry')


def main():
    started = time.perf_counter()
    checks = (unit_square_energy_and_reciprocity,
              fixed_reference_area_under_current_stretch,
              constant_gap_refinement_invariance,
              finite_difference_gradient,
              eligible_pair_multiplicity_and_order,
              inactive_gap_and_empty_candidates,
              finite_edges_not_infinite_planes,
              intersecting_faces_require_independent_guard)
    rows = []
    for check in checks:
        tick = time.perf_counter()
        try:
            detail = check()
            row = dict(name=check.__name__, passed=True, **detail)
        except Exception as exc:
            row = dict(name=check.__name__, passed=False, error=str(exc),
                       traceback=traceback.format_exc())
        row['seconds'] = time.perf_counter()-tick
        rows.append(row)
        print(check.__name__, 'PASS' if row['passed'] else 'FAIL', flush=True)
    passed = sum(row['passed'] for row in rows)
    report = dict(
        status='synthetic_area_quadrature_passed' if passed == len(rows) else 'synthetic_area_quadrature_failed',
        tests_passed=passed, tests_total=len(rows), tests=rows,
        wall_seconds=time.perf_counter()-started,
        coefficient_N_per_mm3=COEFFICIENT, activation_clearance_mm=CLEARANCE,
        energy_definition='0.5 * k * sum(reference vertex area * b(nearest eligible finite triangle gap)), both directions',
        reference_measure='Lumped incident reference triangle area divided by three; positive per used vertex.',
        scope='Bounded manufactured flat-patch diagnostics only. No actual source, hotspot, or dynamics result is claimed by this suite.',
        force_law_adopted=False, actual_facial_frames=0, actual_geometry_changed=False,
        collision_guards_replaced=False, scientific_validation=False,
        limitations=[
            'Point quadrature is an energy discretization, never a finite triangle collision or swept-path certificate.',
            'Constant-gap refinement invariance does not establish general quadrature convergence or calibrated traction.',
            'Nearest eligible target changes can be nonsmooth; only first derivatives at these bounded fixtures are tested.',
            'The coefficient has units N/mm^3, unlike the prior unit-facet-pair coefficient N/mm.',
        ],
        source_sha256={name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                       for name in ('area_quadrature_probe.py', 'test_area_quadrature_probe.py')},
    )
    (HERE/'area-quadrature-synthetic.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: report[key] for key in ('status', 'tests_passed', 'tests_total', 'wall_seconds')}, indent=2))
    if passed != len(rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
