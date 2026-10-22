"""One source-reference linear algebra probe, NOT a guarded physical step.

Contact terms cannot be included until finite source geometry is qualified.
This diagnostic therefore exports no q, velocity, or motion/frame artifact.
"""
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
from scipy.sparse import diags

from source_transfer import load_geometry_with_source_fibres
from clipped_origin_support import load_support_bundle
from implicit_dynamics import source_activation
from linear_equilibration import solve_kkt


def span(x):
    x = np.abs(np.asarray(x))
    positive = x[x > 0]
    return dict(min_positive=float(positive.min()), maximum=float(positive.max()),
        ratio=float(positive.max()/positive.min()), zero_entries=int(np.sum(x == 0)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain', type=Path, required=True)
    p.add_argument('--phases', type=Path, required=True)
    p.add_argument('--support-bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Use a new report path; previous audits are immutable')
    started = time.perf_counter()
    s = load_geometry_with_source_fibres(a.domain, a.phases)
    c, _ = load_support_bundle(s, a.support_bundle)
    dt = .001
    activation = source_activation(dt, case='zygomaticus_frontalis')
    energy, material_g, j = s.energy(s.X, activation)
    gravity_force = s.mass[:, None]*np.array([0., -9810., 0.])
    physical_g = material_g-gravity_force
    material_K = s.tangent(s.X, activation)
    K = material_K+diags(np.repeat(s.mass/dt**2, 3))
    before = dict(mass_N_s2_per_mm=span(s.mass),
        material_tangent_diagonal_N_per_mm=span(material_K.diagonal()),
        implicit_1ms_diagonal_N_per_mm=span(K.diagonal()))
    d, multiplier, linear = solve_kkt(K, -physical_g.ravel(), c.C3, equilibrate=True)
    direction = d.reshape(s.X.shape)
    stationarity = (K@d+c.C3.T@multiplier+physical_g.ravel()).reshape(s.X.shape)
    D = 1/np.sqrt(np.abs(K.diagonal()))
    report = dict(scope='One unconstrained-in-geometry source-state linear direction, no nonlinear trial/line search or physical step',
        domain_sha256=s.provenance['domain_sha256'], engineering_scale_mm_per_L0=s.mm_per_L0,
        diagnostic_dt_s=dt, physical_time_advanced_s=0., activation=activation,
        source_minimum_J=j, source_material_energy_N_mm=energy,
        source_material_force_max_N=float(np.linalg.norm(material_g, axis=1).max()),
        source_gravity_force_max_N=float(np.linalg.norm(gravity_force, axis=1).max()),
        source_material_plus_gravity_force_max_N=float(np.linalg.norm(physical_g, axis=1).max()),
        source_inertial_force_N=0., physical_damping_per_second=0.,
        physical_mass_kg=float(s.mass.sum()*1000), original_spans=before,
        equilibrated_primal_diagonal_abs_span=span(D*K.diagonal()*D),
        linear_system=linear, unaccepted_direction_max_mm=float(np.linalg.norm(direction, axis=1).max()),
        direction_constraint_violation_max_mm=float(np.abs(c.C3@d).max()),
        linearized_stationarity_max_N=float(np.linalg.norm(stationarity, axis=1).max()),
        directional_energy_slope_N_mm=float(physical_g.ravel()@d),
        original_physical_equations_used=True, no_artificial_mass_stiffness_or_pins=True,
        finite_contact_terms_included=False, finite_contact_absence_reason='Native repaired-v1 nonintersection is still failed; this is algebra only, not the complete physical step',
        no_actual_trial_coordinates_saved=True, nonlinear_solve_run=False, accepted_facial_frames=0,
        total_seconds=time.perf_counter()-started,
        peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
