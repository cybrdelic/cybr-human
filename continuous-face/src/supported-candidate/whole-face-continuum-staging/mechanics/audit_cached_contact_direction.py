"""One cached-force KKT probe; no trial state, line search, or frame export."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.sparse import diags
from source_transfer import load_geometry_with_source_fibres, sha
from clipped_origin_support import load_support_bundle
from implicit_dynamics import source_activation
from linear_equilibration import solve_kkt


def peak(vectors, positions, *, parents=None):
    magnitude = np.linalg.norm(vectors, axis=1)
    i = int(magnitude.argmax())
    out = dict(index=i, maximum_mm=float(magnitude[i]), reference_point_mm=positions[i].tolist(),
        correction_vector_mm=vectors[i].tolist())
    if parents is not None:
        out['parent_tet'] = int(parents[i])
    return out


def main():
    p = argparse.ArgumentParser()
    for name in ('domain', 'phases', 'support-bundle', 'cached-force', 'fine-cut', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Use a new report; preserve previous immutable diagnostics')
    started = time.perf_counter()
    solid = load_geometry_with_source_fibres(a.domain, a.phases)
    supports, _ = load_support_bundle(solid, a.support_bundle)
    force_provenance = json.loads(a.cached_force.with_suffix('.json').read_text())
    with np.load(a.cached_force, allow_pickle=False) as cache:
        if str(cache['source_domain_sha256']) != solid.provenance['domain_sha256']:
            raise ValueError('Cached force belongs to a different immutable domain')
        contact_g = cache['self_gradient_N']
    if contact_g.shape != solid.X.shape or not np.isfinite(contact_g).all():
        raise ValueError('Invalid cached contact gradient')
    dt = .001
    activation = source_activation(dt, case='zygomaticus_frontalis')
    energy, material_g, _ = solid.energy(solid.X, activation)
    g = material_g+contact_g-solid.mass[:, None]*np.array([0., -9810., 0.])
    K = solid.tangent(solid.X, activation)+diags(np.repeat(solid.mass/dt**2, 3))
    d, multiplier, linear = solve_kkt(K, -g.ravel(), supports.C3, equilibrate=True)
    direction = d.reshape(solid.X.shape)
    dF = solid.deformation(direction)
    gradient_norm = np.linalg.norm(dF, axis=(1, 2))
    strain_norm = np.linalg.norm(.5*(dF+dF.transpose(0, 2, 1)), axis=(1, 2))
    rows = {}
    rows['carrier'] = peak(direction, solid.X)
    rows['carrier']['node_mass_N_s2_per_mm'] = float(solid.mass[rows['carrier']['index']])
    with np.load(a.domain, allow_pickle=False) as z:
        parents, bary = z['quadrature_tet'], z['quadrature_barycentric']
        vectors = np.einsum('qa,qai->qi', bary, direction[solid.T[parents]])
        rows['integrated_quadrature_centroids'] = peak(vectors, z['quadrature_points']*solid.mm_per_L0, parents=parents)
        parents, bary = z['exterior_tet'], z['exterior_barycentric']
        vectors = np.einsum('qa,qai->qi', bary, direction[solid.T[parents]])
        rows['finite_boundary_vertices'] = peak(vectors, z['vertices']*solid.mm_per_L0, parents=parents)
        v = rows['finite_boundary_vertices']['index']
        faces = np.flatnonzero(np.any(z['triangles'] == v, axis=1))
        rows['finite_boundary_vertices']['incident_boundary_classes'] = np.unique(z['boundary_class'][faces]).tolist()
    with np.load(a.fine_cut, allow_pickle=False) as fine:
        parents = fine['parent_tet']
        centers = fine['raw_cut_centroid']*solid.mm_per_L0
        vectors = direction[solid.T[parents, 0]]+np.einsum('qab,qb->qa', dF[parents], centers-solid.X[solid.T[parents, 0]])
        rows['actual_convex_cut_piece_centroids'] = peak(vectors, centers, parents=parents)
    contact_boundary = Path(force_provenance['boundary_path'])
    if sha(contact_boundary) != force_provenance['boundary_sha256']:
        raise ValueError('Cached force boundary changed')
    with np.load(contact_boundary, allow_pickle=False) as z:
        parents = z['exterior_tet']
        vectors = np.einsum('qa,qai->qi', z['exterior_barycentric'], direction[solid.T[parents]])
        rows['cached_contact_boundary_vertices'] = peak(vectors, z['vertices'], parents=parents)
    for key, values in (('dF_Frobenius', gradient_norm), ('symmetric_linear_strain_Frobenius', strain_norm)):
        i = int(values.argmax())
        rows[key] = dict(parent_tet=i, maximum=float(values[i]),
            carrier_centroid_mm=solid.X[solid.T[i]].mean(0).tolist(),
            occupied_fraction=float(solid.parent_physical_volume[i]/solid.carrier_volume[i]),
            physical_volume_mm3=float(solid.parent_physical_volume[i]),
            carrier_nodes=solid.T[i].tolist(), dF=dF[i].tolist())
    raw_accel = np.linalg.norm(contact_g, axis=1)/solid.mass
    j = int(raw_accel.argmax())
    report = dict(scope='ONE linear diagnostic with cached unit-pair self-contact gradient only; not a physical step',
        source_domain_sha256=solid.provenance['domain_sha256'], cached_force_sha256=sha(a.cached_force),
        cached_force_provenance=force_provenance['numerical_model'], diagnostic_dt_s=dt,
        activation=activation, physical_mass_kg=float(solid.mass.sum()*1000),
        cached_self_gradient_max_N=float(np.linalg.norm(contact_g, axis=1).max()),
        raw_self_force_over_mass_peak=dict(node=j, mm_per_s2=float(raw_accel[j]), reference_point_mm=solid.X[j].tolist()),
        correction_peaks=rows, linear_system=linear, energy_directional_slope_N_mm=float(g.ravel()@d),
        contact_tangent_included=False, exact_material_tangent_and_physical_inertia_included=True,
        physical_damping_per_second=0., artificial_mass_stiffness_pins_or_aggregation=False,
        native_geometry_still_intersecting=True, geometry_guard_bypass=False,
        no_geometry_or_nonlinear_acceptance_attempt=True, no_q_or_velocity_artifact_saved=True,
        physical_time_advanced_s=0., accepted_facial_frames=0, total_seconds=time.perf_counter()-started)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
