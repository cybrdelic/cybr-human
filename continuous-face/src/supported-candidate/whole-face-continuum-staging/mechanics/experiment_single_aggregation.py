"""Isolated one-node affine aggregation experiment. Never a motion producer."""
import json
from collections import deque
from pathlib import Path
import time
import numpy as np
from scipy.sparse import csr_matrix, diags, eye, kron, vstack
from source_transfer import load_geometry_with_source_fibres, sha
from clipped_origin_support import load_support_bundle
from implicit_dynamics import source_activation
from linear_equilibration import solve_kkt
from audit_cached_contact_direction import peak


def main():
    here = Path(__file__).resolve().parent
    out = here/'cut-pilot-repaired-v1-supports/single-node7550-aggregation-experiment.json'
    if out.exists():
        raise FileExistsError('The one authorized isolated experiment already has a report')
    started = time.perf_counter()
    root = here.parent/'geometry/cut-pilot/repaired-v1'
    graphdir = here.parent/'geometry/cut-pilot/contact-connectivity-audit'
    evidence = json.loads((graphdir/'connectivity-report.json').read_text())
    for name in ('cut-continuum.npz', 'cut-fine-tetrahedra.npz'):
        if sha(root/name) != evidence['source_sha256'][name]:
            raise ValueError('Connectivity witness is not for this immutable geometry')
    s = load_geometry_with_source_fibres(root/'cut-continuum.npz', root/'cut-muscle-phases.npz')
    supports, _ = load_support_bundle(s, here/'cut-pilot-repaired-v1-supports')
    graph = np.load(graphdir/'local-positive-face-graph.npz', allow_pickle=False)
    fine = np.load(root/'cut-fine-tetrahedra.npz', allow_pickle=False)
    fine_phi, fine_parent = fine['fine_node_phi'], fine['parent_tet']
    rows, nodeids = graph['fine_tet_rows'], graph['fine_node_ids']
    lookup = {int(row): i for i, row in enumerate(rows)}
    adjacency = {int(row): [] for row in rows}
    edges, areas = graph['graph_fine_tet_edges'], graph['graph_positive_material_area']
    for e, ((a, b), area) in enumerate(zip(edges, areas)):
        ia, ib = lookup[int(a)], lookup[int(b)]
        common = np.intersect1d(nodeids[ia], nodeids[ib])
        if len(common) != 3 or not np.array_equal(common, np.sort(graph['graph_shared_fine_node_ids'][e])) or area <= 0:
            raise ValueError('An alleged material graph edge is not a genuine positive-area shared face')
        phia = fine_phi[a][np.isin(nodeids[ia], common)]
        if np.max(phia) <= 0:
            raise ValueError('Shared face contains no positive material')
        adjacency[int(a)].append((int(b), float(area)))
        adjacency[int(b)].append((int(a), float(area)))
    slave, start = 7550, 63631
    if start not in lookup or slave not in s.T[int(fine_parent[start])]:
        raise ValueError('Hotspot fine-tet ownership mismatch')
    previous, distance, edgearea = {start: None}, {start: 0}, {}
    queue = deque([start])
    while queue:
        a = queue.popleft()
        for b, area in adjacency[a]:
            if b not in previous:
                previous[b], distance[b], edgearea[b] = a, distance[a]+1, area
                queue.append(b)
    candidates = []
    for row in rows:
        parent = int(fine_parent[row])
        nodes = s.T[parent]
        occupancy = s.parent_physical_volume[parent]/s.carrier_volume[parent]
        minmass = float(s.mass[nodes].min())
        if int(row) in previous and slave not in nodes and occupancy >= .5 and minmass >= 1e-9:
            bary123 = s.inv[parent]@(s.X[slave]-s.X[nodes[0]])
            bary = np.r_[1-bary123.sum(), bary123]
            candidates.append((distance[int(row)], float(np.abs(bary).sum()), -float(occupancy), int(row), parent, bary))
    if not candidates:
        out.write_text(json.dumps(dict(status='failed_no_genuinely_connected_healthy_root', physical_solve_run=False), indent=2)+'\n')
        print(out.read_text())
        return
    print('Witness checked and healthy connected root candidates found; no linear solve yet.', flush=True)
    _, _, _, end, parent, weights = min(candidates, key=lambda x: x[:5])
    rootnodes = s.T[parent]
    path, cursor = [], end
    while cursor is not None:
        path.append(cursor)
        cursor = previous[cursor]
    path.reverse()
    columns = np.r_[slave, rootnodes]
    coefficients = np.r_[1., -weights]
    row = csr_matrix((coefficients, (np.zeros(5, int), columns)), shape=(1, len(s.X)))
    reference_error = float(np.max(np.abs(row@s.X)))
    constant_error = float(abs(coefficients.sum()))
    affine_probe = s.X@np.array([[1.2, .3, -.1], [.2, .8, .1], [0., -.2, 1.1]])+np.array([5., -8., 3.])
    affine_error = float(np.max(np.abs(row@affine_probe)))
    projected, _ = supports.projected_residual(np.asarray(row.toarray()).T)
    independent_norm = float(np.linalg.norm(projected))
    if max(reference_error, constant_error, affine_error) > 1e-10 or independent_norm < 1e-8:
        raise ValueError('Affine reproduction or independent relative-constraint test failed')
    C = vstack([supports.C3, kron(row, eye(3), format='csr')], format='csr')
    force_path = here.parent/'contact/repaired-v1-initial-self-force.npz'
    with np.load(force_path, allow_pickle=False) as force:
        if str(force['source_domain_sha256']) != s.provenance['domain_sha256']:
            raise ValueError('Cached gradient identity mismatch')
        contact_g = force['self_gradient_N']
    dt = .001
    activation = source_activation(dt, case='zygomaticus_frontalis')
    _, material_g, _ = s.energy(s.X, activation)
    g = material_g+contact_g-s.mass[:, None]*np.array([0., -9810., 0.])
    K = s.tangent(s.X, activation)+diags(np.repeat(s.mass/dt**2, 3))
    print('Starting the single unchanged-equation KKT experiment.', flush=True)
    d, multipliers, linear = solve_kkt(K, -g.ravel(), C, equilibrate=True)
    direction = d.reshape(s.X.shape)
    dF = s.deformation(direction)
    star = np.unique(graph['carrier_parents'])
    peaks = {'global_carrier': peak(direction, s.X), 'slave_correction_mm': float(np.linalg.norm(direction[slave]))}
    with np.load(root/'cut-continuum.npz', allow_pickle=False) as domain:
        for name, parentkey, barykey, pointkey in (('boundary', 'exterior_tet', 'exterior_barycentric', 'vertices'),
                ('quadrature', 'quadrature_tet', 'quadrature_barycentric', 'quadrature_points')):
            parents = domain[parentkey]
            correction = np.einsum('qa,qai->qi', domain[barykey], direction[s.T[parents]])
            peaks['global_'+name] = peak(correction, domain[pointkey], parents=parents)
            ids = np.flatnonzero(np.isin(parents, star))
            value = peak(correction[ids], domain[pointkey][ids], parents=parents[ids])
            value['index'] = int(ids[value['index']])
            peaks['local_star_'+name] = value
    norm = np.linalg.norm(dF, axis=(1, 2))
    hot = int(norm.argmax())
    peaks['global_dF_Frobenius'] = dict(parent_tet=hot, maximum=float(norm[hot]), carrier_centroid_mm=s.X[s.T[hot]].mean(0).tolist())
    peaks['local_star_dF_Frobenius'] = float(norm[star].max())
    baseline = json.loads((here/'cut-pilot-repaired-v1-supports/cached-self-force-linear-direction.json').read_text())
    report = dict(status='one_isolated_relative_MPC_linear_experiment_completed', slave_node=slave,
        source_domain_sha256=s.provenance['domain_sha256'], cached_force_sha256=sha(force_path),
        connectivity_graph_sha256=sha(graphdir/'local-positive-face-graph.npz'),
        root_parent=parent, root_nodes=rootnodes.tolist(), root_weights=weights.tolist(),
        root_occupied_fraction=float(s.parent_physical_volume[parent]/s.carrier_volume[parent]),
        root_vertex_mass_N_s2_per_mm=s.mass[rootnodes].tolist(),
        root_selection='Shortest positive-material-face path among local roots >=50% occupied with all nodal integrated basis volumes >=1L0^3; then smallest extrapolation L1 norm, then greatest occupancy.',
        positive_area_fine_tet_path=path, path_carrier_parents=fine_parent[path].tolist(),
        path_bottleneck_area_L0_2=float(min(edgearea[x] for x in path[1:])),
        constant_reproduction_error=constant_error, affine_reproduction_error=affine_error,
        reference_coordinate_change_mm=0., reference_constraint_error_mm=reference_error,
        additional_scalar_constraint_rank=1, additional_coordinate_constraint_rank=3,
        relative_row_independence_norm=independent_norm, approximation_constraint_is_not_an_anatomical_pin=True,
        physical_mass_kg=float(s.mass.sum()*1000), original_mass_force_and_tangent_unchanged=True,
        original_source_support_rows_unchanged=True, linear_system=linear, correction_peaks=peaks,
        baseline_local_boundary_correction_mm=baseline['correction_peaks']['finite_boundary_vertices']['maximum_mm'],
        baseline_local_quadrature_correction_mm=baseline['correction_peaks']['integrated_quadrature_centroids']['maximum_mm'],
        baseline_local_dF_Frobenius=baseline['correction_peaks']['dF_Frobenius']['maximum'],
        aggregation_constraint_multiplier_N=multipliers[-3:].tolist(), aggregation_reaction_is_internal_kinematic_not_bone_support=True,
        contact_tangent_included=False, physical_time_advanced_s=0., accepted_facial_frames=0,
        nonlinear_solve_run=False, guard_bypass=False, geometry_unchanged=True,
        no_q_or_velocity_artifact_saved=True, broader_aggregation_implemented=False,
        total_seconds=time.perf_counter()-started)
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
