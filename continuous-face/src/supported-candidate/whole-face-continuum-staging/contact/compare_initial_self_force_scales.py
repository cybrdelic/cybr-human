"""Read-only saved-force/physical-mass comparison; no contact or motion solve.

Run with Python -B. Only the two named analysis outputs in this directory are
written. Existing raw domain, phase, boundary, diagnostic and mechanics remain
unchanged. Force attribution to individual pairs cannot be reconstructed from
the saved aggregate gradient and is deliberately not attempted.
"""
from pathlib import Path
import hashlib
import json
import sys

sys.dont_write_bytecode = True
import numpy as np
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
sys.path.insert(0, str(STAGE / "mechanics"))
from source_transfer import load_geometry_with_source_fibres
from continuum import PHASE_NAMES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    a = np.asarray(values)
    return {"count": int(a.size), "min": float(a.min()),
            "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def main():
    paths = dict(
        raw_domain=STAGE / "geometry/cut-pilot/repaired-v1/cut-continuum.npz",
        muscle_phases=STAGE / "geometry/cut-pilot/repaired-v1/cut-muscle-phases.npz",
        canonical_boundary=HERE / "repaired-v1-dyadic-boundary.npz",
        saved_force=HERE / "repaired-v1-initial-self-force.npz",
        saved_force_report=HERE / "repaired-v1-initial-self-force.json",
        mechanics=STAGE / "mechanics/continuum.py",
        loading_interface=STAGE / "mechanics/source_transfer.py",
        dynamics=STAGE / "mechanics/implicit_dynamics.py",
        mass_audit=STAGE / "mechanics/cut-pilot-repaired-v1-supports/domain-mechanics-audit.json",
        native_audit=STAGE / "geometry/cut-pilot/repaired-v1/cut-native-audit.json")
    before = {key: sha(path) for key, path in paths.items()}
    assert before["raw_domain"] == "47049ef9ccb4106d8170948809f33dfb222271daa04bc576afc48ed3c3d053cd"
    diagnostic = json.loads(paths["saved_force_report"].read_text())
    audit = json.loads(paths["mass_audit"].read_text())
    native = json.loads(paths["native_audit"].read_text())
    solid = load_geometry_with_source_fibres(paths["raw_domain"], paths["muscle_phases"])
    with np.load(paths["saved_force"], allow_pickle=False) as z:
        gradient = z["self_gradient_N"].copy()
        assert str(z["source_domain_sha256"]) == before["raw_domain"]
        assert not bool(z["accepted_motion"])
    assert gradient.shape == solid.X.shape and np.isfinite(gradient).all()
    assert diagnostic["source_sha256"] == before["raw_domain"]
    assert diagnostic["boundary_sha256"] == before["canonical_boundary"]
    force = -gradient
    norms = np.linalg.norm(force, axis=1)
    mass = solid.mass
    gravity = np.array([0., -9810., 0.])
    gravity_norm = float(np.linalg.norm(gravity))
    gravity_force = mass * gravity_norm
    acceleration = force / mass[:, None]
    acceleration_norm = norms / mass
    ratios = norms / gravity_force
    affected = norms > 0
    density = np.array([solid.materials[p].density_N_s2_mm4 for p in PHASE_NAMES])
    site_mass = solid.weights * (solid.fractions @ density)
    parent_mass_basis = np.zeros((len(solid.T), 4))
    np.add.at(parent_mass_basis, solid.parent, site_mass[:, None] * solid.bary)
    parent_mass = parent_mass_basis.sum(1)
    volume = solid.parent_physical_volume
    incident = [[] for _ in solid.X]
    for parent, nodes in enumerate(solid.T):
        for slot, node in enumerate(nodes):
            if parent_mass_basis[parent, slot] > 0:
                incident[int(node)].append((parent, slot))
    max_support = np.array([max(volume[p] for p, _ in row) for row in incident])
    with np.load(paths["raw_domain"], allow_pickle=False) as z:
        raw_basis = z["node_integrated_basis_weight"].copy()
        raw_weights = z["quadrature_weights"].copy()
        raw_fraction = z["quadrature_material_fraction"].copy()
    with np.load(paths["canonical_boundary"], allow_pickle=False) as z:
        vertices, faces, parent, bary = [z[k].copy() for k in
            ("vertices", "triangles", "exterior_tet", "exterior_barycentric")]
        assert str(z["source_domain_sha256"]) == before["raw_domain"]
    carrier = csr_matrix((bary.ravel(), (np.repeat(np.arange(len(vertices)), 4),
        solid.T[parent].ravel())), shape=(len(vertices), len(solid.X)))
    carrier.eliminate_zeros()
    assert np.array_equal(carrier @ solid.X, vertices)
    areas = np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                   vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1) / 2

    def node_row(node, with_supports=True):
        node = int(node)
        support = sorted(incident[node], key=lambda pair: volume[pair[0]])
        result = dict(node=node, reference_mm=solid.X[node].tolist(),
            self_force_N=force[node].tolist(), self_force_norm_N=float(norms[node]),
            lumped_mass_N_s2_per_mm=float(mass[node]), lumped_mass_kg=float(1000 * mass[node]),
            own_gravity_force_N=(mass[node] * gravity).tolist(),
            own_gravity_force_norm_N=float(gravity_force[node]),
            force_to_own_gravity_ratio=float(ratios[node]),
            formal_contact_only_lumped_acceleration_mm_s2=acceleration[node].tolist(),
            formal_contact_only_lumped_acceleration_norm_mm_s2=float(acceleration_norm[node]),
            formal_contact_only_lumped_acceleration_norm_m_s2=float(acceleration_norm[node] / 1000),
            acceleration_to_gravity_ratio=float(acceleration_norm[node] / gravity_norm),
            positive_mass_parent_supports=len(support),
            incident_occupied_volume_sum_mm3=float(sum(volume[p] for p, _ in support)),
            minimum_parent_occupied_volume_mm3=float(volume[support[0][0]]),
            maximum_parent_occupied_volume_mm3=float(volume[support[-1][0]]),
            effective_basis_volume_mm3=float(mass[node] / density[0]))
        if with_supports:
            result["mass_supports_ascending_volume"] = [dict(
                parent_tet=int(p), occupied_volume_mm3=float(volume[p]),
                carrier_volume_mm3=float(solid.carrier_volume[p]),
                occupancy_fraction=float(volume[p] / solid.carrier_volume[p]),
                total_parent_mass_N_s2_per_mm=float(parent_mass[p]),
                contribution_to_this_node_mass_N_s2_per_mm=float(parent_mass_basis[p, slot]),
                fraction_of_this_node_mass=float(parent_mass_basis[p, slot] / mass[node]),
                volume_averaged_parent_basis_at_node=float(parent_mass_basis[p, slot] / parent_mass[p]))
                for p, slot in support]
        return result

    top_force = np.argsort(norms)[-20:][::-1]
    top_ratio = np.argsort(ratios)[-20:][::-1]
    above_gravity = np.flatnonzero(ratios > 1)
    above_gravity = above_gravity[np.argsort(ratios[above_gravity])[::-1]]
    selected = sorted(set(top_force) | set(top_ratio) | set(above_gravity))
    parent_affected = np.any(affected[solid.T] & (parent_mass_basis > 0), axis=1)
    smallest = np.flatnonzero(parent_affected)
    smallest = smallest[np.argsort(volume[smallest])[:20]]
    support_strata = []
    for threshold in (0., 1e-6, 1e-4, .01, .36, 1., 3.6, 10.):
        ids = np.flatnonzero(affected & (max_support >= threshold))
        index = int(ids[np.argmax(ratios[ids])])
        support_strata.append(dict(minimum_largest_parent_volume_mm3=threshold,
            affected_nodes=int(len(ids)), nodes_force_exceeds_own_gravity=int(np.sum(ratios[ids] > 1)),
            maximum_ratio_node=index, maximum_force_to_own_gravity_ratio=float(ratios[index])))
        if index not in selected:
            selected.append(index)
    energy_hotspots = []
    for hot in diagnostic["hot_pairs"]:
        entry = dict(hot)
        entry["reference_facet_areas_mm2"] = areas[hot["faces"]].tolist()
        entry["parent_occupied_volumes_mm3"] = volume[hot["parent_tets"]].tolist()
        entry["parent_nodes"] = solid.T[hot["parent_tets"]].tolist()
        energy_hotspots.append(entry)
    all_input_unchanged = all(sha(path) == before[key] for key, path in paths.items())
    assert all_input_unchanged
    assert float(norms.max()) == diagnostic["self_max_node_force_N"]
    assert float(mass.sum()) == audit["mass_N_s2_per_mm"]
    assert np.allclose(mass, raw_basis * density[0], rtol=1e-12, atol=0)
    report = dict(
        status="saved_initial_self_force_compared_to_unchanged_physical_mass",
        actual_facial_frames=0, accepted_motion=False, force_recomputed=False,
        solver_invoked=False, new_contact_law_implemented=False,
        inputs={key: dict(path=str(path), sha256=before[key]) for key, path in paths.items()},
        checks=dict(all_input_artifacts_unchanged=all_input_unchanged,
            canonical_boundary_reconstructs_bitwise=True, saved_force_max_matches=True,
            loader_mass_matches_existing_audit=True,
            max_absolute_mass_difference_from_stored_integrated_basis_N_s2_per_mm=float(np.max(np.abs(mass - raw_basis * density[0]))),
            max_relative_mass_difference_from_stored_integrated_basis=float(np.max(np.abs(mass / (raw_basis * density[0]) - 1))),
            raw_volume_mm3=float(raw_weights.sum()), split_loader_volume_mm3=float(solid.weights.sum()),
            raw_phase_volume_mm3=dict(zip(PHASE_NAMES, map(float, raw_weights @ raw_fraction)))),
        physical_conventions=dict(length_scale="1 source L0 = 1 mm, explicit engineering assumption, not patient calibration",
            mass_unit="N s^2/mm; multiply by 1000 for kg", gravity_mm_s2=gravity.tolist(),
            uniform_density_N_s2_mm4=float(density[0]), uniform_density_kg_m3=1000.,
            mass_rule="exact existing load_geometry_with_source_fibres -> CutContinuum._mass_matrices: sum_q w_q rho_q N_i(X_q)",
            force_rule="negative of saved self_gradient_N, no re-evaluation",
            ratio_rule="norm(F_self_i)/(m_i norm(gravity)) = norm(F_self_i/m_i)/norm(gravity)",
            acceleration_interpretation="Formal contact-only diagonal lumped F/m diagnostic, not the constrained coupled continuum acceleration or an accepted trajectory.",
            self_law=diagnostic["numerical_model"]),
        totals=dict(physical_nodes=len(solid.X), parent_tets=len(solid.T), affected_nodes=int(affected.sum()),
            unaffected_nodes=int((~affected).sum()), mass_kg=float(mass.sum() * 1000),
            gravity_force_norm_N=float(gravity_force.sum()), self_force_node_norm_sum_N=float(norms.sum()),
            net_self_force_N=force.sum(0).tolist(), self_energy_N_mm=diagnostic["self_energy_N_mm"],
            active_separated_pairs=diagnostic["active_separated_pairs"],
            nodes_exceeding_gravity_ratio={str(k): int(np.sum(ratios > k)) for k in (1, 10, 100, 1000, 1e6, 1e9, 1e10)},
            maximum_self_force_node=int(np.argmax(norms)), maximum_force_to_gravity_node=int(np.argmax(ratios)),
            maximum_force_to_gravity_ratio=float(ratios.max()),
            maximum_formal_acceleration_mm_s2=float(acceleration_norm.max())),
        distributions=dict(nodal_mass_N_s2_per_mm=stats(mass), force_norm_N_all_nodes=stats(norms),
            force_to_gravity_ratio_all_nodes=stats(ratios), force_to_gravity_ratio_affected_nodes=stats(ratios[affected]),
            parent_occupied_volume_mm3=stats(volume), canonical_reference_facet_area_mm2=stats(areas)),
        top_force_nodes=top_force.tolist(), top_force_to_gravity_nodes=top_ratio.tolist(),
        all_nodes_above_own_gravity=above_gravity.tolist(),
        node_details={str(i): node_row(i) for i in sorted(selected)},
        support_strata=dict(note="Descriptive reporting bins only, not geometry thresholds, exclusions, mass floors, or new support rules. A node may share a tiny parent while its other parents carry substantial mass.", rows=support_strata),
        smallest_occupied_parent_supports_affecting_nonzero_force_nodes=[dict(
            parent_tet=int(p), occupied_volume_mm3=float(volume[p]),
            carrier_volume_mm3=float(solid.carrier_volume[p]), occupancy_fraction=float(volume[p] / solid.carrier_volume[p]),
            nodes=[dict(**node_row(n, False), mass_contribution_from_this_parent_N_s2_per_mm=float(parent_mass_basis[p, slot]))
                   for slot, n in enumerate(solid.T[p])]) for p in smallest],
        saved_top_energy_pairs_with_area_and_volume=energy_hotspots,
        attribution_limit="Saved NPZ contains only the aggregate nodal gradient, not every active pair or its force. Parent rows prove mass support and saved hot pairs prove their own geometry; they do not attribute the full nodal contact force to a particular parent or pair.",
        conclusion="Meaningful severe weak-cut force/mass imbalance is present. The largest absolute-force nodes mostly have substantial mass and subgravity contact loading; the worst normalized hotspot is instead the minimum-mass node. This warrants a declared continuum contact quadrature investigation, not a physical mass floor or contact exclusions. F/m alone does not predict the constrained implicit step.",
        proposed_reference_area_alternative=dict(status="proposal only, not implemented or calibrated",
            formulation="For each physical opposing patch pair A,B: E_AB=(k_A/2)[sum_(r in A) a_r^0 b(d(x_r(q),Gamma_B(q));h) + sum_(s in B) a_s^0 b(d(x_s(q),Gamma_A(q));h)], with b=-(d-h)^2 log(d/h) for 0<d<h. Each distance is to the full eligible finite opposing surface, not an infinite plane or nearest carrier node.",
            units="b: mm^2; fixed reference quadrature area a^0: mm^2; new coefficient k_A: N/mm^3; E: N mm. The existing k=0.001 N/mm must not silently be reused as k_A. Relating them requires an explicitly declared reference area A_ref in mm^2, with k_A=k/A_ref, not an inferred calibration.",
            physical_choice="Use positive reference triangle quadrature weights summing to each triangle's area, or a vertex's one-third sum of incident reference facet areas, fixed in the reference state. Evaluate both directions with the one-half factor to avoid master/slave bias. Define physical opposing patches and same-material local eligibility explicitly; do not make the measure depend on the number of facets returned by broadphase.",
            geometry_independence="Changing energetic quadrature is not permission to remove any geometric contact. Preserve the exhaustive eligible finite-triangle endpoint and whole-segment guards independently of the quadrature samples, including features missed by quadrature and all currently required pairs. No pair exclusions or guard relaxation are proposed.",
            required_before_use=["Choose and disclose the physical surface measure, interaction-area construction, reference area and k_A independently of a desired trajectory or the hotspot F/m.",
                "Validate refinement and triangulation/subdivision invariance, including multiplicity and near shared-feature limits, with conservation and energy-gradient checks.",
                "Preserve finite-facet endpoint and whole-segment collision guards, unchanged h=0.02 mm, all physical material, and every required pair; no mass/volume cutoffs or artificial barriers on true same material.",
                "Re-measure force/mass scales under the declared new law. Area weighting alone is not shown to cure tiny basis masses or local-adjacency limitations.",
                "Keep actual motion blocked until the native/reference geometry certificates are satisfied."]),
        motion_blockers=dict(source_state_certified=False, reference_embedding_certified=False,
            native_boundary_faces_with_crossings=native["native_finite_triangle_intersections"]["boundary_faces_with_crossings"],
            saved_self_diagnostic_intersecting_or_unresolved_pairs=diagnostic["intersecting_or_unresolved_pairs"],
            local_partition_assumption=diagnostic["conditional_assumption"],
            note="Zero remaining self-diagnostic intersections is conditional and does not remove the independent native crossing blocker."))
    output = HERE / "repaired-v1-self-force-scales.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    highest = node_row(int(np.argmax(ratios)), False)
    largest = node_row(int(np.argmax(norms)), False)
    text = f"""Saved initial self-contact force scales, repaired-v1

No force recomputation, solve, accepted motion, input edit, or contact-law change.

The physical loader uses 1 L0 = 1 mm, rho = 1e-9 N s^2/mm^4 (1000 kg/m^3),
row-sum physical quadrature mass, and gravity (0,-9810,0) mm/s^2.
Total tissue mass is {mass.sum()*1000:.15g} kg. Saved self energy is
{diagnostic['self_energy_N_mm']:.15g} N mm across {diagnostic['active_separated_pairs']} active separated pairs.

Severe normalized hotspot: node {highest['node']} at {highest['reference_mm']} mm.
Force {highest['self_force_norm_N']:.10g} N; mass {highest['lumped_mass_N_s2_per_mm']:.10g} N s^2/mm
({highest['lumped_mass_kg']:.10g} kg); own gravity {highest['own_gravity_force_norm_N']:.10g} N.
Force/gravity = {highest['force_to_own_gravity_ratio']:.10g}; formal contact-only F/m =
{highest['formal_contact_only_lumped_acceleration_norm_mm_s2']:.10g} mm/s^2.
All six contributing parents are tiny: total occupied volume {highest['incident_occupied_volume_sum_mm3']:.10g} mm^3,
range {highest['minimum_parent_occupied_volume_mm3']:.10g} to {highest['maximum_parent_occupied_volume_mm3']:.10g} mm^3.

The largest absolute-force node, {largest['node']} at {largest['reference_mm']} mm, is not this pathology:
force {largest['self_force_norm_N']:.10g} N, mass {largest['lumped_mass_kg']:.10g} kg,
force/gravity {largest['force_to_own_gravity_ratio']:.10g}, and formal F/m {largest['formal_contact_only_lumped_acceleration_norm_mm_s2']:.10g} mm/s^2.
Its parents contain {largest['incident_occupied_volume_sum_mm3']:.10g} mm^3 in total, including
a largest occupied support of {largest['maximum_parent_occupied_volume_mm3']:.10g} mm^3.

Across all {len(mass)} nodes, {int(affected.sum())} have a nonzero saved force. Ratios exceed
1 at {int(np.sum(ratios>1))} nodes, 10 at {int(np.sum(ratios>10))}, 100 at {int(np.sum(ratios>100))}, and 1000 at {int(np.sum(ratios>1000))}.
Among affected nodes with any occupied parent at least 3.6 mm^3 (10% of the 36 mm^3 carrier),
none exceeds own gravity; the largest ratio is {support_strata[-2]['maximum_force_to_own_gravity_ratio']:.10g}.
These bins are descriptive only. Tiny-parent membership alone is not a weak-node diagnosis:
parent 27582 (7.247728419e-10 mm^3) also contains node 7079, whose other supports give
mass 8.695134602e-5 kg and force/gravity only 0.002789049723.

The highest saved pair energy involves facets of areas {energy_hotspots[0]['reference_facet_areas_mm2']}
mm^2 in parents with occupied volumes {energy_hotspots[0]['parent_occupied_volumes_mm3']} mm^3.
Tiny facet area and tiny total nodal mass are distinct phenomena. Aggregate saved gradients
do not identify every pair's contribution to a node.

Conclusion: unit finite-facet pair weights have a meaningful weak-cut force/mass imbalance.
F/m is a diagnostic, not the constrained coupled continuum acceleration or a simulated trajectory.
A candidate alternative is a declared reciprocal reference-area quadrature over each
physical opposing patch pair: half the area-weighted barrier sum from A to the closest
finite surface B, plus half the reciprocal sum from B to the closest finite surface A.
Use fixed positive triangle or lumped-vertex reference weights in mm^2 and a new
coefficient k_A in N/mm^3, since b has units mm^2. Reusing the old 0.001 N/mm as
k_A would be dimensionally wrong. Reference area must count physical patches rather
than raw neighbor multiplicity; the coefficient, patch eligibility and measure require
explicit declaration and refinement/conservation validation. Retain exhaustive finite
triangle geometric guards separately, including features missed by quadrature samples.
No area-weighted law was implemented, and area weighting alone is not proven to cure tiny masses.
Do not inflate mass, remove material/contacts, weaken clearance, or loosen endpoint/path guards.

Actual motion remains blocked by {native['native_finite_triangle_intersections']['boundary_faces_with_crossings']} native crossing boundary faces and incomplete
reference/native certification. Zero unresolved self pairs in the saved conditional diagnostic
does not remove those blockers.

Full node/support evidence and input hashes: {output.name}
"""
    (HERE / "repaired-v1-self-force-scales.txt").write_text(text)
    print(json.dumps(dict(output=str(output), max_ratio_node=highest,
        max_force_node=largest, all_inputs_unchanged=all_input_unchanged), indent=2))


if __name__ == "__main__":
    main()
