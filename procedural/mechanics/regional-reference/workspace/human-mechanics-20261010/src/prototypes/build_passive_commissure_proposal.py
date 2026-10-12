"""Author passive oral fibre families on the unchanged connected native domain.

This builds a small proposal, not a domain or solved state. All paths and widths
are authored chart controls; none is a measured anatomical length. No geometry,
attachment, activation, dense transfer, or existing source file is changed.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import hashlib
import json
import sys

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from build_bilateral_mechanics_case import boundary, surface_audit


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unit(x):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("Degenerate proposed fibre direction")
    return x / norm


def signed_angle(a, center):
    return (np.asarray(a) - center + 240) % 480 - 240


def build(out):
    domain_path = ROOT / "output/bilateral-mechanics-v1/domain.json"
    geometry_path = ROOT / "output/bilateral-midface-bone-aware-v2/rest_geometry.npz"
    charts_path = ROOT / "inputs/native_oral_charts_v16.npz"
    d = json.loads(domain_path.read_text())
    a = np.load(geometry_path, allow_pickle=False)
    oral = np.load(charts_path, allow_pickle=False)
    v, t = a["vertices"], a["tetrahedra"]
    params, tags = a["cell_parameters"], a["chart_tags"]
    tet_tags = tags[params[:, 0]]
    grid = a["oral_node_grid"]
    fixed = np.zeros(len(v), dtype=bool)
    fixed[d["fixed_nodes"]] = True
    data = np.memmap(domain_path.parent / d["binary"], mode="r", dtype=np.uint8)
    q = d["arrays"]["rest"]
    rest = np.ndarray(q["shape"], dtype=q["dtype"], buffer=data, offset=q["byte_offset"])
    q = d["arrays"]["nodes"]
    nodes = np.ndarray(q["shape"], dtype=q["dtype"], buffer=data, offset=q["byte_offset"])
    if not np.array_equal(rest, v * .001) or not np.array_equal(nodes, t):
        raise ValueError("Case does not have the exact proposed rest and topology")
    if d["source_geometry_sha256"] != sha(geometry_path):
        raise ValueError("Case geometry provenance differs")

    # Actual tetrahedron-face adjacency, not proximity ties or a screen corridor.
    faces = np.sort(t[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]], axis=2)
    uf, inv, count = np.unique(faces.reshape(-1, 3), axis=0, return_inverse=True, return_counts=True)
    face_owners = defaultdict(list)
    for flat, face_id in enumerate(inv):
        if count[face_id] == 2:
            face_owners[int(face_id)].append(flat // 4)
    internal_ids = np.array(list(face_owners))
    pairs = np.array(list(face_owners.values()))
    all_active = np.concatenate([m["tetrahedra"] for m in d["muscles"]])
    oral_mask = tet_tags[:, 0] == 5
    if np.any(oral_mask[all_active]):
        raise ValueError("Source active body overlaps oral tets")

    regions = []
    summaries = []
    interface_records = []

    def add_region(name, ids, directions, fractions, role):
        ids = np.asarray(ids, dtype=int)
        directions = unit(np.asarray(directions))
        fractions = np.broadcast_to(fractions, len(ids)).copy()
        if not len(ids) or np.any(fractions <= 0) or np.any(fractions > 1):
            raise ValueError("Invalid proposed mixture fraction")
        if np.intersect1d(ids, all_active).size:
            raise ValueError("Proposal cannot change incoming body energy")
        member = np.zeros(len(t), dtype=bool)
        member[ids] = True
        edges = pairs[member[pairs].all(1)]
        local = np.full(len(t), -1)
        local[ids] = np.arange(len(ids))
        graph = sparse.coo_matrix((np.ones(2 * len(edges)),
                                  (local[np.r_[edges[:, 0], edges[:, 1]]],
                                   local[np.r_[edges[:, 1], edges[:, 0]]])),
                                 shape=(len(ids), len(ids))).tocsr()
        ncomp = connected_components(graph, directed=False, return_labels=False)
        surf = surface_audit(boundary(t[ids]))
        if ncomp != 1 or surf["open_or_nonmanifold_edges"] or surf["nonmanifold_vertex_count"] or surf["inconsistent_winding_edges"]:
            raise ValueError("Proposed finite region is not connected/manifold: " + name)
        regions.append({"id": name, "tetrahedra": ids.tolist(),
                        "fibreDirections": np.round(directions, 10).tolist(),
                        "volumeFractions": np.round(fractions, 10).tolist(),
                        "activation": 0., "stiffness": 24000., "scope": role})
        summaries.append({"id": name, "tetrahedra": len(ids),
                          "face_connected_components": int(ncomp), "boundary": surf,
                          "touching_fixed_node_tets": int(fixed[t[ids]].any(1).sum()),
                          "fully_fixed_tets": int(fixed[t[ids]].all(1).sum()),
                          "depth_bands": np.unique(params[ids, 1]).tolist(),
                          "fraction_range": [float(fractions.min()), float(fractions.max())]})

    # Fixed-depth chart derivatives use only existing FEM node positions. The
    # coarse angular sector contains 4 native samples, and each radial cell 4
    # native rows. All tetrahedra in one hexahedral/wedge cell get the same line
    # field. Sign is immaterial for the tension-only strain energy.
    candidate = np.flatnonzero(oral_mask & np.isin(params[:, 1], [1, 2]))
    ia, ir, ik = tet_tags[candidate, 1], tet_tags[candidate, 2], params[candidate, 1]
    radial = np.zeros((len(candidate), 3))
    angular = np.zeros_like(radial)
    for j, (i, r, k) in enumerate(zip(ia, ir, ik)):
        p = v[grid[np.array([i, (i + 1) % 120])[:, None, None],
                        np.array([r, r + 1])[None, :, None],
                        np.array([k, k + 1])[None, None, :]]]
        radial[j] = (p[:, 1].mean((0, 1)) - p[:, 0].mean((0, 1))) / 4
        angular[j] = (p[1].mean((0, 1)) - p[0].mean((0, 1))) / 4
    add_region("passive-circumoral-chart-family", candidate, angular, .25,
               "Authored superficial/intermediate circumoral connective family; zero activation; not a complete orbicularis anatomy")

    landmarks = []
    for side, body in zip([1, -1], d["muscles"]):
        side_name = "right" if side == 1 else "left"
        corner = int(np.argmax(side * oral["external_positions"][0, :, 0]))
        bm = np.zeros(len(t), dtype=bool)
        bm[body["tetrahedra"]] = True
        crosses = ((bm[pairs[:, 0]] & oral_mask[pairs[:, 1]]) |
                   (bm[pairs[:, 1]] & oral_mask[pairs[:, 0]]))
        shared_faces = uf[internal_ids[crosses]]
        shared_nodes = np.unique(shared_faces)
        if len(shared_nodes) != 44:
            raise ValueError("Incoming body footprint changed; re-author rather than silently rebind")
        angles = np.array([i * 4 for i in range(120) if np.isin(grid[i, -1], shared_nodes).any()])
        delta = signed_angle(angles, corner)
        low, high = float(delta.min()), float(delta.max())
        rmid = 4 * ir + 2
        amid = 4 * ia + 2
        rel = signed_angle(amid, corner)
        width = .4 + .6 * rmid / 16
        # Converging entry fan retains 40% of the actual incoming angular
        # footprint at the wetline, avoiding a point insertion/singularity.
        zeta = np.where(rel < 0, rel / low, rel / high) / width
        selected = zeta < 1
        # Compact transverse taper; no displacement taper exists anywhere.
        frac = .50 * (1 - zeta[selected] ** 2) ** 2
        dalpha_dr = .6 * rel[selected] / (16 * width[selected])
        fan_direction = radial[selected] + dalpha_dr[:, None] * angular[selected]
        name = "passive-" + side_name + "-commissure-fan"
        add_region(name, candidate[selected], fan_direction, frac,
                   "Authored passive distributed insertion fan from actual incoming shared-face footprint toward the true native commissure; no new active continuation")
        fan_mask = np.zeros(len(t), dtype=bool)
        fan_mask[candidate[selected]] = True
        direct = ((bm[pairs[:, 0]] & fan_mask[pairs[:, 1]]) |
                  (bm[pairs[:, 1]] & fan_mask[pairs[:, 0]]))
        direct_faces = uf[internal_ids[direct]]
        if not len(direct_faces):
            raise ValueError("Fan does not touch its incoming body by full faces")
        free_faces = direct_faces[~fixed[direct_faces].any(1)]
        if not len(free_faces):
            raise ValueError("Fan has no entirely free shared-face entrance")
        interface_records.append({"side": side_name, "incoming_body_id": body["id"],
                                  "all_incoming_oral_faces": shared_faces.tolist(),
                                  "fan_entry_faces": direct_faces.tolist(),
                                  "fully_free_fan_entry_faces": free_faces.tolist(),
                                  "incoming_shared_nodes": len(shared_nodes),
                                  "incoming_shared_fixed_nodes": int(fixed[shared_nodes].sum()),
                                  "source_native_angle_range_relative_to_corner": [low, high],
                                  "source_native_angles": angles.tolist()})
        lo = corner // 4
        hi = (lo + 1) % 120
        alpha = (corner % 4) / 4
        native_point = oral["external_positions"][0, corner]
        chord = (1 - alpha) * v[grid[lo, 0, 0]] + alpha * v[grid[hi, 0, 0]]
        landmarks.append({"side": side_name, "native_angle": corner,
                          "native_vertex_id": int(oral["external_vertex_ids"][0, corner]),
                          "native_commissure_design_units": native_point.tolist(),
                          "native_row16_exterior_design_units": oral["external_positions"][16, corner].tolist(),
                          "fem_wetline_bracket_nodes": [int(grid[lo, 0, 0]), int(grid[hi, 0, 0])],
                          "bracket_weights": [1 - alpha, alpha],
                          "native_minus_bracket_design_units": (native_point - chord).tolist(),
                          "bracket_error_design_units": float(np.linalg.norm(native_point - chord)),
                          "note": "Exact dense native landmark; not a force target or a new FEM node. Preserve the existing exact native transfer remainder."})

    occupancy = np.zeros(len(t))
    for region in regions:
        occupancy[region["tetrahedra"]] += region["volumeFractions"]
    if occupancy.max() > 1:
        raise ValueError("Overfilled fibre mixture")
    arrays = {}
    for i, region in enumerate(regions):
        arrays[f"family_{i}_tetrahedra"] = np.array(region["tetrahedra"], dtype=np.uint32)
        arrays[f"family_{i}_directions"] = np.array(region["fibreDirections"])
        arrays[f"family_{i}_fractions"] = np.array(region["volumeFractions"])
    metadata = {
        "schema": "native-passive-commissure-proposal-v1",
        "status": "Geometry-compatible authored material proposal only; no solve, transfer, render or motion claim",
        "base_domain_sha256": sha(domain_path),
        "base_arrays_sha256": sha(domain_path.parent / d["binary"]),
        "source_geometry_sha256": sha(geometry_path), "source_chart_sha256": sha(charts_path),
        "incoming_muscle_records_sha256": hashlib.sha256(json.dumps(d["muscles"], sort_keys=True).encode()).hexdigest(),
        "identity": {"nodes": len(v), "tetrahedra": len(t), "rest_unchanged": True, "topology_unchanged": True},
        "implementation": "Append these families to unchanged incoming records and use existing InterdigitatedMuscleFEM. The passive matrix is counted once. Keep activation=0 for every new family. Never call global setMuscleActivation with a nonzero value.",
        "controls": {"depth_bands": [1, 2], "circumoral_fraction": .25,
                     "peak_insertion_fraction": .50, "wetline_fan_width_fraction_of_entry": .4,
                     "family_stiffness_solver_units": 24000., "stiffness_sensitivity_multipliers": [.5, 1, 2],
                     "note": "All material, mixture, depth and fan controls are authored, uncalibrated; 0.001 solver length per design unit is not an anatomical millimetre."},
        "landmarks": landmarks, "interfaces": interface_records, "regions": summaries,
        "maximum_passive_fibre_mixture_fraction": float(occupancy.max()),
        "oral_fixed_grid_counts_by_radial_row_and_depth": fixed[grid].sum(0).tolist(),
        "oral_fully_fixed_tetrahedra": int(fixed[t[oral_mask]].all(1).sum()),
        "ablations": [
            {"id": "C0", "families": [], "purpose": "Unchanged incoming body control at identical origin support, materials and activation"},
            {"id": "C1", "families": ["passive-right-commissure-fan", "passive-left-commissure-fan"], "purpose": "Passive distributed insertion alone"},
            {"id": "C2", "families": ["passive-circumoral-chart-family"], "purpose": "Passive circumoral transmission alone"},
            {"id": "C3", "families": [r["id"] for r in regions], "purpose": "Interdigitated combination at unchanged cheek activation; no oral activation"},
            {"id": "C4", "families": [r["id"] for r in regions], "right_incoming_activation": 0., "purpose": "All-zero rest and unload check; no hidden local active force"},
            {"id": "BC", "purpose": "Repeat C0/C3 only after legitimate native cavity-return/transfer extension relocates the deep cutoff. The present 360-node oral collar is a separate causal confound; do not silently remove pins or hide the seam with transfer taper."}],
        "required_future_checks": ["Same active-region strain/energy and source activation commands across C0-C3", "Same origin supports across C0-C3; support expansion is an independent factor", "Convergence, positive J, complete static-bone path feasibility, native transfer and cavity continuity", "Exact native commissure plus upper/lower lip marker displacements and area/normal checks", "Passive family energies/forces and oral fixed-collar reactions", "No promise that passive reinforcement amplifies motion; it may redistribute or suppress it", "Lip/tissue self-contact remains a separate unverified requirement"],
        "references": [
            {"url": "https://pubmed.ncbi.nlm.nih.gov/16547604/", "use": "Cadaver study motivates a superficial functional association of perioral muscles and outer orbicularis. No specimen dimensions or geometry imported."},
            {"url": "https://arxiv.org/abs/0803.3924", "use": "Supports mechanically embedded insertion modelling and the importance of anatomical description/contact; does not validate this authored face."}]
    }
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "passive_fibre_arrays.npz", **arrays)
    (out / "regions.json").write_text(json.dumps({"muscles": regions}, separators=(",", ":")) + "\n")
    (out / "proposal.json").write_text(json.dumps(metadata, indent=2) + "\n")
    size = sum(p.stat().st_size for p in out.iterdir())
    if size > 1024 ** 2:
        raise ValueError("Proposal exceeds 1 MiB budget")
    print(json.dumps({"output": str(out), "bytes": size, "regions": summaries,
                      "interfaces": [{k: x[k] for k in ["side", "incoming_shared_nodes", "incoming_shared_fixed_nodes"]} |
                                     {"fan_entry_faces": len(x["fan_entry_faces"]), "fully_free_fan_entry_faces": len(x["fully_free_fan_entry_faces"])} for x in interface_records],
                      "maximum_mixture_fraction": float(occupancy.max())}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output)
