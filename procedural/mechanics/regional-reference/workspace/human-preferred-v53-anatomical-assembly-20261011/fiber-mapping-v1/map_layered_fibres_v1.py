#!/usr/bin/env python3
"""Geometry-pinned, containment-only authored fibre mapping. Never runs a solve.

Run in the project's guarded Python/Blender environment (mathutils + SciPy).
Outputs are NEW files only. Fractions are four-point quadrature estimates of
authored occupancy, not exact clipped volumes, measured anatomy, or histology.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicHermiteSpline

FAMILIES = ("zygomatic_reference", "superficial_oral", "deep_oral",
            "buccinator_deep", "mandibular_depressor")
SOURCE_REGIONS = FAMILIES + ("junction", "zygomatic_passive_adapter")
ALPHA, BETA = .5854101966249685, .1381966011250105
QUADRATURE = np.full((4, 4), BETA) + np.eye(4) * (ALPHA - BETA)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Input pin mismatch: {path}: {actual} != {expected}")
    return actual


def closed_boundary(tets):
    """Extract the complete outward boundary of positive source tetrahedra."""
    faces = np.concatenate([tets[:, [1, 2, 3]], tets[:, [0, 3, 2]],
                            tets[:, [0, 1, 3]], tets[:, [0, 2, 1]]])
    _, first, counts = np.unique(np.sort(faces, axis=1), axis=0,
                                return_index=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("Nonmanifold tetrahedron boundary")
    return faces[first[counts == 1]]


def check_closed(vertices, faces):
    if not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Invalid source closed volume")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Source face index out of bounds")
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, inverse, count = np.unique(np.sort(edges, axis=1), axis=0,
                                 return_inverse=True, return_counts=True)
    winding = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if np.any(count != 2) or np.any(winding != 0) or np.any(np.linalg.norm(cross, axis=1) <= 0):
        raise ValueError("Containment requires a closed, consistently wound, nondegenerate source")
    return {"used_vertices": int(len(np.unique(faces))), "triangles": int(len(faces)),
            "closed_edges": True, "consistent_edge_winding": True}


class ClosedVolume:
    """Three-ray BVH parity, with double-precision winding on uncertain points.

    Outside the actual volume AABB is outside. Surface-near, disagreeing, or
    over-budget ray results are resolved using all original triangles, never
    by the nearest centreline. Nonintegral winding remains ambiguous (-1).
    """
    def __init__(self, name, vertices, faces):
        from mathutils import Vector
        from mathutils.bvhtree import BVHTree
        self.name = name
        self.checks = check_closed(vertices, faces)
        self.offset = vertices[np.unique(faces)].mean(axis=0)
        local = vertices - self.offset
        self.tri = local[faces]
        self.edges1 = self.tri[:, 1] - self.tri[:, 0]
        self.edges2 = self.tri[:, 2] - self.tri[:, 0]
        self.normals = np.cross(self.edges1, self.edges2)
        self.normal_lengths = np.linalg.norm(self.normals, axis=1)
        self.minimum, self.maximum = self.tri.min(axis=(0, 1)), self.tri.max(axis=(0, 1))
        self.diagonal = float(np.linalg.norm(self.maximum - self.minimum))
        self.near_tolerance = max(1e-6, self.diagonal * 3e-7)
        self.step = self.near_tolerance * .5
        self.tree = BVHTree.FromPolygons(local.tolist(), faces.tolist(), all_triangles=True)
        self.directions = [Vector(d).normalized() for d in
                           ((1, .331, -.713), (-.813, .143, .719), (.341, -.823, .119))]
        self.Vector = Vector
        self.stats = {"candidate_queries": 0, "winding_fallbacks": 0,
                      "ambiguous_queries": 0, "ray_disagreements": 0}

    def exact_winding(self, point):
        a, b, c = (self.tri[:, k] - point for k in range(3))
        # A solid-angle sum alone can return either 0 or 1 on a triangle due
        # to signed floating-point zero. Explicitly quarantine exact faces,
        # edges and vertices before using it as an inside/outside predicate.
        exact_tolerance = 128 * np.finfo(float).eps * max(1, self.diagonal)
        on_plane = np.abs(np.einsum("ij,ij->i", a, self.normals)) <= exact_tolerance * self.normal_lengths
        if on_plane.any():
            d, e1, e2, n = -a[on_plane], self.edges1[on_plane], self.edges2[on_plane], self.normals[on_plane]
            nn = np.square(self.normal_lengths[on_plane])
            u = np.einsum("ij,ij->i", np.cross(d, e2), n) / nn
            v = np.einsum("ij,ij->i", np.cross(e1, d), n) / nn
            if ((u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10)).any():
                return -1
        na, nb, nc = (np.linalg.norm(q, axis=1) for q in (a, b, c))
        if min(na.min(), nb.min(), nc.min()) <= np.finfo(float).eps * max(1, self.diagonal):
            return -1
        numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
        denominator = (na * nb * nc + np.einsum("ij,ij->i", a, b) * nc
                       + np.einsum("ij,ij->i", b, c) * na
                       + np.einsum("ij,ij->i", c, a) * nb)
        winding = abs(float(np.arctan2(numerator, denominator).sum() / (2 * np.pi)))
        if not np.isfinite(winding):
            return -1
        if winding < .005:
            return 0
        if abs(winding - 1) < .005:
            return 1
        return -1

    def classify(self, points):
        result = np.zeros(len(points), dtype=np.int8)
        local = points - self.offset
        candidates = np.flatnonzero(((local >= self.minimum - self.near_tolerance)
                                    & (local <= self.maximum + self.near_tolerance)).all(axis=1))
        for index in candidates:
            self.stats["candidate_queries"] += 1
            point = local[index]
            p = self.Vector(point)
            distance = self.tree.find_nearest(p)[3]
            parities = []
            if distance > self.near_tolerance:
                for direction in self.directions:
                    origin = p.copy()
                    for hits in range(64):
                        hit, _, _, _ = self.tree.ray_cast(origin, direction, self.diagonal * 3 + 1)
                        if hit is None:
                            parities.append(hits % 2)
                            break
                        origin = hit + direction * self.step
                    else:
                        parities.append(-1)
            if len(parities) == 3 and parities[0] == parities[1] == parities[2] and parities[0] >= 0:
                result[index] = parities[0]
            else:
                self.stats["winding_fallbacks"] += 1
                self.stats["ray_disagreements"] += int(len(set(parities)) > 1)
                result[index] = self.exact_winding(point)
                self.stats["ambiguous_queries"] += int(result[index] < 0)
        return result


class AuthoredCurve:
    """C1 centreline tangent, projected only AFTER actual volume membership."""
    def __init__(self, points, tangents=None):
        self.points = np.asarray(points, dtype=float)
        self.s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))]
        if len(self.s) < 2 or not np.all(np.diff(self.s) > 1e-12):
            raise ValueError("Repeated or degenerate authored centreline")
        if tangents is None:
            self.curve = PchipInterpolator(self.s, self.points, axis=0)
            self.method = "C1 componentwise PCHIP through authored centreline samples, chord-length parameter"
        else:
            tangent = np.asarray(tangents, dtype=float)
            tangent /= np.linalg.norm(tangent, axis=1)[:, None]
            self.curve = CubicHermiteSpline(self.s, self.points, tangent, axis=0)
            self.method = "C1 cubic Hermite through original ring centres with original normalized ring tangents"
        self.derivative = self.curve.derivative()

    def directions(self, points):
        output = np.empty_like(points, dtype=float)
        parameters = np.empty(len(points), dtype=float)
        # Bounded batches avoid a cells-by-all-source-triangles allocation.
        a, edge = self.points[:-1], np.diff(self.points, axis=0)
        edge2 = np.einsum("ij,ij->i", edge, edge)
        for start in range(0, len(points), 512):
            p = points[start:start + 512]
            delta = p[:, None, :] - a[None, :, :]
            t = np.clip(np.einsum("nsi,si->ns", delta, edge) / edge2, 0, 1)
            distance2 = np.square(delta - t[:, :, None] * edge).sum(axis=2)
            segment = distance2.argmin(axis=1)
            lo = self.s[np.maximum(segment - 1, 0)].copy()
            hi = self.s[np.minimum(segment + 2, len(self.s) - 1)].copy()
            # Golden-section projection on the local three-segment bracket.
            # No route selection is made here; membership has already passed.
            ratio = (np.sqrt(5) - 1) / 2
            x1, x2 = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
            f1 = np.square(self.curve(x1) - p).sum(axis=1)
            f2 = np.square(self.curve(x2) - p).sum(axis=1)
            for _ in range(36):
                left = f1 < f2
                hi = np.where(left, x2, hi)
                lo = np.where(left, lo, x1)
                x1, x2 = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
                f1 = np.square(self.curve(x1) - p).sum(axis=1)
                f2 = np.square(self.curve(x2) - p).sum(axis=1)
            parameter = (lo + hi) * .5
            tangent = self.derivative(parameter)
            length = np.linalg.norm(tangent, axis=1)
            if not np.isfinite(length).all() or (length < 1e-12).any():
                raise ValueError("Degenerate authored tangent")
            output[start:start + len(p)] = tangent / length[:, None]
            parameters[start:start + len(p)] = parameter / self.s[-1]
        return output, parameters


def assign_fractions(membership):
    """Each source status is outside=0, inside=1, or ambiguous=-1.

    Junction and original passive zygomatic adapter override active families.
    An ambiguous source makes the entire sample unresolved, conservatively.
    Otherwise genuine active overlaps share a unit budget equally. No source
    membership is invented. The unresolved fraction is quarantined, not tissue.
    """
    ambiguous = (membership < 0).any(axis=2)
    passive = (membership[:, :, 5:] == 1).any(axis=2) & ~ambiguous
    candidate = (membership[:, :, :5] == 1) & ~ambiguous[:, :, None] & ~passive[:, :, None]
    count = candidate.sum(axis=2)
    sample_weights = candidate / np.maximum(count, 1)[:, :, None]
    missing = (count == 0) & ~passive & ~ambiguous
    return {"sample_weights": sample_weights,
            "family_weights": sample_weights.mean(axis=1),
            "passive_fraction": passive.mean(axis=1),
            "unresolved_fraction": (missing | ambiguous).mean(axis=1),
            "missing": missing, "ambiguous": ambiguous,
            "overlap": count > 1, "passive_samples": passive}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tet", "parts", "zygomatic", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("tet", "parts", "zygomatic"):
        parser.add_argument("--expect-" + name + "-sha256", required=True)
    parser.add_argument("--constraints", type=Path, required=True)
    parser.add_argument("--chunk-tets", type=int, default=1024)
    parser.add_argument("--mesh-status", choices=("diagnostic-only", "admitted-geometry"), default="diagnostic-only")
    parser.add_argument("--mesh-report", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.chunk_tets <= 4096:
        raise ValueError("chunk-tets must be between 1 and 4096")
    if args.output.exists():
        raise FileExistsError("Output already exists; do not replace preserved mapping")
    pins = {name: {"path": str(getattr(args, name)), "sha256": pinned(
        getattr(args, name), getattr(args, "expect_" + name + "_sha256"))}
        for name in ("tet", "parts", "zygomatic")}
    constraints = json.loads(args.constraints.read_text())
    pins["constraints"] = {"path": str(args.constraints), "sha256": sha256(args.constraints)}
    mesh_evidence = {}
    if args.mesh_report is not None:
        mesh_evidence = json.loads(args.mesh_report.read_text())
        if mesh_evidence.get("geometry_sha256") != pins["tet"]["sha256"]:
            raise ValueError("Mesh report is not evidence for the pinned tetrahedral geometry")
        pins["mesh_report"] = {"path": str(args.mesh_report), "sha256": sha256(args.mesh_report)}
    if args.mesh_status == "admitted-geometry" and not mesh_evidence.get("admitted_reference_geometry"):
        raise ValueError("Reference-geometry admission needs the matching admitted mesh report")
    tet, parts, original = (np.load(getattr(args, name), allow_pickle=False)
                            for name in ("tet", "parts", "zygomatic"))
    vertices = tet["vertices"]
    cells = tet["tetrahedra"]
    if cells.ndim != 2 or cells.shape[1] != 4 or cells.min() < 0 or cells.max() >= len(vertices):
        raise ValueError("Invalid final tetrahedral connectivity")
    cell_points = vertices[cells]
    volumes = np.linalg.det(np.stack([cell_points[:, k] - cell_points[:, 0]
                                     for k in (1, 2, 3)], axis=-1)) / 6
    if not np.isfinite(volumes).all() or (volumes <= 0).any():
        raise ValueError("Final tetrahedra must have finite positive signed volumes")
    active_original = original["active_tetrahedra"]
    passive_ids = np.setdiff1d(np.arange(len(original["tetrahedra"])), active_original)
    passive_boundary = closed_boundary(original["tetrahedra"][passive_ids])
    volumes_source = [ClosedVolume(name, parts[name + "_vertices"], parts[name + "_triangles"])
                      for name in SOURCE_REGIONS[:-1]]
    volumes_source.append(ClosedVolume(SOURCE_REGIONS[-1], original["vertices"], passive_boundary))
    curves = [AuthoredCurve(original["ring_centers"], original["ring_tangents"])]
    curves.extend(AuthoredCurve(parts[name + "_centreline"]) for name in FAMILIES[1:])
    membership = np.zeros((len(cells), 4, len(SOURCE_REGIONS)), dtype=np.int8)
    sample_points = np.einsum("qi,tij->tqj", QUADRATURE, cell_points)
    for start in range(0, len(cells), args.chunk_tets):
        stop = min(start + args.chunk_tets, len(cells))
        points = sample_points[start:stop].reshape(-1, 3)
        for k, volume in enumerate(volumes_source):
            membership[start:stop, :, k] = volume.classify(points).reshape(-1, 4)
        print(json.dumps({"classified_tets": stop, "total": len(cells)}), flush=True)
    assigned = assign_fractions(membership)
    weights = assigned["family_weights"]
    direction_fields = np.zeros((len(cells), len(FAMILIES), 3))
    curve_parameters = np.full((len(cells), len(FAMILIES)), np.nan)
    family_records, family_reports = [], {}
    for k, (name, curve) in enumerate(zip(FAMILIES, curves)):
        ids = np.flatnonzero(weights[:, k] > 0)
        if not len(ids):
            raise ValueError(f"No contained active cells for required family {name}")
        w = assigned["sample_weights"][ids, :, k]
        # Project the weighted centroid of that family's confirmed interior
        # samples, rather than the whole element's potentially exterior centre.
        interior_representative = (sample_points[ids] * w[:, :, None]).sum(axis=1) / w.sum(axis=1)[:, None]
        directions, parameters = curve.directions(interior_representative)
        direction_fields[ids, k] = directions
        curve_parameters[ids, k] = parameters
        family_records.append({"id": name, "activationChannel": name, "activation": 0.0,
                               "tetrahedra": ids.tolist(), "volumeFractions": weights[ids, k].tolist(),
                               "fibreDirections": directions.tolist()})
        family_reports[name] = {"tetrahedra": len(ids), "weighted_volume_design_units_cubed":
                                float(volumes @ weights[:, k]), "curve_method": curve.method,
                                "direction_norm_error": float(abs(np.linalg.norm(directions, axis=1) - 1).max()),
                                "parameter_range": [float(parameters.min()), float(parameters.max())]}
    residual = weights.sum(axis=1) + assigned["passive_fraction"] + assigned["unresolved_fraction"] - 1
    if abs(residual).max() > 1e-12:
        raise ValueError("Family + passive + unresolved fractions do not sum to one")
    missing_cells = np.flatnonzero(assigned["missing"].any(axis=1))
    ambiguous_cells = np.flatnonzero(assigned["ambiguous"].any(axis=1))
    mixed_cells = np.flatnonzero((membership != membership[:, :1, :]).any(axis=(1, 2)))
    overlap_cells = np.flatnonzero(assigned["overlap"].any(axis=1))
    quarantine_cells = np.flatnonzero(assigned["unresolved_fraction"] > 0)
    admitted = len(quarantine_cells) == 0
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "mapping.npz", family_names=np.array(FAMILIES),
                        source_region_names=np.array(SOURCE_REGIONS), membership=membership,
                        family_weights=weights, fibre_directions=direction_fields,
                        curve_parameters=curve_parameters, passive_fraction=assigned["passive_fraction"],
                        unresolved_fraction=assigned["unresolved_fraction"],
                        cell_volumes=volumes, missing_cells=missing_cells, ambiguous_cells=ambiguous_cells,
                        mixed_membership_cells=mixed_cells, overlap_cells=overlap_cells,
                        quarantine_cells=quarantine_cells, quadrature_barycentric=QUADRATURE)
    mapping = {"schema": "authored-layered-fibre-mapping-v1", "pins": pins,
               "mesh_status": args.mesh_status,
               "inherited_mesh_limits": mesh_evidence.get("limits", []),
               "family_order": list(FAMILIES), "families": family_records,
               "material_stiffness": "Required explicitly by a future mechanics assembly; not calibrated or assigned here",
               "passive_regions": ["junction", "zygomatic_passive_adapter"],
               "solver_integration": "Use InterdigitatedMuscleFEM after choosing per-family stiffness and clearing all gates. Never pass overlapping records to single-family VolumetricMuscleFEM.",
               "default_activation": 0, "mapping_admitted_no_unresolved_samples": admitted,
               "solve_ready": False}
    (args.output / "families.json").write_text(json.dumps(mapping, separators=(",", ":")))
    report = {"schema": mapping["schema"], "pins": pins, "tetrahedra": len(cells),
              "mesh_status": args.mesh_status,
              "inherited_mesh_quality": mesh_evidence.get("quality"),
              "inherited_mesh_limits": mesh_evidence.get("limits", []),
              "families": family_reports, "source_volume_checks": {
                  q.name: {**q.checks, **q.stats, "near_surface_fallback_tolerance_design_units": q.near_tolerance}
                  for q in volumes_source},
              "quadrature": "Four symmetric interior tetra samples with equal weights; not exact clipped volumes",
              "overlap_policy": "Central junction and original passive root adapter take precedence; each actual nonpassive muscle overlap sample shares a unit budget equally among its contained families",
              "missing_samples": int(assigned["missing"].sum()), "ambiguous_samples": int(assigned["ambiguous"].sum()),
              "missing_cells": len(missing_cells), "ambiguous_cells": len(ambiguous_cells),
              "mixed_membership_cells": len(mixed_cells), "genuine_overlap_cells": len(overlap_cells),
              "quarantine_cells": len(quarantine_cells),
              "missing_cell_examples": missing_cells[:30].tolist(), "ambiguous_cell_examples": ambiguous_cells[:30].tolist(),
              "passive_fraction_weighted_volume_design_units_cubed": float(volumes @ assigned["passive_fraction"]),
              "unresolved_fraction_weighted_volume_design_units_cubed": float(volumes @ assigned["unresolved_fraction"]),
              "maximum_partition_error": float(abs(residual).max()),
              "mapping_admitted_no_unresolved_samples": admitted, "solve_ready": False,
              "mapping_sha256": sha256(args.output / "mapping.npz"),
              "families_sha256": sha256(args.output / "families.json"),
              "source_code_sha256": sha256(__file__),
              "limitations": ["Authored fibre-direction and overlap estimates; no physiological calibration or validated physical units.",
                              "No geometry, head surface, attachment, constitutive law or solved state is changed.",
                              "Finite samples do not certify exact cell/volume intersection or internal interface conformance.",
                              "Mixed cells carry explicit integration-point occupancy estimates; refine or clip interfaces for resolved material boundaries.",
                              "Any unresolved occupancy is quarantined and blocks mechanics assembly, even if some other samples in that cell are assigned.",
                              "Stiffness, activation schedules, boundary conditions, traction transmission, quality/contact and a new solve remain separate gates.",
                              "Source construction has finite unbound oral/posterior cuts; alveolar and raphe attachments remain absent."],
              "anatomical_source_scope": constraints.get("scope")}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("source_volume_checks", "families")}, indent=2), flush=True)
    # Write diagnostics even for a failure; never silently approve the result.
    return 0 if admitted else 2


if __name__ == "__main__":
    # Blender places script arguments after --; plain Python does not.
    cli = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    raise SystemExit(main(cli))
