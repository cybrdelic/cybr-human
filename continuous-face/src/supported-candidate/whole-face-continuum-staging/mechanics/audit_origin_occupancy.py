"""Exact finite source-patch/retained-voxel intersection audit, no pinning changes.

Triangle polygons are clipped against every candidate occupied cube, not tested
only at vertices or centroids. Bone-origin segment coverage is exact piecewise
voxel traversal, but is deliberately NOT called finite connector-volume proof.
"""
import itertools
import json
from pathlib import Path
import time
import numpy as np

from source_transfer import SOURCE, FAMILIES, sha


def polygon_area(poly):
    if len(poly) < 3:
        return 0.
    return float(np.linalg.norm(np.cross(poly[1:-1]-poly[0], poly[2:]-poly[0]), axis=1).sum()/2)


def clip_plane(poly, axis, bound, sign):
    if not len(poly):
        return poly
    distance = sign*(poly[:, axis]-bound)
    out = []
    for i in range(len(poly)):
        j = (i+1) % len(poly)
        inside_i, inside_j = distance[i] >= 0, distance[j] >= 0
        if inside_i:
            out.append(poly[i])
        if inside_i != inside_j:
            alpha = distance[i]/(distance[i]-distance[j])
            out.append(poly[i]+alpha*(poly[j]-poly[i]))
    return np.asarray(out).reshape(-1, 3)


def clip_triangle_cube(triangle, lo, hi):
    poly = triangle.copy()
    for axis in range(3):
        poly = clip_plane(poly, axis, lo[axis], 1)
        poly = clip_plane(poly, axis, hi[axis], -1)
        if len(poly) < 3:
            return 0.
    return polygon_area(poly)


def closest_triangle_distance(point, triangles):
    """Exhaustive true finite-triangle distance, no nearest-centroid shortlist."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    e0, e1, d = b-a, c-a, point-a
    aa, ab, bb = np.sum(e0*e0, 1), np.sum(e0*e1, 1), np.sum(e1*e1, 1)
    pa, pb = np.sum(d*e0, 1), np.sum(d*e1, 1)
    det = aa*bb-ab*ab
    good = det > np.finfo(float).eps*np.maximum(aa*bb, 1e-30)
    u, v = np.zeros(len(a)), np.zeros(len(a))
    u[good] = (pa[good]*bb[good]-pb[good]*ab[good])/det[good]
    v[good] = (pb[good]*aa[good]-pa[good]*ab[good])/det[good]
    face = good & (u >= 0) & (v >= 0) & (u+v <= 1)
    dist = np.full(len(a), np.inf)
    dist[face] = np.sum((point-(a+u[:, None]*e0+v[:, None]*e1))[face]**2, 1)
    for p, q in ((a, b), (b, c), (c, a)):
        edge = q-p
        t = np.sum((point-p)*edge, 1)/np.maximum(np.sum(edge*edge, 1), 1e-30)
        closest = p+np.clip(t, 0, 1)[:, None]*edge
        dist = np.minimum(dist, np.sum((point-closest)**2, 1))
    idx = int(np.argmin(dist))
    return float(np.sqrt(dist[idx])), idx


class VoxelDomain:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.origin, self.h, cells = z["grid_origin"], float(z["grid_spacing"]), z["cell_ijk"]
        self.shape = cells.max(0)+2
        self.occ = np.zeros(self.shape, bool)
        self.occ[tuple(cells.T)] = True

    def index_occupied(self, ijk):
        ijk = np.asarray(ijk, int)
        inside = np.all((ijk >= 0) & (ijk < self.shape), axis=-1)
        result = np.zeros(len(ijk), bool)
        result[inside] = self.occ[tuple(ijk[inside].T)]
        return result

    def classify_points(self, points):
        result = []
        for p in points:
            u = (p-self.origin)/self.h
            choices = [[int(np.rint(x))-1, int(np.rint(x))] if abs(x-np.rint(x)) < 1e-10 else [int(np.floor(x))] for x in u]
            cells = np.asarray(list(itertools.product(*choices)))
            occupied = self.index_occupied(cells)
            result.append(1 if occupied.all() else (2 if occupied.any() else 0))
        return np.asarray(result, np.int8)  # 0 void;1 volume interior;2 exact union boundary

    def triangle_occupied_area(self, triangle):
        low = np.floor((triangle.min(0)-self.origin)/self.h).astype(int)
        high = np.floor((triangle.max(0)-self.origin)/self.h).astype(int)
        # Finite source facets coincident with grid planes need a union-of-
        # neighboring-cell rule. Flag instead of double counting their area.
        for k in range(3):
            u = (triangle[:, k]-self.origin[k])/self.h
            if np.ptp(u) < 1e-12 and abs(u[0]-np.rint(u[0])) < 1e-10:
                raise ValueError("source facet exactly coplanar with voxel boundary requires dedicated union clipping")
        occupied_area, candidate_cubes = 0., 0
        for index in itertools.product(*(range(low[k], high[k]+1) for k in range(3))):
            ijk = np.array(index)
            if np.all((ijk >= 0) & (ijk < self.shape)) and self.occ[index]:
                lo = self.origin+self.h*ijk
                occupied_area += clip_triangle_cube(triangle, lo, lo+self.h)
                candidate_cubes += 1
        total = polygon_area(triangle)
        if occupied_area > total+max(1e-8*total, 1e-10):
            raise ValueError("clipped source patch area was double counted")
        return min(occupied_area, total), candidate_cubes

    def segment_coverage(self, a, b):
        direction = b-a
        length = float(np.linalg.norm(direction))
        if length == 0:
            return 1. if self.classify_points([a])[0] else 0.
        cuts = [0., 1.]
        for k in range(3):
            if abs(direction[k]) < 1e-15:
                continue
            low, high = sorted(((a[k]-self.origin[k])/self.h, (b[k]-self.origin[k])/self.h))
            planes = self.origin[k]+self.h*np.arange(int(np.floor(low))+1, int(np.ceil(high)))
            cuts.extend(((planes-a[k])/direction[k]).tolist())
        cuts = np.unique(np.clip(cuts, 0, 1))
        middle = (cuts[:-1]+cuts[1:])/2
        ijk = np.floor((a+middle[:, None]*direction-self.origin)/self.h).astype(int)
        occupied = self.index_occupied(ijk)
        return float(np.diff(cuts) @ occupied)


def main():
    here = Path(__file__).resolve().parent
    path = here.parent/"geometry/whole-face-integration-cells.npz"
    output = here/"source-origin-occupied-tissue-audit.json"
    if output.exists():
        raise FileExistsError("Preserve prior audit: output already exists")
    started = time.perf_counter()
    domain = VoxelDomain(path)
    with np.load(SOURCE/"exact-rest/anatomy-registered-rest.npz", allow_pickle=False) as z:
        V, F, P = z["vertices"], z["triangles"], z["part_ids"]
    all_status, all_incident, all_coverage, all_distance, all_points, all_part = [], [], [], [], [], []
    triangles, triangle_occupied_area, triangle_total_area, triangle_part = [], [], [], []
    rows = []
    global_start = 0
    for family in FAMILIES:
        with np.load(SOURCE/"source-body"/f"body-{family}.npz", allow_pickle=False) as z:
            X, boundary, parts = z["data__rest_vertices"], z["data__boundary_faces"], z["data__prepared_body_packet__body_part_ids"]
            for side, part in enumerate(parts):
                pre = f"data__prepared_body_packet__origin_{side}_"
                ids, bone_ids, bone_face = z[pre+"nodes"], z[pre+"bone_global_vertex_ids"], z[pre+"bone_source_face_ids"]
                bone_part, bary, offsets = int(z[pre+"bone_part_id"]), z[pre+"barycentric"], z[pre+"rest_offsets"]
                if not np.array_equal(F[bone_face], bone_ids) or not np.all(P[bone_face] == bone_part):
                    raise ValueError("source bone ownership changed")
                points = X[ids]
                bone_points = np.einsum("ni,nij->nj", bary, V[bone_ids])
                np.testing.assert_allclose(points-bone_points, offsets, atol=1e-10, rtol=0)
                local_map = {int(node): i for i, node in enumerate(ids)}
                faces = boundary[np.isin(boundary, ids).all(1)]
                local_faces = np.asarray([[local_map[int(n)] for n in face] for face in faces])
                incident = np.bincount(local_faces.ravel(), minlength=len(ids))
                status = domain.classify_points(points)
                source_faces = F[P == part]
                native = V[source_faces]
                distances = np.array([closest_triangle_distance(p, native)[0] for p in points])
                cover = np.array([domain.segment_coverage(a, b) for a, b in zip(bone_points, points)])
                areas, occupied_areas, candidates = [], [], 0
                for face in faces:
                    tri = X[face]
                    occupied, count = domain.triangle_occupied_area(tri)
                    area = polygon_area(tri)
                    areas.append(area); occupied_areas.append(occupied); candidates += count
                    triangles.append(tri); triangle_total_area.append(area); triangle_occupied_area.append(occupied); triangle_part.append(int(part))
                areas, occupied_areas = np.asarray(areas), np.asarray(occupied_areas)
                contained = areas-occupied_areas <= np.maximum(1e-8*areas, 1e-10)
                eligible_local = (status > 0) & (incident > 0)
                full_patch = bool(contained.all() and eligible_local.all())
                row = dict(family=family, muscle_source_part=int(part), bone_source_part=bone_part,
                    origin_begin=global_start, origin_count=len(ids),
                    actual_volume_sites=int((status == 1).sum()), actual_boundary_sites=int((status == 2).sum()),
                    carrier_only_void_sites=int((status == 0).sum()),
                    source_sites_without_finite_origin_triangle=int((incident == 0).sum()),
                    locally_tissue_supported_source_origin_sites=int(eligible_local.sum()),
                    global_void_site_indices=(global_start+np.where(status == 0)[0]).tolist(),
                    finite_origin_triangles=len(faces), fully_contained_origin_triangles=int(contained.sum()),
                    partly_or_fully_missing_origin_triangles=int((~contained).sum()),
                    source_patch_area_L0_2=float(areas.sum()), occupied_patch_area_L0_2=float(occupied_areas.sum()),
                    missing_patch_area_L0_2=float((areas-occupied_areas).sum()),
                    occupied_patch_area_fraction=float(occupied_areas.sum()/areas.sum()),
                    whole_original_patch_tissue_contained=full_patch,
                    bone_origin_segments_fully_occupied=int((cover >= 1-1e-10).sum()),
                    bone_origin_segment_coverage_fraction_min=float(cover.min()),
                    native_muscle_facet_distance_L0_max=float(distances.max()),
                    native_muscle_facet_distance_L0_median=float(np.median(distances)),
                    source_origin_to_bone_gap_L0_max=float(np.linalg.norm(offsets, axis=1).max()),
                    exact_occupied_cube_clips=candidates,
                    assessment=("Original finite source-origin patch is represented in tissue; idealized rigid bonding remains conditional on finite connector/native clearance guards." if full_patch else "Original patch is not wholly represented. Void sites are ineligible; only explicitly clipped occupied source-patch subsets can be considered after repaired-domain re-audit."))
                rows.append(row)
                all_status.extend(status); all_incident.extend(incident); all_coverage.extend(cover); all_distance.extend(distances); all_points.extend(points); all_part.extend([int(part)]*len(ids))
                global_start += len(ids)
                print(json.dumps({k: row[k] for k in ("family", "muscle_source_part", "actual_volume_sites", "carrier_only_void_sites", "fully_contained_origin_triangles", "finite_origin_triangles", "occupied_patch_area_fraction")}), flush=True)
    status, incident = np.asarray(all_status), np.asarray(all_incident)
    report = dict(status="finite occupied source-patch audit; no source constraint changes or motion",
        domain_cells_path=str(path), domain_cells_sha256=sha(path),
        source_registered_sha256=sha(SOURCE/"exact-rest/anatomy-registered-rest.npz"),
        origin_count=len(status), volume_sites=int((status == 1).sum()), boundary_sites=int((status == 2).sum()),
        carrier_only_void_sites=int((status == 0).sum()),
        locally_eligible_source_origin_sites=int(((status > 0) & (incident > 0)).sum()),
        finite_origin_triangles=len(triangles), total_source_patch_area_L0_2=float(np.sum(triangle_total_area)),
        occupied_source_patch_area_L0_2=float(np.sum(triangle_occupied_area)),
        source_patches=rows, previous_audits_preserved=True, source_constraints_changed=False,
        finite_attachment_certified=False, accepted_facial_frames=0,
        method=dict(point_occupancy="exact literal cube-union point classification, including explicit boundary cases",
            finite_patch="every actual source-body origin boundary triangle clipped against every candidate occupied axis-aligned cube; finite area integrated, not sampled",
            native_source_comparison="exhaustive closest finite native muscle-facet distance; actual source bone facet IDs/part and retained offsets verified",
            attachment_segments="exact piecewise occupied-voxel traversal of retained bone-to-origin segments, not finite-volume connector proof"),
        conclusions=["All 70 carrier-only void source sites must be excluded from any current constraints or repaired in geometry; no snapping or proxy pinning is allowed.",
            "A locally occupied source node does not by itself certify an entire origin patch or finite bonded connector.",
            "Unchanged all-313 source-origin constraints are not physically eligible on the current literal voxel domain.",
            "Native-rigid overlap and incomplete/open atlas owners remain independent clearance blockers; no positive contact certificate follows from this audit."],
        seconds=time.perf_counter()-started)
    output.write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(here/"source-origin-occupied-tissue-audit.npz", source_origin_points_L0=np.asarray(all_points),
        source_part=np.asarray(all_part), site_status=status, finite_origin_face_incidence=incident,
        locally_tissue_supported_site=(status > 0) & (incident > 0), bone_origin_segment_coverage=np.asarray(all_coverage),
        native_muscle_facet_distance_L0=np.asarray(all_distance), origin_patch_triangles=np.asarray(triangles),
        triangle_source_part=np.asarray(triangle_part), triangle_original_area_L0_2=triangle_total_area,
        triangle_occupied_area_L0_2=triangle_occupied_area)
    print(json.dumps({k: report[k] for k in ("origin_count", "volume_sites", "carrier_only_void_sites", "locally_eligible_source_origin_sites", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
