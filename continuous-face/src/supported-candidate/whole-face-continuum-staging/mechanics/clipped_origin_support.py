"""Source-owned positive-area supports on the actual repaired cut volume.

Each constraint is the exact P1 average on one convex occupied source-triangle
polygon. Its centroid is inside that polygon. Lost source area is disclosed,
not a requirement that every old source vertex/triangle must be recovered.
"""
import argparse
import itertools
import json
from pathlib import Path
import time
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz

from source_transfer import SOURCE, FAMILIES, sha, load_geometry_with_source_fibres
from implicit_dynamics import OriginConstraints
from audit_origin_occupancy import polygon_area


PERMUTATIONS = np.array(list(itertools.permutations(range(3))))
OFFSETS = np.zeros((6, 4, 3))
for i, p in enumerate(PERMUTATIONS):
    OFFSETS[i, 1, p[0]] = 1
    OFFSETS[i, 2, p[:2]] = 1
    OFFSETS[i, 3] = 1


def clip_halfspace(poly, normal, offset):
    if len(poly) < 3:
        return np.empty((0, 3))
    dist = poly@normal+offset
    if np.all(dist >= 0):
        return poly
    if np.all(dist < 0):
        return np.empty((0, 3))
    points = []
    for i in range(len(poly)):
        j = (i+1) % len(poly)
        inside_i, inside_j = dist[i] >= 0, dist[j] >= 0
        if inside_i:
            points.append(poly[i])
        if inside_i != inside_j:
            alpha = dist[i]/(dist[i]-dist[j])
            points.append(poly[i]+alpha*(poly[j]-poly[i]))
    return np.asarray(points).reshape(-1, 3)


def area_centroid(poly):
    if len(poly) < 3:
        return 0., np.full(3, np.nan)
    area = np.linalg.norm(np.cross(poly[1:-1]-poly[0], poly[2:]-poly[0]), axis=1)/2
    total = float(area.sum())
    if total <= 0:
        return 0., np.full(3, np.nan)
    centroid = np.sum(area[:, None]*(poly[0]+poly[1:-1]+poly[2:])/3, axis=0)/total
    return total, centroid


def triangle_barycentric(point, triangle):
    u, v = triangle[1]-triangle[0], triangle[2]-triangle[0]
    gram = np.array([[u@u, u@v], [u@v, v@v]])
    tail = np.linalg.solve(gram, [u@(point-triangle[0]), v@(point-triangle[0])])
    return np.r_[1-tail.sum(), tail]


class CutFineDomain:
    """Positive convex fine-tet cuts, independent of mechanical interpolation."""
    def __init__(self, path):
        self.path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            self.cells = z["fine_cell_ijk"].astype(int)
            self.permutation = z["fine_permutation"].astype(int)
            self.phi = z["fine_node_phi"].astype(float)
            self.parent = z["parent_tet"].astype(int)
            self.origin, self.h = z["geometry_origin"], float(z["geometry_spacing"])
        if not np.isfinite(self.phi).all() or self.phi.shape != (len(self.cells), 4):
            raise ValueError("finite per-fine-tet phi data required")
        self.shape = self.cells.max(0)+1
        self.lookup = np.full(tuple(self.shape)+(6,), -1, np.int32)
        self.lookup[tuple(self.cells.T)+(self.permutation,)] = np.arange(len(self.cells))
        self.unit_inverse = np.array([np.linalg.inv(np.stack([v[1]-v[0], v[2]-v[0], v[3]-v[0]], axis=1)) for v in OFFSETS])/self.h

    def pieces(self, triangle):
        low = np.maximum(0, np.floor((triangle.min(0)-self.origin)/self.h).astype(int))
        high = np.minimum(self.shape-1, np.floor((triangle.max(0)-self.origin)/self.h).astype(int))
        for cell in itertools.product(*(range(low[k], high[k]+1) for k in range(3))):
            for fine in self.lookup[cell]:
                if fine < 0:
                    continue
                perm = self.permutation[fine]
                base = self.origin+self.h*self.cells[fine]
                inv = self.unit_inverse[perm]
                # lambda_1..3 >=0, lambda_0>=0, then the one scalar cut.
                normals = np.r_[inv, -inv.sum(0)[None]]
                offsets = np.r_[-inv@base, 1+inv.sum(0)@base]
                normal_phi = (self.phi[fine, 1:]-self.phi[fine, 0])@inv
                offset_phi = self.phi[fine, 0]-normal_phi@base
                poly = triangle.copy()
                for normal, offset in zip(normals, offsets):
                    poly = clip_halfspace(poly, normal, offset)
                    if len(poly) < 3:
                        break
                if len(poly) >= 3:
                    poly = clip_halfspace(poly, normal_phi, offset_phi)
                    area, centroid = area_centroid(poly)
                    if area > 1e-12:
                        yield int(fine), int(self.parent[fine]), poly, area, centroid


def prepare_clipped_support(solid, cut_path, *, minimum_polygon_area_L0_2=.01, physical_domain=None):
    if not np.isfinite(minimum_polygon_area_L0_2) or minimum_polygon_area_L0_2 <= 0:
        raise ValueError("positive explicit minimum source support area required")
    domain = CutFineDomain(cut_path) if physical_domain is None else physical_domain
    with np.load(SOURCE/"exact-rest/anatomy-registered-rest.npz", allow_pickle=False) as z:
        V, F, P = z["vertices"], z["triangles"], z["part_ids"]
    sample, parent, bary, area, source_part, source_face, source_bary, bone_point, offsets, fine_ids = ([] for _ in range(10))
    polygons, polygon_offsets, polygon_sizes = [], [], []
    patch_reports = []
    face_counter = 0
    for family in FAMILIES:
        path = SOURCE/"source-body"/f"body-{family}.npz"
        with np.load(path, allow_pickle=False) as z:
            X, boundary, parts = z["data__rest_vertices"], z["data__boundary_faces"], z["data__prepared_body_packet__body_part_ids"]
            for side, part in enumerate(parts):
                pre = f"data__prepared_body_packet__origin_{side}_"
                ids, bone_ids, bone_faces = z[pre+"nodes"], z[pre+"bone_global_vertex_ids"], z[pre+"bone_source_face_ids"]
                bone_part = int(z[pre+"bone_part_id"])
                if not np.array_equal(F[bone_faces], bone_ids) or not np.all(P[bone_faces] == bone_part):
                    raise ValueError("source origin bone ownership mismatch")
                bp = np.einsum("ni,nij->nj", z[pre+"barycentric"], V[bone_ids])
                np.testing.assert_allclose(X[ids]-bp, z[pre+"rest_offsets"], atol=1e-10, rtol=0)
                lookup = {int(n): i for i, n in enumerate(ids)}
                faces = boundary[np.isin(boundary, ids).all(1)]
                original_area = retained_area = support_area = 0.
                before = len(sample)
                small_count = 0
                for face in faces:
                    triangle = X[face]
                    original = polygon_area(triangle)
                    original_area += original
                    cut_area = 0.
                    local = np.array([lookup[int(n)] for n in face])
                    for fid, par, poly, aa, centroid in domain.pieces(triangle):
                        cut_area += aa
                        if aa < minimum_polygon_area_L0_2:
                            small_count += 1
                            continue
                        q = centroid*solid.mm_per_L0
                        tail = solid.inv[par]@(q-solid.X[solid.T[par, 0]])
                        bb = np.r_[1-tail.sum(), tail]
                        if bb.min() < -1e-8:
                            raise ValueError("cut support polygon is not in the stated carrier parent")
                        sb = triangle_barycentric(centroid, triangle)
                        if sb.min() < -1e-8:
                            raise ValueError("cut support point left its actual source-origin triangle")
                        anchor = sb@bp[local]
                        sample.append(centroid); parent.append(par); bary.append(bb); area.append(aa)
                        source_part.append(int(part)); source_face.append(face_counter); source_bary.append(sb)
                        bone_point.append(anchor); offsets.append(centroid-anchor); fine_ids.append(fid)
                        polygon_offsets.append(sum(len(p) for p in polygons)); polygon_sizes.append(len(poly)); polygons.append(poly)
                        support_area += aa
                    if cut_area > original+max(1e-8*original, 1e-9):
                        raise ValueError("source-origin polygon area double counted at fine interfaces; no support generated")
                    retained_area += min(cut_area, original)
                    face_counter += 1
                patch_reports.append(dict(family=family, muscle_source_part=int(part), bone_source_part=bone_part,
                    source_body_path=str(path), source_body_sha256=sha(path), original_source_triangles=len(faces),
                    original_area_L0_2=original_area, occupied_clipped_area_L0_2=retained_area,
                    geometric_lost_area_L0_2=original_area-retained_area,
                    retained_constraint_area_L0_2=support_area,
                    small_polygon_excluded_area_L0_2=retained_area-support_area,
                    small_polygon_count=small_count, finite_support_rows=len(sample)-before))
    if not sample:
        raise ValueError("no positive-area source-origin patches remain; no fallback pins")
    sample, parent, bary, area = map(np.asarray, (sample, parent, bary, area))
    C = csr_matrix((bary.ravel(), (np.repeat(np.arange(len(sample)), 4), solid.T[parent].ravel())), shape=(len(sample), len(solid.X)))
    span = sample-sample.mean(0)
    rigid = np.zeros((len(sample)*3, 6))
    for i, p in enumerate(span):
        rigid[3*i:3*i+3, :3] = np.eye(3)
        rigid[3*i:3*i+3, 3:] = np.array([[0, p[2], -p[1]], [-p[2], 0, p[0]], [p[1], -p[0], 0]])
    rigid_rank = int(np.linalg.matrix_rank(rigid))
    if rigid_rank != 6:
        raise ValueError("retained source support patches do not constrain all6 rigid modes")
    provenance = dict(source_origin_only=True, finite_material_attachment_qualified=True,
        qualification_scope="positive-area physical receiving patches and source ownership, not native collision clearance",
        constraint_kind="exact affine area-average over each convex occupied source-origin polygon",
        receiving_points_are_in_actual_cut_polygons=True, no_ghost_point_pinning=True,
        physical_integration_domain=getattr(domain, 'description', 'positive affine-phi clipped fine tetrahedra'),
        cut_fine_geometry_path=str(Path(cut_path).resolve()), cut_fine_geometry_sha256=sha(cut_path),
        domain_path=solid.provenance["domain_path"], domain_sha256=solid.provenance["domain_sha256"],
        source_atlas_sha256=sha(SOURCE/"exact-rest/anatomy-registered-rest.npz"),
        minimum_polygon_area_L0_2=minimum_polygon_area_L0_2,
        source_patch_records=patch_reports, support_rows=len(sample),
        affected_parent_tetrahedra=int(len(np.unique(parent))), rigid_motion_rank=rigid_rank,
        physical_source_origin_offsets_retained=True,
        bone_anchor_mapping="barycentric interpolation of verified source-origin bone recipients on same rigid bone part",
        bonding_assumption="idealized rigid source-origin bonding with retained affine rest offset; no new finite connector volume is invented",
        connector_volume_created=False, connector_collision_certified=False,
        no_exterior_targets=True, no_source_vertex_snapping=True,
        lost_area_is_disclosed_not_an_absolute_veto=True)
    constraints = OriginConstraints(C, solid.X, provenance=provenance)
    source = dict(provenance)
    report = dict(status="occupied clipped source-origin constraint candidate; no facial motion",
        finite_material_attachment_qualified=True, constraint_source=source,
        rank_audit={k: v for k, v in constraints.report.items() if k != "source"},
        source_patch_records=patch_reports, accepted_facial_frames=0,
        native_contact_qualification_not_implied=True)
    arrays = dict(receiving_points_L0=sample, parent_tet=parent, parent_barycentric=bary,
        polygon_area_L0_2=area, source_part=np.asarray(source_part), source_origin_triangle_index=np.asarray(source_face),
        source_triangle_barycentric=np.asarray(source_bary), bone_anchor_affine_points_L0=np.asarray(bone_point),
        idealized_rest_offsets_L0=np.asarray(offsets), fine_tet_index=np.asarray(fine_ids),
        polygon_vertices_L0=np.concatenate(polygons), polygon_vertex_offset=np.asarray(polygon_offsets), polygon_vertex_count=np.asarray(polygon_sizes))
    arrays["constraint_selected_polygon_index"] = constraints.selected_source_rows
    return constraints, arrays, report


def load_support_bundle(solid, directory):
    """Load only matching geometry evidence and its actual occupied sample map."""
    directory = Path(directory)
    report = json.loads((directory/"clipped-origin-support.json").read_text())
    source = report["constraint_source"]
    if report.get("finite_material_attachment_qualified") is not True or source.get("domain_sha256") != solid.provenance.get("domain_sha256"):
        raise ValueError("support evidence does not qualify this exact physical domain")
    if sha(source["cut_fine_geometry_path"]) != source["cut_fine_geometry_sha256"]:
        raise ValueError("underlying finite cut geometry changed; re-audit source supports")
    C = load_npz(directory/"constraint-matrix.npz").tocsr()
    with np.load(directory/"clipped-origin-support.npz", allow_pickle=False) as z:
        chosen = z["constraint_selected_polygon_index"]
        parent, bary = z["parent_tet"][chosen], z["parent_barycentric"][chosen]
        points = z["receiving_points_L0"][chosen]*solid.mm_per_L0
        area = z["polygon_area_L0_2"][chosen]
    if np.any(area < source["minimum_polygon_area_L0_2"]) or np.any(bary.min(1) < -1e-8):
        raise ValueError("invalid occupied support quadrature/area")
    expected = csr_matrix((bary.ravel(), (np.repeat(np.arange(len(chosen)), 4), solid.T[parent].ravel())), shape=C.shape)
    difference = C-expected
    if np.max(np.abs(difference.data), initial=0) > 1e-12 or not np.allclose(C@solid.X, points, atol=1e-8, rtol=0):
        raise ValueError("constraint matrix differs from certified occupied polygon samples")
    return OriginConstraints(C, solid.X, provenance=source), report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=Path, required=True)
    parser.add_argument("--phases", type=Path, required=True)
    parser.add_argument("--fine-cut", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-area", type=float, default=.01)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output directory required to preserve previous audit")
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    solid = load_geometry_with_source_fibres(args.domain, args.phases)
    constraints, arrays, report = prepare_clipped_support(solid, args.fine_cut, minimum_polygon_area_L0_2=args.minimum_area)
    np.savez_compressed(args.output/"clipped-origin-support.npz", **arrays)
    save_npz(args.output/"constraint-matrix.npz", constraints.C)
    report["seconds"] = time.perf_counter()-started
    (args.output/"clipped-origin-support.json").write_text(json.dumps(report, indent=2)+"\n")
    (args.output/"domain-mechanics-audit.json").write_text(json.dumps(solid.audit(), indent=2)+"\n")
    print(json.dumps({"rank": report["rank_audit"], "seconds": report["seconds"], "accepted_facial_frames": 0}, indent=2))


if __name__ == "__main__":
    main()
