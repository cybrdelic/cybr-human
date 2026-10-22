"""Approximate exterior support queries on the common anatomical assembly.

No preferred-skin positional target or local anatomy warp is used. The only
transform is the caller-visible rigid translation (default Z -4.5). An
explicit body allowlist rejects internal atlas structures and attachment
patches. Fixed outward-normal and angular-depth filters retain approximate
external cortex / superficial muscle / nasal cartilage candidates. The actual
Gingiva contributes only explicitly selected labial/buccal arch walls, never
its palatal walls, top/bottom caps, posterior ends or small detached fragment.

This is NOT a watertight union, exact visibility test, signed distance field,
or collision certificate. Centroid candidates can miss a large triangle;
the radial visibility approximation can discard concave exterior regions.
It is suitable for limited head-envelope support, not oral/nostril cavities,
ears, neck, shoulders, or lids. Globes/lids require their separate constraint.
Unmatched query rows are explicitly invalid and exert no implied support.
"""

from pathlib import Path
import json

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


BONES = (
    "Zygomatic_bone", "Temporal_bone", "Parietal_bone", "Occipital_bone",
    "Frontal_bone", "Nasal_bone", "Mandible_bone", "Maxilla_bone",
)
MUSCLES = (
    "Corrugator supercilii", "Depressor anguli oris", "Depressor labii inferioris",
    "Frontalis muscle", "Levator labii superioris", "Levator nasolabialis",
    "Mentalis muscle", "Nasalis muscle", "Occipitalis muscle",
    "Orbicularis oris muscle", "Orbital part of orbicularis oculi",
    "Procerus muscle", "Risorius muscle", "Superficial part of masseter",
    "Temporalis muscle", "Temporoparietalis muscle",
    "Zygomaticus major muscle", "Zygomaticus minor muscle",
)
CARTILAGES = ("Lateral process of nasal septal cartilage", "Major alar cartilage")
CONNECTIVE = ("Epicranial aponeurosis", "Masseteric fascia", "Superficial layer of temporal fascia", "Parotid gland")
KIND_NAMES = ("bone", "muscle", "cartilage", "connective", "gum")


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-20)


def _external_gingiva_domain(vertices, faces):
    """Select real buccal/labial walls of the two principal gingival arches.

    This is an explicit anatomical face-domain rule, not a skin fit. Native
    positions and winding stay unchanged. Horizontal outwardness is measured
    from each actual arch's area-weighted XZ center; superior/inferior caps and
    backward-facing ends are excluded. Inward palatal/lingual walls therefore
    cannot become exterior support merely because their atlas body is closed.
    """
    ids, inverse = np.unique(faces, return_inverse=True)
    local = inverse.reshape(-1, 3)
    edges = np.concatenate([local[:, [0, 1]], local[:, [1, 2]], local[:, [2, 0]]])
    count, component = connected_components(coo_matrix((np.ones(len(edges)),
        (edges[:, 0], edges[:, 1])), shape=(len(ids), len(ids))), directed=False)
    sizes = np.bincount(component)
    tri = vertices[faces]
    centers = tri.mean(1)
    cross = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    normal = _unit(cross)
    face_component = component[local[:, 0]]
    # Physical area, not vertex count: the corrected lower arch is sampled
    # much more densely than the original upper arch. Mesh density must not
    # make an anatomically large upper gum disappear from contact support.
    component_area=np.bincount(face_component,weights=.5*area2,minlength=count)
    principal=np.argsort(component_area)[-min(2,count):]
    principal=principal[component_area[principal]>=.05*component_area.max()]
    keep = np.zeros(len(faces), bool)
    arch_info = []
    for arch in principal:
        rows = np.flatnonzero(face_component == arch)
        center = np.average(centers[rows], axis=0, weights=np.maximum(area2[rows], 1e-20))
        radial_xz = _unit(centers[rows][:, [0, 2]] - center[[0, 2]])
        outward = np.sum(normal[rows][:, [0, 2]] * radial_xz, axis=1)
        selected = (outward >= .20) & (abs(normal[rows, 1]) <= .85) & (normal[rows, 2] >= -.25) & (area2[rows] > 1e-12)
        keep[rows[selected]] = True
        arch_info.append({'component': int(arch), 'source_vertices': int(sizes[arch]),
            'physical_surface_area':float(component_area[arch]),
            'source_faces': int(len(rows)), 'selected_wall_faces': int(selected.sum()),
            'actual_area_center': center.tolist()})
    return keep, {'source_faces': len(faces), 'source_components': int(count),
        'selected_wall_faces': int(keep.sum()), 'arches': arch_info,
        'rule': 'major active arches selected by physical surface area, independent of mesh sampling density; outward horizontal normal dot arch radial >= .20; abs(normal Y) <= .85; normal Z >= -.25; standard head outward/visibility gates still apply',
        'geometry': 'input active Gingiva triangles unchanged by this selector; may be original or explicitly corrected anatomy; no skin positions or offsets',
        'limits': 'face-domain selection, not a solid gum union; internal palatal/lingual walls, axial caps and posterior-facing ends intentionally excluded'}


def _candidate_points(points, triangle_candidates):
    """Exact closest point on each supplied candidate, not global search."""
    a, b, c = (triangle_candidates[:, :, i] for i in range(3))
    p = points[:, None, :]
    ab, ac, ap = b - a, c - a, p - a
    d00, d01, d11 = (np.sum(a * b, axis=-1) for a, b in ((ab, ab), (ab, ac), (ac, ac)))
    d20, d21 = np.sum(ap * ab, axis=-1), np.sum(ap * ac, axis=-1)
    den = d00 * d11 - d01 * d01
    u = np.divide(d11 * d20 - d01 * d21, den, out=np.zeros_like(den), where=den > 1e-20)
    v = np.divide(d00 * d21 - d01 * d20, den, out=np.zeros_like(den), where=den > 1e-20)
    bary = np.stack([1 - u - v, u, v], axis=-1)
    q = a + u[:, :, None] * ab + v[:, :, None] * ac
    distance2 = np.sum((q - p) ** 2, axis=-1)
    distance2[(bary.min(axis=-1) < 0) | (den <= 1e-20)] = np.inf
    for ia, ib in ((0, 1), (1, 2), (2, 0)):
        start, end = triangle_candidates[:, :, ia], triangle_candidates[:, :, ib]
        edge = end - start
        t = np.clip(np.sum((p - start) * edge, axis=-1) /
                    np.maximum(np.sum(edge * edge, axis=-1), 1e-20), 0, 1)
        candidate = start + t[:, :, None] * edge
        d2 = np.sum((candidate - p) ** 2, axis=-1)
        take = d2 < distance2
        q[take], distance2[take] = candidate[take], d2[take]
        edge_bary = np.zeros_like(bary)
        edge_bary[:, :, ia], edge_bary[:, :, ib] = 1 - t, t
        bary[take] = edge_bary[take]
    return q, distance2, bary


class ExteriorSupport:
    """Fixed candidate mesh; query current positions and current normals.

    query() returns Q, normal, kind (0 bone / 1 muscle / 2 cartilage /
    3 connective / 4 externally selected gum), valid,
    distance, signed_gap, face_id, part_id, and barycentric coordinates.
    Invalid rows have Q equal to the input, zero normal, kind/ids -1, and
    infinite distance. signed_gap is a local plane distance, NOT inside/outside.
    Negative gaps are allowed: skin starting inside a support can be pushed out.
    """

    def __init__(self, packet, meta=None, *, already_registered=False,
                 translation=(0.0, 0.0, -4.5),
                 radial_center=(0.0, 20.0, 20.0), outward_gate=0.05,
                 angular_radius=0.035, depth_tolerance=2.5,
                 visibility_candidates=48, include_superficial_connective=False,
                 include_external_gingiva=True):
        if already_registered:
            translation = (0.0, 0.0, 0.0)
        if isinstance(packet, (str, Path)):
            path = Path(packet)
            self.manifest = json.loads(path.with_suffix(".json").read_text()) if meta is None else meta
            with np.load(path, allow_pickle=False) as a:
                packet = {k: a[k] for k in ("vertices", "triangles", "part_ids")}
        else:
            if meta is None:
                raise ValueError("An in-memory anatomy packet requires its part metadata")
            self.manifest = meta
        self.vertices = np.array(packet["vertices"], dtype=float) + np.asarray(translation)
        self.faces = np.asarray(packet["triangles"], dtype=np.int64)
        part_ids = np.asarray(packet["part_ids"], dtype=np.int64)
        self.radial_center = np.asarray(radial_center, dtype=float)
        parts = self.manifest["parts"]
        part_kind = np.full(len(parts), -1, dtype=np.int8)
        for i, part in enumerate(parts):
            name, kind = part["name"], part["kind"]
            allowed = {"bone": BONES, "muscle": MUSCLES, "cartilage": CARTILAGES, "connective": CONNECTIVE if include_superficial_connective else ()}.get(kind, ())
            if any(name == prefix or name.startswith(prefix + ".") for prefix in allowed):
                part_kind[i] = KIND_NAMES.index(kind)
            if include_external_gingiva and name == 'Gingiva' and kind == 'gum':
                part_kind[i] = KIND_NAMES.index('gum')
        allowed_faces = np.flatnonzero(part_kind[part_ids] >= 0)
        tri = self.vertices[self.faces[allowed_faces]]
        centers = tri.mean(axis=1)
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normal = _unit(cross)
        radii = np.linalg.norm(centers - self.radial_center, axis=1)
        directions = _unit(centers - self.radial_center)
        # Nasal lateral faces need a local midline direction; their normals
        # can be nearly perpendicular to a head-center radial vector.
        radial = directions.copy()
        nasal = np.array([parts[i]["kind"] == "cartilage" or
                          parts[i]["name"].startswith("Nasal_bone") for i in part_ids[allowed_faces]])
        radial[nasal] = _unit(centers[nasal] - np.array([0.0, -12.0, 112.0]))
        outward = (np.sum(normal * radial, axis=1) >= outward_gate) & (np.linalg.norm(cross, axis=1) > 1e-12)
        gingiva_rows = np.flatnonzero(part_kind[part_ids[allowed_faces]] == KIND_NAMES.index('gum'))
        gingiva_info = None
        if len(gingiva_rows):
            gum_keep, gingiva_info = _external_gingiva_domain(self.vertices, self.faces[allowed_faces[gingiva_rows]])
            outward[gingiva_rows] &= gum_keep
        visible = np.ones(len(tri), dtype=bool)
        if angular_radius > 0:
            # Approximate occlusion only among already outward-facing source
            # faces, never against a preferred rendered skin reference.
            candidates = np.flatnonzero(outward)
            direction_tree = cKDTree(directions[candidates])
            for start in range(0, len(candidates), 4096):
                ids = candidates[start:start + 4096]
                distance, local = direction_tree.query(directions[ids],
                    k=min(visibility_candidates, len(candidates)))
                if local.ndim == 1:
                    local, distance = local[:, None], distance[:, None]
                sampled_radius = radii[candidates[local]].copy()
                sampled_radius[distance > angular_radius] = -np.inf
                visible[ids] = radii[ids] >= sampled_radius.max(axis=1) - depth_tolerance
        keep = outward & visible
        self.face_ids = allowed_faces[keep]
        self.triangles = tri[keep]
        self.centers = centers[keep]
        self.areas = 0.5 * np.linalg.norm(cross[keep], axis=1)
        self.normals = normal[keep]
        # Source parts have disjoint vertex ranges. Accumulate within those
        # original surfaces, not across overlapping anatomical bodies.
        vertex_normal = np.zeros_like(self.vertices)
        for corner in range(3):
            np.add.at(vertex_normal, self.faces[allowed_faces, corner], cross)
        self.vertex_normals = _unit(vertex_normal)
        self.triangle_vertex_normals = self.vertex_normals[self.faces[self.face_ids]]
        self.part_ids = part_ids[self.face_ids]
        self.kind = part_kind[self.part_ids]
        self.tree = cKDTree(self.centers)
        self.stats = dict(source_faces=len(self.faces), allowed_faces=len(allowed_faces),
                          outward_faces=int(outward.sum()), retained_faces=len(self.face_ids),
                          translation=list(map(float, translation)), include_superficial_connective=bool(include_superficial_connective),
                          include_external_gingiva=bool(include_external_gingiva),
                          selected_parts=[p["name"] for p, k in zip(parts, part_kind) if k >= 0],
                          face_counts_by_kind={k: int(np.sum(self.kind == i)) for i, k in enumerate(KIND_NAMES)},
                          limits="Approximate centroid visibility and candidate nearest-triangle query; no global union/contact guarantee")
        self.summary = self.stats
        self.summary["retained_area"] = float(self.areas.sum())
        if gingiva_info is not None:
            gingiva_info['retained_wall_faces'] = int(np.sum(self.kind == KIND_NAMES.index('gum')))
            self.summary['external_gingiva_domain'] = gingiva_info

    def query(self, points, normals=None, *, candidates=64, normal_gate=0.15,
              max_distance=40.0, chunk=1024):
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
            raise ValueError("Expected finite current positions with shape (n, 3)")
        n = _unit(p - self.radial_center) if normals is None else _unit(np.asarray(normals, dtype=float))
        if n.shape != p.shape or not np.isfinite(n).all():
            raise ValueError("Current query normals must match positions")
        if candidates < 1 or chunk < 1 or max_distance <= 0 or not -1 <= normal_gate <= 1:
            raise ValueError("Invalid candidate/query limit")
        size = len(p)
        out = dict(Q=p.copy(), normal=np.zeros_like(p), kind=np.full(size, -1, dtype=np.int8),
                   valid=np.zeros(size, dtype=bool), distance=np.full(size, np.inf),
                   signed_gap=np.full(size, np.nan), face_id=np.full(size, -1, dtype=np.int64),
                   part_id=np.full(size, -1, dtype=np.int64), barycentric=np.zeros_like(p))
        k = min(candidates, len(self.triangles))
        for start in range(0, size, chunk):
            end = min(start + chunk, size)
            ids = self.tree.query(p[start:end], k=k)[1]
            if ids.ndim == 1:
                ids = ids[:, None]
            q, d2, bary = _candidate_points(p[start:end], self.triangles[ids])
            candidate_normals = _unit(np.einsum("nki,nkij->nkj", bary, self.triangle_vertex_normals[ids]))
            bad_normal = np.sum(candidate_normals * self.normals[ids], axis=-1) < 0.05
            candidate_normals[bad_normal] = self.normals[ids][bad_normal]
            compatible = np.sum(candidate_normals * n[start:end, None, :], axis=-1) >= normal_gate
            d2[~compatible] = np.inf
            best = np.argmin(d2, axis=1)
            row = np.arange(end - start)
            valid = d2[row, best] <= max_distance ** 2
            destination = start + row[valid]
            chosen = ids[row[valid], best[valid]]
            out["valid"][destination] = True
            out["Q"][destination] = q[row[valid], best[valid]]
            out["normal"][destination] = candidate_normals[row[valid], best[valid]]
            out["kind"][destination] = self.kind[chosen]
            out["distance"][destination] = np.sqrt(d2[row[valid], best[valid]])
            out["signed_gap"][destination] = np.sum((p[destination] - out["Q"][destination]) * out["normal"][destination], axis=1)
            out["face_id"][destination] = self.face_ids[chosen]
            out["part_id"][destination] = self.part_ids[chosen]
            out["barycentric"][destination] = bary[row[valid], best[valid]]
        out["points"], out["normals"] = out["Q"], out["normal"]
        return out
