"""Read verified registered source rest only; never accepted neutral coordinates."""
from pathlib import Path
import hashlib
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix

from continuum import B, CutContinuum, PHASE_NAMES, MUSCLE_COLUMN
from implicit_dynamics import OriginConstraints

SOURCE = B / "full-face-coverage-plan" / "restored"
FAMILIES = ("frontalis", "corrugator", "depressor", "zygomaticus", "orbicularis")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fibres():
    """Index exact source fibre directions by the actual bilateral part IDs."""
    parts = {}
    for family in FAMILIES:
        path = SOURCE / "source-body" / f"body-{family}.npz"
        with np.load(path, allow_pickle=False) as z:
            X, T = z["data__rest_vertices"], z["data__tetrahedra"]
            ft, ff, fd = z["data__fibre_tets"], z["data__fibre_family"], z["data__fibre_directions"]
            part_ids = z["data__prepared_body_packet__body_part_ids"]
            for side, part in enumerate(part_ids):
                selected = ff == side
                if not np.any(selected):
                    continue
                centers = X[T[ft[selected]]].mean(1)
                parts[int(part)] = dict(family=family, source_tet=ft[selected], centers=centers,
                    fibre=fd[selected], tree=cKDTree(centers), path=str(path), sha256=sha(path))
    return parts


def load_geometry_with_source_fibres(domain_path, phase_path, *, mm_per_L0=1.):
    """Split integration weights, not mechanical nodes, for source muscle phases.

    Generic/unverified muscle sources remain passive. Verified fibre transfer
    uses the nearest source rest tetrahedron centroid within the SAME native
    source part. This is a declared coarse constitutive transfer approximation,
    not an assertion that the agglomerated centroid lies inside that muscle.
    """
    with np.load(domain_path, allow_pickle=False) as z:
        domain = {k: z[k] for k in ("nodes", "tets", "quadrature_points", "quadrature_weights",
            "quadrature_tet", "quadrature_barycentric", "quadrature_material_fraction")}
    with np.load(phase_path, allow_pickle=False) as z:
        row_q = z["muscle_quadrature_index"].astype(int)
        row_part = z["muscle_source_part"].astype(int)
        row_fraction = z["muscle_fraction"].astype(float)
        row_points = z["muscle_source_centroid"] if "muscle_source_centroid" in z else domain["quadrature_points"][row_q]
        centroid_kind = "source-phase volume centroid" if "muscle_source_centroid" in z else "common parent physical centroid"
    n = len(domain["quadrature_weights"])
    fractions = np.asarray(domain.pop("quadrature_material_fraction"), float)
    if fractions.shape != (n, len(PHASE_NAMES)) or np.any(row_q < 0) or np.any(row_q >= n) or np.any(row_fraction < 0):
        raise ValueError("invalid phase packet")
    source_total = np.bincount(row_q, row_fraction, minlength=n)
    partition_error = float(np.max(np.abs(source_total-fractions[:, MUSCLE_COLUMN]), initial=0))
    if partition_error > 1e-10:
        raise ValueError(f"source muscle shares do not conserve muscle phase: {partition_error}")
    sources = source_fibres()
    verified = np.isin(row_part, list(sources))
    active_total = np.bincount(row_q[verified], row_fraction[verified], minlength=n)
    # One residual site holds all passive phases; one pure-muscle site per
    # verified source owns its exact share. Their volumes sum to original w.
    remainder = fractions.copy()
    remainder[:, MUSCLE_COLUMN] -= active_total
    if remainder.min() < -1e-12:
        raise ValueError("verified active source phase exceeds physical muscle volume")
    remainder = np.maximum(remainder, 0)
    remainder_volume = remainder.sum(1)
    passive = remainder_volume > 0
    active_rows = np.where(verified & (row_fraction > 0))[0]
    pq = row_q[active_rows]
    qindex = np.r_[np.where(passive)[0], pq]
    split = np.r_[remainder_volume[passive], row_fraction[active_rows]]
    new_fraction = np.zeros((len(qindex), len(PHASE_NAMES)))
    new_fraction[:passive.sum()] = remainder[passive]/remainder_volume[passive, None]
    new_fraction[passive.sum():, MUSCLE_COLUMN] = 1.
    family_names = tuple(f for f in FAMILIES if any(sources[int(p)]["family"] == f for p in np.unique(row_part[verified])))
    family = np.full(len(qindex), -1, int)
    fibre = np.zeros((len(qindex), 3))
    mapped_tet = np.full(len(qindex), -1, int)
    new_source = np.full(len(qindex), -1, int)
    reports = []
    for part in np.unique(row_part[active_rows]):
        select = np.where(row_part[active_rows] == part)[0]
        rows = active_rows[select]
        src = sources[int(part)]
        distance, nearest = src["tree"].query(row_points[rows])
        target = passive.sum()+select
        family[target] = family_names.index(src["family"])
        fibre[target] = src["fibre"][nearest]
        mapped_tet[target] = src["source_tet"][nearest]
        new_source[target] = part
        reports.append(dict(source_part=int(part), family=src["family"], rows=len(rows),
            physical_volume_L0_3=float(np.sum(domain["quadrature_weights"][row_q[rows]]*row_fraction[rows])),
            nearest_source_centroid_distance_L0_max=float(distance.max()),
            nearest_source_centroid_distance_L0_p95=float(np.percentile(distance, 95)),
            path=src["path"], sha256=src["sha256"]))
    arrays = {k: domain[k] for k in ("nodes", "tets")}
    for name in ("quadrature_points", "quadrature_tet", "quadrature_barycentric"):
        arrays[name] = domain[name][qindex]
    arrays["quadrature_weights"] = domain["quadrature_weights"][qindex]*split
    arrays.update(material_fractions=new_fraction, quadrature_muscle_family=family,
        family_names=family_names, quadrature_fibre=fibre, source_part=new_source)
    original_volume = float(np.sum(domain["quadrature_weights"]))
    error = float(arrays["quadrature_weights"].sum()-original_volume)
    if abs(error) > 1e-8*max(1., original_volume):
        raise ValueError("integration splitting changed physical volume")
    provenance = dict(domain_path=str(Path(domain_path).resolve()), domain_sha256=sha(domain_path),
        source_phase_path=str(Path(phase_path).resolve()), source_phase_sha256=sha(phase_path),
        source_registered_anatomy_sha256=sha(SOURCE / "exact-rest" / "anatomy-registered-rest.npz"),
        source_phase_partition_error=partition_error, integration_split_volume_error_L0_3=error,
        fibre_transfer="nearest source rest tetrahedron centroid, restricted to exact same native muscle part; no direction fitting",
        fibre_query_position=centroid_kind, fibre_transfer_records=reports,
        unverified_muscle_parts_passive=[int(p) for p in np.unique(row_part[~verified])],
        registration_reapplied=False, accepted_or_pending_regional_coordinates_loaded=False)
    solid = CutContinuum(**arrays, mm_per_L0=mm_per_L0, provenance=provenance)
    solid.source_tet_per_integration_site = mapped_tet
    solid.source_geometry_quadrature_index = qindex
    return solid


def locate_points_in_carrier(points, X, T, *, tolerance=1e-10):
    xyz = X[T]
    lo, hi = xyz.min(1), xyz.max(1)
    inv = np.linalg.inv(np.stack([xyz[:, j]-xyz[:, 0] for j in (1, 2, 3)], axis=2))
    parents = np.full(len(points), -1, int)
    bary = np.full((len(points), 4), np.nan)
    for i, p in enumerate(points):
        candidates = np.where(np.all((p >= lo-tolerance) & (p <= hi+tolerance), axis=1))[0]
        tail = np.einsum("tij,tj->ti", inv[candidates], p-xyz[candidates, 0])
        b = np.c_[1-tail.sum(1), tail]
        valid = np.where(b.min(1) >= -tolerance)[0]
        if len(valid):
            j = valid[np.argmax(b[valid].min(1))]
            parents[i], bary[i] = candidates[j], b[j]
    return parents, bary


def source_origin_candidate(solid):
    """Return provenance-rich candidates; no projection/snap of missing origins.

    Completeness of source geometry ownership is verified here. Finite occupied
    cut-domain support/attachment containment is NOT certified by the carrier
    locator, so this candidate alone cannot authorize actual facial motion.
    """
    with np.load(SOURCE / "exact-rest" / "anatomy-registered-rest.npz", allow_pickle=False) as a:
        V, F, P = a["vertices"], a["triangles"], a["part_ids"]
    points, bone_points, offsets, part_ids, rows = [], [], [], [], []
    for family in FAMILIES:
        path = SOURCE / "source-body" / f"body-{family}.npz"
        with np.load(path, allow_pickle=False) as z:
            X, boundary = z["data__rest_vertices"], z["data__boundary_faces"]
            muscle_parts = z["data__prepared_body_packet__body_part_ids"]
            for side, muscle_part in enumerate(muscle_parts):
                prefix = f"data__prepared_body_packet__origin_{side}_"
                ids, bone, bone_faces = z[prefix+"nodes"], z[prefix+"bone_global_vertex_ids"], z[prefix+"bone_source_face_ids"]
                bone_part, bary, off = int(z[prefix+"bone_part_id"]), z[prefix+"barycentric"], z[prefix+"rest_offsets"]
                if not np.array_equal(F[bone_faces], bone) or not np.all(P[bone_faces] == bone_part):
                    raise ValueError("source-origin bone ownership mismatch")
                bp = np.einsum("ni,nij->nj", bary, V[bone])
                error = float(np.max(np.linalg.norm(X[ids]-bp-off, axis=1)))
                if error > 1e-10:
                    raise ValueError("source origin does not reconstruct from original bone plus retained rest offset")
                patch = boundary[np.isin(boundary, ids).all(1)]
                if not len(patch):
                    raise ValueError("origin set has no complete finite source surface triangles")
                area = np.linalg.norm(np.cross(X[patch[:, 1]]-X[patch[:, 0]], X[patch[:, 2]]-X[patch[:, 0]]), axis=1).sum()/2
                start = sum(len(p) for p in points)
                gap = np.linalg.norm(off, axis=1)
                rows.append(dict(family=family, muscle_source_part=int(muscle_part), bone_source_part=bone_part,
                    origin_begin=start, origin_count=len(ids), finite_source_patch_triangles=len(patch),
                    finite_source_patch_area_L0_2=float(area), retained_origin_bone_gap_L0_min=float(gap.min()),
                    retained_origin_bone_gap_L0_max=float(gap.max()), rest_reconstruction_error_L0=error,
                    assumption="existing origin-to-bone rest offset is idealized rigid bonded connection, not shifted geometry",
                    body_path=str(path), body_sha256=sha(path)))
                points.append(X[ids]); bone_points.append(bp); offsets.append(off); part_ids.append(np.full(len(ids), muscle_part))
    points, bone_points, offsets, part_ids = map(np.concatenate, (points, bone_points, offsets, part_ids))
    parent, bary = locate_points_in_carrier(points*solid.mm_per_L0, solid.X, solid.T)
    valid = parent >= 0
    report = dict(source_origin_only=True, status="candidate only, finite occupied-support/attachment containment pending",
        total_origin_sites=len(points), mapped_origin_sites=int(valid.sum()), unmapped_origin_sites=int((~valid).sum()),
        missing_origins_not_snapped_or_silently_dropped=True, source_origin_patches=rows,
        affected_carrier_tetrahedra=int(len(np.unique(parent[valid]))),
        actual_finite_cut_support_verified=False, origin_bone_source_geometry_verified=True,
        coarse_interpolation_caveat="multiple source points can constrain much of a coarse parent; rank and affected cells reported",
        broad_periosteal_support=False, exterior_targets=False)
    constraints = None
    if valid.all():
        C = csr_matrix((bary.ravel(), (np.repeat(np.arange(len(points)), 4), solid.T[parent].ravel())), shape=(len(points), len(solid.X)))
        constraints = OriginConstraints(C, solid.X, provenance=dict(report))
        report["constraint_rank_audit"] = constraints.report
    arrays = dict(source_origin_points_L0=points, source_bone_points_L0=bone_points,
        retained_source_offsets_L0=offsets, muscle_source_part=part_ids,
        parent_tet=parent, parent_barycentric=bary)
    return constraints, arrays, report
