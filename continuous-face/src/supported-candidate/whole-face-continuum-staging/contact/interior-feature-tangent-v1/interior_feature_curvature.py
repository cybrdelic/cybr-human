"""Numerical tangent only for stable interior finite closest features.

Reuses the unchanged native radius/AABB enumeration and finite-triangle witness
routine. Not a new distance law, geometry engine, force, or collision guard.
Exterior features and unresolved/transition/multiple-witness cases get no added
curvature. A shared edge is counted once per existing physical site/owner term.
"""
from collections import Counter
import time
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


def _feature(native, point, face, witness, normal, tolerance):
    """Classify a finite witness, with strict Voronoi margins for edge/vertex."""
    if not native.resolved_faces[face]:
        return None, 'unresolved_triangle'
    tri = native.triangles[face]
    ab, ac = tri[1]-tri[0], tri[2]-tri[0]
    gram = np.array([[ab@ab, ab@ac], [ab@ac, ac@ac]])
    try:
        uv = np.linalg.solve(gram, np.array([ab@(witness-tri[0]), ac@(witness-tri[0])]))
    except np.linalg.LinAlgError:
        return None, 'unresolved_barycentric'
    if not np.isfinite(uv).all():
        return None, 'unresolved_barycentric'
    bary = np.r_[1-uv.sum(), uv]
    altitude = np.linalg.norm(np.cross(ab, ac))/max(np.linalg.norm(ab), np.linalg.norm(ac), np.linalg.norm(tri[2]-tri[1]))
    btol = max(1e-10, tolerance/max(altitude, np.finfo(float).tiny))
    if btol >= .01 or np.min(bary) < -btol:
        return None, 'unresolved_barycentric'
    positive = np.flatnonzero(bary > btol)
    zeros = np.flatnonzero(abs(bary) <= btol)
    if len(positive)+len(zeros) != 3:
        return None, 'unresolved_barycentric'
    if len(positive) == 3:
        return dict(kind='face', key=('face',), basis=np.empty((0, 3)), barycentric=bary), None
    residual = point-witness
    if len(positive) == 2:
        a, b = tri[positive]; edge = b-a; length = np.linalg.norm(edge)
        if length <= tolerance:
            return None, 'unresolved_edge'
        e = edge/length
        # Exact projected seam/face transitions are not stable edge regions.
        inward = tri[zeros[0]]-witness
        inward -= (inward@e)*e
        ilen = np.linalg.norm(inward)
        if ilen <= tolerance or residual@(inward/ilen) >= -tolerance:
            return None, 'edge_face_transition'
        if abs(normal@e) > 1e-8 or abs(np.linalg.norm(normal)-1) > 1e-8:
            return None, 'inconsistent_edge_normal'
        tangent = np.cross(e, normal); tangent /= np.linalg.norm(tangent)
        key = ('edge', *sorted(int(x) for x in native.faces[face][positive]))
        return dict(kind='edge', key=key, basis=tangent[None], edge_unit=e, barycentric=bary), None
    if len(positive) == 1:
        a = tri[positive[0]]
        for j in zeros:
            edge = tri[j]-a; length = np.linalg.norm(edge)
            if length <= tolerance or residual@(edge/length) >= -tolerance:
                return None, 'vertex_edge_transition'
        if abs(np.linalg.norm(normal)-1) > 1e-8:
            return None, 'inconsistent_vertex_normal'
        axis = np.eye(3)[np.argmin(abs(normal))]
        t1 = np.cross(normal, axis); t1 /= np.linalg.norm(t1)
        t2 = np.cross(normal, t1); t2 /= np.linalg.norm(t2)
        key = ('vertex', int(native.faces[face, positive[0]]))
        return dict(kind='vertex', key=key, basis=np.array([t1, t2]), barycentric=bary), None
    return None, 'unresolved_feature'


def interior_feature_data(guard, state):
    """Recover ALL potentially tied closest faces for each interior owner row.

    Radius = existing owner distance + numerical pad, never nearest-k. Every
    candidate in that radius is evaluated by the actual native finite routine.
    The unchanged winning distance/normal must be reproduced exactly.
    """
    started = time.perf_counter()
    if not state['valid'].all() or not np.isfinite(state['energy']):
        raise ValueError('Finite feasible unchanged contact state required')
    native = getattr(guard.rigid, 'native', guard.rigid)
    selected = np.flatnonzero(state['active'] & (state['force_gap'] < 0))
    sites = state['force_ids'][selected]; owners = state['force_owner'][selected]
    distances = -state['force_gap'][selected]
    points = state['points'][sites]
    scale = np.max(abs(native.vertices))
    pads = 2048*np.finfo(float).eps*(np.linalg.norm(points, axis=1)+scale+guard.h+1)
    candidates = [[] for _ in selected]
    for start in range(0, len(selected), 256):
        x = points[start:start+256]; distance = distances[start:start+256]; pad = pads[start:start+256]
        for face_ids, tree, radius in native.distance_bins:
            choices = tree.query_ball_point(x, radius+distance+pad)
            count = np.fromiter(map(len, choices), int, len(x))
            if not count.sum():
                continue
            pi = np.repeat(np.arange(len(x)), count)
            fi = face_ids[np.concatenate(choices).astype(int)]
            keep = native.face_owner[fi] == owners[start+pi]
            pi, fi = pi[keep], fi[keep]
            for begin in range(0, len(pi), 16384):
                ii, jj = pi[begin:begin+16384], fi[begin:begin+16384]
                tri = native.triangles[jj]
                delta = np.maximum(np.maximum(tri.min(1)-x[ii], x[ii]-tri.max(1)), 0.)
                keep = np.sum(delta**2, axis=1) <= (distance[ii]+pad[ii])**2
                ii, jj = ii[keep], jj[keep]
                if not len(ii):
                    continue
                witness, d2 = native._finite_witness(x[ii], jj)
                dd = np.sqrt(d2)
                keep = dd <= distance[ii]+pad[ii]
                for i, f, w, d in zip(ii[keep], jj[keep], witness[keep], dd[keep]):
                    candidates[start+i].append((float(d), int(f), w))
    rows = []; skipped = Counter(); face_count = 0
    for group, force_row in enumerate(selected):
        if not native.owners[int(owners[group])]['closed']:
            skipped['not_closed_owner'] += 1; continue
        p, d, pad = points[group], distances[group], pads[group]
        if d <= pad:
            skipped['zero_or_unresolved_distance'] += 1; continue
        choices = sorted(candidates[group], key=lambda x:(x[0], x[1]))
        if not choices:
            skipped['missing_complete_witness'] += 1; continue
        dd, face, witness = choices[0]
        normal = state['normal'][force_row]
        if dd != d or not np.array_equal(-(p-witness)/d, normal):
            skipped['winner_not_exactly_reproduced'] += 1; continue
        tied = [c for c in choices if c[0] <= d+pad]
        if any(np.linalg.norm(c[2]-witness) > pad for c in tied):
            skipped['ambiguous_different_closest_points'] += 1; continue
        features = []; reason = None
        for _, f, w in tied:
            feature, reason = _feature(native, p, f, w, normal, pad)
            if reason:
                break
            features.append(feature)
        if reason:
            skipped[reason] += 1; continue
        if len({x['key'] for x in features}) != 1:
            skipped['ambiguous_different_features'] += 1; continue
        feature = features[0]
        if feature['kind'] == 'face':
            face_count += 1; continue
        # R already contains physical site volume in BOTH penalty and barrier.
        # Assembly below uses only carrier basis weights, never volume again.
        reaction = guard.k*guard.weights[sites[group]]*(guard.h-state['force_gap'][force_row])+state['barrier']['reaction'][force_row]
        rows.append(dict(force_row=int(force_row), site=int(sites[group]), owner=int(owners[group]),
            gap_mm=float(-d), reaction_N=float(reaction), stiffness_N_per_mm=float(reaction/d),
            closest_point_mm=witness.copy(), normal=normal.copy(),
            tied_faces=np.array([c[1] for c in tied], int), **feature))
    return dict(rows=rows, report=dict(interior_owner_terms=len(selected), stable_edge_terms=sum(r['kind']=='edge' for r in rows),
        stable_vertex_terms=sum(r['kind']=='vertex' for r in rows), stable_face_terms_zero_addition=face_count,
        skipped_ambiguous_or_unresolved=dict(skipped), complete_finite_witness_enumeration=True,
        exterior_added_terms=0, site_volume_applied_once=True, wall_seconds=time.perf_counter()-started))


def assemble_interior_curvature(guard, data):
    """Sum C_i^T [(R/d) P_tangent] C_i using PSD factored sparse rows."""
    rows = data['rows']
    if not rows:
        return csr_matrix((3*len(guard.solid.X),)*2)
    sites=[]; vectors=[]
    for row in rows:
        for tangent in row['basis']:
            sites.append(row['site']); vectors.append(np.sqrt(row['stiffness_N_per_mm'])*tangent)
    A = guard.C[np.asarray(sites, int)].tocoo(); vectors = np.asarray(vectors)
    r = np.repeat(A.row, 3); c = (3*A.col[:,None]+np.arange(3)).ravel()
    values = (A.data[:,None]*vectors[A.row]).ravel()
    B = coo_matrix((values, (r,c)), shape=(len(sites),3*len(guard.solid.X))).tocsr()
    return (B.T@B).tocsr()
